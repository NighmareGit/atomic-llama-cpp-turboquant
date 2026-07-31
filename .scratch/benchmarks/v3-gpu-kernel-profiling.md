# V3 — GPU Kernel Profiling: 2-GPU Layer-Split Analysis

**Date:** 2026-07-24  
**Build:** 83605ae27 (10202)  
**Status:** COMPLETE (7900 XTX profiled; 5060 Ti blocked by ncu permissions)

## Profiling Method

- **7900 XTX (ROCm):** `rocprof --hsa-trace`, 4-token decode (tg4=15.65 with profiling overhead)
- **5060 Ti (CUDA):** `ncu --set full --target-processes all` → **BLOCKED** — `ERR_NVGPUCTRPERM` (requires root on remus)
- **Scheduler trace:** `GGML_SCHED_TRACE=1`, 32-token decode (tg32=7.18 with trace overhead)

## 7900 XTX Kernel Breakdown

Total GPU kernel time: ~40.6ms for 4 tokens ≈ **10.2ms/token**

| Kernel Category | % of GPU Time | Calls (4 tokens) | Avg Duration | Notes |
|----------------|---------------|------------------|-------------|-------|
| Q6_K matmul (main weights) | 27.0% | 615 | 17.8 µs | Dominant operation — weight × activation |
| Dequantization (q8_1) | 8.1% | 1140 | 2.9 µs | Pre-matmul overhead |
| GPU mem copies (rocclr) | 6.5% | 323 | 8.2 µs | H2D/D2D copies |
| MoE experts (Q4_K/Q8_0 matmul) | 12.3% | 290 | 15-25 µs | Expert FFN matmuls |
| MoE routing/topk | 3.6% | 130 | 11.1 µs | Expert selection |
| float matmul | 3.5% | 260 | 5.5 µs | fp32 matmuls |
| k_get_rows | 4.7% | 200 | 9.6 µs | KV cache lookups |
| rms_norm | 2.9% | 260 | 4.6 µs | Layer norms |
| All others (rope, bin ops, etc.) | ~31.4% | — | <4 µs | Scattered small kernels |

## Scheduler Trace Analysis (GGML_SCHED_TRACE=1)

For one decode step (trace_id=33), the 7900 XTX split:
- **compute_us:** 100,586 µs (100.6 ms)
- **idle_us:** 92,913 µs (92.9 ms)
- **split_total:** 193,520 µs (193.5 ms)

The idle time (48%) is the time between `t_split_start` and `t_compute_start` — input copying including:
- GET_TENSOR from 5060 Ti (previous token's layer output)
- Cross-backend COPY operations

Split 2 (5060 Ti) shows only `sync_copy_fallback` operations (311 µs for `l_out-25` transfer), no compute nodes. This suggests the 5060 Ti's compute happens in a separate split not captured in this trace window, or the 5060 Ti's compute is tiny relative to the 7900's.

**Layer distribution (from trace):** Split 1 (backend 0, 7900) handles layers 0-25 (26 layers). Split 2 (backend 1, 5060 Ti) handles layers 26-27 (2 layers via RPC). The 7900 carries 93% of the layer compute.

## Throughput Decomposition

| Component | Time/token | Source |
|-----------|-----------|--------|
| 7900 GPU compute | ~10.2 ms | rocprof total kernel time ÷ 4 tokens |
| RPC overhead (GET_TENSOR + copies) | ~2.2 ms | tg time - GPU compute |
| **Total** | **~12.4 ms** | 80.89 tg64 |

## Bottleneck Analysis

The 2-GPU layer-split has a severe **compute imbalance**: the 7900 XTX handles 26/28 layers (93%) while the 5060 Ti handles only 2/28 layers (7%). This isn't scaling — it's offloading a tiny fraction of work across the network.

**Why the imbalance?** The 35B Q6_K model requires ~22 GiB VRAM. The 7900 XTX has 24 GiB and can fit nearly the entire model. The 5060 Ti adds only 16 GiB but isn't needed for capacity — the auto layer split puts minimal layers on it because the 7900 can absorb most.

**Is this bad?** Actually, the throughput at 80.89 t/s is excellent (76% of the 7900's single-GPU baseline of 106.84 t/s). The 2.2ms RPC overhead per token is modest — mainly the GET_TENSOR for the layer-25→26 boundary crossing plus sync.

## Layer-Split Tuning Potential

The rocprof data suggests the per-layer GPU time on the 7900 is:
- Per-layer compute: ~10.2ms / 26 layers ≈ 0.39ms/layer
- The 5060 Ti's per-layer time with CUDA is likely similar or slightly slower (memory bandwidth limited: 288 vs 960 GB/s)

**Optimization options:**
1. **More balanced split:** Shift 4-6 layers to the 5060 Ti (e.g., 20/8 split). This would reduce 7900 compute to ~7.8ms and increase 5060 Ti compute to ~3.1ms. But the GET_TENSOR overhead would still apply, so net benefit is modest.
2. **Split at natural boundary:** Place the split at layer 13 (midpoint) for ~13/15 or 15/13 distribution. Requires forcing the split via `--tensor-split` or similar parameter.
3. **No split → single GPU:** For the 35B model that fits on one GPU, the best throughput is single-GPU (106.84 t/s). Multi-GPU only helps for models that truly don't fit.

## Key Insight

For **models that fit on a single GPU**, multi-GPU layer-split adds overhead without benefit. The 35B model at 80.89 t/s on 2 GPUs vs 106.84 t/s on 1 GPU shows a 24% penalty. Multi-GPU scaling is only beneficial for:
- Models too large for any single GPU (e.g., 80B MoE at 56.8 GiB → V6)
- Row-split parallelism (where compute is truly parallel, not sequential)

## Next Steps

- [x] 7900 XTX kernel profile complete
- [x] 5060 Ti kernel profile attempted — **BLOCKED**: ncu 2022.4.1.0 only supports up to Ada Lovelace (sm_89). RTX 5060 Ti is Blackwell (sm_120). Requires ncu 2025.x+.
- [ ] Experiment with forced layer-split ratios (20/8, 15/13) — low priority since 2-GPU is inherently imbalanced
- [x] V6 5-GPU Mixtral stress test — BLOCKED by Docker 3060 Ti NVML mismatch (requires reboot)
- [ ] V5 MTP self-speculation — can amortize RPC overhead with 2-3 tokens/decode

## 5060 Ti Profiling Attempt (2026-07-24)

**Method:** `sudo ncu --set full -o /tmp/ncu-5060-sudo ./bin/rpc-server --host 0.0.0.0 --port 50051`
**Result:** `==ERROR== Profiling is not supported on device 0.`
**Root cause:** ncu version 2022.4.1.0 supports chips: ad102-ad107, ga100-ga107, gv100, tu102-tu117. The RTX 5060 Ti (Blackwell, sm_120) is unsupported.
**Fix:** Upgrade ncu to 2025.x or later on remus. Alternatively, use `nsys` (NVIDIA Nsight Systems) for timeline profiling.

## References

- rocprof data: `/tmp/v3-traces/rocprof-7900.stats.csv`
- Scheduler trace: `/tmp/v3-traces/sched-trace-output.txt`
- NCU attempt: `ERR_NVGPUCTRPERM` — needs `sudo` on remus
