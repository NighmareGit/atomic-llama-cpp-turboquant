# V6 — 5-GPU 80B MoE + Mixtral Final Attempt (2026-07-24)

**Date:** 2026-07-24  
**Build:** 83605ae27 (10202)  
**Status:** 5-GPU 80B COMPLETE; 5-GPU Mixtral BLOCKED (VRAM, not Docker)

## Precondition: All 5 GPU Servers Running

| GPU | RPC Address | Status |
|-----|------------|--------|
| 7900 XTX (ROCm) | local HIP_VISIBLE_DEVICES=0 | ✅ |
| RTX 3060 Ti (Docker) | 127.0.0.1:50051 | ✅ (NVML fixed via `nvidia-persistenced` restart) |
| RTX 5060 Ti (remus) | 192.168.8.22:50051 | ✅ |
| RTX 3090 (triton) | 192.168.8.23:50051 | ✅ |
| RTX 3070 (triton) | 192.168.8.23:50052 | ✅ |

**Prior blocker resolved:** Docker 3060 Ti NVML driver/library mismatch fixed by `sudo systemctl restart nvidia-persistenced` (no reboot needed).

## 5-GPU 80B MoE (7900 XTX + 3090 + 5060 Ti + 3070 + 3060 Ti)

| Config | tg64 (t/s) | vs 3-GPU (84.21) | vs 4-GPU (71.81) |
|--------|-----------|-------------------|-------------------|
| 3-GPU (7900+3090+5060) | **84.21** | baseline | — |
| 4-GPU (+3070) | **71.81** | -14.7% | baseline |
| 5-GPU (+3060 Ti) | **70.19** | -16.7% | **-2.3%** |

**Analysis:** Near-flat from 4→5 GPU. The 3060 Ti (8 GiB, weakest GPU in the fleet) handles very few layers (~1-2) due to its tiny VRAM share. The additional GET_TENSOR call (~2ms UDP) nearly perfectly offsets the already-minimal compute benefit of distributing to yet another weak GPU.

**Scaling curve is asymptotically flat after 3 GPUs.** The 3-GPU sweet spot holds — it's the minimum count that fits the 80B model with comfortable VRAM headroom.

## 5-GPU Mixtral 8×22B (72 GiB IQ4_XS) — BLOCKED (VRAM)

**Model:** `/mnt/980pro/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS.gguf` (72 GiB)

### Attempt 1: 5-GPU layer-split (all GPUs)

```
E alloc_tensor_range: failed to allocate RPC0[127.0.0.1:50051] buffer of size 8057880576
```

**Root cause:** The layer-split allocator distributes ~10% of model layers to the RTX 3060 Ti (8 GiB / 80 GiB total). For a 72 GiB model, this means ~7.2 GiB of model weights + CUDA context overhead = >8 GiB. The 3060 Ti cannot fit its proportional share.

### Attempt 2: 4-GPU (without 3060 Ti)

Failed silently — model requires >72 GiB total, 4-GPU aggregate (7900+5060+3090+3070 = ~72.4 GiB) has insufficient headroom for CUDA context, scratch buffers, and KV cache.

### Attempt 3: 5-GPU + --no-kv-offload

Flag not supported by llama-bench.

### Conclusion

Mixtral 8×22B (72 GiB) **requires a GPU with ≥12 GiB VRAM in every slot** for layer-split to work. Our hardware has two 8 GiB GPUs (3060 Ti, 3070) that cannot serve as layer-split targets for a 72 GiB model. The model would need a different distribution strategy (expert offload, hybrid split) to work on this hardware.

## Completed Throughput Scaling Pattern

| GPU Count | Model | tg64 (t/s) | GET_TENSOR/token | Δ from best |
|-----------|-------|-----------|-------------------|-------------|
| 1 (7900) | 35B | 106.84 | 0 | baseline |
| 2 (7900+5060) | 35B | 80.89 | 1 | -24.3% |
| 3 (7900+3090+5060) | 80B | **84.21** | 2 | **sweet spot** |
| 4 (+3070) | 80B | 71.81 | 3 | -14.7% vs 3-GPU |
| 5 (+3060 Ti) | 80B | 70.19 | 4 | -16.7% vs 3-GPU |

**Pattern confirmed across all 5 GPU counts:** Layer-split throughput peaks at the minimum GPU count that fits the model. Each additional GPU adds GET_TENSOR overhead (~2ms/token with UDP) that exceeds the compute benefit of distributing layers to a weak GPU. The curve flattens asymptotically — 4→5 GPU shows only -2.3% regression, indicating the overhead has saturated.

## Implications

1. **3-GPU sweet spot is definitive.** No benefit to 4+ GPU for the 80B model.
2. **Mixtral can't be layer-split on this hardware.** The 8 GiB GPUs are too small. Row-split (V0) or expert-aware distribution would be needed.
3. **For single-request throughput, fewer strong GPUs > more weak GPUs.** This is the inverse of throughput-oriented multi-user serving.
4. **V0 row-split remains the only path to positive scaling beyond minimum GPU count**, since it avoids the per-GPU GET_TENSOR tax.

## References

- `v6-80b-udp.md`: 3-GPU 80B with UDP (84.21 t/s)
- `v6-4gpu-80b-mixtral.md`: 4-GPU 80B + initial Mixtral attempt
- `v6-layer-split-80b.md`: Original 3-GPU 80B (21.16 t/s, no UDP)
