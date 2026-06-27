# RPC Bug Hunt Results (36B NL MoE)

Instrumentation-led diagnosis for Qwen3.6-35B-A3B-UD-IQ4_NL_XL GPU underutilization on Config F. Complements [`PROFILING.md`](PROFILING.md) (hardware rule-out) and [`RPC-WAIT-MAP.md`](RPC-WAIT-MAP.md) (static wait-site map).

## Verdict

**Not a hardware bottleneck.** The client scheduler runs graph splits **serially**; each RPC device adds a TCP-bound stage. The RX6600 third hop (`:50052`) adds ~10 ms/token of critical-path time while holding only 2.1 GB weights.

Path B events are active (`sched copies = 4`) but overlap **tokens**, not **splits within one token**. Measured split times sum to observed token period on 3-device F.

**Immediate ops fix:** drop `:50052` for this model (`ts=50,50`, 2-device F). **Code fix target:** reduce cross-RPC `COPY_TENSOR` / drain stalls and/or collapse remus GPUs into one RPC backend (Path C).

## Trace matrix (2026-06-27)

Built with `GGML_RPC_TRACE=1`, `GGML_SCHED_TRACE=1` (rebuilt `build-cuda-b-bin/portable/`). 128 gen tokens, `q4_0` KV, `-ngl 99`, `--fit off`.

| Run | Topology | splits | G (t/s) | split_total / 130 tok | RPC split ms (backend1) | 6600 split ms (backend2) |
|-----|----------|--------|---------|----------------------|-------------------------|--------------------------|
| `trace-f-3gpu` | 5070 + 5060 + 6600 `ts=30,12,58` | **4** | **38.0** | 3484 ms (**26.8 ms/tok**) | 1309 ms (**10.1 ms/tok**) | 2106 ms (**16.2 ms/tok**) |
| `trace-f-2gpu` | 5070 + 5060 `ts=50,50` | **3** | **48.9** | 2117 ms (**16.3 ms/tok**) | 2080 ms (**16.0 ms/tok**) | - |
| `trace-e-2gpu` | 5070 + 5060 (Config E) | **3** | **47.6** | 2291 ms (**17.6 ms/tok**) | 2236 ms (**17.2 ms/tok**) | - |

Backend IDs from `sched-trace.jsonl`: `backend3`=CPU, `backend0`=CUDA0, `backend1`=RPC `:50051`, `backend2`=RPC `:50052` (3-device only).

**Cross-check:** 1000 / 26.8 ms = **37.3 t/s** vs measured **38.0** on 3-device F. Scheduler split sum explains throughput without invoking PCIe/NIC limits.

Artifacts: `benchmarks/trace-f-3gpu/`, `trace-f-2gpu/`, `trace-e-2gpu/`; summaries in each `telemetry/trace-summary.txt`.

## Stall waterfall (3-device F, per token)

```text
Token critical path (~26.8 ms) -- serial splits in ggml_backend_sched_compute_splits
|
|-- CPU split (backend3)           ~0.3 ms   graph_compute_async dominates
|-- CUDA0 split (backend0)         ~0.2 ms   local, negligible
|-- RPC :50051 split (backend1)   ~10.1 ms   input_wait_copy + GRAPH_RECOMPUTE RTT
|-- RPC :50052 split (backend2)   ~16.2 ms   input_wait_copy + weak GPU + 2nd hop
|
+-- Between tokens (not in split_total): EVENT_RECORD drain ~9 ms avg,
    COPY_TENSOR cross-endpoint (GET+SET fallback), sched_synchronize on graph reuse
```

First-token example from `sched-trace.jsonl` (split 2/3 on RPC backends):

| Split | backend | input_wait_copy | graph_compute_async | split_total |
|-------|---------|-----------------|---------------------|-------------|
| 2 | RPC :50051 | 62.6 ms | 0.9 ms | 63.7 ms |
| 3 | RPC :50052 | 147.0 ms | 277.9 ms | 425.8 ms |

Steady-state gen averages are lower (table above) but **input_wait_copy** on RPC backends remains the dominant phase vs local `graph_compute_async`.

## RPC trace budget (128-token window)

Aggregated client-side (`rpc-trace.jsonl`). Totals include **load + gen** (server lifetime); use split totals for gen-critical-path, RPC categories for stall **class**.

| Opcode / phase | trace-f-3gpu | trace-f-2gpu | trace-e-2gpu |
|----------------|-------------|-------------|-------------|
| COPY_TENSOR total ms | **42454** | **47407** | **49096** |
| COPY_TENSOR avg us | 294822 | 282186 | 292236 |
| GRAPH_RECOMPUTE total ms | 2554 | 4527 | 1846 |
| EVENT_RECORD drain ms | 2382 | 1776 | 1768 |
| blocking_events ms | 32281 | 36192 | 34308 |
| sched input_wait_copy ms | 2569 | 1878 | 1889 |
| sched graph_compute_async ms | 899 | 223 | 385 |

**COPY_TENSOR** is the largest RPC wall-time bucket (~280-295 ms avg per call). `ggml-rpc.cpp` disables cross-socket async copy (`cpy_tensor_async` NULL); cross-endpoint copies fall back to blocking GET+SET. This aligns with [`RPC-WAIT-MAP.md`](RPC-WAIT-MAP.md) site list.

**GRAPH_RECOMPUTE** count scales with splits x pipeline copies: 3712 (4-split) vs 4352 (3-split with heavier 5060 share).

## Why more TFLOPS hurts (confirmed)

1. **Amdahl / serial splits:** Token time ~= sum of per-split latencies, not aggregate TFLOPS.
2. **6600 straggler:** +1 split, +2.1 GB on `:50052`, ~16 ms/tok stage for ~12% weight share (`ts=30,12,58`).
3. **Low util%:** Bursty compute (CUDA0 peak 24% util, 17.7% TDP in trace-f-3gpu) between RPC waits; consistent with [`PROFILING.md`](PROFILING.md).
4. **Not NIC/RAM:** Peak ~1.6 Gbps vs 2.5G ceiling; GEN pages/sec low. PCIe not saturated during gen.

Dropping 6600: **38.0 -> 48.9 t/s (+29%)** with trace matrix; matches prior profile run `profile-f-36b-nl-no6600` (**48.9 t/s**).

## Path A / B / C fit (evidence-based)

| Path | Status | Fit score | Rationale from traces |
|------|--------|-----------|----------------------|
| **A** (batch SET, async GET) | Partial (SET batch exists) | **Medium** | Would trim fixed RTT overhead; does not remove serial **split loop** or cross-endpoint COPY_TENSOR |
| **B** (events, pipeline copies) | **Shipped** | **High but plateaued** | `sched copies=4` confirmed; token overlap works, split serialism remains bottleneck |
| **C** (server multi-GPU) | Planned | **High for remus** | Collapse `:50051` + `:50052` into one RPC backend -> eliminate client-side split between 5060/6600 and halve cross-endpoint copies |

**Topology recommendation (no code):** Config F for 35B/36B A3B MoE -> **2-device** `ts=50,50`, single remus endpoint. Do not attach `:50052` unless model needs the VRAM and accepts the serial hop tax.

## Reproduce

```bat
cmd /c "D: && cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant && powershell -NoProfile -File scripts\cuda-windows-5070ti\build.ps1"
cmd /c "D: && cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant && powershell -NoProfile -File scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-3gpu,trace-f-2gpu,trace-e-2gpu"
```

Parse only:

```powershell
.\scripts\cuda-windows-5070ti\pathb-rpc-trace-parse.ps1 -TraceDir docs\cuda-windows-5070ti\benchmarks\trace-f-3gpu\telemetry
```

Env vars (set by `rpc-server-bench.ps1 -Trace`): `GGML_RPC_TRACE=1`, `GGML_SCHED_TRACE=1`, optional `*_TRACE_FILE` paths.

## Deferred (L3-L5)

Not run in this pass; static + lightweight trace sufficient for root cause.

- Nsight Systems on `llama-server.exe` during gen (CUDA0 idle gaps vs RPC thread)
- `nsys` on remus `rpc-server` (server `graph_compute` vs recv idle)
- `tcpdump` / Wireshark RTT histogram on `:50051`/`:50052` (confirm single-stream ordering)

Server-side `rpc_trace_emit` in remus docker requires **rebuilt rpc-server image**; client-side traces above are sufficient for split/RTT classification.

## Files touched in hunt

| File | Role |
|------|------|
| `ggml/src/ggml-backend.cpp` | `GGML_SCHED_TRACE` per-split phases |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | `GGML_RPC_TRACE` send/drain/server |
| `scripts/cuda-windows-5070ti/rpc-server-bench.ps1` | `-Trace`, fixed `Log-Meta` order |
| `scripts/cuda-windows-5070ti/pathb-rpc-trace-parse.ps1` | jsonl -> `trace-summary.txt` |
| `scripts/cuda-windows-5070ti/pathb-trace-runbook.ps1` | trace matrix driver |

## RX6600 slot-init hang (2026-06-27)

Separate from gen-time straggler analysis above: **4-GPU with RX6600 hangs at `initializing slots`**, not during `load_tensors` fan-out.

| Run | Client | Endpoints | Stall | Result |
|-----|--------|-----------|-------|--------|
| `trace-g-4gpu-romulus-q8-pp1` | romulus 7900 | 5060+6600+3060 | slot init after ~64s load | HANG |
| `trace-g-4gpu-plus` | Windows 5070 | 5060+6600+3060 | slot init ~45min | RPC recv failed |
| `trace-g-4gpu-primary` (x3) | romulus 7900 | 5060+3060+5070 | slot init OK ~85s load | **PASS** G 38-43 |

Load-phase `SET_TENSOR_HASH` completes on all variants. Failure is **first slot graph warmup** with `:50052` in the scheduler (likely EVENT_RECORD/COPY drain ordering).

**Production 4-GPU:** drop `:50052`; use Windows 5070 `:50053`. See [CLUSTER-4GPU-PRIMARY.md](CLUSTER-4GPU-PRIMARY.md).

## 4-GPU primary trace (7900 + 3060 + 5060 + 5070)

`trace-g-4gpu-primary-trace` hotpath (romulus client, build `833ad4429`):

| Metric | Value |
|--------|-------|
| Serial split sum | 20.7 ms/tok |
| 5060 straggler | 9.6 ms/tok |
| 3060 / 5070 | 5.5 / 5.4 ms/tok |
| SET_TENSOR_HASH | 6.5s / 280 calls (load) |
| assembly_overlap | 1075 (B+1 pass) |

## Next steps (fixes -- separate from diagnosis)

1. **Ops:** 2-device F for Windows-client 36B NL (`ts=50,50`, drop `:50052`). **4-GPU romulus:** use primary topology (no 6600).
2. **RX6600 debug:** dedicated session for ROCm docker slot-init hang (not blocking primary cluster).
3. **Path A3 / RPC copy:** cross-endpoint `cpy_tensor_async` to cut COPY_TENSOR bucket on gen path.
4. **Optional:** Split-level overlap in scheduler (large change).