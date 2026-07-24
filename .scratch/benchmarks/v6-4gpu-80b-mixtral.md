# V6 — 4-GPU 80B MoE + Mixtral Attempt (2026-07-24)

**Date:** 2026-07-24  
**Build:** 83605ae27 (10202)  
**Status:** 4-GPU 80B COMPLETE; 5-GPU Mixtral → see `v6-5gpu-80b-mixtral.md`

## 4-GPU 80B MoE (7900 XTX + 3090 + 5060 Ti + 3070)

| Config | tg64 (t/s) | vs 3-GPU (84.21) | Notes |
|--------|-----------|-------------------|-------|
| 3-GPU (7900+3090+5060) | **84.21** | baseline | Sweet spot |
| 4-GPU (+3070) | **71.81** | **-14.7%** | 3070 bottleneck confirmed |

**Analysis:** Adding the RTX 3070 (8 GiB, weakest GPU) increases GET_TENSOR calls from 2/token to 3/token without proportionally reducing per-GPU compute time. The 3070 handles very few layers (~2-3) due to its small VRAM, making the network overhead outweigh the compute distribution benefit.

**Pattern:** Each additional GPU adds one GET_TENSOR per token (~2ms with UDP). When the added GPU contributes minimal compute (small VRAM → few layers), the net is negative. The sweet spot is the minimum GPU count that fits the model.

## 5-GPU Mixtral 8×22B (72 GiB IQ4_XS) — BLOCKED

**Model:** `/mnt/980pro/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS.gguf` (71.1 GiB)

**Attempt:** 4-GPU load failed — "failed to load model" (VRAM insufficient at ~72 GiB total). Requires the 5th GPU (3060 Ti, 8 GiB) for ~80 GiB total.

**Blocker:** Docker 3060 Ti RPC server fails to start:
```
nvidia-container-cli: initialization error: nvml error: driver/library version mismatch
```
Root cause: NVML driver/library version mismatch on romulus host (NVIDIA kernel module vs userspace library). Requires system-level fix (kernel module rebuild or reboot).

## Throughput Scaling Pattern

Across all V6 experiments:

| GPU Count | Model | tg64 (t/s) | GET_TENSOR/token | Scaling Efficiency |
|-----------|-------|-----------|-------------------|-------------------|
| 1 (7900) | 35B | 106.84 | 0 | baseline |
| 2 (7900+5060) | 35B | 80.89 | 1 | 75.7% |
| 3 (7900+3090+5060) | 80B | 84.21 | 2 | n/a (no 1-GPU) |
| 4 (7900+3090+5060+3070) | 80B | 71.81 | 3 | -14.7% vs 3-GPU |

**Conclusion:** Layer-split multi-GPU throughput peaks at the minimum GPU count that fits the model. Adding GPUs beyond this threshold is counterproductive — each additional GPU adds network overhead that exceeds its compute contribution.

## Recommendations

1. **3-GPU sweet spot:** For models requiring 50-65 GiB, use the 3 strongest GPUs (7900+3090+5060)
2. **5-GPU Mixtral:** Needs Docker 3060 Ti fix. Projected ~25-35 t/s based on linear scaling of 3-GPU result with +2 GET_TENSOR overhead
3. **Future work:** Row-split parallelism (V0) avoids the GET_TENSOR-per-GPU penalty and could scale positively with GPU count

## References

- `v6-80b-udp.md`: 3-GPU 80B with UDP (84.21 t/s)
- `v6-layer-split-80b.md`: Original 3-GPU 80B (21.16 t/s, no UDP)
- `T2-5gpu-multi-backend.md`: Phase 2 5-GPU experiment (10.10 t/s)
