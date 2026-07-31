# V7 — 72B Dense Model 3-GPU Stress Test (2026-07-24)

**Date:** 2026-07-24  
**Build:** 83605ae27 (10202) — UDP default + V1a cache + V1b GET_TENSOR_BATCH  
**Model:** Qwen3-72B-Instruct.IQ4_XS (37.40 GiB, 72.7B params, **dense**)  
**GPUs:** 3-GPU layer-split (7900 XTX + RTX 3090 + RTX 5060 Ti)  

## Motivation

All previous multi-GPU benchmarks used MoE models (35B/3B and 80B/3B), where only ~10% of parameters are active per token. The 72B dense model computes **all 72.7B parameters every token**, making it a fundamentally different stress test: higher compute per token, higher memory bandwidth pressure, and no expert-sparsity to mask RPC overhead.

## Results

| Config | tg32 (t/s) | tg64 (t/s) | Notes |
|--------|-----------|-----------|-------|
| 3-GPU 72B dense | 32.43 | **34.79** | All params active every token |

## Comparison: Dense vs MoE at 3-GPU

| Model | Size (GiB) | Active Params | tg64 (t/s) | Compute/TFLOPS |
|-------|-----------|--------------|-----------|----------------|
| 80B MoE Q5_K_M | 52.90 | ~3B | **84.21** | Low (sparse) |
| 72B Dense IQ4_XS | 37.40 | **72.7B** | **34.79** | High (dense) |

The 72B dense model is 2.42× slower than the 80B MoE despite being 30% smaller on disk. This is expected — dense models perform ~24× more FLOPs per token (72.7B all-active vs ~3B MoE). The 2.42× slowdown is much lower than the 24× FLOP difference because:
- Memory bandwidth dominates at small batch sizes, and the 72B model is IQ4_XS (37 GiB) vs the 80B at Q5_K_M (53 GiB)
- MoE routing overhead partially offsets sparsity gains at single-token inference

## Interactive Viability

At 34.79 t/s, the 72B dense model provides **acceptable** interactive performance (>10 t/s). It's slower than the MoE sweet spot but faster than reading speed (~5-10 t/s for most users).

## Comparison with V6 Scaling

Unlike the 80B MoE where 3-GPU is the sweet spot (84.21 t/s → 71.81 → 70.19), the 72B dense model was only tested at 3-GPU. The dense model's higher compute intensity means it would likely see less relative benefit from adding GPUs (compute is already saturating), but more absolute benefit from RPC overhead reduction.

## Key Insight

Dense models stress **compute** and **memory bandwidth**, not just RPC overhead. This makes them a complementary stress test to MoE models. For dense models:
- Layer-split throughput is compute-bound, not RPC-bound
- Adding GPUs has even less benefit than for MoE (compute is the bottleneck, not GET_TENSOR frequency)
- RPC overhead optimization (UDP, memory cache) still helps but proportionally less

## References

- `v6-80b-udp.md`: 3-GPU 80B MoE (84.21 tg64)
- `v6-5gpu-80b-mixtral.md`: 5-GPU 80B + Mixtral VRAM analysis
