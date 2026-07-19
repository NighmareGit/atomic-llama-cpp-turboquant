# Historical split, ROCm-max experiment, kernel profiles

**Date:** 2026-07-19

## 1. What split did the fast tip use?

| Source | Split | Interpretation |
|--------|-------|----------------|
| Peer 148 t/s (`f2d627a6a`, summary) | **TS 24,76** | Client ROCm + RPC 3060 Ti; Plus=1; ctk/ctv q8_0 |
| D7 / hot-paths docs | often **30,70** | 35B MoE |
| Live retest with 24,76 | layers ~**RPC 66 / ROCm 228** of 294 assign rows | ~22% on 3060, ~78% on 7900 |
| Device order with `-rpc` | typically **RPC0 first, then ROCm0** | `-ts A,B` => A% first device (RPC), B% ROCm |

So the documented peer “fast” dual-GPU split was **already ROCm-majority** (76% on the fast card if RPC is first), **not** 50/50 and not “all on 3060.”

## 2. Shovel max on 7900 XTX (bare profiler, n=128, r=3)

Artifacts: `/tmp/rocm-max-split-20260719165956`

| Run | Tip | TS / mode | TG t/s | Layer mass (approx) |
|-----|-----|-----------|-------:|---------------------|
| **fast-1gpu-rocm** | 19db22abb | no RPC | **104.3** | 100% ROCm |
| **fast-ts2-98** | 19db22abb | 2,98 | **98.0** | ~2% RPC / 98% ROCm |
| fast-ts5-95 | 19db22abb | 5,95 | **96.0** | |
| fast-ts10-90 | 19db22abb | 10,90 | **92.8** | |
| fast-ts24-76 | 19db22abb | **24,76 (peer)** | **87.6** | more on RPC |
| cur-1gpu-rocm | 802ccb4c6 | no RPC | **103.7** | 100% ROCm |
| cur-ts5-95 | 802ccb4c6 | 5,95 | **83.9** | |
| cur-ts95-5 | 802ccb4c6 | 95,5 (RPC-first = **wrong way**) | **FAIL** OOM on 3060 | almost all RPC |

### Takeaways

1. **More on 7900 helps a lot** vs peer 24,76: **87.6 → 96–98** dual, **104** single-GPU.
2. **Still not 148.** Single-GPU ceiling here is ~**104**, not 148 — so 148 was either different harness/env/RPC era, or not the same measurement we think.
3. **1-GPU fast vs current almost equal** (104.3 vs 103.7) — code tip gap is tiny without RPC; dual-GPU gap remains (98 vs 84 on ROCm-heavy).
4. Wrong-way ts (95,5 with RPC-first) **OOMs** the 8GB card — confirms enum order.

## 3. GPU kernel profiles (rocprofv3)

Artifacts: `/tmp/kernel-prof-20260719170530` (rocpd `*.db` under `Romulus/`)

Note: under rocprof, wall TG collapses to ~12 t/s (instrumentation overhead). Compare **kernel time mix**, not TG t/s.

Dominant kernels (all arms, total GPU time order):

1. **`mul_mat_vec_q`** (Q4_K / quant matvec) — by far largest  
2. **`quantize_q8_1`**  
3. **`k_get_rows_float`**  
4. **`flash_attn_ext_vec`** (FA path active; vec FA, not only WMMA name)  
5. **`topk_moe_cuda`**  
6. **`rms_norm_f32`**

### 1-GPU: fast tip vs current

| Kernel class | fast-1gpu | cur-1gpu |
|--------------|----------:|---------:|
| mul_mat_vec_q (type14) | ~480 ms | ~481 ms |
| quantize_q8_1 | ~129 ms | ~129 ms |
| flash_attn_ext_vec | ~71 ms | ~69 ms |

**Nearly identical.** Local HIP kernels are not where the dual-GPU 89→75 (or 98→84) regression lives.

### Dual ROCm-heavy: fast-ts2-98 vs cur-ts5-95

Same top kernel set; dual has slightly less local matvec time (some work on RPC, not in this HIP-only kernel DB). **No new pathological kernel** on current tip.

## 4. Implication for “network hop / sync”

- Kernel mix healthy and **matched** across tips on 1-GPU.  
- Dual-GPU still slower than 1-GPU even with **ts 2,98** (98 vs 104) — residual RPC hop cost.  
- Dual-GPU **tip gap** (fast ~98 vs cur ~84 on ROCm-heavy) is more likely **RPC/sched/sync path** than matvec kernels.  
- Documented **148** remains **unreproduced**; local ceiling ~**104** on this box today.

## 5. Practical split recommendation

For dual-GPU on romulus **right now**:

```text
# RPC-first device order (typical with -rpc):
-ts 2,98   # or 5,95  — max shovel onto 7900 XTX
# Avoid 95,5 with RPC-first (OOM on 3060)
```

If you can force ROCm-first device list, invert: `-ts 98,2`.
