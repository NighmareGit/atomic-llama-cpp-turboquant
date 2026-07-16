# D7.7 — WMMA vec_dot Prototype Findings

**Date:** 2026-07-16
**Status:** DECISION — WMMA vec_dot is NOT viable for M=1 decode. Pivot to dp4a micro-optimizations.

## 1. Prototype Summary

Replaced the Q4_K `vec_dot_q4_K_q8_1` dp4a path with a WMMA-accelerated
alternative using `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` (RDNA3/gfx1100).
The prototype dequantized both Q4_K weights and q8_1 activations to fp16, loaded
them into WMMA tiles, and computed the 16x16 outer product — then summed the
diagonal to recover the scalar dot product.

## 2. Benchmark Results (gemma-4-12B-it-Q4_K_M, tg16, dual-GPU RPC)

| Build | Run 1 | Run 2 (steady) |
|-------|:-----:|:--------------:|
| **WMMA vec_dot** | 109.0 t/s (146.7 ms) | **138.6 t/s (115.4 ms)** |
| **dp4a baseline** | 61.5 t/s (260.2 ms) | **52.6 t/s (304.3 ms)** |
| Delta | +77% | **+163%** |

WMMA shows 2.2-2.6x faster TPS than dp4a.

## 3. Why This Is Suspicious

A vec_dot replacement should not yield a 2x speedup because:

1. **Architectural overhead**: WMMA computes the full 16x16 outer product (256
   multiply-adds) when only the diagonal sum is needed. The dp4a path directly
   computes the scalar dot product with 8 dp4a ops — far less work.

2. **Dequant cost**: Converting 256 4-bit Q4_K weights to fp16 + 256 q8_1
   activations to fp16 adds significant overhead. The dp4a path works directly
   on quantized data with integer SIMD.

3. **Profiling magnification**: Even if WMMA hardware were 8x faster than dp4a,
   the dequant overhead would dominate for single-token (M=1) operations.

4. **Theoretical maximum**: If Q4_K matmul is ~35% of GPU time (from D7.6
   profiling), even eliminating it entirely could only yield ~54% improvement.
   A 163% improvement implies something else changed.

**Most likely cause**: The WMMA kernel produces numerically incorrect results
(output garbage). The speedup is because the computation is effectively a no-op
or the WMMA intrinsic produces zeros/small values that reduce faster.

## 4. Root Cause Analysis

### 4.1 WMMA Tile Layout Complexity

RDNA3 WMMA requires very specific tile layouts:
- A (activations): `I_MAJOR_MIRRORED` — data duplicated in a mirror pattern
- B (weights): `I_MAJOR_MIRRORED` — same mirror layout
- D (accumulator): `I_MAJOR` — standard row-major

The prototype used a simplified tile packing that does NOT match the expected
mirror layout. The `__builtin_amdgcn_wmma` intrinsic operates on halfx16_t
vectors with specific lane assignments — our naive packing likely puts data
on the wrong lanes, causing the WMMA to multiply wrong elements.

### 4.2 M=1 Inefficiency

WMMA minimum tile is M=N=K=16. For a single-token decode (M=1), we compute 256
multiply-adds to produce a 16x16 result, then reduce to a scalar. The dp4a path
computes exactly 256 multiply-adds directly — no reduction needed. WMMA adds
tile load/store overhead and 16x reduction cost with no benefit.

### 4.3 Timing Anomaly

The 115 ms tg16 wall time with WMMA (138.6 t/s) is close to the **MTP-enabled**
Qwen 35B performance (143 t/s from D7.3). For a 12B dense model without MTP
on dual GPU, expected tg16 would be ~50-70 t/s (matching the DP4A baseline).
The WMMA result is physically implausible for this model architecture.

## 5. Decision

**WMMA vec_dot is NOT viable for MMVQ (single-token decode).**

Reasons:
1. Architecturally wrong — computes 16x too much work for M=1
2. Tile layout complexity makes correct implementation fragile
3. Dequant overhead negates any WMMA throughput advantage
4. Suspected incorrect numerical results (unverified but highly probable)
5. Even if correct, the 2x speedup is likely from other factors (compiler
   optimizations removing dead code, different code paths being taken)

**Prototype code is NOT merged.** It remains behind `#ifdef
GGML_HIP_WMMA_VECDOT_EXPERIMENTAL` guards in the working tree for reference.
The CMake option defaults to OFF.

## 6. Where WMMA IS Useful

WMMA is already correctly used in:
- **MMQ (batch matmul)**: `mmq.cuh` — processes M >= 8 rows simultaneously,
  amortizing WMMA tile overhead
- **Flash Attention**: `fattn-wmma-f16.cuh` — processes full attention heads,
  M >= 128
- **MMF (FP16 matmul)**: `mmf.cuh` — FP16/BF16 dense matmul with M >= 1-16

The key insight: WMMA wins when M (batch dimension) is large enough to amortize
tile setup cost. For M=1 (single-token decode), dp4a is the correct choice.

## 7. Next Steps: dp4a Micro-Optimizations

Pivot to optimizing the existing dp4a path for Q4_K/Q4_K_S:

1. **Instruction scheduling**: Reorder loads/computes for gfx1100 dual-issue
   capability. The RDNA3 architecture can dual-issue scalar + vector ops.
2. **Loop structure**: The QR4_K=8 inner loop in `vec_dot_q4_K_q8_1_impl_vmmq`
   may benefit from different unroll factors.
3. **Prefetch hints**: Scale tables (`sc`, `m`) are loaded per-block; adding
   `__builtin_prefetch` may hide L1 miss latency.
4. **Register analysis**: Check if the current 64-register usage per thread
   is optimal for gfx1100 (256 registers per SIMD, 2 SIMDs per CU).
5. **Benchmark infrastructure**: Use the same gemma-4-12B-Q4_K_M model with
   tg16, repeat=3, no-warmup for controlled A/B testing.

## 8. Artifacts

- Prototype source: `ggml/src/ggml-cuda/vecdotq.cuh` (line ~910, behind `#ifdef GGML_HIP_WMMA_VECDOT_EXPERIMENTAL`)
- Dispatch: `ggml/src/ggml-cuda/mmvq.cu` (line ~22, behind same `#ifdef`)
- CMake: `ggml/CMakeLists.txt:221`, `ggml/src/ggml-hip/CMakeLists.txt:144`
- Benchmark data: `/tmp/d77-wmma-test/`, `/tmp/d77-baseline/` (cleaned up)
