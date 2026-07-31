# kernel-anvil Research: Performance Optimization Methods

**Date:** 2026-07-17
**Source:** `~/projects/kernel-anvil` (commit history through July 2026)
**Goal:** Identify optimization methods transferable to Path-D GPipe assembly line

---

## 1. kernel-anvil Overview

Profile-guided GPU kernel optimizer for AMD. Reads a GGUF model, profiles each unique (quant_type, N, K) GEMV shape on the actual GPU, finds optimal kernel configs via guided sweep, and writes a JSON config that llama.cpp loads at runtime.

**Headline result:** 2.25x decode speedup on Qwen3.5-27B (12 -> 27 tok/s on 7900 XTX) from shape-specific kernel tuning alone.

### Core Mechanism

The MMVQ (mul_mat_vec_q) path handles quantized matrix-vector products for batch=1-8 (decode). llama.cpp uses hardcoded `nwarps` and `rows_per_block` based only on (type, ncols_dst, arch) -- NOT on the actual matrix dimensions (N, K). kernel-anvil makes these shape-aware.

| Parameter | Currently depends on | Could depend on |
|-----------|---------------------|-----------------|
| nwarps | type, ncols_dst, arch | N (nrows), K (ncols) |
| rows_per_block | ncols_dst, arch, small_k | N, K |

---

## 2. Optimization Methods

### M1: Shape-Specific Kernel Config Tuning

**What:** For each unique (quant, N, K) shape in the model, profile candidate (nwarps, rows_per_block) configs and pick the fastest.

**How:** Parse GGUF -> enumerate unique shapes -> for each shape, sweep configs on real GPU -> verify correctness -> write JSON config -> llama.cpp loads at dispatch time.

**Results:**
- Q4_K 5120x6144: 1.54x speedup (rows_per_block=2)
- Q5_K 5120x10240: 1.38x speedup (nwarps=8)
- Q4_K 6144x5120: 1.17x speedup (nwarps=8)
- Q4_K 5120x1024: 1.13x speedup (rows_per_block=2)

**Relevance to Path-D:** HIGH. D7.6 rocprofv3 analysis shows MatMul is 55-76% of GPU time across all models. Speeding up the dominant matmul kernel directly reduces pipeline stage compute time, improving assembly-line balance. The q6_K matmul was identified as Path-D's #1 optimization target (35.2% of GPU).

### M2: small_k Optimization Fix

**What:** When K is small enough, the kernel can process multiple rows per block (`rows_per_block = nwarps`) instead of 1. This reduces kernel launch count and improves occupancy.

**Critical finding:** An off-by-one in the threshold condition (`<` instead of `<=`) means the optimization never triggers for K=4096 -- the most common weight dimension in modern models (gate_proj, up_proj, q_proj, o_proj).

```cpp
// Current (broken for K=4096):
const bool use_small_k = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
// blocks_per_row_x = 4096/256 = 16, threshold = 8*2 = 16, 16 < 16 = false

// Fix: change < to <=
```

**Relevance to Path-D:** HIGH. This is a one-line fix that activates small_k for the majority of MMVQ dispatches. Combined with kernel-anvil's profiling, it's a guaranteed win with zero risk.

### M3: Cell Ablation (Honest A/B per Shape)

**What:** Instead of trusting proxy metrics (bandwidth %, occupancy), A/B test each candidate cell individually against a baseline using real llama-bench decode runs. Only cells that measurably win are included in the config.

**Why:** Proxy metrics can lie. The only honest measurement is "does this shape actually run faster with this config on the real model?"

**Relevance to Path-D:** MEDIUM. The methodology is transferable: when evaluating pipeline depth strategies (D7.1 n_copies>1), the honest A/B approach would have saved time -- the +0.8-1.4% result was noise, but it took a prototype to discover. Apply this discipline to future pipeline experiments.

### M4: Bottleneck Classification -> Guided Sweep

**What:** Profile a kernel -> classify bottleneck (bandwidth_bound, occupancy_limited_vgpr, occupancy_limited_lds, register_spill, compute_bound, launch_overhead) -> generate targeted config search space based on bottleneck class.

**Search spaces:**
- bandwidth_bound: larger BLOCK_K, try SPLIT_K
- occupancy_limited_vgpr: lower BLOCK_N, fewer warps
- register_spill: smaller blocks
- compute_bound: larger blocks, more warps

**Relevance to Path-D:** MEDIUM. The profile-classify-sweep loop is a systematic optimization methodology. Path-D's D7 vectors (A/B/B2/C) were essentially ad-hoc bottleneck hunts. A more systematic approach (profile -> classify -> targeted sweep) could accelerate future optimization work.

### M5: Operator Fusion (from Vulkan Analysis)

**What:** Fuse multiple operations into single kernel dispatches to eliminate intermediate memory round-trips and reduce dispatch count.

**Key fusions identified:**
- quantize_q8_1 into MMVQ (eliminates 5.7% of decode time, 224 launches/token)
- RMS_NORM + MUL (eliminates 73+73 launches/token)
- RMS_NORM + MUL + ROPE (3-way fusion, eliminates 108 launches/token)
- MUL_MAT + ADD (eliminates 71 launches/token)

**Relevance to Path-D:** HIGH. The quantize_q8_1 -> MMVQ fusion alone targets 5.7% of decode time. This is a kernel-level optimization that directly reduces per-token stage time. The D7.6 stall hunt confirmed no remaining sync stalls -- the next frontier is reducing kernel count via fusion.

### M6: wave64 Mode for RDNA3

**What:** RDNA3 WGP contains two SIMD32 units. Vulkan's RADV driver uses wave64 (both units in lockstep), while ROCm HIP uses wave32. wave64 gives:
- Half the scheduling overhead
- Wider memory coalescing (64 threads vs 32)
- More efficient cross-lane reductions

**Measured bandwidth for Q4_K 12288x4096:**
- Vulkan wave64: 839 GB/s (87% of peak)
- ROCm wave32: 598 GB/s (62% of peak)

**Relevance to Path-D:** LOW. D7.7 already tried WMMA-accelerated vec_dot (architecturally similar -- wider SIMD) and found it wrong for single-token decode (M=1 overhead). wave64 would require restructuring all warp-level primitives (shared memory reduction, warp_reduce). High effort, uncertain payoff given D7.7 findings.

### M7: HIP Graphs

**What:** Capture the dispatch sequence and replay without per-kernel CPU overhead. Analogous to Vulkan's command buffers.

**Impact:** Eliminates ~3 ms/token of HIP dispatch overhead (904 kernels/token * ~3-5 us each = ~3-4 ms).

**Relevance to Path-D:** MEDIUM. Currently marked "experimental, slow" in llama.cpp. If matured, it would reduce per-kernel overhead -- but Path-D's bottleneck is compute (MatMul 55-76%), not dispatch overhead. The RPC hop latency (20.4% of GPU time per D7.2) is a network issue, not a dispatch issue.

### M8: Autoforge (Custom Kernel Generation)

**What:** Generate purpose-built HIP kernels with hardcoded N/K dimensions, optimal nwarps/rows_per_block, unrolled inner loops. Compile with hipcc and load via dlopen.

**Relevance to Path-D:** LOW. The generated kernels are single-GPU optimizations. Path-D's challenge is multi-GPU pipeline balancing, not single-kernel performance. The concept of "generate optimal config per shape" is more applicable than the kernel generation itself.

---

## 3. Transferability Assessment

| Method | Relevance | Effort | Risk | Priority |
|--------|-----------|--------|------|----------|
| M2: small_k fix | HIGH | 1 line | negligible | **P0 -- do immediately** |
| M1: Shape-specific tuning | HIGH | medium (integrate smithy patch) | low | **P1 -- integrate for romulus** |
| M5: quantize_q8_1 fusion | HIGH | medium (kernel modification) | medium | **P1 -- high impact** |
| M3: Cell ablation discipline | LOW | (methodology) | none | **P2 -- apply to future experiments** |
| M4: Guided sweep methodology | LOW | (methodology) | none | **P2 -- systematic optimization** |
| M7: HIP Graphs | MEDIUM | high (upstream maturity) | medium | **P3 -- wait for upstream** |
| M6: wave64 mode | LOW | very high | high | **D7.7 already explored, skip** |
| M8: Autoforge custom kernels | LOW | very high | high | **skip** |

---

## 4. Recommended Actions for Path-D

### Immediate (P0)

**Fix the small_k off-by-one** in `ggml/src/ggml-cuda/mmvq.cu`:
```cpp
// Change < to <= in should_use_small_k
const bool use_small_k = nwarps > 1 && blocks_per_row_x <= nwarps * blocks_per_iter_1warp;
```
This is a one-line, zero-risk change that activates small_k for K=4096 shapes (the majority of MMVQ time). Benchmark on romulus to quantify.

### Short-term (P1)

**Integrate kernel-anvil's smithy patch** for romulus local:
1. Apply `kernel-anvil/patches/apply.sh` to the llama.cpp tree
2. Run `kernel-anvil gguf-optimize` for the models in `/mnt/models`
3. Benchmark with `SMITHY_CONFIG=... llama-bench` to quantify per-model speedup
4. Expected: 10-30% decode improvement (based on published results for similar models)

This directly reduces pipeline stage compute time, which is the bottleneck for GPipe saturation.

**Investigate quantize_q8_1 fusion** into the MMVQ kernel:
- The Vulkan path already does this (reads float activations directly)
- ROCm path quantizes to Q8_1 first, then does integer dot product
- On RDNA3, the integer DP4A advantage is less clear than on NVIDIA
- Fusing would eliminate 5.7% of decode time + 224 kernel launches/token

### Medium-term (P2)

**Adopt cell ablation methodology** for pipeline depth experiments:
- When testing n_copies>1, adaptive depth, or split overhead strategies, use honest A/B llama-bench measurements rather than proxy metrics
- This avoids chasing noise (as happened with D7.1 n_copies>1)

**Apply bottleneck classification** to D7.8 LDS activation caching:
- Profile the Q4_K MMVQ kernel with rocprofv3
- Classify the bottleneck (likely bandwidth_bound for activation loads)
- Generate targeted optimization search space based on classification
- This is more systematic than the current prototype-and-measure approach

### Long-term (P3)

**Track HIP Graphs maturity** in upstream llama.cpp:
- When GGML_HIP_GRAPHS becomes reliable, it would reduce dispatch overhead
- More relevant for multi-GPU where per-kernel RPC overhead compounds

---

## 5. Key Insight: Complementary Optimization Axes

kernel-anvil optimizes **within a single GPU** (making each matmul faster).
Path-D optimizes **across GPUs** (pipelining matmuls across devices).

These are orthogonal and complementary:
- kernel-anvil makes each stage faster -> less time per stage -> easier pipeline balance
- Path-D overlaps stages -> hides latency -> better GPU utilization

The combination of "faster stages" + "better overlap" could yield multiplicative improvements. For example:
- kernel-anvil's 1.54x MMVQ speedup on Q4_K 5120x6146 would directly reduce the bottleneck stage time on the 3060 Ti server
- With faster stages, the pipeline fill/drain time decreases, reducing the bubble percentage
- This could push global_3bk_pct from current levels toward the 25% target

---

## 6. Key Files for Integration

| File | Purpose |
|------|---------|
| `kernel-anvil/patches/smithy-config.h` | Runtime config loader (copy to `ggml/src/ggml-cuda/`) |
| `kernel-anvil/patches/README.md` | Patch instructions |
| `kernel-anvil/kernel_anvil/cell_ablation.py` | A/B testing methodology reference |
| `kernel-anvil/kernel_anvil/sweep.py` | Guided config sweep implementation |
| `kernel-anvil/kernel_anvil/analyze.py` | Bottleneck classification logic |
| `kernel-anvil/docs/decode-pipeline-breakdown.md` | MMVQ analysis + optimization targets |
| `kernel-anvil/docs/vulkan-vs-rocm-analysis.md` | Fusion opportunities + dispatch overhead |
| `kernel-anvil/docs/llama-cpp-kernel-analysis.md` | Deep dive into MMVQ kernel internals |
