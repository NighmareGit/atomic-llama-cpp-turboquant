# Slice 7 Research: Layer 1-3 Performance via kernel-anvil Integration

**Date:** 2026-07-17
**Context:** Slice 6 (D7) attacked pipeline-level bottlenecks (event_wait_slot, RPC overlap, MTP asymmetry). D7.6 rocprofv3 profiling revealed that **MatMul is 55.4% of GPU time** -- the real bottleneck is single-GPU kernel performance, not pipeline coordination.
**Goal:** Reframe optimization into Layer 1-3 framework and introduce kernel-anvil vectors targeting the dominant MatMul bottleneck.

---

## 1. The Layer 1-3 Optimization Framework

Path-D's pipeline optimization operates at three layers. Slice 6 attacked Layers 2-3 while Layer 1 was largely untouched.

| Layer | Scope | Current Status | Dominant Bottleneck |
|-------|-------|---------------|---------------------|
| **Layer 1** | Single-GPU kernel performance | D7.8 LDS prototype (in progress), D7.7 WMMA (closed) | MatMul = 55.4% of GPU time |
| **Layer 2** | Multi-GPU pipeline overlap | D5 (n_stages>2), D6 (multi-seq) complete | Pipeline bubbles, stage imbalance |
| **Layer 3** | System-level (RPC, dispatch, memory) | D6.10 GPU event fix, D7.3 FA enabled | RPC latency 20.4%, dispatch overhead |

**Key insight:** Layer 1 is the foundation. Faster kernels make every pipeline stage faster, which makes pipeline balance easier, which improves assembly-line saturation. kernel-anvil's methods directly target Layer 1.

---

## 2. D7.6 Profiling Data: The Layer 1 Targets

From `docs/research/d76-rocprofv3-kernel-profile.md` (Qwen3.6-35B-A3B on 7900XTX):

| Category | % GPU | Kernel | Layer |
|----------|-------|--------|-------|
| **MatMul** | **55.4%** | q6_K (35.2%), iq4_xs (8.0%), q8_0 (6.1%), q5_K (2.7%), fp32 (3.5%) | **Layer 1** |
| quantize_q8_1 | 7.9% | Input quantization | **Layer 1** |
| MoE routing | 4.0% | topk_moe_cuda | Layer 1 |
| FlashAttn | 3.4% | flash_attn_ext_vec | Layer 1 |
| SSM | 2.4% | Gated delta net | Layer 1 |
| Other | 26.9% | copies, norms, etc. | mixed |

**Layer 1 total: ~73% of GPU time.** kernel-anvil's shape-specific tuning targets the 55.4% MatMul portion. The small_k fix and quantize_q8_1 fusion target another ~8%.

---

## 3. kernel-anvil Methods Mapped to Layer 1

### 3.1 Shape-Specific MMVQ Tuning (kernel-anvil core)

**What it does:** For each unique (quant_type, N, K) shape in the model, profiles candidate (nwarps, rows_per_block) configs on the actual GPU and picks the fastest.

**Why it targets Layer 1:** The MMVQ kernel is the MatMul workhorse. llama.cpp uses hardcoded nwarps=8 for all shapes on RDNA3. But optimal nwarps varies by shape:
- Small N (GQA k/v, 1024 rows): fewer warps = less reduction overhead
- Large K (down_proj, 14336): more warps = better K parallelism
- Large N (vocab, 32000+): nwarps affects per-block throughput vs bandwidth

**Published results (kernel-anvil README):**

| Shape | Count | Speedup | Optimal Config |
|-------|------:|--------:|---------------|
| Q4_K 5120x6144 | 48 | **1.54x** | rows_per_block=2 |
| Q5_K 5120x10240 | 48 | **1.38x** | nwarps=8 |
| Q4_K 6144x5120 | 64 | **1.17x** | nwarps=8 |
| Q4_K 5120x1024 | 22 | **1.13x** | rows_per_block=2 |

**End-to-end: 12 -> 27 tok/s (2.25x) on Qwen3.5-27B.**

**Path-D relevance:** D7.6 shows q6_K matmul is 35.2% of GPU time. If kernel-anvil's tuning achieves even half the published speedup on q6_K shapes, that's a 10-15% end-to-end TG improvement. This is the single highest-leverage optimization available.

### 3.2 small_k Off-by-One Fix

**What it does:** The `should_use_small_k` threshold uses strict `<` instead of `<=`, so K=4096 shapes (the most common) never trigger the multi-row optimization.

**Why it targets Layer 1:** Activating small_k for K=4096 means gate_proj, up_proj, q_proj, o_proj all process multiple rows per block instead of one. This reduces kernel launch count and improves occupancy.

**Path-D relevance:** Zero-risk, one-line change. The D7.6 breakdown shows Q4_K and Q6_K shapes at K=4096 are the majority of MatMul dispatches. This is a guaranteed win.

### 3.3 quantize_q8_1 Fusion

**What it does:** Fuses the input quantization step into the MMVQ kernel, eliminating a separate kernel dispatch.

**Why it targets Layer 1:** D7.6 shows quantize_q8_1 at 7.9% of GPU time (90.9M ns). Fusing eliminates:
- 224 kernel launches/token (dispatch overhead)
- 573 us/token of quantization work
- One round-trip through global memory (quantized activations)

**Path-D relevance:** The Vulkan analysis (kernel-anvil docs) proves this works on RDNA3 -- Vulkan reads float activations directly. The ROCm path quantizes to Q8_1 first because DP4A integer dot products are faster on NVIDIA. On RDNA3, the advantage is less clear.

### 3.4 Cell Ablation Methodology

**What it does:** Instead of trusting proxy metrics (bandwidth %, occupancy), A/B test each candidate shape individually against baseline using real llama-bench decode runs.

**Why it matters for Path-D:** D7.1 (n_copies>1) wasted effort on a hypothesis that turned out to be noise (+0.8-1.4%). The honest A/B approach would have killed it in one experiment. Apply this discipline to all future Layer 1 optimization decisions.

---

## 4. New Slice 7 Vectors

### Vector D: kernel-anvil Shape-Specific Tuning Integration

**Target:** MatMul = 55.4% of GPU time (q6_K 35.2%, iq4_xs 8.0%, q8_0 6.1%)

**Mechanism:**
1. Apply kernel-anvil's smithy patch to llama.cpp (`kernel-anvil/patches/apply.sh`)
2. Run `kernel-anvil gguf-optimize` for each model in `/mnt/models`
3. Benchmark with `SMITHY_CONFIG=... llama-bench` to quantify per-model speedup
4. Integrate winning configs into the build pipeline

**Expected gain:** 10-30% TG improvement (conservative, based on kernel-anvil's published results for similar models).

**Effort:** Medium (patch application + profiling run + benchmark verification)

**Risk:** Low (smithy patch is opt-in via env var, no behavioral change without config)

**Prerequisites:** None. Can run on romulus standalone (no RPC needed for profiling).

### Vector E: small_k Off-by-One Fix

**Target:** K=4096 shapes (gate_proj, up_proj, q_proj, o_proj) = majority of MatMul dispatches

**Mechanism:** Change `<` to `<=` in `should_use_small_k` threshold in `mmvq.cu`.

**Expected gain:** 5-15% on affected shapes (per kernel-anvil's analysis).

**Effort:** 1 line change + rebuild + benchmark.

**Risk:** Negligible (only affects the threshold boundary, not the optimization itself).

### Vector F: quantize_q8_1 Fusion into MMVQ

**Target:** quantize_q8_1 = 7.9% of GPU time

**Mechanism:** Modify the MMVQ kernel to accept float activations directly, eliminating the separate quantization step. Follow the Vulkan path's approach (dequantize weights to float in-shader, multiply with float activations).

**Expected gain:** 5-10% TG improvement.

**Effort:** Medium (kernel modification + correctness verification).

**Risk:** Medium (requires careful numerical verification -- integer DP4A vs float dot product may differ in edge cases).

### Vector G: Autoforge Custom Kernels for Top Shapes

**Target:** The top 5 shapes that dominate GPU time (from D7.6 breakdown)

**Mechanism:** Use kernel-anvil's `autoforge` to generate purpose-built HIP kernels with hardcoded N/K dimensions, optimal nwarps/rows_per_block, unrolled inner loops.

**Expected gain:** 15-25% on targeted shapes (kernel-anvil's Q4_K 12288x4096 shows 1.94x).

**Effort:** High (custom codegen pipeline + integration with llama.cpp dispatch).

**Risk:** Medium (custom kernels need per-model generation, complicate the build).

---

## 5. Prioritization Within Slice 7

| Priority | Vector | Target | Gain | Effort | Risk |
|----------|--------|--------|------|--------|------|
| **P0** | E: small_k fix | K=4096 shapes | 5-15% | 1 line | negligible |
| **P1** | D: kernel-anvil tuning | All MatMul (55.4%) | 10-30% | medium | low |
| **P1** | F: quantize fusion | quantize_q8_1 (7.9%) | 5-10% | medium | medium |
| **P2** | G: autoforge custom | Top 5 shapes | 15-25% | high | medium |

---

## 6. Complementarity with D7.8 LDS Prototype

The D7.8 LDS activation caching prototype (currently in progress) attacks the same Layer 1 bottleneck from a different angle:

| Approach | Mechanism | Target |
|----------|-----------|--------|
| D7.8 LDS | Cache q8_1 activations in `__shared__` to eliminate redundant global reads | Activation memory traffic |
| kernel-anvil D | Tune nwarps/rows_per_block per shape | Thread utilization + occupancy |
| kernel-anvil E | Fix small_k threshold | Multi-row processing |
| kernel-anvil F | Fuse quantize into MMVQ | Kernel count + dispatch |

These are **complementary, not competing**. LDS reduces memory traffic while kernel-anvil tuning improves compute utilization. The expected combined effect is multiplicative:
- LDS: reduces activation read traffic 16x per half-warp
- kernel-anvil: ensures the compute units are fully utilized with optimal thread config
- Combined: faster kernels with less memory wait = maximum MatMul throughput

---

## 7. Updated Layer 1-3 Optimization Landscape

### Layer 1: Single-GPU Kernel Performance

| Lead | Status | Result |
|------|--------|--------|
| D7.7 WMMA vec_dot | CLOSED | Not viable for M=1 (16x too much work) |
| D7.8 LDS activation caching | IN PROGRESS | Blocked on GPU reset, code verified |
| **Vector D: kernel-anvil tuning** | **NEW** | **Highest priority after E** |
| **Vector E: small_k fix** | **NEW** | **P0 - do immediately** |
| **Vector F: quantize fusion** | **NEW** | **P1** |
| **Vector G: autoforge custom** | **NEW** | **P2** |

### Layer 2: Multi-GPU Pipeline Overlap

| Lead | Status | Result |
|------|--------|--------|
| D5 n_stages > 2 | COMPLETE | No gain on 2-GPU (within noise) |
| D6 multi-seq Mode B | COMPLETE | Working, 10/10 tests pass |
| D7.1 n_copies > 1 | CLOSED | +0.8-1.4% (noise) |

### Layer 3: System-Level

| Lead | Status | Result |
|------|--------|--------|
| D6.10 GPU event pipelining | COMPLETE | input_copy_slow -98.6% |
| D7.3 FA on HIP | COMPLETE | +7.5% TG |
| D7.4 skip-SSM verify | CLOSED | Output collapses |
| D7.5 RPC overlap | RESOLVED | Redirect to D6.10 |

---

## 8. Key Files for Integration

| File | Purpose |
|------|---------|
| `kernel-anvil/patches/smithy-config.h` | Runtime config loader |
| `kernel-anvil/patches/README.md` | Patch instructions |
| `kernel-anvil/kernel_anvil/cell_ablation.py` | A/B testing methodology |
| `ggml/src/ggml-cuda/mmvq.cu` | Target for small_k fix + smithy patch |
| `docs/research/d76-rocprofv3-kernel-profile.md` | Per-kernel timing breakdown |
| `docs/wayfinder/HANDOFF-D7.8-LDS-prototype.md` | LDS prototype status |

---

## 9. Success Metrics

| Vector | Metric | Target |
|--------|--------|--------|
| E: small_k fix | TG improvement on K=4096 model | >= 5% |
| D: kernel-anvil tuning | TG improvement on 35B model | >= 10% |
| F: quantize fusion | quantize_q8_1 time reduction | >= 50% |
| Combined (D+E+F) | End-to-end TG vs D7.3 baseline (143 t/s) | >= 157 t/s (+10%) |

---

*Research complete -- ready for agent grilling and ticket creation*
