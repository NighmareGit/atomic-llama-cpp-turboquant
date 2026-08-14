# Optimization Landscape: Layer 1-3 Performance

**Branch:** Path-D-Gpipeline-Assembly-Line
**Date:** 2026-07-17
**Source:** `docs/research/slice-7-kernel-anvil-integration.md`
**Status:** Slice 7 planning -- kernel-anvil vectors identified, ready for execution

---

## Framework

Path-D optimization operates at three layers. Improvements at lower layers compound through higher layers.

| Layer | Scope | Bottleneck (from D7.6 rocprofv3) | % GPU |
|-------|-------|----------------------------------|-------|
| **Layer 1** | Single-GPU kernel performance | MatMul (q6_K, iq4_xs, q8_0) | 55.4% |
| **Layer 2** | Multi-GPU pipeline overlap | Pipeline bubbles, stage imbalance | ~15% |
| **Layer 3** | System-level (RPC, dispatch) | RPC latency, dispatch overhead | ~25% |

**Key insight:** Layer 1 is 55.4% of GPU time but has received the least attention (only D7.7 WMMA and D7.8 LDS). kernel-anvil's methods directly target Layer 1.

---

## Layer 1: Single-GPU Kernel Performance

### Accomplished/Tried

| Lead | Ticket | Result | Verdict |
|------|--------|--------|---------|
| WMMA vec_dot | D7.7 | 2.2x TPS but numerically incorrect (M=1 overhead) | CLOSED -- not viable |
| LDS activation caching | D7.8 | Code verified, CMake misconfiguration resolved. No significant throughput delta. | COMPLETE |

### New Leads (from kernel-anvil research)

| Lead | Vector | Mechanism | Target | Est. Gain | Effort | Priority |
|------|--------|-----------|--------|-----------|--------|----------|
| small_k off-by-one | E | Change `<` to `<=` in threshold | K=4096 shapes | 5-15% | 1 line | **P0** |
| kernel-anvil tuning | D | Shape-specific nwarps/rows_per_block | All MatMul (55.4%) | 10-30% | medium | **P1** |
| quantize fusion | F | Fuse quantize_q8_1 into MMVQ | quantize (7.9%) | 5-10% | medium | **P1** |
| autoforge custom | G | Purpose-built HIP kernels per shape | Top 5 shapes | 15-25% | high | **P2** |

### D7.6 Per-Kernel Breakdown (Qwen3.6-35B-A3B, 7900XTX)

| Kernel | % GPU | Layer 1 Sub-target |
|--------|-------|-------------------|
| q6_K matmul | 35.2% | MMVQ kernel config |
| quantize_q8_1 | 7.9% | Fusion candidate |
| iq4_xs matmul | 8.0% | MMVQ kernel config |
| q8_0 matmul | 6.1% | MMVQ kernel config |
| fp32 matmul | 3.5% | cuBLAS path |
| q5_K matmul | 2.7% | MMVQ kernel config |
| FlashAttn | 3.4% | Already optimized (D7.3) |
| MoE routing | 4.0% | topk_moe_cuda |
| SSM | 2.4% | Gated delta net |

---

## Layer 2: Multi-GPU Pipeline Overlap

### Accomplished

| Lead | Ticket | Result |
|------|--------|--------|
| n_stages > 2 | D5.1-D5.7 | Complete. No gain on 2-GPU (within noise). Benefit expected on 3+ GPU. |
| Multi-seq Mode B | D6.1-D6.9 | Complete. Per-seq events, GRAPH_COMPUTE_STAGE. 10/10 tests pass. |
| n_copies > 1 | D7.1 | CLOSED. +0.8-1.4% (noise). GPipe bypasses pipeline_barrier. |

### Open Leads

None currently. Layer 2 is mature for 2-GPU. Further gains require 3+ GPU cluster.

---

## Layer 3: System-Level

### Accomplished

| Lead | Ticket | Result |
|------|--------|--------|
| GPU event pipelining | D6.10 | COMPLETE. input_copy_slow -98.6%. |
| FA on HIP | D7.3 | COMPLETE. +7.5% TG (133.0 -> 143.0 t/s). |
| Skip-SSM verify | D7.4 | CLOSED. Output collapses at +75% upper bound. |
| RPC overlap | D7.5 | RESOLVED. Redirect to D6.10. |

### Open Leads

None currently. Layer 3 optimizations are exhausted for 2-GPU config.

---

## Priority Order

1. **P0:** Vector E (small_k fix) -- 1 line, guaranteed win
2. **P1:** Vector D (kernel-anvil tuning) -- highest Layer 1 leverage
3. **P1:** Vector F (quantize fusion) -- eliminates 7.9% GPU time
4. **P2:** Vector G (autoforge custom) -- high effort, defer until D proven
5. **Continue:** D7.8 LDS prototype -- complementary to kernel-anvil vectors

---

## Baseline Metrics (for measuring Slice 7 gains)

| Config | TG (t/s) | Source |
|--------|----------|--------|
| 2-GPU RPC, Qwen3.6-35B, n_max=2, GPipe ON, FA ON | **143.0** | D7.3 (current best) |
| 2-GPU RPC, Qwen3.6-35B, n_max=2, GPipe ON, FA OFF | 133.0 | D7.1 |
| 2-GPU RPC, Qwen3.6-35B, n_max=2, GPipe OFF | 131.1 | D7.1 |
| 1-GPU 7900XTX, n_max=1 | 102.5 | D7.1 |

---

*Optimization landscape last updated: 2026-07-17*
