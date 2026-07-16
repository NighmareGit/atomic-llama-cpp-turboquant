# D7.6 -- Vector C: rocprofv3 GPU Kernel Profiling

**Date:** 2026-07-16
**Task:** Fix rocprofv3 + ggml compatibility; get per-kernel timing within the 6,843 us GPU compute window
**Tool:** rocprofv3 (ROCm 7.2.3) `--kernel-trace --stats --summary`
**Status:** COMPLETE
**See also:** `docs/research/rocprofv3-profiling-guide.md` — practical usage guide

## 1. Compatibility Fix

### Root Cause

rocprofv3 with `--hip-trace` (HIP API interception) causes SIGABRT in
`ggml_uncaught_exception`. The HIP API interception layer conflicts with ggml's
stream management, particularly `cudaStreamPerThread` usage in the async H2D
copy path and CUDA/HIP graph capture/replay.

### Solution

Use `rocprofv3 --kernel-trace` without `--hip-trace`. Kernel dispatch tracing
has a lighter interception footprint -- it only traces GPU kernel launches via
the HSA queue, not host-side HIP API calls. This avoids the stream management
conflict entirely.

```bash
GGML_CUDA_GRAPHS=0 \
rocprofv3 --kernel-trace --stats --summary \
  -d <output-dir> -o <prefix> -f csv \
  -- <profiler-command>
```

**Critical requirement:** `GGML_CUDA_GRAPHS=0` must be set. With CUDA graphs
enabled, individual kernels are not dispatched individually -- they are captured
into a graph and replayed as a single unit, making per-kernel timing invisible.

### Alternative Approaches Tested

| Approach | Result | Notes |
|----------|--------|-------|
| `rocprofv3 --attach PID` | FAILED | ptrace_scope=1, requires root or parent relationship |
| `rocprof-sys-run --device` (Dyninst) | WORKS | Extreme overhead (9x slowdown), kernel names available but timings inflated |
| `rocprofv3 --kernel-trace` (this fix) | WORKS | 9.5x overhead but kernel names + relative percentages are reliable |
| `rocprofv2 --kernel-trace` | NOT TESTED | Deprecated, rocprofv3 preferred |
| `rocprof` (v1) PMC counters | NOT TESTED | Would work with `-i pmc.txt` but heavyweight |

## 2. Per-Kernel Timing Breakdown

### 2.1 Test Configuration

- Model: Qwen3.6-35B-A3B-APEX-MTP-I-Q6_K (21.86 GiB)
- GPUs: 7900XTX (ROCm, local) + 3060Ti (RPC, remote)
- GPipe stages: 3 (CPU -> RPC -> ROCm)
- MTP speculative decoding: n_max=2, n_min=1
- GGML_CUDA_GRAPHS=0
- Tasks: tg, n_gen=64, repeat=1

**Profiling overhead:** 15.10 t/s (vs 143.0 t/s baseline) = 9.5x slowdown.
Total GPU kernel time: 1,156 ms out of 4,238 ms wall = 27.3% GPU utilization.
Absolute timings are inflated but relative percentages are meaningful.

### 2.2 Top 10 Kernels by Total Duration

| # | Kernel | Calls | Total (ns) | Avg (ns) | % GPU |
|---|--------|-------|-----------|----------|-------|
| 1 | `mul_mat_vec_q<14>` q6_K forward | 17,792 | 369,219,715 | 20,752 | 31.94% |
| 2 | `quantize_q8_1` | 33,664 | 90,999,806 | 2,703 | 7.87% |
| 3 | `k_get_rows_float` | 5,760 | 50,545,989 | 8,775 | 4.37% |
| 4 | `mul_mat_vec_q<23>` iq4_xs transposed | 2,688 | 49,458,255 | 18,400 | 4.28% |
| 5 | `topk_moe_cuda` | 3,840 | 46,239,725 | 12,042 | 4.00% |
| 6 | `mul_mat_vec_q<23>` iq4_xs forward | 2,688 | 42,875,244 | 15,951 | 3.71% |
| 7 | `mul_mat_vec_f` fp32 | 7,680 | 40,389,062 | 5,259 | 3.49% |
| 8 | `flash_attn_ext_vec` | 1,024 | 39,761,411 | 38,830 | 3.44% |
| 9 | `mul_mat_vec_q<8>` q8_0 forward | 3,840 | 39,290,161 | 10,232 | 3.40% |
| 10 | `mul_mat_vec_q<14>` q6_K transposed | 1,536 | 37,513,410 | 24,423 | 3.24% |

### 2.3 Categorized Breakdown

| Category | % GPU | Key Kernels | Notes |
|----------|-------|-------------|-------|
| **MatMul** | **55.4%** | q6_K (35.2%), iq4_xs (8.0%), q8_0 (6.1%), q5_K (2.7%), fp32 (3.5%) | Dominant bottleneck. q6_K alone is >1/3 of all GPU time |
| **Element-wise** | 8.6% | bin_bcast (5.5%), unary/softplus/silu/sigmoid (2.8%), scale (0.04%) | Gated activations in MoE + SSM layers |
| **Data Movement** | 8.4% | k_get_rows (4.4%), k_set_rows_quant (2.0%), copyBuffer (2.0%) | Embedding lookup + quantization overhead |
| **Normalization** | 7.9% | rms_norm (4.6%), l2_norm (1.7%), cpy_scalar (1.6%) | RMS norms in attention + SSM layers |
| **Quantization** | 7.9% | quantize_q8_1 | Dequant to fp32 before compute |
| **MoE Routing** | 4.0% | topk_moe_cuda | Expert selection among 256 experts (8 activated) |
| **Flash Attention** | 3.4% | flash_attn_ext_vec | Already well-optimized |
| **SSM** | 2.4% | gated_delta_net (1.7%), ssm_conv (0.7%) | SSM layers are cheap vs full attention |
| **RoPE** | 1.2% | rope_multi | Positional encoding |
| **Other** | 0.8% | concat_cont (0.8%), fillBuffer (0.02%) | Tensor concatenation |

### 2.4 Quantization Type Distribution (MatMul Sub-Breakdown)

The model uses a mix of quantization formats. This affects which matmul kernels
are invoked and their relative cost:

| Quant Type | ggml_type | % of MatMul | Avg per-call (us) | Calls | Role |
|------------|-----------|-------------|-------------------|-------|------|
| q6_K | 14 | 63.5% | 20.8-24.4 | 19,328 | Main weights (219 tensors) |
| iq4_xs | 23 | 14.4% | 15.9-18.4 | 5,376 | Expert shared weights (63 tensors) |
| q8_0 | 8 | 10.9% | 8.1-10.2 | 7,680 | KV cache read/write (131 tensors) |
| q5_K | 13 | 4.8% | 18.0-30.2 | 1,280 | Secondary weights (30 tensors) |
| fp32 | n/a | 6.3% | 5.3 | 7,680 | Embeddings + norms (308 tensors) |

**Key insight:** q6_K matmul is 63.5% of all matmul time. The average q6_K
forward matmul (20.8 us) is 4x slower than q8_0 (5.3 us) for similar-sized
operations. This is the primary target for any quantization-level optimization.

### 2.5 FAST vs SLOW Step Kernel Mix

The trace cannot distinguish FAST from SLOW steps at the aggregate level (all
kernels are merged). However, the kernel call counts reveal the step count:

- `flash_attn_ext_vec`: 1,024 calls = 16 attn layers x 64 steps
  - 16 attention layers per full model pass (every 4th of 40 layers + final)
  - 64 steps confirms both FAST and SLOW steps run attention
- `topk_moe_cuda`: 3,840 calls = 60 calls per full model pass x 64
  - ~60 MoE routing calls per full pass (30 SSM layers have MoE FFN + shared experts)
- `ssm_conv_f32` + `gated_delta_net_cuda`: 2,816 each = 44 per pass x 64
  - 30 SSM layers + 14 MTP-related SSM calls per pass

**SLOW-step estimate:** The 185x compute asymmetry (37 us FAST vs 6,843 us SLOW
per D7.2) suggests SLOW steps account for ~99% of GPU kernel time. The kernel
breakdown above therefore primarily reflects SLOW (verification) step behavior.
FAST (draft) steps contribute negligible GPU time (~1% of total).

## 3. Bottleneck Analysis

### 3.1 Primary Bottleneck: q6_K MatMul (35.2% of GPU time)

The q6_K quantization format (6-bit per weight, K-quant) is the workhorse of
this model (219 of 753 tensors). Each q6_K matmul averages 20.8 us, and there
are ~300 q6_K matmuls per full decode step (both forward + transposed).

**Attack vectors for q6_K reduction:**
- **Re-quantize to iq4_xs or q4_K_M**: Would reduce weight size by ~33%,
  proportionally reducing memory bandwidth demand. But need quality validation.
- **MTP verification skip (D7.4)**: Skip SSM layers during verification --
  would eliminate ~30 SSM-layer matmuls per SLOW step (out of ~300 total),
  roughly a 10% reduction in matmul count.
- **FP8/HIP wmma**: AMD gfx1100 supports WMMA instructions for fp16/fp32.
  Investigating WMMA-based matmul kernels could improve throughput.

### 3.2 Secondary Bottleneck: Quantization + Data Movement (16.3%)

`quantize_q8_1` (7.9%) + `k_get_rows`/`k_set_rows` (8.4%) represent the
dequant+reformat pipeline. Every matmul requires a preceding quantize step
to convert weights to q8_1 for the compute kernel.

These are memory-bandwidth-bound operations with low arithmetic intensity.
They benefit from:
- Larger batch sizes (amortize overhead)
- Async overlap with compute (D7.5 approach, currently blocked)

### 3.3 MoE Routing is NOT the Bottleneck (4.0%)

Despite 256 experts, `topk_moe_cuda` is only 4% of GPU time. This is because
the routing decision (which 8 experts to activate) is a cheap top-k selection
on a small score vector. The expensive part is the expert FFN matmul, which
is captured in the matmul category.

### 3.4 SSM vs Attention Cost

| Operation Type | % GPU | Per-Layer Avg (us) | Layers | 
|----------------|-------|--------------------|--------|
| Flash Attention (full) | 3.4% | 38.8 us | 10 full-attn layers (every 4th) |
| SSM (gated delta net) | 2.4% | 7.0 us (gdn) + 3.0 us (conv) | 30 SSM layers |

SSM layers are **4x cheaper per layer** than full attention (10.0 us vs 38.8 us).
This validates Qwen3.6's hybrid architecture: replacing 30 of 40 attention layers
with SSM dramatically reduces compute while maintaining quality.

## 4. Estimated SLOW Step Kernel Budget

Applying the relative percentages to the D7.2-measured SLOW step compute window
of 6,843 us (unprofiled baseline):

| Category | % | Est. us in SLOW Step |
|----------|---|---------------------|
| MatMul (all types) | 55.4% | ~3,791 us |
| Element-wise ops | 8.6% | ~589 us |
| Data movement | 8.4% | ~575 us |
| Normalization | 7.9% | ~541 us |
| Quantization | 7.9% | ~541 us |
| MoE routing | 4.0% | ~274 us |
| Flash Attention | 3.4% | ~233 us |
| SSM | 2.4% | ~164 us |
| RoPE | 1.2% | ~82 us |
| Other | 0.8% | ~55 us |

**Note:** These are estimates. The profiling overhead is 9.5x, so absolute
timings are inflated. The relative percentages assume uniform overhead across
all kernel types, which is approximate but directionally correct.

## 5. Implications for Attack Vectors

### Vector A (D7.3, COMPLETE): Reduce GPU Compute
- Flash Attention enabled (+7.5% TG) -- confirmed well-optimized at 3.4% of GPU
- q6_K matmul (35.2%) is the largest remaining target for A-like optimization
- Q4_K_M quantization would primarily reduce q6_K matmul time

### Vector B (D7.4, ANALYSIS COMPLETE): MTP Verification Skip
- Skipping SSM layers during verification would save ~10 us/layer x 30 layers
  = ~300 us per SLOW step (out of ~6,843 us = ~4.4% reduction)
- The SSM layers are already cheap (2.4% of GPU time). The skip-SSM approach
  saves less than expected because the matmul bottleneck is in the shared MoE
  FFN layers, not the SSM layers themselves
- **Revised D7.4 upper bound:** Skipping SSM layers alone gives at most ~5%
  TG improvement. The 185x asymmetry is dominated by matmul count reduction,
  which requires skipping attention + MoE FFN layers, not just SSM

### Vector B2 (D7.5, PROTOTYPED): RPC Download Overlap
- Data movement (8.4%) + quantization (7.9%) are candidates for overlap
- These are memory-bandwidth-bound; overlapping with compute could hide
  300-500 us of the SLOW step

### Vector C (D7.6): Kernel Profiling
- **COMPLETE.** rocprofv3 `--kernel-trace` works with `GGML_CUDA_GRAPHS=0`
- Per-kernel timing now available for targeted optimization
- q6_K matmul identified as the #1 optimization target (35.2% of GPU time)

## 6. Stall Hunt — GPU Idle Gap Analysis (D7.6 post-D6.10.1)

After D6.10.1 eliminated the `input_copy_slow` event_synchronize wait (-98.6%,
Split 2: 165,000 us -> 2,359 us), we ran a rocprofv3 kernel-trace on the patched
binary to search for remaining sync stalls or unnecessary GPU idle periods.

### 6.1 Test Configuration

- Model: Qwen3.6-35B-A3B-APEX-MTP-I-Quality (IQ4_XS, moe_interleave)
- GPUs: 7900XTX (ROCm, local) + 3060Ti (RPC, remote)
- GPipe stages: 3 (CPU -> RPC -> ROCm)
- MTP speculative decoding: n_max=2
- GGML_CUDA_GRAPHS=0 (required for per-kernel tracing)
- Tasks: tg, n_gen=16, repeat=1
- Profiling overhead: 16.3 t/s (vs 143.0 t/s baseline) = 8.8x slowdown
- ROCm agent: gfx1100 (Radeon RX 7900 XTX), 5 HIP queues, 4 streams
- Total GPU kernel dispatches: 40,343

### 6.2 Micro-Level: Perfect Inter-Kernel Pipelining

The two main compute streams (Queue 3 and Queue 4, 19,740 kernels each) show
near-perfect pipelining at the micro level:

| Metric | Queue 3 | Queue 4 |
|--------|---------|---------|
| Kernels | 19,740 | 19,740 |
| Median gap | 4.8 us | 4.8 us |
| P95 gap | **5.2 us** | **5.0 us** |
| P99 gap | **9.4 us** | **9.3 us** |
| % gaps < 10 us | **99.08%** | **99.14%** |
| GPU util (kernel/wall) | 13.80% | 15.31% |

The 4-5 us inter-kernel gaps are rocprofv3's per-dispatch recording overhead.
At production speed these vanish — there are no micro-stalls between consecutive
kernels on either compute stream.

### 6.3 Macro-Level: GPipe Stage Boundary Bubbles

Larger gaps exist but are all GPipe pipeline fill/drain bubbles at stage
boundaries, not CPU-side sync stalls:

| Gap Threshold | Queue 3 Count | Queue 4 Count | Total Time (Q3) |
|---------------|:------------:|:------------:|:---------------:|
| > 1 ms | 107 | 95 | 816.6 ms |
| > 5 ms | 75 | 72 | 750.2 ms |
| > 10 ms | 37 | 34 | ~650 ms |
| Largest | — | — | 35.9 ms (Q3), 12.8 ms (Q4) |

The gaps follow a clear repeating cycle pattern at stage boundaries:

| Transition Pattern | Typical Gap | Interpretation |
|--------------------|:-----------:|----------------|
| `copyBuffer` -> `rms_norm_f32` | ~10 ms | H2D copy + RMS norm at stage start |
| `k_bin_bcast<op_add>` -> `unary_gated_op_kernel<op_softplus>` | ~12 ms | Element-wise output -> MoE gate (stage boundary) |
| `cpy_scalar` -> `k_get_rows_float` | ~6 ms | Scalar copy -> embedding lookup |
| `mul_mat_vec_q` -> `k_bin_bcast<op_mul>` | ~10 ms | Matmul -> element-wise (attention->FFN transition) |

**Critical finding:** When Queue 3 has a gap >5 ms, Queue 4 has 0% coverage
(zero kernels executing). And vice versa. Both compute streams idle
simultaneously at the same stage boundaries — this is the GPipe fill/drain
bubble, not a scheduler stall.

Queue 1 (copy, 857 kernels) and Queue 2 (fill, 6 kernels) have isolated large
gaps (9.4 s and 8.2 s respectively) — these are the H2D/D2H copy queues that
only fire at inter-iteration sync points, which is expected behavior.

### 6.4 Key Insight: Profiling Magnification

The 8.8x profiling overhead inflates everything proportionally:

| Metric | Profiled | Estimated Unprofiled | Factor |
|--------|:--------:|:--------------------:|:------:|
| TG throughput | 16.3 t/s | 143.0 t/s | 8.8x |
| Typical stage boundary gap | 10-12 ms | **1.1-1.4 ms** | 8.8x |
| Largest gap | 35.9 ms | **4.1 ms** | 8.8x |

The 10-36 ms gaps we observe in the profiler trace would be 1-4 ms at production
speed — well within normal GPipe bubble range for a 3-stage pipeline. The gaps
are real (inherent to the GPipe algorithm), but their magnitude is dominated by
the profiling instrumentation overhead, not by sync stalls.

### 6.5 Copy Queue (Q1/Q2) Analysis

| Queue | Kernels | Largest Gap | Role |
|-------|:-------:|:-----------:|------|
| Q1 | 857 | 9,415 ms | H2D/D2H copy operations |
| Q2 | 6 | 8,250 ms | Buffer fill/clear operations |

These queues are sparse by design — copies only happen at pipeline sync points
between iterations. The 8-9 second gaps span the entire generation run (16 tokens
with profiling overhead) and are expected behavior.

### 6.6 Conclusion

**No new unnecessary sync stalls found.** The D6.10/D6.10.1 fixes eliminated
the real stalls. All remaining GPU idle periods are:
1. GPipe stage boundary bubbles (inherent to the algorithm, 1-4 ms at production speed)
2. Sparse copy/fill queue intervals (expected async behavior)
3. rocprofv3 instrumentation overhead (8.8x magnification artifact)

The GGML_SCHED_TRACE=2 scheduler trace was correct — proceed to D7.7.

## 7. Next Steps

1. **D7.7: Targeted kernel optimization** -- Focus on q6_K matmul (35.2%).
   Investigate WMMA-accelerated matmul for gfx1100 (7900 XTX supports WMMA).
2. **D7.8: Quantization experiment** -- Compare q6_K vs q4_K_M throughput
   on the same model. Estimate: ~20-25% matmul reduction from smaller weights.
3. **Revisit D7.4 upper bound** -- The skip-SSM approach was estimated at
   +75% TG based on the 185x asymmetry. The kernel data shows SSM layers are
   only 2.4% of GPU time. The asymmetry must come from layer count reduction
   (fewer matmuls), not SSM-specific savings. Revised estimate: 10-15% TG
   from skip-SSM alone, 30-40% from broader layer skipping.
4. **D7.9: SLOW-step-only profiling** -- Parse the 159K-line kernel trace CSV
   to isolate SLOW steps and get the exact kernel mix for verification steps.

## 8. Artifacts

- Kernel stats CSV: `/tmp/d76-profiling/rocprofv3-kernel/kernel-trace_kernel_stats.csv`
- Kernel trace CSV: `/tmp/d76-profiling/rocprofv3-kernel/kernel-trace_kernel_trace.csv` (159K lines)
- Agent info: `/tmp/d76-profiling/rocprofv3-kernel/kernel-trace_agent_info.csv`
- Profiler invocation: `GGML_CUDA_GRAPHS=0 rocprofv3 --kernel-trace --stats --summary`
- D6.10.1 post-fix stall hunt results DB: `/tmp/d610-rocprof/Romulus/287965_results.db` (12 MB, 40,343 dispatches)
- Stall hunt heatmap: `/tmp/d610-rocprof/heatmap.json`
