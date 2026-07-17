# Optimization Landscape: Layer 1-3 Overview

**Date:** 2026-07-17
**Node:** ROMULUS (7900XTX, ROCm 7.2.3)
**Baseline TG:** 133.0 t/s (Qwen3.6-35B-A3B Q6_K, dual-GPU RPC, MTP n_max=2)
**Current TG:** 143.0 t/s (after Layer 3 Vector A: FA on HIP, +7.5%)

---

## Layer Framework

The optimization surface is organized into three layers, from outermost (system) to innermost (kernel):

| Layer | Scope | Levers | Current % of bottleneck |
|-------|-------|--------|------------------------|
| **Layer 1 -- Pipeline/RPC** | How work flows between GPUs and stages | GPipe depth, event pipelining, multi-seq dispatch, RPC overlap | ~10% (mostly eliminated by D6.10) |
| **Layer 2 -- Kernel/MatMul** | How individual GPU kernels compute | WMMA acceleration, LDS caching, dp4a/sudot4, quantization format | **55-76%** (primary bottleneck) |
| **Layer 3 -- Architecture/Model** | What the model architecture enables | Flash Attention, SSM skipping, quantization choice, MTP | Secondary (FA done, SSM risky, MTP already on) |

---

## Layer 1: Pipeline / RPC / System

### Accomplished / Shipped

| Lead | Ticket | Result | TG Delta |
|------|--------|--------|----------|
| GPipe multi-stage depth (n_stages = n_backends + 1) | D5.x | Topology-aware pipeline; stage-level event model. 2-GPU: already saturated by 2-stage copy-slot. 3+ GPU: expected to show gains. | ~0% (2-GPU) |
| Multi-seq Mode B dispatch | D6.5-D6.9 | Stage-available scheduling with per-seq tracking. Double-buffered events. 10/10 GPipe tests pass. | Structural (enables concurrent requests) |
| GPU event pipelining (copy_event per-call) | D6.10 | `input_copy_slow`: 165,000 us -> 2,359 us (-98.6%). Split-time `event_synchronize` replaced with per-stage HIP events on GPU backends. | +4.3% TPS (124.4 -> 129.8) |
| D6.10.1: skip event wait for H2D INPUT copies | D6.10.1 | Host-to-GPU INPUT copies skip `event_synchronize` when n_copies > 1. 12/12 GPipe tests pass. | Included in D6.10 delta |

### Tried and Rejected / Closed

| Lead | Ticket | Result | Why Rejected |
|------|--------|--------|--------------|
| n_copies > 1 (copy-slot rotation) | D7.1 | +0.8-1.4% | Within noise. GPipe uses per-stage events, not `pipeline_barrier()` -- n_copies has no effect on GPipe path. |
| RPC download overlap (H2D async) | D7.5 | No H2D bottleneck found | `input_copy_slow` (2,645 us) is GPU `event_synchronize` wait, not actual H2D copy. 97.6% is 16-byte `leaf_70`. Pivoted to D6.10. |
| Coalescing / Lazy / Gating / Reorder | D7.0 strats 2-5 | 0 impact | Only 2 splits with GPipe stage filtering; no multi-split dispatch to optimize. |

### Open / Pending

| Lead | Ticket | Status |
|------|--------|--------|
| Adaptive pipeline depth (straggler detection) | R3.1-R3.5 | Deferred. Homogeneous-collapse heuristic shipped; adaptive logic planned for 3+ GPU cluster. |
| Pareto optimizer (hot-on-fast placement) | D4.11-D4.14 | Planned + ticketed. Requires heatmap data from 5-GPU cluster. |

### Key Insight

The pipeline layer is **mostly done** for 2-GPU. With only 2 splits (CPU -> RPC -> ROCm) and GPipe stage filtering, there is minimal pipeline-level waste left. The `event_wait_slot` (9ms, 143:1 wait/compute ratio) is a GPU compute bottleneck masquerading as a pipeline bottleneck -- the ROCm stage has only 64 us of work but waits 9ms for the previous decode's GPU event. Reducing GPU compute time (Layer 2) is the actual fix.

**Source docs:** `docs/wayfinder/D7.0-pipeline-depth-research.md`, `docs/wayfinder/D6.10-implementation-analysis.md`, `docs/research/d75-rpc-overlap-research.md`

---

## Layer 2: Kernel / MatMul (PRIMARY BOTTLENECK)

### Research Foundation

**Source:** `docs/research/d76-rocprofv3-kernel-profile.md` + `docs/research/d76b-multi-model-kernel-comparison.md`

The D7.6 rocprofv3 profiling campaign produced per-kernel timing breakdowns across 6 models. Key findings:

| Metric | Qwen MoE Q6_K | Qwen MoE Q4_K | Gemma-4 Dense Q4_K |
|--------|:------------:|:------------:|:-----------------:|
| MatMul % GPU | 55.4% | 60.7% | **76.0%** |
| q6_K alone | 35.2% | 5.4% | -- |
| q4_K alone | -- | 24.0% | 36.4% |
| Attention % GPU | 3.4% | 2.8% | 9.4% |
| Quant (q8_1) % GPU | 7.9% | 7.0% | 2.4% |
| SSM % GPU | 2.4% | 2.1% | -- (dense) |

**Cross-model pattern:** MatMul dominates everywhere (55-76%). q4_K matmul is 29% faster than q6_K per call (14.8 us vs 20.8 us). Dense models (Gemma-4) spend proportionally more in matmul (76% vs 55%) because MoE routing + SSM layers in Qwen dilute the matmul share.

### Accomplished / Shipped

| Lead | Ticket | Result |
|------|--------|--------|
| rocprofv3 kernel profiling | D7.6 | Per-kernel breakdown for 6 models. Fix: `--kernel-trace` without `--hip-trace` avoids HIP interception crash. Guidance: `docs/research/rocprofv3-profiling-guide.md`. |
| Multi-model comparison | D7.6b | 4-model comparison dataset. Q4_K identified as 29% faster matmul vs Q6_K. Dense models have different optimization profile (matmul 76%, attention 10%). |
| q6_K matmul identified as #1 target | D7.6 | 35.2% of all GPU time for Qwen Q6_K models. Each call ~20.8 us, ~300 calls per verify step. |

### In Progress / Prototype

| Lead | Ticket | Status | Mechanism |
|------|--------|--------|-----------|
| **WMMA vec_dot acceleration** | D7.7 | Prototype behind `GGML_HIP_WMMA_VECDOT_EXPERIMENTAL` (OFF). HEAD commit `19db22abb`. | Uses `rocwmma` matrix instructions on RDNA3 (gfx1100) for `vec_dot_q4_K_q8_1_impl_vmmq` -- wraps existing `dp4a` (via `__builtin_amdgcn_sudot4`) in WMMA fragment ops. |
| **LDS activation caching** | D7.8 | Prototype behind `GGML_HIP_MMVQ_LDS_PROTOTYPE` (OFF). COMPLETE -- CMake misconfiguration resolved. No significant throughput delta. Files: `ggml/src/ggml-cuda/mmvq.cu` lines 745-870, 1060. | Cooperatively loads 24 `block_q8_1` structures from global into `__shared__` once per MMVQ iteration, eliminating 16x redundant reads per half-warp. Inlined `vec_dot` reads `__shared__` directly. |

### Open / Pending

| Lead | Priority | Rationale | Estimated Gain |
|------|----------|-----------|---------------|
| **q4_K quantization re-target** | HIGH | q4_K matmul is 29% faster than q6_K. Re-quantizing weights to q4_K_M reduces matmul time proportionally. Already proven for Qwen (Q4_K model shows 60.7% matmul share at lower absolute time). | ~15-20% TG |
| **WMMA for additional matmul types** | MEDIUM | D7.7 prototypes q4_K only. Extend to q5_K, q6_K, q8_0, iq4_xs. Requires per-quant-format WMMA fragment mapping. | Cumulative with D7.7 |
| **LDS caching for MMQ (batched) path** | LOW | D7.8 prototypes LDS for MMVQ (single-token decode). Batched MMQ (prompt processing) could also benefit, but prompt processing is already fast (240-280 t/s). | Marginal |
| **FP8 compute** | EXPLORATORY | AMD gfx1100 supports FP8 via WMMA. Would require new quantization format + kernel. | Potentially large but high engineering cost |

### Key Insight

Layer 2 is where the real throughput gains live. 55-76% of GPU time is matmul. The q4_K vs q6_K gap alone suggests a ~15-20% TG improvement from re-quantization. The WMMA and LDS prototypes attack from the kernel side -- making each matmul call faster rather than reducing the number of calls.

**Source docs:** `docs/research/d76-rocprofv3-kernel-profile.md`, `docs/research/d76b-multi-model-kernel-comparison.md`, `docs/research/d73-vector-a-gpu-compute-reduction.md`

---

## Layer 3: Architecture / Model-Level

### Accomplished / Shipped

| Lead | Ticket | Result | TG Delta |
|------|--------|--------|----------|
| **Flash Attention on HIP** | D7.3 | `GGML_HIP_ROCWMMA_FATTN=ON`. Uses rocWMMA for WMMA flash attention on RDNA3 (gfx1100). WMMA FA kernel verified in `libggml-hip.so`. | **+7.5%** (133.0 -> 143.0 t/s) |
| **MTP speculative decoding** | (existing) | n_max=2, n_min=1. FAST/SLOW alternating pattern: 5 draft steps (3,229 us) + 4 verify steps (12,946 us). Each verify step covers the full 40-layer model. | +74% TPS (vs no-MTP baseline) |

### Tried and Rejected / Deferred

| Lead | Ticket | Result | Why Rejected |
|------|--------|--------|--------------|
| **Skip-SSM verify (full)** | D7.4 | **+75% TG upper bound, output collapses.** Skip 30 SSM layers during MTP verification -- saves ~1,000 us SSM compute + SSM portion of MoE FFN. Output quality degrades severely because `h_nextn` (MTP head seed) comes from layer 39 (last full-attention layer), and skipping SSM layers corrupts the hidden state that feeds it. | Quality collapse. 5 refinement approaches (R1-R5) cataloged with decision matrix + revisit criteria. |

### Open / Pending

| Lead | Priority | Rationale | Estimated Gain |
|------|----------|-----------|---------------|
| **Skip-SSM refinement R1-R5** | MEDIUM | Partial skip, soft skip, selective MoE skip, quality-gated skip, or 2-pass verify. Each trades quality for speed differently. | +10-50% TG depending on approach |
| **Re-quantize model to q4_K_M** | HIGH | 29% faster matmul per call. No code changes needed -- purely a model conversion decision. Requires quality validation (perplexity comparison). | ~15-20% TG |
| **Dense model optimization** | EXPLORATORY | Gemma-4-12B spends 76% in matmul (vs 55% for Qwen MoE). MoE routing + SSM layers "dilute" matmul share in Qwen. Dense models need different optimization strategy -- matmul kernel speed is even more critical. | Model-dependent |

### Key Insight

Layer 3 is about making smart architectural choices rather than writing new code. FA on HIP was a one-line CMake change (+7.5%). Re-quantization to q4_K is a model conversion with no code changes (+15-20%). These are the highest ROI-per-engineering-hour optimizations available. Skip-SSM has massive theoretical upside but quality risk -- the 5 refinement approaches need careful evaluation.

**Source docs:** `docs/research/d73-vector-a-gpu-compute-reduction.md`, `docs/research/d74-code-skip-ssm-verify.md`, `docs/research/d74-mtp-verification-analysis.md`

---

## Cross-Layer Interactions

| Interaction | Effect |
|-------------|--------|
| Layer 2 -> Layer 1 | Faster matmul (Layer 2) directly reduces the `event_wait_slot` (Layer 1) because the ROCm stage finishes sooner, unblocking the next decode earlier. |
| Layer 3 -> Layer 2 | Re-quantizing to q4_K (Layer 3) changes which matmul kernels are invoked and their per-call cost (Layer 2). WMMA/LDS prototypes for q4_K (Layer 2) compound with q4_K model (Layer 3). |
| Layer 3 -> Layer 1 | MTP (Layer 3) creates the FAST/SLOW alternating pattern that Layer 1's multi-seq dispatcher exploits for pipeline fill. |
| Layer 1 -> Layer 2 | GPipe per-stage events (Layer 1) gate when Layer 2 kernels can launch -- the 143:1 wait/compute ratio means Layer 2 runs for 64 us then waits 9ms. |

---

## Current Bottleneck Map (Qwen3.6-35B-A3B Q6_K, SLOW decode step)

```
12,946 us SLOW decode step
├── Layer 1 (Pipeline): ~1,366 us (10.6%)
│   ├── input_wait_copy: 1,319 us (waiting for RPC stage completion)
│   └── other dispatch overhead: ~47 us
└── Layer 2+3 (GPU compute): 6,843 us (52.9%)
    ├── MatMul (Layer 2): ~3,790 us (55.4% of GPU = q6_K 35.2%, iq4_xs 8.0%, etc.)
    ├── Quantize q8_1 (Layer 2): ~540 us (7.9%)
    ├── Element-wise (Layer 2): ~589 us (8.6%)
    ├── Data movement (Layer 2): ~575 us (8.4%)
    ├── RMS/L2 Norm (Layer 2): ~541 us (7.9%)
    ├── MoE routing (Layer 3): ~274 us (4.0%)
    ├── Flash Attention (Layer 3): ~233 us (3.4%)  [already optimized via WMMA]
    ├── SSM (Layer 3): ~164 us (2.4%)
    └── RoPE/Other (Layer 2): ~137 us (2.0%)
```

Remaining 4,737 us (36.6%) is outside GPU compute window -- synchronization gaps between stages.

---

## Priority Order (Highest ROI First)

| # | Lead | Layer | Est. TG Gain | Engineering Cost | Risk |
|---|------|-------|:-----------:|:----------------:|------|
| 1 | Re-quantize model to q4_K_M | 3 | +15-20% | Low (model conversion) | Quality validation needed |
| 2 | WMMA vec_dot (D7.7) -- complete prototype | 2 | +5-15% | Medium (kernel work) | Needs correctness validation |
| 3 | LDS caching (D7.8) -- complete prototype | 2 | +3-8% | Medium (kernel work) | Needs correctness validation |
| 4 | Extend WMMA to q5_K/q6_K/q8_0 | 2 | +5-10% cumulative | High (per-quant work) | Per-quant format mapping |
| 5 | Skip-SSM refinement (R1-R5) | 3 | +10-50% | High (quality-sensitive) | Output quality collapse |
| 6 | Adaptive pipeline depth (R3.x) | 1 | 0-5% | Medium | Only for 3+ GPU |
| 7 | Pareto optimizer (D4.11+) | 1 | Variable | High (full system) | Requires 5-GPU cluster |

---

## Reference: All Source Documents

| Doc | Covers |
|-----|--------|
| `docs/research/d76-rocprofv3-kernel-profile.md` | Per-kernel GPU timing (Qwen MoE Q6_K baseline) |
| `docs/research/d76b-multi-model-kernel-comparison.md` | 4-model comparison, q4_K vs q6_K matmul speed |
| `docs/research/d73-vector-a-gpu-compute-reduction.md` | FA on HIP (+7.5% TG) |
| `docs/research/d74-code-skip-ssm-verify.md` | Skip-SSM architecture + 5 refinement approaches |
| `docs/research/d75-rpc-overlap-research.md` | RPC overlap investigation (resolved: no H2D bottleneck) |
| `docs/research/d72-gpu-timeline-profile.md` | FAST/SLOW step pattern, per-stage timing |
| `docs/research/split-overhead-mitigation.md` | Original n_copies research |
| `docs/research/rocprofv3-profiling-guide.md` | How to run rocprofv3 with ggml |
| `docs/wayfinder/D7.0-pipeline-depth-research.md` | Pipeline depth + n_copies closure |
| `docs/wayfinder/D6.10-implementation-analysis.md` | GPU event pipelining fix (-98.6% input_copy_slow) |
| `docs/wayfinder/TRACKING.md` | Full ticket status for all D7.x + R3.x |
| `docs/tickets/path-d-slices.md` | Slice definitions + D7.x acceptance criteria |
| `ggml/src/ggml-cuda/mmvq.cu` | D7.7 WMMA + D7.8 LDS prototype code |
| `docs/wayfinder/HANDOFF-D7.8-LDS-prototype.md` | Current state + next steps for D7.8 |
