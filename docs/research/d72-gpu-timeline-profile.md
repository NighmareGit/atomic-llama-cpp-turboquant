# D7.2 — GPU Timeline Profiling of event_wait_slot

**Date:** 2026-07-16
**Task:** Vector C — Decompose 8.6ms event_wait_slot into components
**Tool:** GGML_SCHED_TRACE=1 (sched-trace fallback)
**rocprofv3:** Initially crashed (SIGABRT in `ggml_uncaught_exception`). **Resolved in D7.6** — use `--kernel-trace` without `--hip-trace` to avoid HIP interception conflict. See `docs/research/d76-rocprofv3-kernel-profile.md`.

## 1. Tool Status

### rocprofv3 (initially FAILED — RESOLVED in D7.6)
- rocprofv3 available at `/opt/rocm/bin/rocprofv3` (ROCm 7.2.3)
- `--kernel-trace` and `--hip-trace` flags accepted
- **Initial problem (D7.2):** Profiler crashes with SIGABRT when rocprofv3 tracing is enabled
  - Error: `ggml_uncaught_exception()` thrown from `llama_gpipe_profiler`
  - Likely cause: rocprofv3 HIP interception conflicts with ggml's HIP usage
- **Resolution (D7.6):** Use `--kernel-trace` **without** `--hip-trace`. The HIP interception was the crash root cause; kernel-trace alone avoids it. See `docs/research/d76-rocprofv3-kernel-profile.md` for the full per-kernel breakdown (MatMul 55.4%, quantize_q8_1 7.9%, etc.).

### GGML_SCHED_TRACE=1 (SUCCESS)
- Lightweight built-in tracing, no external tools needed
- Captures per-phase timing for each GPipe stage
- Limitation: No GPU kernel-level granularity (only async compute submission time)

## 2. Test Configuration

- Model: Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf
- GPUs: 7900XTX (ROCm, local) + 3060Ti (RPC, remote)
- GPipe stages: 3 (CPU → RPC → ROCm)
- MTP speculative decoding: n_max=2, n_min=1
- KV cache: q8_0/q8_0
- n_gen=8, batch=1, ubatch=256
- GGML_CUDA_GRAPHS=0

## 3. Per-Phase Timing Results

### 3.1 Decode Step Classification

The trace reveals **two distinct decode step types** (alternating pattern):

| Step Type | Count | Split 0 (CPU) | Split 1 (RPC/3060Ti) | Split 2 (ROCm/7900XTX) | Total |
|-----------|-------|---------------|----------------------|------------------------|-------|
| **FAST**  | 20    | 12.3 µs       | 2,975 µs             | 242 µs                 | 3,229 µs |
| **SLOW**  | 16    | 19.6 µs       | 4,665 µs             | 8,261 µs               | 12,946 µs |

Pattern: 5 FAST + 4 SLOW repeating (MTP draft + verification cycles).

### 3.2 FAST Steps — Per-Phase Breakdown

| Phase | Mean (µs) | % of Split Total |
|-------|-----------|------------------|
| **Split 0 (CPU):** | **12.3** | **0.4%** |
| graph_compute_async | 5.7 | 0.2% |
| **Split 1 (RPC/3060Ti):** | **2,975.1** | **92.1%** |
| graph_compute_async | 2,779.3 | 85.5% |
| input_wait_copy | 182.5 | 5.7% |
| input_copy_slow | 60.9 | 1.9% |
| event_record | 6.8 | 0.2% |
| rpc_prefetch_start | 14.2 | 0.4% |
| **Split 2 (ROCm/7900XTX):** | **241.8** | **7.5%** |
| graph_compute_async | 36.7 | 1.1% |
| input_wait_copy | 156.2 | 4.8% |
| rpc_defer_flush | 41.5 | 1.3% |
| rpc_gather_flush | 37.4 | 1.2% |

### 3.3 SLOW Steps — Per-Phase Breakdown

| Phase | Mean (µs) | % of Split Total |
|-------|-----------|------------------|
| **Split 0 (CPU):** | **19.6** | **0.2%** |
| graph_compute_async | 8.7 | 0.1% |
| **Split 1 (RPC/3060Ti):** | **4,665.2** | **36.0%** |
| graph_compute_async | 4,291.1 | 33.1% |
| input_wait_copy | 347.0 | 2.7% |
| input_copy_slow | 152.6 | 1.2% |
| event_record | 17.3 | 0.1% |
| rpc_prefetch_start | 22.9 | 0.2% |
| **Split 2 (ROCm/7900XTX):** | **8,260.6** | **63.8%** |
| graph_compute_async | 6,843.4 | 52.9% |
| input_wait_copy | 1,319.4 | 10.2% |
| input_copy_slow | 2,644.8 | 20.4% |
| rpc_gather_flush | 177.2 | 1.4% |
| rpc_defer_flush | 83.2 | 0.6% |

### 3.4 Key Observation: event_wait_slot = 0

**The `event_wait_slot` phase is NOT present in this trace.** This means:
- The GPU compute completes BEFORE the next decode step starts waiting
- The event wait completes in <1 µs (filtered out by `ev_us > 0` check)
- The 8.6ms event_wait_slot from D7.0 research was from a **different configuration** (1-GPU setup)

This is a critical finding: with 2 GPUs, the compute is fast enough that the inter-decode synchronization barrier is effectively zero.

## 4. The Real Bottleneck: Split 2 (ROCm/7900XTX)

### 4.1 SLOW Step Analysis

In SLOW steps, the ROCm GPU is the bottleneck:
- **graph_compute_async: 6,843 µs** — actual GPU kernel execution
- **input_wait_copy: 1,319 µs** — waiting for RPC download + H2D copy
- **input_copy_slow: 2,645 µs** — RPC download of tensors from 3060Ti

The ROCm GPU spends **42% of its time waiting for input** (input_wait_copy + input_copy_slow = 3,964 µs out of 8,261 µs).

### 4.2 FAST vs SLOW Asymmetry

| Metric | FAST | SLOW | Ratio |
|--------|------|------|-------|
| Split 1 compute | 2,779 µs | 4,291 µs | 1.54x |
| Split 2 compute | 37 µs | 6,843 µs | **185x** |
| Split 2 input wait | 156 µs | 1,319 µs | 8.5x |

The 185x difference in Split 2 compute suggests:
- FAST steps: MTP draft token (minimal layers evaluated)
- SLOW steps: MTP verification (full model evaluation across all layers)

## 5. Pie Chart — SLOW Step Breakdown (12.9ms total)

```
SLOW Decode Step (12,946 µs total)
============================================================

Split 0 (CPU)       19.6 µs   [█]                                    0.2%
Split 1 (RPC)     4,665.2 µs   [█████████████████████████]           36.0%
  graph_compute   4,291.1 µs   [████████████████████████]            33.1%
  input_wait_copy   347.0 µs   [█]                                    2.7%
  input_copy_slow   152.6 µs   [█]                                    1.2%
  other             125.5 µs                                        1.0%
Split 2 (ROCm)    8,260.6 µs   [████████████████████████████████████] 63.8%
  graph_compute   6,843.4 µs   [████████████████████████████████]     52.9%
  input_wait_copy 1,319.4 µs   [█████]                               10.2%
  input_copy_slow 2,644.8 µs   [█████████]                           20.4%
  rpc_gather       177.2 µs                                        1.4%
  rpc_defer_flush   83.2 µs                                        0.6%
  other             192.6 µs                                        1.5%

============================================================
```

**Note:** input_copy_slow for Split 2 (2,645 µs) overlaps with input_wait_copy (1,319 µs) in the async pipeline. The actual wall time is less than the sum.

## 6. Bottleneck Identification

### Primary Bottleneck: Split 2 (ROCm/7900XTX) — 63.8% of SLOW step

Within Split 2:
1. **GPU kernel execution: 6,843 µs (52.9%)** — the dominant cost
2. **RPC download (input_copy_slow): 2,645 µs (20.4%)** — waiting for 3060Ti output
3. **Input wait (input_wait_copy): 1,319 µs (10.2%)** — H2D copy synchronization

### Secondary Bottleneck: Split 1 (RPC/3060Ti) — 36.0% of SLOW step

Within Split 1:
1. **GPU kernel execution: 4,291 µs (33.1%)** — 3060Ti compute
2. **Input wait: 347 µs (2.7%)** — waiting for CPU embedding

## 7. Recommended Attack Vectors

### Vector A: Overlap Split 2 Compute with RPC Download

**Target:** Split 2 input_wait_copy (1,319 µs) + input_copy_slow (2,645 µs) = 3,964 µs

**Mechanism:** The ROCm GPU waits for the 3060Ti to produce tensors, then does H2D copy, then computes. If we can:
- Start RPC download earlier (before previous compute finishes)
- Pipeline the H2D copy with compute (multiple streams)

**Expected gain:** Hide 1,300-2,600 µs of the 8,261 µs Split 2 time (16-31% reduction)

### Vector B: Reduce MTP Verification Cost

**Target:** Split 2 graph_compute_async (6,843 µs in SLOW steps)

**Mechanism:** The 185x difference between FAST (37 µs) and SLOW (6,843 µs) suggests MTP verification is expensive. Options:
- Reduce spec-draft-n-max (fewer tokens to verify)
- Use a smaller draft model
- Skip verification for low-confidence drafts

**Expected gain:** If SLOW steps can be reduced to FAST-like compute, total step time drops from 12,946 µs to ~3,229 µs (4x speedup)

### Vector C: rocprofv3 GPU Kernel Profiling (RESOLVED in D7.6)

**Target:** Decompose 6,843 µs GPU compute into individual kernels

**Status:** ~~BLOCKED~~ **COMPLETE.** D7.6 successfully ran rocprofv3 with `--kernel-trace` (no `--hip-trace`), producing per-kernel breakdowns for 4 models. Key finding: MatMul = 55.4% of GPU time (q6_K alone = 35.2%). See `docs/research/d76-rocprofv3-kernel-profile.md` for full data.

## 8. Comparison with D7.0 Background Data

| Metric | D7.0 (1-GPU) | D7.2 (2-GPU) |
|--------|-------------|-------------|
| event_wait_slot | 8,594 µs (89.9%) | **0 µs (0%)** |
| Split 1 total | ~9,560 µs | 2,975-4,665 µs |
| Split 2 total | N/A | 242-8,261 µs |
| Total step | ~10,600 µs | 3,229-12,946 µs |

**Key insight:** The 2-GPU setup eliminates the event_wait_slot bottleneck by distributing compute across two GPUs. The new bottleneck is the ROCm GPU's compute time in MTP verification steps.

## 9. Next Steps

1. ~~**Fix rocprofv3 compatibility**~~ — **COMPLETE (D7.6).** `--kernel-trace` without `--hip-trace` works. Per-kernel data captured.
2. **Profile GPU kernels** — **COMPLETE (D7.6).** See `docs/research/d76-rocprofv3-kernel-profile.md` for the full breakdown.
3. **Analyze MTP pattern** — Understand why FAST/SLOW steps alternate 5:4. Open lead (see `docs/wayfinder/D7-REEXAMINATION.md`).
4. ~~**Test Vector A**~~ — **COMPLETE (D7.3).** FA on HIP: +7.5% TG.
5. **Test Vector B** — Evaluate MTP verification cost vs. acceptance rate. Superseded by D7.6 finding that SSM = 2.4% of GPU time (skip-SSM upper bound revised to ~5%).
