# D7.8 — Pipeline Barrier RPC Synchronization Stall

**Date:** 2026-07-19
**Status:** OPEN — Root cause identified, fix pending
**Task:** Identify why 9B-MTP Q4_K_M achieves only 55.7 t/s on dual-GPU ROMULUS (7900XTX ROCm + 3060Ti RPC via PCIe) with PPLUS=1
**Finding:** Pipeline barrier blocks on RPC backend event synchronize (12ms per token, 46% of wall time). The per-split input handling already synchronizes RPC events, making the barrier's RPC wait redundant.
**Target:** Skip RPC backend event wait in pipeline barrier during graph reuse. Per-split input handling provides sufficient synchronization.

---

## 1. Executive Summary

A 9B Q4_K_M model running on a 7900 XTX with `-sm layer -ts 75,25` and a 3060 Ti via RPC achieves only **55.7 t/s** in token generation mode. The 7900 XTX alone is capable of ~260 t/s for this model size. The bottleneck is **not GPU compute** (only 15.5% GPU utilization) and **not PCIe bandwidth** (local loopback + PCIe 3.0 x4 = 4 GB/s). The bottleneck is a **software synchronization stall** in `ggml_backend_sched_pipeline_barrier`.

ROCprofv3 kernel tracing reveals:
- 22,632 kernel dispatches, 523 ms total GPU time across 3,173 ms wall span
- **123 × 12ms stalls** between compute kernel completion and DMA copy start
- These 12ms stalls consume **1,465 ms = 46.2% of wall time**
- Pattern: Q4_K matvec (queue 3) -- 12ms gap -- copyBuffer (queue 2)

The stall is caused by `pipeline_barrier` calling `ggml_backend_event_synchronize` on the RPC backend's event, which does a **blocking socket read** waiting for the event response from the RPC server (3060 Ti Docker container). With `RPC_EVENT_DEFER_BARRIER=1`, the event record is sent asynchronously, but the response is consumed synchronously at the barrier.

The per-split input handling (`compute_splits`, lines 2729-2736 of ggml-backend.cpp) already synchronizes RPC events before using input tensors. The pipeline barrier's RPC wait is therefore **redundant** and creates unnecessary blocking.

---

## 2. Evidence

### 2.1 Profiler Configuration

```bash
GGML_CUDA_GRAPHS=0 GGML_PIPELINE_PLUS=1 \
rocprofv3 --kernel-trace --stats --summary \
  -d /tmp/rocprof-data-20260719-191812 -o roc-kernel -f csv \
  -- llama-server -m Qwen3.5-9B-MTP-Q4_K_M.gguf --rpc 127.0.0.1:50051 \
    -sm layer -ts 75,25 -ngl 99 -c 2048 --host 127.0.0.1 --port 8080
```

Inference: "What is the capital of France?" max_tokens=128, temp=0, seed=42

### 2.2 Key Metrics

| Metric | Value |
|--------|-------|
| Wall span (inference) | 3,173 ms |
| Total GPU time | 493 ms |
| GPU utilization | **15.5%** |
| Inter-kernel gaps >100us | 601 (3.0% of all gaps) |
| Total gap time | 2,686 ms |
| **12ms stalls (q3->q2, matvec->copyBuffer)** | **123 stalls, 1,465 ms total = 46.2%** |

### 2.3 The 12ms Stall Pattern

Every instance follows the same pattern:

```
Queue 3 compute: void mul_mat_vec_q<(ggml_type)14, 1, false, false>  (~300 us)
  → 12ms STALL
Queue 2 copy:   __amd_rocclr_copyBuffer
```

Queue 3 = compute engine (HIP), Queue 2 = DMA copy engine. The copy cannot start because it depends on data from the RPC backend (3060 Ti) that hasn't arrived yet.

### 2.4 Queue Transition Analysis

| Transition | Gaps >100us | Description |
|-----------|-------------|-------------|
| q3->q3 | 419 | Compute-to-compute (tiny gaps, normal) |
| q2->q2 | 156 | Copy-to-copy (normal) |
| **q3->q2** | **129** | **Compute-to-copy: includes all 12ms stalls** |
| q3->q2 (12ms) | 123 | The 12ms = RPC sync stall |

---

## 3. Root Cause Analysis

### 3.1 Control Flow

In `src/llama-context.cpp:1605-1611`, during graph reuse:

```cpp
if (cparams.pipeline_parallel) {
    if (llama_pipeline_plus_enabled() && !llama_pipeline_reuse_full_sync()) {
        ggml_backend_sched_pipeline_barrier(sched.get());  // <-- called here
    } else {
        ggml_backend_sched_synchronize(sched.get());
    }
}
```

### 3.2 Pipeline Barrier (ggml-backend.cpp:3385-3490)

The barrier:
1. Computes `wait_mask` from `barrier_copy_src_mask & pending_mask`
2. For each backend in `wait_mask`:
   - GPU backends with `wf_cross`: `ggml_backend_event_wait` → `cudaStreamWaitEvent` (**non-blocking**)
   - **RPC backends: `ggml_backend_event_synchronize` → `rpc_finish_event_response` → blocking socket read** (12ms stall)
3. Then drains RPC backends via `ggml_backend_synchronize` (if `EVENT_DEFER_BARRIER`)

### 3.3 The Redundancy

The per-split input handling in `ggml_backend_sched_compute_splits` (lines 2729-2736) **already** waits for RPC events:

```cpp
if (sched->events[input_bid][sched->cur_copy] != NULL) {
    if (wf_cross && !ggml_backend_is_rpc_backend(split_backend)) {
        ggml_backend_event_wait(split_backend, ...);  // non-blocking stream wait
    } else {
        ggml_backend_event_synchronize(sched->events[input_bid][sched->cur_copy]);  // blocking
    }
}
```

Every split that consumes RPC backend output synchronizes with it before using the data. The pipeline barrier's RPC wait is **doubly redundant**:
- The barrier waits => 12ms
- The compute split waits again (but the event is already signaled, so it returns instantly)

### 3.4 Why Copy-Slot Rotation Doesn't Help

The fix `f68e17b9b` disables copy-slot rotation during graph reuse to prevent KV cache garbling. Without rotation, every token maps to the same copy slot (`cur_copy`), and the barrier must wait for all producers. With rotation, different tokens would use different slots and no wait would be needed — but rotation caused the garble bug that f68e17b9b fixed.

---

## 4. Proposed Fix

### 4.1 Fix Location

`ggml_backend_sched_pipeline_barrier` in `ggml/src/ggml-backend.cpp`, lines 3437-3448.

### 4.2 Fix: Skip RPC Backend Event Wait in the Barrier Loop

The barrier should not call `ggml_backend_event_synchronize` on RPC backend events, because the per-split compute handling already handles this synchronization. Instead, the RPC backend should be excluded from the event wait loop, and the deferred RPC drain (lines 3452-3467) should be the sole RPC synchronization point in the barrier.

Current code (lines 3437-3448):
```cpp
for (int i = 0; i < sched->n_backends; i++) {
    if (!(wait_mask & (1u << i)) || sched->events[i][new_copy] == NULL) {
        continue;
    }
    // B+14 W2: non-blocking event_wait on GPU backends when cross wavefront is on.
    if (wf_cross && !ggml_backend_is_rpc_backend(sched->backends[i])) {
        ggml_backend_t b = sched->backends[i];
        if (b->iface.event_wait != NULL) {
            ggml_backend_event_wait(b, sched->events[i][new_copy]);
            continue;
        }
    }
    ggml_backend_event_synchronize(sched->events[i][new_copy]);
}
```

The fix: when `wf_cross` is enabled (which it is with PPLUS=1), RPC backends in the wait_mask should be skipped in the event wait loop. The deferred RPC drain after the loop will handle them.

Reasoning:
- `wf_cross` is only enabled with `PPLUS=1` and `GGML_SCHED_WAVEFRONT_CROSS` (or wavefront master)
- With PPLUS=1, the per-split compute already synchronizes RPC inputs before using them
- The deferred RPC drain (`ggml_backend_synchronize` on RPC backends) provides the final synchronization
- During graph reuse (no copy-slot rotation), skipping the RPC event wait is safe because the copy slot is frozen

---

## 5. Expected Improvement

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| Wall time per token | 17.96 ms | ~5-6 ms |
| t/s | 55.7 | ~170-200 |
| GPU utilization | 15.5% | ~50-60% |

The remaining bottleneck will be the 3060 Ti compute time (25% of model layers at ~3-5ms) plus PCIe transfer overhead.

---

## 6. Related Work

- **D7.5 — RPC Overlap Research** (`docs/research/d75-rpc-overlap-research.md`): Previous investigation into RPC download overlap. Found that GPU event synchronization (not H2D copies) was the bottleneck.
- **f68e17b9b**: Fix for multi-GPU KV garble under PPLUS=1. Disables copy-slot rotation during graph reuse.
- **D6.10**: GPU event pipelining (the original proposed fix for the bottleneck D7.5 identified).

---

## 7. Test Results (Before Fix, 9B-MTP Q4_K_M)

| Config | t/s | Accept Rate | Notes |
|--------|-----|-------------|-------|
| No RPC (single GPU) | ~260 | — | Estimated from GPU compute time per token |
| RPC + PPLUS=1 + blessed | 55.7 | 97.6% | Current: bottlenecked by barrier RPC sync |
| RPC + GPipe=1 (triton) | 117.8 | 97.6% | GPipe hides the sync on triton |
