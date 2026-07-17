# D7.5 — Overlap RPC Download with GPU Compute (RESOLVED: No H2D bottleneck found)

**Date:** 2026-07-16
**Status:** RESOLVED — Pivot to D6.10 GPU event pipelining
**Task:** Vector B2 — Investigate hiding Split 2's `input_copy_slow` (2,645 µs, 20.4% of SLOW decode) behind GPU kernel execution
**Finding:** `input_copy_slow` is GPU event synchronization, not H2D copies. No copy to overlap.
**Target:** Redirect effort to D6.10 (GPU event pipelining) which addresses the real bottleneck.
**Context:** Qwen3.6-35B-A3B-APEX-MTP on dual-GPU ROMULUS (7900XTX ROCm + 3060Ti RPC), GPipe stages=3, MTP n_max=2

---

## 1. Executive Summary

Split 2 (ROCm/7900XTX) spends 42% of its 8,261 µs SLOW-decode window waiting for inputs (input_wait_copy 1,319 µs + input_copy_slow 2,645 µs = 3,964 µs). The existing `rpc_prefetch_start` mechanism overlaps the RPC wire transfer (GET request / response) with Split 1's compute, but the **local H2D copy** — moving data from staging buffers to the ROCm GPU tensor — remains synchronous and sequential.

The fix requires enabling async H2D copy on the ROCm backend, launching copies on a separate stream, and starting them BEFORE the GPU synchronize barrier. This would overlap the 2,645 µs copy with Split 1's RPC compute, effectively hiding it behind the 4,291 µs that Split 1 spends in `graph_compute_async`.

---

## 2. Current RPC Download Architecture

### 2.1 Split Processing Pipeline

`ggml_backend_sched_compute_splits()` (`ggml/src/ggml-backend.cpp:2305`) iterates splits sequentially:

```
Split 0 (CPU, 12-20 µs)
  -> Split 1 (RPC/3060Ti, 2,975-4,665 µs)
  -> Split 2 (ROCm/7900XTX, 242-8,261 µs)
```

For the SLOW verify step (12,946 µs total), Split 2 dominates at 63.8% of total time.

### 2.2 Per-Split Processing Phases

Each split goes through these phases in order:

| Phase | Split 1 (RPC) | Split 2 (ROCm) SLOW |
|-------|:-------------:|:-------------------:|
| rpc_gather_prefetch_early | — | — |
| input_wait_copy (event wait) | 347 µs | 1,319 µs |
| rpc_gather_flush | — | — |
| INPUT tensor copy | 152 µs | **2,645 µs** |
| rpc_defer_flush / rpc_flush | — | 41 µs |
| graph_compute_async | 4,291 µs | 6,843 µs |
| rpc_prefetch_start | 22 µs | — |
| event_record | 17 µs | — |

### 2.3 Two-Phase Input Copy Design

The scheduler handles inputs in two separate loops:

**Phase A: INPUT-tagged tensors** (lines 2426-2460) — User-provided inputs (embeddings, positions, masks). ALWAYS synchronous:
```cpp
if (input->flags & GGML_TENSOR_FLAG_INPUT) {
    ggml_backend_event_synchronize(event);  // wait for GPU ready
    ggml_backend_tensor_copy(input, input_cpy);  // synchronous copy
}
```

**Phase B: Gather tensors** (lines 2461-2650) — Tensors from other splits/backends. Two-pass:
- Pass 0: RPC-sourced tensors → issue async GET via `try_async_tensor_copy` or fallback to sync
- Pass 1: Non-RPC-sourced tensors → host-to-device copies, some async via `cpy_tensor_async`

The `input_copy_slow` (2,645 µs) is the total time spent in **Phase A** for INPUT-tagged tensors on Split 2.

### 2.4 RPC Prefetch Mechanism (What It Overlaps)

The `rpc_prefetch_start` mechanism (line 2367-2379) runs AFTER Split 1's graph_compute_async completes. It issues deferred RPC GET_TENSOR requests for Split 2's gather inputs:

```
Split 1: graph_compute_async (4,291 µs)
  -> rpc_prefetch_start: issues GET_TENSOR for Split 2's gather inputs (22 µs)
Split 2: rpc_gather_prefetch_early: issues more GET_TENSOR if needed
  -> rpc_defer_flush: waits for GET responses + copies staging -> GPU (41 µs)
```

The wire transfer (network round-trip for GET_TENSOR) IS overlapped with Split 1's compute because the prefetch starts before Split 2's event wait. However, the **local H2D copy** (staging buffer → GPU tensor) happens during Split 2's copy phase and is NOT overlapped with any compute.

### 2.5 RPC get_tensor_async Infrastructure

The RPC backend already has an async get_tensor mechanism (`ggml/src/ggml-rpc/ggml-rpc.cpp`):

```cpp
// ggml_backend_rpc_buffer_get_tensor_async (line 1454)
// Sends RPC_CMD_GET_TENSOR, adds to tls_pending_get_tensor queue
// Response NOT waited for — handled later by flush_pending_get_tensor_for_socket

struct rpc_pending_download {
    socket_ptr sock;
    std::vector<uint8_t> staging;
    ggml_tensor * dst;
    ggml_backend_t dst_backend;
};
// tls_pending_downloads: deferred downloads pending flush
```

The flush at line 1635 (`ggml_backend_rpc_flush_pending_downloads_for_dst`) does:
1. `flush_pending_get_tensor_for_socket(sock)` — waits for RPC response (wire data → staging buffer)
2. `ggml_backend_tensor_set(dst, staging.data(), ...)` — copies staging → GPU (synchonous, default path)

There's a `defer_h2d_sync` path that uses `ggml_backend_tensor_set_async` instead, but this requires a separate stream and proper synchronization.

### 2.6 The Critical Gap

**What's overlapped:** RPC wire transfer (GET request → response received) with Split 1's compute.

**What's NOT overlapped:**
1. Local H2D copy (staging buffer → GPU tensor) — synchronous, sequential
2. Any INPUT-tagged tensors that source from RPC backends — `ggml_backend_tensor_copy()` on an RPC→GPU path falls through to the slow path:
```cpp
// ggml-backend.cpp:795 — the "slow copy" fallback
size_t nbytes = ggml_nbytes(src);
void * data = malloc(nbytes);
ggml_backend_tensor_get(src, data, 0, nbytes);  // RPC network call!
ggml_backend_tensor_set(dst, data, 0, nbytes);  // H2D copy
free(data);
```

---

## 3. What input_copy_slow Actually Contains

The `input_copy_slow` (2,645 µs) is the sum of all INPUT-tagged tensor copies that individually exceed 50 µs. These are Phase A copies. Candidate tensors include:

| Candidate | Source Buffer | Dest Buffer | Est. Size | Copy Path |
|-----------|---------------|-------------|-----------|-----------|
| Token embeddings | Host (CPU) | ROCm (GPU) | ~6 KB (3 tokens × 2048-dim) | Host→GPU set |
| Position IDs | Host (CPU) | ROCm (GPU) | ~12 B (3 int32) | Host→GPU set |
| inp_Z (MTP seed) | RPC? | ROCm (GPU) | ~49 MB (3×16384-dim fp16) | RPC→Host→GPU |
| KV cache refs | Varies | ROCm (GPU) | — | Varies |
| inp_embd (input embeddings) | ROCm (GPU) | ROCm (GPU) | ~6 KB | Same-device |

The most likely cause of 2,645 µs is **inp_Z** — the MTP seed hidden state tensor. For n_draft=2, this is 3 positions × hidden_dim 16384 × fp16 = ~98 KB. But actually, the input tensor might also include tensors whose source buffer is on RPC backend, triggering the slow-path RPC network copy.

**To confirm:** Re-run the profiler with GGML_SCHED_TRACE=2 to capture individual tensor names and sizes in the `input_copy_slow` trace entries (the `sched_trace_emit_sync_detail` function logs tensor name, size, src_buft, dst_buft).

---

## 4. Approach Catalog

### 4.1 Approach A1: Async H2D via Separate Copy Stream

**Concept:** For INPUT-tagged tensor copies on non-RPC GPU backends, use `ggml_backend_tensor_set_async` instead of `ggml_backend_tensor_copy`, launching H2D copies on a stream separate from the compute stream. The copies run concurrently with Split 1's RPC compute (since ROCm GPU is idle during Split 1). Synchronize the copy stream before graph_compute_async.

**Current blocking code** (`ggml-backend.cpp:2426-2460`):
```cpp
// INPUT tensor copy — always synchronous
ggml_backend_event_synchronize(event);
ggml_backend_tensor_copy(input, input_cpy);
```

**Proposed change:**
```cpp
// INPUT tensor copy — async when possible
ggml_backend_event_synchronize(event);
if (ggml_backend_sched_try_async_tensor_copy(sched, input_backend, split_backend, input, input_cpy)) {
    // H2D copy issued on separate stream, no blocking
    // Sync point: before graph_compute_async
} else {
    ggml_backend_tensor_copy(input, input_cpy);
}
```

**Infrastructure needed:**
- `try_async_tensor_copy` already exists and handles host→GPU async via `cpy_tensor_async` (line 2059-2067)
- Need a sync point before `graph_compute_async`: `ggml_backend_synchronize_copy_stream(split_backend)`
- The ROCm HIP backend needs a dedicated copy stream

**Estimated savings:** 2,645 µs → ~100 µs (event_synchronize overhead). The 2,645 µs of H2D copies overlap with Split 1's 4,291 µs compute.

**Risks:** Low. The ROCm GPU is idle during Split 1, so copy stream concurrency has no resource contention.

### 4.2 Approach A2: Move H2D Copy Before Copy-Slot Wait

**Concept:** Issue the H2D copies BEFORE the Split 2 copy-slot wait (event_synchronize). The copies run while the previous Split 2 GPU work drains.

This requires knowing which inputs are "ready" (don't depend on Split 1's outputs). User-provided INPUT tensors are always ready — they're not computed by any split.

```cpp
// B+13b-style early-issue: launch H2D copies before event_wait
if (need_copy_slot_wait) {
    for (int input_id = 0; ...) {
        if (input->flags & GGML_TENSOR_FLAG_INPUT) {
            // This copy doesn't depend on Split 1; issue it early
            try_async_tensor_copy(sched, input, input_cpy);
        }
    }
}
// Then do the event_wait
ggml_backend_event_synchronize(event);
// Then do gather handling...
```

**Estimated savings:** Same as A1, but cleaner integration with existing B+13b prefetch pattern.

**Risks:** Low. Must ensure the copy stream is separate from the event stream or the copy completes before graph_compute reads the tensors.

### 4.3 Approach A3: Async RPC Flush (defer_h2d_sync)

**Concept:** In `ggml_backend_rpc_flush_pending_downloads_for_dst`, use `ggml_backend_tensor_set_async` for the staging→GPU copy instead of `ggml_backend_tensor_set`. Then synchronize the copy stream before graph_compute_async.

**Current code** (`ggml-rpc.cpp:1657-1663`):
```cpp
const bool defer_h2d_sync = ggml_backend_rpc_get_tensor_defer();
if (defer_h2d_sync) {
    ggml_backend_tensor_set(it->dst, it->staging.data(), 0, it->staging.size());  // SYNC
} else {
    ggml_backend_tensor_set_async(it->dst_backend, it->dst, ...);  // ASYNC
}
```

The `defer_h2d_sync` path currently uses SYNCHRONOUS `tensor_set`. The non-defer path uses async `tensor_set_async`. But the non-defer path is rarely taken in the GPipe GET_TENSOR_DEFER mode.

**Proposed change:** Always use `tensor_set_async` with deferred sync:
```cpp
// Always async — sync point moved to before graph_compute_async
ggml_backend_tensor_set_async(it->dst_backend, it->dst, it->staging.data(), 0, it->staging.size());
```

**Estimated savings:** The `rpc_defer_flush` (41 µs) is already small, so this approach alone doesn't address the main bottleneck. Combine with A1.

### 4.4 Approach A4: Double-Buffered RPC Downloads

**Concept:** While Split 2's GPU is computing (graph_compute_async), prefetch Split 2's NEXT-cycle inputs via RPC. This is the analog of `rpc_prefetch_start` but for Split 2 rather than Split 1.

Currently, `rpc_prefetch_start` runs after Split 1's compute to prefetch Split 2's inputs. During Split 2's compute, no prefetch happens for the NEXT token's Split 2 needs.

**Implementation:**
- After Split 2's graph_compute_async, before event_record, issue RPC GETs for NEXT token's gather inputs
- These GETs overlap with event_record + Split 0/1 of the next token

**Estimated savings:** ~300-500 µs per SLOW step (the portion of gather input handling that can be pre-overlapped).

**Risks:** Medium. Requires double-buffered staging buffers (one being filled by current prefetch, one holding previous prefetch data).

### 4.5 Approach A5: GPU-Side RPC Download Stream

**Concept:** Create a dedicated RPC download stream on the ROCm GPU. This stream runs concurrently with the compute stream. When RPC GET responses arrive, the staging→GPU copy is issued on the download stream. The compute stream uses events to wait for specific tensors when it needs them.

This is the most sophisticated approach — similar to CUDA's async memory copy model.

**Implementation:**
1. Create `ggml_rpc_download_stream_t` per GPU backend
2. Each pending download has a completion event
3. graph_compute_async inserts wait events for tensors it needs
4. The download stream runs independently, filling GPU tensors as RPC responses arrive

**Estimated savings:** 2,645 µs + 1,319 µs input_wait_copy → near-zero (fully overlapped).

**Risks:** HIGH. Requires careful stream synchronization, GPU-side staging buffers, and significant scheduler changes.

---

## 5. Decision Matrix

| Approach | Est. Savings | Quality Risk | Implementation Effort | Prerequisite |
|----------|:-----------:|:------------:|:---------------------:|-------------|
| A1 (async H2D for INPUT tensors) | 2,645 µs → <200 µs | Low | Low (plumb existing `try_async_tensor_copy`) | None |
| A2 (early-issue before event wait) | Same as A1 | Low | Low-Medium | None |
| A3 (async RPC flush) | 41 µs → ~5 µs | Low | Low | None |
| A4 (double-buffered prefetch) | ~300-500 µs | Medium | Medium | A1 or A2 |
| A5 (GPU-side download stream) | All 3,964 µs → near-zero | High | High | A1-A4, ROCm HIP stream support |

---

## 8. Prototype Findings (2026-07-16)

### 8.1 Approach A1: Async H2D via `cpy_tensor_async`

**Attempt:** Route INPUT-tagged tensors through `ggml_backend_sched_try_async_tensor_copy()` instead of `ggml_backend_tensor_copy()`. The CUDA/HIP backend's `cpy_tensor_async` uses `cudaStreamPerThread` for H2D copies with a shared `copy_event`.

**Blockers discovered:**

1. **Shared `copy_event` overwrite** (`ggml-cuda/common.cuh:1413`): `ggml_cuda_issue_pinned_h2d_async` uses a single `cudaEvent_t copy_event` per CUDA context. Each call records on `cudaStreamPerThread` then waits on the compute stream. Multiple calls overwrite the event; only the last copy is waited on. **Fixed** by replacing with per-call events (`pending_copy_events` vector in `ggml_backend_cuda_context`).

2. **Illegal memory access with fixed events**: Even with per-call events, `ggml_backend_tensor_set_async` (via `ggml_cuda_issue_pinned_h2d_async`) causes "illegal memory access" during TG decode. The copies appear to target correct GPU addresses, but the error surfaces during `graph_compute_async` kernel execution. Suspected root cause: **CUDA graph capture incompatibility** — the profiler captures CUDA graphs for the compute graph, which may hard-code tensor addresses. Async H2D copies that modify tensor data outside the graph capture may create stale references.

3. **Profiler build time**: Full rebuild after CUDA backend changes takes ~10 minutes (includes HIP compilation of all CUDA source). This slows iteration significantly.

### 8.2 Approach A2: Early-Issue (Synchronous Pre-Copies)

**Attempt:** At Split 1's entry (before RPC compute on 3060Ti), do synchronous H2D copies for Split 2's INPUT tensors. The ROCm GPU is idle during Split 1's RPC compute on the 3060Ti.

**Result:** TG degraded from 104.65 t/s to 87.40 t/s (**-16.5%**). Root cause: the synchronous pre-copies block the CPU pipeline — Split 1's processing (including RPC compute submission) is delayed. The CPU-blocking cost (2,645 µs of H2D copies) exceeds any overlap benefit, since the ROCm GPU may still be busy from the previous iteration's compute.

### 8.3 What We Shipped

| File | Change | Status |
|------|--------|--------|
| `ggml-cuda/common.cuh` | Add `pending_copy_events` vector to context | ✅ Kept (legitimate fix) |
| `ggml-cuda/ggml-cuda.cu` | Per-call event creation in `issue_pinned_h2d_async` | ✅ Kept (fixes copy_event overwrite) |
| `ggml-cuda/ggml-cuda.cu` | Cleanup in `synchronize` and destructor | ✅ Kept |
| `ggml-backend.cpp` | `GGML_SCHED_INPUT_COPY_ASYNC` env var | ✅ Kept (infrastructure for future) |
| `ggml-backend.cpp` | Async INPUT copy logic | ❌ Reverted (crashes) |
| `ggml-backend.cpp` | Early-issue H2D pre-copy | ❌ Reverted (regression) |

### 8.4 Key Insight: What `input_copy_slow` Actually Is

The 2,645 µs `input_copy_slow` is emitted by `sched_trace_emit_sync_detail` for individual INPUT tensor copies that exceed 50 µs. Running the profiler with `GGML_SCHED_TRACE=2` would reveal the tensor names, sizes, and source/dest buffer types. This would confirm whether the copies are:
- Host→GPU H2D (embeddings, positions) — target for async H2D
- GPU→GPU (intra-device) — already fast
- RPC→GPU (network transfer) — target for RPC overlap (D7.5 original scope)
- Something else (unknown)

### 8.5 Recommendation After Prototype

**Defer D7.5 implementation until D7.6 (rocprofv3) is unblocked.** The GPU kernel profiler is needed to:
1. Confirm whether `input_copy_slow` is H2D, RPC, or something else
2. Verify CUDA graph compatibility with async stream operations
3. Profile the impact of async H2D on graph-compute kernel timing

The copy_event fix and env var infrastructure remain in the codebase behind `GGML_SCHED_INPUT_COPY_ASYNC=1` for quick re-testing when rocprofv3 is available.

---

## 9. Post-D7.6 Diagnostic (2026-07-16): `input_copy_slow` is GPU Event Wait

### 9.1 Method

Ran `GGML_SCHED_TRACE=2` on Qwen3.6-35B-MTP (no rocprofv3, 124 t/s unprofiled, n_gen=32, MTP n_max=2, GPipe stages=3, default tensor split):

```bash
env GGML_SCHED_TRACE=2 GGML_SCHED_TRACE_FILE=/tmp/d75-diagnostic/sched-trace.jsonl \
  GGML_CUDA_GRAPHS=0 \
  build/bin/llama-gpipe-profiler \
  -m /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  -rpc 127.0.0.1:50051 --n-gpu-layers 99 --gpipe-stages 3 \
  --tasks tg --n-gen 32 --repeat 1 --no-warmup \
  --spec-type draft-mtp --spec-draft-n-max 2
```

The `sched_trace_emit_sync_detail` function logs tensor name, nbytes, src_buft, dst_buft, host_src/host_dst, and rpc_src/rpc_dst for each `input_copy_slow` entry.

### 9.2 Results: 97.6% is `leaf_70` (16 bytes)

142 `input_copy_slow` entries totaled 172,651 µs across 32 decode steps:

| Tensor | Count | Size | Split(s) | Total Time | Avg | % of Total |
|--------|-------|------|----------|:----------:|:---:|:----------:|
| `leaf_70` | 108 | 16 B | 1, 2 | 168,512 µs | 1,560 µs | **97.6%** |
| `attn_inp_kq_mask` | 9 | 512 B | 1, 2 | 1,200 µs | 133 µs | 0.7% |
| `leaf_76` | 11 | 8 B | 1 | 1,106 µs | 100 µs | 0.6% |
| `leaf_74` | 10 | 8 B | 1 | 1,073 µs | 107 µs | 0.6% |
| `model.input_embed` | 3 | 8 KB | 1 | 692 µs | 230 µs | 0.4% |

Split-by-split `leaf_70` breakdown:

| Split | Backend | Avg | Range | Interpretation |
|-------|---------|:---:|:-----:|---------------|
| Split 1 | RPC/3060Ti | ~100 µs | 55-526 µs | RPC GPU finishes compute quickly; short wait |
| Split 2 | ROCm/7900XTX | **~3,500 µs** | 1,811-5,236 µs | Waiting for previous iteration's ROCm GPU to drain |

### 9.3 Root Cause

The `input_copy_slow` measurement is NOT a copy — it's the `ggml_backend_event_synchronize` call at `ggml-backend.cpp:2447` waiting for the previous iteration's GPU compute to finish:

```cpp
// ggml-backend.cpp:2446-2448
if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
}
```

The ROCm GPU takes 6,843 µs of compute per SLOW step (D7.2 data). The next iteration's Split 2 starts after Split 0 (20 µs) + Split 1 (4,665 µs) = 4,685 µs have elapsed. The remaining 6,843 - 4,685 = **2,158 µs** of GPU work is still in-flight, causing the event_synchronize to block.

The copy itself (`leaf_70`, 16 bytes, CPU→ROCm) takes <1 µs. The remaining 3,500 µs is pure GPU wait.

### 9.4 Revised Assessment

**D7.5's original premise (overlap H2D with GPU compute) is invalid.** There is no H2D bottleneck. The `input_copy_slow` time is GPU synchronization overhead from sequential iteration processing.

| Original D7.5 Hypothesis | Diagnostic Finding | Verdict |
|--------------------------|-------------------|---------|
| "H2D copies are slow (2,645 µs)" | Copies are 16 bytes, <1 µs each | **DISPROVEN** |
| "Async H2D would hide copy time" | No copy time to hide | **N/A** |
| "Early-issue pre-copies would help" | Copies aren't the bottleneck; GPU is | **Explains A2 regression** |
| "RPC download is the bottleneck" | RPC tensors handled via gather path, not INPUT | **Wrong path** |

### 9.5 New Direction: D6.10 GPU Event Pipelining (COMPLETE)

> **D6.10 shipped** as commit `f29a92eb1`. See `docs/wayfinder/D6.10-implementation-analysis.md` for the full implementation analysis and `docs/wayfinder/D6.10-module-design.md` for the module design.

The original D7.5 pivot proposed enabling overlap between iterations via GPU event pipelining. This was implemented in D6.10:

- **D6.10 fix:** Scan backends for one with `event_new != NULL`, create gpipe_events on it for async event pipelining. Fallback to full sync if no GPU backends.
- **D6.10.1 fix:** Skip `event_synchronize` for host->GPU INPUT copies when `n_copies > 1` (copy-slot rotation prevents buffer conflict).
- **Result:** `input_copy_slow`: 165,000 us -> 2,359 us (**-98.6%**). TPS: 124.4 -> 129.8 (**+4.3%**). 12/12 GPipe tests pass.

The original D7.5 overlap concept (Split 0+1 of N+1 concurrent with Split 2 of N) was validated by the D6.10.1 result: with event pipelining active, the inter-iteration overlap works as intended. The remaining `input_copy_slow` (2,359 us) is now within noise of the GPU compute time.

**D7.5 is resolved: no H2D overlap needed. The copy_event fix stays. D6.10 delivered the overlap via a different mechanism.**

### 9.6 Artifacts

- Trace file: `/tmp/d75-diagnostic/sched-trace.jsonl` (1756 lines, 142 input_copy_slow entries)
- Heatmap: `/tmp/d75-diagnostic/heatmap.json`
- Raw tool: `docs/research/rocprofv3-profiling-guide.md`

---

## 6. Recommendation

**Start with Approach A1 + A2 (combined):** Enable async H2D copy for INPUT-tagged tensors on GPU backends, and issue the copies early (before event_synchronize) when the tensors don't depend on preceding splits. This requires:

1. **Enable `try_async_tensor_copy` for INPUT-tagged tensors** — currently the INPUT loop only uses synchronous `ggml_backend_tensor_copy`. Extend the try_async path to handle INPUT tensors.

2. **Add copy-stream synchronize** before `graph_compute_async` — a lightweight sync point that ensures all async H2D copies complete before GPU kernels read the tensors.

3. **Early-issue pattern** — move the async INPUT tensor copies before the copy-slot wait, following the B+13b prefetch pattern already used for gather tensors.

This is the minimal viable change with the highest ROI. It reuses existing infrastructure (`try_async_tensor_copy`, `cpy_tensor_async` on HIP backend) and follows established patterns in the codebase.

If A1+A2 proves insufficient (e.g., the INPUT tensors are small and the bottleneck is elsewhere), escalate to A4 (double-buffered prefetch) and then A5 (full stream separation).

---

## 7. Next Steps

1. ~~**Confirm the bottleneck**~~ — **COMPLETE (D7.5).** `input_copy_slow` is GPU event sync, not H2D.
2. ~~**Prototype A1 + A2**~~ — **SUPERSEDED by D6.10.** Event pipelining achieved the overlap without async H2D.
3. **D6.10 follow-up** — **COMPLETE.** See `docs/wayfinder/D6.10-implementation-analysis.md`.
4. **Layer 1 kernel optimization** — With Layers 2-3 exhausted (D6.10, D7.3), the remaining bottleneck is GPU compute. See Slice 7 vectors in `docs/research/slice-7-kernel-anvil-integration.md` and `docs/wayfinder/D7-REEXAMINATION.md`.

---

*Research based on code analysis of `ggml/src/ggml-backend.cpp` (lines 2305-2797, scheduler loop) and `ggml/src/ggml-rpc/ggml-rpc.cpp` (lines 1454-1670, RPC download infrastructure).*
