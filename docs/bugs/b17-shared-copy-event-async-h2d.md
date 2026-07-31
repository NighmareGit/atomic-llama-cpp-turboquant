# B+17: Shared `copy_event` overwrite corrupts async H2D transfers on ROCm

**Bug class:** Data corruption (silent) — model weights loaded incorrectly, producing garbled output
**Discovered:** 2026-07-16 during MTP benchmark profiling
**Fixed:** `e2fcdaf63` ("ggml-cuda: fix shared copy_event overwrite with per-call events for async H2D")
**Branch:** `path-d-udp-test` (not yet merged to `path-d-good`)

---

## 1. Symptom

Passing `--no-mmap` to `llama-cli` or `llama-server` causes the model to produce garbled output (`???????` instead of coherent text). The generation speed shows a nonsensical `1000000.0 t/s`, indicating the model generates exactly 1 token (EOS) in near-zero time, then stops. Without `--no-mmap` (default, uses mmap), output is coherent.

**Reproduction:**

```sh
# Works correctly (default mmap path)
./build-rocm/bin/llama-cli -m /mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf \
  -p "Think step by step." -n 64

# Produces garbled output (broken async H2D path)
./build-rocm/bin/llama-cli --no-mmap -m /mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf \
  -p "Think step by step." -n 64
```

**Evidence from debug log (`-lv 10`):**

```
load_all_data: no device found for buffer type CPU for async uploads     # CPU tensors: sync path, OK
load_all_data: using async uploads for device ROCm0, buffer type ROCm0   # GPU tensors: async path, BUGGY
```

---

## 2. Diagnosis

### 2.1 Architecture: two loading paths

`llama-model-loader.cpp` loads tensors via two strategies in `load_all_data()`:

| Path | Activated when | Mechanism | Works? |
|------|---------------|-----------|--------|
| **mmap** | `use_mmap == true` (default) | `tensor->data` points to mmap'd file region; GPU reads via PCIe BAR zero-copy | OK |
| **sync no-mmap** | `use_mmap == false` + ROCm lacks async caps | Read file into temp buffer, `ggml_backend_tensor_set()` synchronous copy | OK |
| **async no-mmap** | `use_mmap == false` + ROCm reports `caps.async, caps.host_buffer, caps.events` | Read chunks into 4 pinned staging buffers, async H2D to GPU | **BROKEN** |

### 2.2 How the async path activates

ROCm's `get_props()` reports:

```cpp
props->caps = {
    .async                = true,
    .host_buffer          = true,   // GGML_CUDA_NO_PINNED env not set
    .buffer_from_host_ptr = false,
    .events               = true,   // GGML_CUDA_NO_PEER_COPY is OFF
};
```

All three required caps are true, so `load_all_data` sets up:
- 4 pinned staging buffers (1 MB each via `hipHostMalloc`)
- 4 CUDA events for buffer-slot synchronization
- An `upload_backend` (ROCm) for async transfers

Tensors then flow: **file read** -> **staging buffer** -> `ggml_backend_tensor_set_async()` -> **GPU**

### 2.3 The broken function

`ggml_cuda_issue_pinned_h2d_async()` in `ggml-cuda.cu:3458`:

```cpp
static void ggml_cuda_issue_pinned_h2d_async(ggml_backend_cuda_context * cuda_ctx,
                                              void * dst, const void * src, size_t nbytes) {
    void * pin = ggml_cuda_pin_host_staging(nbytes);  // shared thread_local buffer!
    memcpy(pin, src, nbytes);
    cudaStream_t stream = cuda_ctx->stream();
    CUDA_CHECK(cudaMemcpyAsync(dst, pin, nbytes, cudaMemcpyHostToDevice, cudaStreamPerThread));

    // BUG: single shared copy_event — each call overwrites the previous event
    if (!cuda_ctx->copy_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx->copy_event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(cuda_ctx->copy_event, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamWaitEvent(stream, cuda_ctx->copy_event, 0));
}
```

---

## 3. Root Cause

Two interacting bugs in the same function:

### 3.1 Primary: Shared `copy_event` overwrite (critical)

`cuda_ctx->copy_event` is a **single CUDA event** shared across all calls to `ggml_cuda_issue_pinned_h2d_async`. Each call:

1. Submits an `cudaMemcpyAsync` on `cudaStreamPerThread`
2. **Re-records** the same event on the per-thread stream — overwriting the previous recording
3. Makes the compute stream wait for this event

**The result:** `cudaStreamWaitEvent(compute_stream, copy_event, 0)` only waits for the **last** copy submitted. All intermediate copies have their event recording overwritten, so the compute stream starts executing before they complete.

```
Timeline:

  Call 1: memcpy(pin, chunk0) -> cudaMemcpyAsync(gpu, pin)     , event_record(copy_event)
  Call 2: memcpy(pin, chunk1) -> cudaMemcpyAsync(gpu, pin)     , event_record(copy_event)  <-- overwrites!
                                      ^-- copy_event now points here
  compute_stream: cudaStreamWaitEvent(compute_stream, copy_event)
                                ^-- waits only for chunk1's copy!
                                    chunk0 may NOT be complete!
```

When chunk 0's data hasn't arrived in GPU memory by the time inference starts, the corresponding tensor weights are garbage, producing garbled model output.

### 3.2 Secondary: Shared `thread_local` pin buffer (race window)

`ggml_cuda_pin_host_staging()` uses a single `static thread_local` pinned buffer. Each call `memcpy`s new data into this buffer before submitting the async H2D. If `hipMemcpyAsync` reads from the source pointer at execution time (not submission time), the DMA engine for copy N may read overwritten data from copy N+1.

**Why this is secondary:** The event overwrite bug alone is sufficient to corrupt data, even if the pin buffer were correctly synchronized. The pin buffer race merely adds another corruption vector.

### 3.3 Why it manifests on ROCm but not NVIDIA CUDA

| Factor | NVIDIA CUDA | AMD ROCm |
|--------|-------------|----------|
| DMA start latency | Immediate after submission | May batch/queue transfers |
| `cudaStreamPerThread` behavior | Well-established per-thread stream | `hipStreamPerThread` — newer feature, may serialize differently |
| Event synchronization | Mature, well-tested | HIP event model — fewer edge cases exercised |

The bug exists on both platforms, but NVIDIA's faster DMA startup and larger user base mean it's either never hit or silently works due to timing. On ROCm, the race window is wide enough to corrupt weights on every run.

---

## 4. Fix

### 4.1 Per-call copy events (replaces shared `copy_event`)

**Commit:** `e2fcdaf63` (branch `path-d-udp-test`)

Instead of reusing `cuda_ctx->copy_event`, create a **new event per call** and store it in a `pending_copy_events` vector:

```cpp
static void ggml_cuda_issue_pinned_h2d_async(ggml_backend_cuda_context * cuda_ctx,
                                              void * dst, const void * src, size_t nbytes) {
    void * pin = ggml_cuda_pin_host_staging(nbytes);
    memcpy(pin, src, nbytes);
    cudaStream_t stream = cuda_ctx->stream();
    CUDA_CHECK(cudaMemcpyAsync(dst, pin, nbytes, cudaMemcpyHostToDevice, cudaStreamPerThread));

    // FIXED: per-call event to avoid overwriting prior copies
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev, 0));
    cuda_ctx->pending_copy_events.push_back(ev);
}
```

Events are cleaned up in `ggml_backend_cuda_synchronize()` (after stream sync ensures all copies complete):

```cpp
static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    // FIXED: destroy per-call copy events (safe after stream sync)
    for (auto ev : cuda_ctx->pending_copy_events) {
        cudaEventDestroy(ev);
    }
    cuda_ctx->pending_copy_events.clear();
}
```

And in the destructor for safety:

```cpp
ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    // ...existing cleanup...
    for (auto ev : pending_copy_events) {
        cudaEventDestroy(ev);
    }
    pending_copy_events.clear();
}
```

### 4.2 Struct addition

`pending_copy_events` member added to `ggml_backend_cuda_context`:

```cpp
struct ggml_backend_cuda_context {
    // ...existing members...
    std::vector<cudaEvent_t> pending_copy_events;
};
```

### 4.3 Optional: Pin buffer per call

For full correctness, `ggml_cuda_pin_host_staging()` could allocate a per-call buffer instead of sharing one. However, the per-call event fix is sufficient to close the data-race window because:

1. Each call's event ensures the compute stream waits for that specific copy
2. The `load_all_data` loop's `ggml_backend_event_synchronize(events[buffer_idx])` waits for the compute-stream-side event, which chains through to the per-thread-stream copy
3. Between loops, the file read latency (~12-20 ms for 64 MB from NVMe) provides more than enough slack for the DMA to complete before the host reuses the pin buffer

If ROCm `hipMemcpyAsync` latency (submission-to-DMA-starts) exceeds the file read latency on slower storage, this secondary issue may resurface. In that case, either:
- Use the staging buffer directly as the DMA source (skip the `memcpy(pin, src)` since staging buffers are already pinned)
- Implement per-call pin buffers

---

## 5. Status

| Item | Status |
|------|--------|
| Fix written | YES — commit `e2fcdaf63` on `path-d-udp-test` branch |
| Merged to `path-d-good` | **NO** — 35 commits behind, not yet cherry-picked |
| Cherry-pick required | `git cherry-pick e2fcdaf63` onto `path-d-good` |
| Tested on ROCm | Yes (author reported fix resolves the crash/garbled output) |
| Tested on NVIDIA | Not explicitly, but the per-call event pattern is strictly more correct |

### Cherry-pick steps

```sh
git checkout path-d-good
git cherry-pick e2fcdaf63
# The fix touches only ggml-cuda.cu — low conflict risk
# Review the diff first: git show e2fcdaf63 -- ggml/src/ggml-cuda/ggml-cuda.cu
```

---

## 6. Related reading

- `docs/rpc-multi-backend-pipeline-plus/IMPLEMENTATION.md` — Path-B+ H2D architecture
- `ggml/src/ggml-cuda/ggml-cuda.cu` — lines 3442-3468 (`ggml_cuda_pin_host_staging` / `ggml_cuda_issue_pinned_h2d_async`)
- `src/llama-model-loader.cpp` — lines 1582-1633 (async upload loop in `load_all_data`)
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_backend_cuda_synchronize` (event cleanup)
