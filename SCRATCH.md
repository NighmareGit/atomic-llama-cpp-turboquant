=== B+17: Shared copy_event overwrite ===

## Question: "why is mmap defective?"

**Answer: mmap is NOT defective. The `--no-mmap` async upload path is broken.**

## Root cause

Two bugs in `ggml_cuda_issue_pinned_h2d_async()` (ggml-cuda.cu:3458), both introduced
by commit 590597110a (B+15, "ggml: B+15 split-1-start gather RPC prefetch"):

1. **PRIMARY: Shared `copy_event` overwrite** — A single `cuda_ctx->copy_event` is
   reused across ALL async H2D calls. Each call re-records the event, so
   `cudaStreamWaitEvent` only waits for the LAST copy, not all intermediate ones.
   This causes the compute stream to start before earlier copies complete, corrupting
   tensor weights on GPU.

2. **SECONDARY: Shared `thread_local` pin buffer** — `ggml_cuda_pin_host_staging()`
   returns a single thread-local pinned buffer. Each call overwrites it before
   the previous async DMA reads from it. (Less critical — timing usually saves this.)

Both were hidden bugs from the moment B+15 introduced the async H2D path. They
never manifested on NVIDIA because fast DMA startup masks the race window. On ROCm,
the timing difference makes it hit reliably, producing garbled output on every run.

## Fix

Written by user in commit `e2fcdaf63` (branch `path-d-udp-test`, NOT merged to `path-d-good`):
- Replace shared `copy_event` with per-call events stored in `pending_copy_events` vector
- Clean up events in `ggml_backend_cuda_synchronize()` and destructor
- Cherry-pick needed: `git cherry-pick e2fcdaf63`

## Documentation written

`docs/bugs/b17-shared-copy-event-async-h2d.md` — full bug report with symptom,
diagnosis, root cause, fix, and cherry-pick instructions.
