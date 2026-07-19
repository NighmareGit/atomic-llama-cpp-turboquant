# Pipeline Barrier RPC Sync Stall — Diagnostics

## Reproduction

**Hardware:** ROMULUS: 7900XTX (ROCm) + 3060Ti (RPC Docker)
**Model:** Qwen3.5-9B-MTP-Q4_K_M.gguf
**Config:** PPLUS=1, blessed env flags, `-sm layer -ts 75,25`, 128 tokens

```bash
GGML_PIPELINE_PLUS=1 \
GGML_PIPELINE_BARRIER_PARTIAL=1 \
GGML_PIPELINE_BARRIER_PARTIAL_STRICT=1 \
GGML_RPC_EVENT_DEFER_BARRIER=1 \
GGML_RPC_GET_TENSOR_DEFER=1 \
GGML_RPC_MULTI_SOCKET_FLUSH=1 \
GGML_SCHED_MOE_ASYNC_COPY=1 \
llama-server -m Qwen3.5-9B-MTP-Q4_K_M.gguf \
  --rpc 127.0.0.1:50051 -sm layer -ts 75,25 -ngl 99 -c 2048 \
  --host 127.0.0.1 --port 8080 --no-warmup
```

## Fix Verification (2026-07-19)

Three changes applied to `ggml/src/ggml-backend.cpp` and built in `/tmp/rocm-prototype-f5824E/`:

1. **wf_cross defaults ON with PPLUS=1** — non-GGML_SCHED_WAVEFRONT_CROSS env no longer needed
2. **RPC backends skipped in pipeline barrier event loop** — no blocking `event_synchronize` on RPC
3. **RPC drain blocked skipped in barrier when wf_cross** — per-split wait_producer handles it

### Trace-Verified

Pipeline barrier with fix: `elapsed_us:8, wf_cross:1` — 8 microseconds vs 12,000 before fix.

### Throughput

| Metric | Before | After |
|--------|--------|-------|
| Predicted t/s | 55.7 | 55.5 |
| Barrier time | 12,000 us | 8 us |

**Root cause of no throughput gain:** The 3060 Ti compute time (25% layers, ~12ms) is the ceiling, not the sync mechanism. The 12ms moved from redundant double-sync to necessary single-sync (wait_producer).

## Profiler Command

```bash
GGML_CUDA_GRAPHS=0 rocprofv3 --kernel-trace --stats --summary \
  -d /tmp/rocprof-output -o roc-kernel -f csv \
  -- <llama-server-command>
```

## Files

| File | Description |
|------|-------------|
| `ISSUE.md` | Ticket with symptom, root cause, fix proposal |
| `SCRATCHPAD.md` | Analysis notes |

## Profiler Data

Raw trace: `/tmp/rocprof-data-20260719-191812/`
- `roc-kernel_kernel_stats.csv` — kernel timing summary
- `roc-kernel_kernel_trace.csv` — per-kernel dispatch trace (22,632 entries)
