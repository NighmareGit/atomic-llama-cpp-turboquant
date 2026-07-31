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

### Single-pair (2026-07-19)

Raw trace: `rocprof-data-20260719-191812/`
- `roc-kernel_kernel_stats.csv` — kernel timing summary
- `roc-kernel_kernel_trace.csv` — per-kernel dispatch trace (22,632 entries)

### Multi-GPU Pipeline Profiling (2026-07-21)

Full report: [rocprof-multi-gpu-2026-07-21/ROCPROF-REPORT.md](rocprof-multi-gpu-2026-07-21/ROCPROF-REPORT.md)

2-GPU (7900 XTX + 3060 Ti) and 4-GPU (7900 XTX + 3060 Ti + 3090 + 3070) pipeline profiling with rocprofv3 kernel traces + GGML_SCHED_TRACE per-backend split timing. Identifies 5 pipeline bubbles: 3090 split imbalance (critical), network RPC data movement stalls, host-to-device issues, 7900 XTX kernel gaps, and 7900 XTX under-utilization.

| Config | 7900 XTX Compute | Bottleneck | Speed (no profiler) |
|--------|-----------------|------------|---------------------|
| 2-GPU | 1,960 ms | 3060 Ti (local) | 64 tok/s |
| 4-GPU | 183 ms | 3090 (network) | 24 tok/s |

Raw data: `rocprof-multi-gpu-2026-07-21/2gpu/` and `rocprof-multi-gpu-2026-07-21/4gpu/`
