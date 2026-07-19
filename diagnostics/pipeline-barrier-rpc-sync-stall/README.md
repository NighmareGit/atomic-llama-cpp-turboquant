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
