# Analysis Scratchpad

## Trace Analysis (rocprofv3)

### Inference GPU Utilization

From roc-kernel_kernel_trace.csv:
- Total inference dispatches: ~21,632 (after skipping ~1,000 model loading dispatches)
- Wall span: 3,173 ms
- Total GPU time: 493 ms
- GPU utilization: **15.5%** — the GPU is idle 84.5% of the time

### Gap Distribution

| Range | Count | Total Time | % of Wall |
|-------|-------|-----------|-----------|
| 10ms+ | 126 | 1,793 ms | 56.5% |
| 1-5ms | 161 | 224 ms | 7.1% |
| 500us-1ms | 207 | 189 ms | 6.0% |
| 100-500us | 651 | 2,553 ms | — (cumulative) |

### The 12ms Stall Pattern

123 instances of: Q4_K matvec (queue 3, ~300us) → 12ms gap → copyBuffer (queue 2)

The q3->q2 queue transition: compute engine finishes, then 12ms later the DMA copy engine starts. This is the RPC sync signature — the CPU blocks in `event_synchronize` waiting for the RPC socket to deliver the event response.

### Test Without RPC (Estimated)

From GPU compute time alone: 493 ms / 128 tokens = 3.85 ms/token → ~260 t/s

The 3060 Ti (RPC backend) handles ~25% of layers. At ~3-5 ms of its own compute time, this is dwarfed by the 12ms sync overhead.

## Code Paths

### Pipeline Barrier (blocking path)

`ggml-backend.cpp:3437-3448`:
```
for backends in wait_mask:
  if wf_cross AND not RPC:
    event_wait → cudaStreamWaitEvent (non-blocking, returns immediately)
  else:
    event_synchronize → rpc_finish_event_response → recv_rpc_cmd_deferred → BLOCKS (12ms)
```

### Per-split Input Handling (redundant sync)

`ggml-backend.cpp:2729-2736`:
```
for each input:
  if backend event exists:
    event_wait or event_synchronize → already waited by barrier!
```

### Event Defer (set but doesn't help)

`RPC_EVENT_DEFER_BARRIER=1`: Event record is sent async, but the response is still consumed synchronously at the barrier via `rpc_finish_event_response`.

## Fix Options

### Option A (Chosen): Skip RPC wait in barrier

Exclude RPC backends from the event wait loop in `pipeline_barrier` when `wf_cross` is enabled. The deferred RPC drain handles the final sync, and per-split input handling synchronizes before use.

### Option B: Non-blocking RPC event_wait

Change `rpc_backend_event_wait` to be truly non-blocking (e.g., post a callback or check a completion flag). Would require significant RPC protocol changes.

### Option C: Enable GPipe

GPipe (`GGML_SCHED_GPIPE=1`) uses a different scheduling model where pipeline stages are independent. Would solve the issue but requires GPipe to be enabled and tested.
