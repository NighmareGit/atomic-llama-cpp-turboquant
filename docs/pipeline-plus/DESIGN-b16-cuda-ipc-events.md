# B+16: CUDA IPC Events for Localhost RPC

## Problem

The current RPC EVENT_RECORD mechanism requires a full TCP round-trip to synchronize
the client with the server's GPU compute. After sending GRAPH_RECOMPUTE (fire-and-forget),
the client blocks on a TCP read of the EVENT_RECORD response. The server's EVENT_RECORD
handler calls `wait_compute_idle()`, which blocks until the GPU finishes. This creates a
CPU-side serialization point: the client's CPU is idle waiting for the server's GPU.

## Solution

Use CUDA IPC (`cudaIpcGetEventEventHandle` / `cudaIpcOpenEventHandle`) to export a GPU
event from the server and import it on the client. The client waits on the event using
`cudaStreamWaitEvent`, which is a GPU-side operation that does not block the CPU.

## Protocol Extension

`rpc_msg_event_record_rsp` gains two new fields appended after `trace_id`:

```cpp
uint8_t has_ipc_event;   // 0 = legacy TCP, 1 = ipc_handle is valid
uint8_t ipc_handle[64];  // cudaIpcEventHandle_t (opaque, 64 bytes)
```

When `has_ipc_event == 1`, the client imports the handle and waits on its own CUDA stream
instead of relying on the TCP response for synchronization.

## Capability Negotiation

New capability flag: `RPC_CAP_CUDA_IPC_EVENTS = (1 << 5)`

- Server advertises in HELLO response (via `get_caps()`) when CUDA is compiled in
- Client advertises in HELLO request (via `get_caps()`) when CUDA is compiled in
- IPC is enabled only when BOTH sides advertise it AND the endpoint is localhost
- Remote RPC always falls back to TCP EVENT_RECORD

## Server Side

After graph compute (`graph_compute` / `graph_recompute`), the compute worker records a
CUDA event on the backend's stream and exports its IPC handle:

```cpp
cudaEvent_t event;
cudaEventCreate(&event, cudaEventInterprocess | cudaEventDisableTiming);
cudaEventRecord(event, stream);
cudaIpcGetEventHandle(&handle, event);
cudaEventDestroy(event);
```

The handle is stored in `rpc_server` and a condition variable is signaled.

The EVENT_RECORD handler waits for the handle (not for GPU completion) and includes it
in the response. This decouples the connection thread from GPU timing.

## Client Side

On receiving an EVENT_RECORD response with `has_ipc_event == 1`:

```cpp
cudaEvent_t server_done;
cudaIpcOpenEventHandle(&server_done, handle);
cudaStreamWaitEvent(client_stream, server_done, 0);
cudaEventDestroy(server_done);
```

The client's CUDA stream waits for the server's GPU event. The CPU is free to continue.

## Docker IPC Host

CUDA IPC requires `--ipc=host` on the Docker run command. The docker-compose files
already include `ipc: host`. If IPC is advertised but `cudaIpcGetEventHandle` fails
(e.g. Docker without `--ipc=host`), the server logs a warning and falls back to TCP.

## Files Changed

- `ggml/include/ggml-cuda.h` - declare `ggml_backend_cuda_get_ipc_event_handle`
- `ggml/src/ggml-cuda/ggml-cuda.cu` - implement IPC event export helper
- `ggml/src/ggml-rpc/transport.h` - add `RPC_CAP_CUDA_IPC_EVENTS` flag and `use_cuda_ipc` socket field
- `ggml/src/ggml-rpc/transport.cpp` - advertise CUDA IPC capability
- `ggml/src/ggml-rpc/ggml-rpc.cpp` - protocol extension, server export, client import, negotiation
