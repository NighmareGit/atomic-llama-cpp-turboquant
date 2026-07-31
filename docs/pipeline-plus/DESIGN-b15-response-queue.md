# B+15: Per-Socket FIFO Response Queue for Deferred EVENT_RECORD

## Problem

Commit 5f8020f77 introduced deferred EVENT_RECORD in
`ggml_backend_rpc_graph_compute()`. The drain uses thread-local
`tls_pending_event`, a single slot per thread. When multiple RPC backends
(multiple sockets) have pending EVENT_RECORD responses, the slot can only
track one. Subsequent `graph_compute` calls overwrite the slot, so responses
interleave on the TCP socket, corrupting CUDA event state:

- 27% throughput regression (RPC-only 80.3 -> 60.9 t/s)
- CUDA graph warmup infinite loop on second model load
- GPU resets

## Root Cause

`tls_pending_event` is a `thread_local` struct with one `sock`/`pending`/`ev`
triple. It assumes at most one pending EVENT_RECORD per thread. With multiple
RPC backends (e.g. ROCm 7900 XTX + CUDA 3060 Ti), the scheduler interleaves
`graph_compute` across backends on the same thread, so the second backend's
`tls_pending_event` write clobbers the first's. The drain then reads the wrong
response off the response channel, desynchronizing the client's event state
from the server's.

## Fix

Replace the single thread-local slot with a per-socket FIFO queue. Each socket
tracks its own deferred responses in order. Drain pops from the front and reads
the response, preserving FIFO ordering regardless of how many sockets have
pending responses.

### Data structures

```cpp
struct rpc_deferred_entry {
    enum rpc_cmd          cmd_type;
    std::vector<uint8_t>  rsp;   // owned response buffer
    struct rpc_event_t *  event; // for EVENT_RECORD entries
};

struct rpc_socket_state {
    socket_ptr sock;
    std::deque<rpc_deferred_entry> queue;
};

static std::unordered_map<socket_ptr, rpc_socket_state> g_rpc_sockets;
static std::mutex g_rpc_sockets_mutex;
```

### Operations

- `rpc_socket_queue_push(sock, entry)`: append to the socket's queue.
- `rpc_socket_drain(sock)`: swap out the queue under the lock, then read each
  response outside the lock (so the global mutex is not held during the
  blocking recv). Each entry's `event` is signaled (`response_pending = false`)
  after its response is read.

### Wiring

- `graph_recompute` / `graph_recompute_all` paths: replace `tls_pending_event`
  writes with `rpc_socket_queue_push`. The entry's `event` is nullptr at push
  time (the event is not yet known); `event_record` links it later.
- `rpc_backend_event_record`: link the event to the un-linked queue entry (if
  graph_compute already sent EVENT_RECORD), else send EVENT_RECORD and push a
  new entry. Then `rpc_socket_drain`.
- `rpc_drain_all_endpoints_pending`: iterate `g_rpc_sockets` and drain each.
- Remove `tls_pending_event` entirely. Remove `last_compute_sock` (redundant;
  socket is computable from endpoint). Keep `last_compute_sent_event`
  (per-context, distinguishes reuse vs non-reuse paths).

## Verification

- Build: `cmake --build build-hip --target llama-server rpc-server`.
- Test: start llama-server with RPC, run 5 sequential requests, restart and
  verify no warmup hang. Check `dmesg` for GPU resets.
