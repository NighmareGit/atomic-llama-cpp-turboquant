# PRD: RPC Transport Overhaul — Fire-and-Forget + UDP

## Phase 1: Audit and Fix All Blocking Points in the Fire-and-Forget Path

### 1.1 Audit Every `send_rpc_cmd` Call Site in `graph_compute`

The function `ggml_backend_rpc_graph_compute` (`ggml-rpc.cpp:2458`) contains five code paths. Each must be classified:

| Line | Call | Type | Blocks? | Condition |
|------|------|------|---------|-----------|
| 2519 | `send_rpc_cmd(sock, GRAPH_COMPUTE_STAGE, ..., resp_buf, resp_size)` | send_recv | **YES** | `gpipe_stage >= 0` |
| 2559 | `send_rpc_cmd(sock, GRAPH_RECOMPUTE_ALL, &req, sizeof(req))` | send_only | no | Path C reuse |
| 2568 | `send_rpc_cmd_deferred(sock, EVENT_RECORD, ...)` | deferred | no (drained later) | Path C reuse |
| 2595 | `send_rpc_cmd(sock, GRAPH_COMPUTE_ALL, ..., resp_buf, resp_size)` | send_recv | **YES** | Path C first-time |
| 2613 | `send_rpc_cmd(sock, GRAPH_RECOMPUTE, &req, sizeof(req))` | send_only | no | single-device reuse |
| 2625 | `send_rpc_cmd_deferred(sock, EVENT_RECORD, ...)` | deferred | no (drained later) | single-device reuse |
| 2648 | `send_rpc_cmd(sock, GRAPH_COMPUTE, ..., resp_buf, resp_size)` | send_recv | **YES** | single-device first-time (telemetry) |
| 2665 | `send_rpc_cmd(sock, GRAPH_COMPUTE, ..., &rsp, sizeof(rsp))` | send_recv | **YES** | single-device first-time (legacy) |

**Blocking paths** (send_recv): lines 2519, 2595, 2648, 2665. These serialize the full graph and wait for the server response.

**Async paths** (send_only + deferred EVENT): lines 2559/2568, 2613/2625. These fire-and-forget.

### 1.2 Verify Deferred EVENT_RECORD Drains Don't Block the Scheduler Loop

The deferred `EVENT_RECORD` response is drained by `rpc_socket_drain`, which is called from:
- `rpc_backend_event_record` (line 2219) — at the next event_record on the same socket.
- `rpc_drain_all_endpoints_pending` (line 466) — at `ggml_backend_synchronize`, `cpy_tensor_async`, etc.
- `flush_pending_get_tensor_for_socket` (line 877) — before GET_TENSOR on the same socket.

**Requirement:** The drain must NOT happen inside `graph_compute` itself. The reuse path (lines 2609-2637) sends the deferred EVENT_RECORD and returns immediately. The drain happens at the NEXT RPC operation or at `event_wait`. This is already the correct behavior — the audit must confirm it is preserved.

**Risk area:** `rpc_event_defer_barrier()` (line 328) controls whether `drain_pending_event_response` is called inside the blocking `send_rpc_cmd` overload (line 1062). When event-defer is enabled (default with pipeline_plus), the blocking path skips the inline drain — correct, because the response would be drained later anyway. The audit must verify this flag is respected consistently.

### 1.3 Async vs Sync Code Path Documentation

Each path in `graph_compute` must be documented with a comment block stating:
- Which `send_rpc_cmd` variant it uses.
- Whether it blocks.
- What condition triggers it.
- Where the EVENT_RECORD response is drained (for async paths).

### 1.4 Deliverables for Phase 1

1. **`rpc-async-audit.h`** — header defining audit macros/constants:
   - `RPC_ASYNC_GRAPH_RECOMPUTE` — marks send_only graph submission paths.
   - `RPC_BLOCKING_GRAPH_COMPUTE` — marks send_recv blocking paths.
   - Per-path trace markers for runtime classification.

2. **Comment block in `ggml_backend_rpc_graph_compute`** — documents every code path's blocking behavior, the conditions that trigger it, and where deferred responses are drained.

3. **`GGML_RPC_TRACE` logging** — the existing `rpc_trace_emit` already logs `blocking=true/false` per call. Enhance it to emit a `path` field (`"stage"`, `"recompute_all"`, `"compute_all"`, `"recompute"`, `"compute"`) so the trace file distinguishes all five paths without reading code.

---

## Phase 2: UDP Transport Design

### 2.1 Message Format

UDP datagram for graph recompute submission:

```
Offset  Size    Field
0       4       magic   — 0x474D4C01 ("GGML" + version byte), big-endian check
4       4       seq     — uint32_t, monotonically increasing per socket (wraps ok)
8       1       cmd     — RPC_CMD_GRAPH_RECOMPUTE (16) or GRAPH_RECOMPUTE_ALL (22)
9       4       device  — uint32_t device index
13      8       graph_uid — uint64_t, identifies cached graph on server
21      4       n_devices — uint32_t (for GRAPH_RECOMPUTE_ALL multi-device)
25      var     devices[] — uint32_t[n_devices] (only for ALL variant)
```

**Total fixed header: 21 bytes** (single-device), **25 + 4*n_devices** (ALL variant).

This mirrors the existing `rpc_msg_graph_recompute_req` (4 bytes) and `rpc_msg_graph_recompute_all_req` (16 + 4*n_devices bytes) but adds a magic + seq for UDP framing.

### 2.2 Port Allocation

- The RPC server's TCP port is known (from the endpoint `host:port`).
- The UDP port = TCP port + 1 by default (configurable via `GGML_RPC_UDP_PORT` env var).
- Client sends UDP datagrams to `host:udp_port`.
- The HELLO handshake does NOT need to negotiate UDP — it's client-side opt-in. If the server isn't listening on UDP, the datagrams are silently lost (but the client only sends UDP when `GGML_RPC_UDP=1` AND the path is a reuse path, so loss falls back safely — see 2.3).

**Alternative (more robust):** Server binds UDP on a separate thread at startup and includes the UDP port in the HELLO response. This requires a protocol change (new hello flag). For the prototype, we use the `TCP+1` convention to avoid touching the HELLO path.

### 2.3 Loss Handling

**Drop is acceptable.** Graph recompute is idempotent and superseded:
- If a UDP datagram is lost, the server simply doesn't recompute that graph.
- The NEXT token's graph call (over TCP or UDP) replaces it.
- Sequence numbers let the server detect gaps: if `seq != last_seq + 1`, a frame was dropped. The server logs a warning (rate-limited) but does NOT request retransmission.
- For the prototype: no retransmit, no ACK, no ordering guarantees.

**Safety mechanism:** The deferred `EVENT_RECORD` (sent over TCP AFTER the UDP graph submit) provides the synchronization point. If the UDP graph frame is lost, the server never computes, so when the client's EVENT_RECORD arrives at the server, the server's `wait_compute_idle()` will wait for any in-flight compute. The client's EVENT_RECORD drain then returns, and the client proceeds — but the output tensor may be stale. **This is why UDP is opt-in and the caller must tolerate stale outputs** (acceptable for benchmarking, not for production without a fallback).

**Fallback:** If `GGML_RPC_UDP=1` but the UDP send fails (e.g., socket not created), fall back to the TCP fire-and-forget path transparently. The caller never sees the difference.

### 2.4 Integration with Existing `rpc_socket` Struct

The `socket_t` class (`transport.h`) gets:
- A new `udp_fd` member (or `sockfd_t udp_fd`) for the client-side UDP socket.
- A new `udp_port` member for the target UDP port.
- Methods: `send_udp(const void * data, size_t size)` — sends a datagram to the pre-configured target.
- The UDP socket is created lazily on first `send_udp` call (or during `get_socket` when `GGML_RPC_UDP=1`).

Server-side:
- A new UDP listener socket bound to `tcp_port + 1`.
- A dedicated thread (`udp_recv_loop`) that `recvfrom()`s datagrams, parses the header, and enqueues `GRAPH_RECOMPUTE` / `GRAPH_RECOMPUTE_ALL` on the `compute_worker` — same as the TCP dispatch path.

### 2.5 Thread Safety

- **Client side:** UDP send happens on the scheduler thread (same thread that calls `graph_compute`). The UDP socket is write-only from this thread. No lock needed for `sendto()` (it's thread-safe for UDP).
- **Server side:** UDP recv happens on a dedicated `udp_recv_loop` thread. It parses the datagram and calls `enqueue_graph_recompute` / `enqueue_graph_recompute_all`, which post to the `compute_worker` thread via the existing `submit_compute_job` + `compute_mtx` + `compute_cv` mechanism. No new locks needed — reuses existing thread-safe queue.
- **Socket lifecycle:** The UDP socket lives in `socket_t::impl` and is closed in the destructor (same as TCP `fd`). No use-after-free as long as the `socket_ptr` is alive.

### 2.6 Deliverables for Phase 2

1. **UDP socket creation in `transport.cpp`** — `create_udp_socket(host, port)`, integrated into `socket_t::impl`.
2. **`rpc_udp_send_graph()`** — packs the UDP header + payload and `sendto()`s it.
3. **Wire into graph_recompute path** — in `ggml_backend_rpc_graph_compute`, the reuse paths (lines 2559, 2613) call `rpc_udp_send_graph()` instead of `send_rpc_cmd(GRAPH_RECOMPUTE)` when `GGML_RPC_UDP=1`. EVENT_RECORD stays on TCP.
4. **Server UDP listener** — `udp_recv_loop` thread that `recvfrom()`s, parses, and enqueues graph recompute.
5. **Env var gating** — `GGML_RPC_UDP=1` enables UDP; default off.

---

## Phase Boundaries

- Phase 1 is pure audit + documentation + tracing. No new transport. Zero risk to the TCP path.
- Phase 2 is additive UDP. TCP path is untouched. UDP is opt-in via env var.
- Neither phase changes the HELLO handshake, the wire protocol for TCP commands, or the server's TCP dispatch loop.
- Neither phase touches RDMA (`GGML_RPC_RDMA` code is independent).
