# SPEC: RPC Transport Overhaul — Fire-and-Forget + UDP

## 1. Current Pain

### 1.1 The Garden-Hose Problem
The RPC pipeline uses TCP with blocking request-response for graph computation. Server-side GPU operations are **26 us median** (from D4.10 telemetry), but the client sees **~700 us per node** — a **28x inflation** from RPC round-trip overhead. We have multi-teraflop GPUs waiting on a TCP socket like a garden hose, while FPS games run 300 fps real-time over UDP with client-side prediction.

### 1.2 Where Time Goes in `graph_compute`
The client `ggml_backend_rpc_graph_compute` (`ggml-rpc.cpp:2458`) has five distinct code paths, each with different blocking behavior:

| Path | Condition | Wire Cmd | Blocks? | Typical Latency |
|------|-----------|----------|---------|-----------------|
| D6.9 stage (GPipe) | `gpipe_stage >= 0` | `GRAPH_COMPUTE_STAGE` | **Yes** (full response + telemetry) | High (full graph serialize + compute) |
| Path C multi-device reuse | `n_devices > 1` AND `uid` matches | `GRAPH_RECOMPUTE_ALL` | **No** (fire-and-forget + deferred EVENT) | ~700 us (TCP send + deferred drain) |
| Path C multi-device first-time | `n_devices > 1` AND new uid | `GRAPH_COMPUTE_ALL` | **Yes** (response + telemetry) | Very high (full graph + compute) |
| Single-device reuse | `uid != 0` AND `uid == last_graph_uid` | `GRAPH_RECOMPUTE` | **No** (fire-and-forget + deferred EVENT) | ~700 us (TCP send + deferred drain) |
| Single-device first-time | new uid | `GRAPH_COMPUTE` | **Yes** (4-byte response) | High (full graph serialize + compute) |

The **reuse paths** (rows 2, 4) are the steady-state token-generation case — they fire-and-forget the graph and synchronize later via deferred `EVENT_RECORD`. The **first-time paths** (rows 3, 5) serialize the entire graph and block until the server finishes computing and sends a response back.

### 1.3 The 250 ms/graph_compute Figure
A full `graph_compute` (first-time) for a 70B model serializes ~100 MB of graph data over TCP, blocks for the full server compute (~26 us GPU + overhead), then waits for the response. With TCP segmentation, Nagle interaction, and the deferred EVENT_RECORD drain, the end-to-end latency balloons. The reuse path avoids the full graph resend but still pays TCP overhead for the 4-byte `GRAPH_RECOMPUTE` + 20-byte `EVENT_RECORD` + the deferred response drain at the next barrier.

## 2. Existing Mitigations

The codebase already has the right primitives — they are functional but underutilized:

### 2.1 GRAPH_RECOMPUTE Fire-and-Forget
`send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request))` sends a 4-byte device index and returns immediately. The server's `enqueue_graph_recompute` (`ggml-rpc.cpp:3948`) posts to a dedicated `compute_worker` thread via `submit_compute_job`. No response is expected on the wire for this command.

### 2.2 Deferred EVENT_RECORD
`send_rpc_cmd_deferred` (`ggml-rpc.cpp:550`) sends the `EVENT_RECORD` command but does NOT wait for the response. The response is drained later by `rpc_socket_drain` — triggered at `event_wait`, the next RPC operation on the socket, or `rpc_drain_all_endpoints_pending`. This decouples the compute submission from the synchronization.

### 2.3 Wavefront Barriers (B+15)
The `rpc_socket_state` per-socket FIFO queue (`g_rpc_sockets`) replaced the old thread-local single-slot `tls_pending_event`. This allows multiple backends to have concurrent deferred EVENT_RECORD responses in-flight without interleaving on the TCP response channel. `rpc_backend_event_record` links the `ggml_backend_event` to the queued entry so the drain signals completion.

### 2.4 What Is Broken
The ticket states fire-and-forget is "broken by UID tracking (fixed separately in ticket #2)". The UID tracking issue is out of scope for this spec — we assume ticket #2 lands first so that `cgraph->uid` reuse detection works correctly for all split configurations. With that fix, the reuse paths (rows 2, 4 above) become available for every split, eliminating full graph serialization on every token.

## 3. Architecture Options

### 3.1 Option A — Fix Fire-and-Forget Only (depends on #2 UID fix)
**What:** Make `GRAPH_RECOMPUTE` + deferred `EVENT_RECORD` work for all splits. No protocol change.

**Mechanism:**
- Rely on ticket #2's UID fix so `cgraph->uid` is stable across tokens for every split.
- The reuse path in `ggml_backend_rpc_graph_compute` (line 2609) already handles this: when `uid != 0 && last_graph_uid == uid`, it sends the 4-byte `GRAPH_RECOMPUTE` + deferred `EVENT_RECORD` and returns immediately.
- No new code in the steady-state path — just unblocking what UID tracking broke.

**Expected gain:** ~200 ms/token (eliminates full graph serialization + blocking response on every token). The reuse path drops from "full graph + compute + response" to "4-byte TCP send + deferred EVENT".

**Risk:** Low. No protocol change, no new transport. Purely depends on UID fix.

**Limitation:** Still pays TCP overhead (~700 us/node) for the fire-and-forget send + deferred EVENT_RECORD drain. The EVENT_RECORD response must still be drained over TCP before the next synchronization point.

### 3.2 Option B — UDP Graph Submission
**What:** Add a UDP socket alongside TCP. Send `GRAPH_RECOMPUTE` over UDP; keep TCP for weight transfers (`SET_TENSOR`), tensor copies (`GET_TENSOR`, `COPY_TENSOR`), and `EVENT_RECORD` synchronization.

**Mechanism:**
- Dedicated UDP port per RPC connection (negotiated via HELLO or env var).
- Message framing: 8-byte header (sequence number + split ID + graph UID) + payload.
- `GRAPH_RECOMPUTE` over UDP = true fire-and-forget. No TCP round-trip for graph submission.
- Loss handling: **drop is acceptable**. For graph recompute, the next token's graph call replaces a lost one. No retransmit. Sequence numbers let the server detect/drop stale frames.
- `EVENT_RECORD` stays on TCP — it needs reliable ordering for synchronization.

**Expected gain:** Eliminates the ~700 us TCP overhead for the steady-state graph submission. Combined with Option A, total per-node cost drops to near-zero (UDP send is ~1-5 us on localhost/LAN).

**Risk:** Medium. New transport, new socket lifecycle, loss handling, out-of-order delivery. But UDP is opt-in and graph recompute is idempotent.

**Wire format (proposed):**
```
| magic(4) | seq(4) | cmd(1) | device(4) | graph_uid(8) | [payload...] |
  0xGGML    uint32   uint8   uint32     uint64        variable
```

### 3.3 Option C — Full UDP
**What:** Everything over UDP with a reliability layer.

**Mechanism:**
- Replace TCP entirely. Implement reliability (ACK/retransmit) for `SET_TENSOR`, `GET_TENSOR`, `COPY_TENSOR`.
- Congestion control, ordering guarantees, flow control.

**Expected gain:** Maximum — eliminates all TCP head-of-line blocking.

**Risk:** High. Essentially reimplementing TCP (or building a QUIC-like protocol). The reliability layer for weight transfers is the hard part — `SET_TENSOR` for a 70B model's embedding table is ~100 MB; losing the last byte means retransmitting everything. TCP already solves this well.

**Verdict:** Overkill. The blocking paths (first-time graph compute, weight transfers) are genuinely benefited by TCP's reliability. UDP only makes sense for the idempotent, drop-tolerant graph recompute.

## 4. Recommendation

**Start with Option A (fix fire-and-forget), design for Option B (UDP) as Phase 2.**

### Phase 1: Audit + Fix Fire-and-Forget
- Audit every `send_rpc_cmd` call site in `graph_compute` and classify blocking vs async.
- Verify deferred `EVENT_RECORD` drains don't block the scheduler loop.
- Document the exact async vs sync code paths.
- Add `GGML_RPC_TRACE` logging to distinguish async vs blocking at runtime.
- Depends on ticket #2 (UID fix) for full coverage.

### Phase 2: UDP Transport Prototype
- Add UDP socket creation to `transport.cpp`.
- Implement `rpc_udp_send_graph()` for `GRAPH_RECOMPUTE` over UDP.
- Wire into the reuse path as opt-in (`GGML_RPC_UDP=1`).
- Server-side: add UDP listener for graph recompute commands.
- Keep TCP 100% backward compatible — UDP is purely additive.

### Out of Scope
- Full UDP (Option C) — rejected as overkill.
- RDMA changes — RDMA already works via `GGML_RPC_RDMA` and is a separate transport layer.
- Changing `EVENT_RECORD` to UDP — it needs TCP ordering for sync correctness.

## 5. Success Criteria

| Metric | Current | Phase 1 Target | Phase 2 Target |
|--------|---------|----------------|----------------|
| Steady-state graph submit latency | ~700 us (TCP send + EVENT drain) | ~700 us (same, but now works for all splits) | ~5 us (UDP send) |
| First-time graph compute latency | ~250 ms | ~250 ms (unchanged) | ~250 ms (unchanged) |
| Tokens/sec (70B, reuse path) | Limited by 700 us/node | Improved (all splits use reuse) | Further improved |
| TCP backward compatibility | 100% | 100% | 100% (UDP opt-in) |
