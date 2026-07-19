# GGML RPC wire protocol (Path-B Plus)

Canonical version constants: `ggml/include/ggml-rpc.h`

```c
#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    4   // CHANNEL_BIND dual-socket
#define RPC_PROTO_PATCH_VERSION    2   // GGML_OP_COUNT=98 (fork turbo op)
```

**Current cluster target:** **4.4.2** on all rpc-servers when running B+11 experiments. Production default client behavior uses **v3 HELLO** (single socket) unless `GGML_RPC_DUAL_SOCKET=1`.

Related: [FEATURE-b11-dual-socket-rpc.md](FEATURE-b11-dual-socket-rpc.md), [rpc-patch/patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md](../../rpc-patch/patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md).

---

## Version history (minor bumps)

| Minor | Patch | Feature | Scheduler impact |
|-------|-------|---------|------------------|
| 4.0 | 0+ | Baseline Path B RPC | Sequential RPC backends |
| 4.1 | - | `SET_TENSOR_BATCH` | Faster weight upload |
| 4.2 | 2 | `RPC_CMD_EVENT_RECORD` (18) | `caps.events=true`; pipeline parallelism |
| 4.3 | 2 | `RPC_CMD_COPY_TENSOR_PEER` (19), HELLO `conn_caps` | Tier-1 peer COPY |
| **4.4** | 2 | `RPC_CMD_CHANNEL_BIND` (20), v4 HELLO | Optional cmd/rsp socket split (B+11) |

Bump **patch** when `GGML_OP_COUNT` changes (RPC graph ops on wire).

---

## Connection lifecycle

```text
Client                                    Server (rpc-server)
  |                                              |
  |-- TCP connect host:port -------------------->|
  |-- HELLO (cmd 14) ---------------------------->|
  |<-- HELLO rsp (version + caps) --------------|
  |                                              |
  | [optional B+11]                              |
  |-- TCP connect host:port (2nd socket) ------->|
  |-- CHANNEL_BIND (cmd 20) + session_id ------->|
  |                         pair rsp_channel ----|
  |                                              |
  |-- cmd byte + input_size + payload ---------->|  (cmd socket)
  |<-- out_size + payload -----------------------|  (rsp socket if paired, else cmd)
  |     ... repeat ...                           |
```

Each endpoint (`host:port`) is one logical RPC backend. Multi-hop llama.cpp passes a comma-separated endpoint list (`-rpc`).

---

## Message framing (all commands)

After TCP connect, every message on the **cmd** socket:

```text
[1 byte cmd][8 byte input_size (uint64 LE)][input_size bytes payload]
```

Response (on **rsp** socket when dual-socket paired, else same cmd socket):

```text
[8 byte out_size (uint64 LE)][out_size bytes payload]
```

`send_only` commands may omit the response read on the client until a later blocking op drains it.

---

## HELLO (cmd 14)

First message on cmd socket. Server reads `input_size` then payload.

### v3 request (8 bytes) — default client wire

Used when `GGML_RPC_DUAL_SOCKET=0` (default). Compatible with proto 4.3 and 4.4 servers.

```c
uint8_t conn_caps[RPC_CONN_CAPS_SIZE];  // 8 bytes
```

### v4 request (16 bytes) — dual-socket intent

Used when `GGML_RPC_DUAL_SOCKET=1`:

```c
struct rpc_msg_hello_req {
    uint8_t  conn_caps[8];
    uint32_t session_id;   // client random, non-zero
    uint8_t  dual_socket;  // 1 = request rsp channel
    uint8_t  reserved[3];
};
```

### v3 response (12 bytes)

```c
struct rpc_msg_hello_rsp_v3 {
    uint8_t major, minor, patch, padding;
    uint8_t conn_caps[8];
};
```

### v4 response (16 bytes)

```c
struct rpc_msg_hello_rsp {
    uint8_t  major, minor, patch;
    uint8_t  flags;        // bit0: dual_socket negotiated
    uint8_t  conn_caps[8];
    uint32_t session_id;   // echoed when dual accepted
};
```

Client version check: reject if `response.minor > RPC_PROTO_MINOR_VERSION` (client too old). Accept server minor <= client minor.

Capability bits in `conn_caps` (see `transport.h`):

- Peer copy (`minor >= 3`)
- Batch SET_TENSOR (`minor >= 1`)

Log example:

```text
RPC 127.0.0.1:50051: proto 4.4 peer_copy=yes dual=no
```

---

## CHANNEL_BIND (cmd 20) — proto 4.4 only

Second TCP connection to **same host:port** immediately after v4 HELLO rsp.

Request payload:

```c
struct rpc_msg_channel_bind_req {
    uint32_t session_id;
};
```

Server looks up pending session from HELLO, attaches this socket as `rsp_channel` on the cmd connection's `socket_t`. Timeout: 10s; on timeout server continues single-socket.

Client API: `rpc_client_bind_response_channel()`.

---

## Command table

| Value | Name | Blocking | Notes |
|-------|------|----------|-------|
| 0 | ALLOC_BUFFER | yes | |
| 1 | GET_ALIGNMENT | yes | |
| 2 | GET_MAX_SIZE | yes | |
| 3 | BUFFER_GET_BASE | yes | |
| 4 | FREE_BUFFER | yes | |
| 5 | BUFFER_CLEAR | yes | |
| 6 | SET_TENSOR | varies | batched via TLS batch |
| 7 | SET_TENSOR_HASH | varies | large tensor dedup |
| 8 | GET_TENSOR | blocking/deferred | B+12 defer path |
| 9 | COPY_TENSOR | varies | B+2 async/defer |
| 10 | GRAPH_COMPUTE | yes | |
| 11 | GET_DEVICE_MEMORY | yes | |
| 12 | INIT_TENSOR | yes | |
| 13 | GET_ALLOC_SIZE | yes | |
| **14** | **HELLO** | yes | must be first; fixed enum value |
| 15 | DEVICE_COUNT | yes | |
| 16 | GRAPH_RECOMPUTE | yes | Path B completion + EVENT ordering |
| 17 | SET_TENSOR_BATCH | varies | minor >= 1 |
| **18** | **EVENT_RECORD** | deferrable | Path B pipeline events |
| **19** | **COPY_TENSOR_PEER** | varies | minor >= 3, same-host |
| **20** | **CHANNEL_BIND** | yes | minor >= 4, 2nd socket only |

`static_assert(RPC_CMD_HELLO == 14)` — do not reorder enum without protocol break.

---

## Path-B Plus client flags (orthogonal to proto minor)

These control behavior on top of negotiated proto; see [IMPLEMENTATION.md](IMPLEMENTATION.md).

| Flag | Default (Plus=1) | RPC interaction |
|------|------------------|-----------------|
| `GGML_RPC_EVENT_DEFER_BARRIER` | ON (B+9) | Defer EVENT recv to barrier |
| `GGML_RPC_GET_TENSOR_DEFER` | ON (B+12) | Defer GET recv to graph boundary |
| `GGML_RPC_MULTI_SOCKET_FLUSH` | ON (B+7a') | Flush all endpoints at synchronize |
| `GGML_RPC_DUAL_SOCKET` | **OFF** (B+11) | v4 HELLO + CHANNEL_BIND |

---

## Server deployment checklist

1. Build `rpc-server` from `Path-B-Event-Support-Pipeline-Plus` @ minor 4.4+
2. Confirm HELLO reports `4.4.x`:

   ```bash
   bash scripts/b6-gate-validate-rpc-matrix.sh
   ```

3. Per-node rebuild scripts:

   | Node | Command |
   |------|---------|
   | remus | `./rpc-patch/scripts/pathb-remus-rpc.sh rebuild` |
   | romulus | `./rpc-patch/scripts/pathb-romulus-rpc.sh rebuild` |
   | triton | `bash scripts/b6-gate-triton-remote.sh rebuild` |
   | jupiter (Windows) | `scripts\b6-gate-jupiter-rebuild-rpc.cmd` then `b6-gate-jupiter-start-rpc-task.ps1` |

4. Mixed-version cluster: safe with `GGML_RPC_DUAL_SOCKET=0` (v3 HELLO). dual=ON requires **all** endpoints 4.4.

---

## Debugging

| Symptom | Check |
|---------|-------|
| `version mismatch` | Client newer than server; rebuild rpc-server |
| `dual-socket bind failed` | Firewall, server not 4.4, or session timeout |
| `HELLO request size mismatch` | Client/server minor skew |
| HOL tails persist | Phase 1.2B Gantt: `scripts/b6-gate-phase12b-gantt.sh <bench-dir>` |

Trace env:

```bash
export GGML_RPC_TRACE=1
export GGML_RPC_TRACE_FILE=/path/rpc-trace.jsonl
```

HELLO + per-op rows include `endpoint`, `minor`, `peer_copy`.

---

## References

- Source: `ggml/src/ggml-rpc/ggml-rpc.cpp`, `ggml/include/ggml-rpc.h`
- B+11 feature doc: [FEATURE-b11-dual-socket-rpc.md](FEATURE-b11-dual-socket-rpc.md)
- Path B events: [rpc-patch/patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md](../../rpc-patch/patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md)
- Ops handover: [rpc-patch/docs/rpc-path-b-plus-handover.md](../../rpc-patch/docs/rpc-path-b-plus-handover.md)