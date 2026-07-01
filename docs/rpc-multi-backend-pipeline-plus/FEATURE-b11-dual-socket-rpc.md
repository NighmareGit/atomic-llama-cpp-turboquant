# FEATURE: B+11 dual-socket RPC (proto 4.4)

| Field | Value |
|-------|-------|
| **ID** | B+11 |
| **Status** | Shipped; bisect **NULL** (default OFF) |
| **Proto** | 4.4.2 (`RPC_CMD_CHANNEL_BIND`) |
| **Gate** | `b6-4gpu-g-triton` n=384 |
| **Design** | [DESIGN-b11-dual-socket.md](DESIGN-b11-dual-socket.md) |
| **Protocol** | [RPC-PROTOCOL.md](RPC-PROTOCOL.md) |

## Summary

B+11 splits each RPC endpoint into two TCP connections after HELLO:

| Socket | Traffic |
|--------|---------|
| **cmd** | Client requests (`send_only`, `send_recv` command half) |
| **rsp** | Server response payloads (`out_size` + data, deferred recv) |

Goal: reduce head-of-line (HOL) blocking when large `GET_TENSOR` / `EVENT_RECORD` responses delay small acks on a single multiplexed socket.

**Bisect verdict (2026-07-01):** NULL on `overlap_pct`; **HURTS G** (-9.1%). Default remains **OFF** (`GGML_RPC_DUAL_SOCKET=0`). Proto 4.4 ships for forward compatibility and optional experiments.

## Problem evidence (Phase 1.2B)

On 4-GPU gate `b6-4gpu-g-n384-romulus-native`:

- `overlap_pct` stuck at ~0.2%
- HOL tails: `drain_event/EVENT` spikes to 428ms+ while p50 RTT ~0.3-1.7ms
- 15 tails / 1968ms total on canonical 4-GPU JUPITER topology

Hypothesis: single-socket multiplexing lets bulk responses block control-plane acks.

## Implementation

### Wire (proto 4.4)

1. Client HELLO (32-byte v4 req) sets `dual_socket=1` + random `session_id`
2. Server registers pending session; HELLO rsp sets `flags&1` + echoes `session_id`
3. Client opens second TCP to same host:port, sends `RPC_CMD_CHANNEL_BIND` (20) + `session_id`
4. Server pairs `rsp_channel` on cmd socket; `send_response()` uses rsp socket

When `GGML_RPC_DUAL_SOCKET=0`, client sends legacy 8-byte v3 HELLO (proto 4.3 wire) for compatibility with older rpc-servers.

### Code map

| Area | File | Notes |
|------|------|-------|
| Cmd enum | `ggml/src/ggml-rpc/ggml-rpc.cpp` | `RPC_CMD_CHANNEL_BIND = 20` |
| Transport | `ggml/src/ggml-rpc/transport.h` | `socket_t::rsp_channel` |
| Client HELLO | `ggml-rpc.cpp` `negotiate_hello()` | v3 vs v4 req selection |
| Server HELLO | `ggml-rpc.cpp` `rpc_serve_client()` | pending session + 10s bind timeout |
| Deferred recv | `recv_rpc_cmd_deferred()` | reads from `rpc_response_sock()` |
| API | `ggml/include/ggml-rpc.h` | `ggml_backend_rpc_dual_socket()` |

### Environment

| Var | Default | Effect |
|-----|---------|--------|
| `GGML_RPC_DUAL_SOCKET` | **0** (OFF) | `=1` enables dual-socket HELLO + CHANNEL_BIND |

Bisect ON arm: `export GGML_RPC_DUAL_SOCKET=1` before `canonical-romulus`.

Log line after HELLO:

```text
RPC 192.168.8.23:50054: proto 4.4 peer_copy=yes dual=yes
```

`dual=no` means single-socket (default or bind fallback).

## Bisect procedure

### Prerequisites

Rebuild **all** rpc-servers to proto 4.4 before dual=ON bisect:

| Endpoint | Script | Port |
|----------|--------|------|
| remus 5060 Ti | `rpc-patch/scripts/pathb-remus-rpc.sh rebuild` | `:50051` |
| romulus ROCm | `rpc-patch/scripts/pathb-romulus-rpc.sh rebuild` | `127.0.0.1:50051` |
| triton 3090 | `scripts/b6-gate-triton-remote.sh rebuild` | `:50054` |
| jupiter 5070 Ti | `scripts/b6-gate-jupiter-rebuild-rpc.cmd` + `b6-gate-jupiter-start-rpc-task.ps1` | `:50053` |

Validate:

```bash
bash scripts/b6-gate-validate-rpc-matrix.sh
```

### 4-GPU gate (romulus background)

```bash
# Uses b6-4gpu-g-triton until jupiter :50053 is 4.4
bash scripts/b6-gate-romulus-b11-4gpu-bisect-bg.sh
```

Or manual:

```bash
export GGML_RPC_DUAL_SOCKET=1
B6_GATE_PRESET=b6-4gpu-g-triton bash scripts/b6-gate-bisect-run.sh canonical-romulus
B6_GATE_PRESET=b6-4gpu-g-triton bash scripts/b6-gate-bisect-run.sh no-dual-socket
```

Output dirs:

- ON: `benches/path-b-plus/b6-4gpu-g-triton-n384-romulus-native`
- OFF: `benches/path-b-plus/b6-4gpu-g-triton-n384-romulus-native-no-dual-socket`

### Pass criteria (M3)

From [DESIGN-b11-dual-socket.md](DESIGN-b11-dual-socket.md):

- `overlap_pct` delta >= 1% vs OFF, **or**
- `hol_tail_count` / `drain_event` ms down >= 50% with stable G

## Bisect results (2026-07-01)

Gate: `b6-4gpu-g-triton` n=384, romulus client, proto 4.4 on remus/romulus/triton (jupiter still 4.3; triton swap).

| Arm | G (t/s) | overlap_pct | stall_ratio | drain_flush_ms | hol_tails | hol_tail_ms |
|-----|---------|-------------|-------------|----------------|-----------|-------------|
| dual ON | 73.2 | 0.2% | 0.955 | 3918 | 5 | **1036** (864ms spike) |
| dual OFF | **80.5** | 0.2% | 0.949 | 2985 | 9 | 171 |
| Delta | **-9.1%** | 0.0 | +0.006 | +31% | -4 | +508% |

Phase 1.2B Gantt (decode_id=1):

- ON: 864ms `drain_event/EVENT` on backend1 (RPC)
- OFF: max 74ms `drain_event/EVENT`; more tails but lower total ms

**Verdict:** NULL on overlap; dual ON introduces worse tail latency and lower throughput. Ship proto 4.4 with default OFF.

Artifacts:

- `benches/path-b-plus/b6-4gpu-g-triton-n384-romulus-native/`
- `benches/path-b-plus/b6-4gpu-g-triton-n384-romulus-native-no-dual-socket/`
- `benches/path-b-plus/b11-4gpu-bisect-logs/`

## Operations

### Enable for experiment

```bash
export GGML_RPC_DUAL_SOCKET=1
# requires proto 4.4 rpc-server on every -rpc endpoint
```

### Rollback

```bash
export GGML_RPC_DUAL_SOCKET=0   # default; v3 HELLO wire
```

Single-socket fallback also occurs if CHANNEL_BIND times out (10s) or second connect fails.

### rpc-server version check

Profiler / validate logs must show `proto 4.4` on all endpoints before dual=ON gate runs. Mixed 4.3 + 4.4 cluster is OK with dual=OFF (v3 HELLO).

## Follow-up

1. ~~Rebuild jupiter `:50053`~~ — **deferred 2026-07-01**; canonical 4-GPU gate is `b6-4gpu-g-triton` (triton `:50054` as RPC2)
2. **B+13** next M3 hunt step per [TRACKING.md](TRACKING.md)
3. Do **not** enable dual-socket in production until a future bisect shows overlap or stable hol_tail improvement