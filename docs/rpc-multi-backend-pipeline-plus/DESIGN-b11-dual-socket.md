# DESIGN: B+11 dual-socket RPC (proto 4.4)

| Field | Value |
|-------|-------|
| **Date** | 2026-07-01 |
| **Status** | Implemented; bisect **NULL** (default OFF) |
| **Feature doc** | [FEATURE-b11-dual-socket-rpc.md](FEATURE-b11-dual-socket-rpc.md) |
| **Protocol** | [RPC-PROTOCOL.md](RPC-PROTOCOL.md) |
| **Gate** | `b6-4gpu-g-n384-romulus-native` |

## Problem

Phase 1.2B showed HOL tail RTT on a single TCP socket per endpoint: `drain_event` spikes (84ms+ on 2-GPU, 854ms on 4-GPU) while p50 RTT stays ~0.3-1.7ms. Multiplexed cmd+response on one socket lets large `GET_TENSOR` / `EVENT_RECORD` responses block small acks.

## Solution

Split each RPC endpoint into two TCP connections after HELLO:

| Socket | Direction | Traffic |
|--------|-----------|---------|
| **cmd** | client -> server | requests (`send_only`, `send_recv` cmd half) |
| **rsp** | server -> client | all response payloads (`out_size` + data, deferred recv) |

Proto bump: **4.4** (`RPC_PROTO_MINOR_VERSION=4`). New cmd: `RPC_CMD_CHANNEL_BIND` (20).

## Wire protocol

1. Client HELLO (32-byte req) sets `dual_socket=1` + `session_id`.
2. Server registers pending session, HELLO rsp sets `flags&1` + echoes `session_id`.
3. Client opens second TCP to same host:port, sends `CHANNEL_BIND` + `session_id`.
4. Server pairs `rsp_channel` on cmd socket; responses use `send_response()` (rsp socket).

Flag: `GGML_RPC_DUAL_SOCKET=1` (default **OFF**; v3 HELLO wire when OFF for 4.3 server compat). Bisect ON: export before `canonical-romulus`; OFF: `no-dual-socket`.

## Bisect

```bash
# Rebuild ALL rpc-servers (proto 4.4) before bisect
B6_GATE_PRESET=b6-4gpu-g bash scripts/b6-gate-bisect-run.sh canonical-romulus
B6_GATE_PRESET=b6-4gpu-g bash scripts/b6-gate-bisect-run.sh no-dual-socket
```

Or background on romulus: `scripts/b6-gate-romulus-b11-4gpu-bisect-bg.sh`

## Pass criteria (M3)

- `overlap_pct` delta >= 1% vs `no-dual-socket` on 4-GPU gate, or
- `hol_tail_count` / `drain_event` ms down >= 50% with stable G

## Rollback

Set `GGML_RPC_DUAL_SOCKET=0` (single-socket fallback; requires proto 4.4 server accepting 32-byte HELLO).

## Bisect result (2026-07-01)

Gate `b6-4gpu-g-triton` n=384 on romulus (proto 4.4 remus/romulus/triton):

| Arm | G (t/s) | overlap_pct | hol_tail_ms |
|-----|---------|-------------|-------------|
| dual ON | 73.2 | 0.2% | 1036 (864ms spike) |
| dual OFF | 80.5 | 0.2% | 171 |

**Verdict:** NULL on overlap; HURTS G (-9.1%). Default OFF retained.