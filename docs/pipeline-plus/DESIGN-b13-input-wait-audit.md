# DESIGN: B+13 input_wait audit (canonical 2-GPU triton)

| Field | Value |
|-------|-------|
| **Date** | 2026-07-01 |
| **Bench** | `b6-2gpu-f-triton-n384-romulus-native` |
| **Status** | Audit complete; fix candidates identified |

## Executive summary

`sync_copy_fallback` is **not** the bottleneck (`0 ms`, `copy_async_ok` = 2310). Split-2 **gather** on the client (`l_out-16` RPC→CUDA0) dominates `input_wait_copy` via:

1. **`event_sync_slot`** (~7 ms/token steady state) — copy-slot `event_wait` before inputs
2. **`rpc_flush_downloads`** (4–113 ms) — blocking GET recv + H2D when B+15 prefetch did not finish in time
3. **`input_copy_slow` with `reject=unknown`** — **telemetry bug**, not a separate slow path; timer includes producer/slot waits while async defer succeeded

## Topology (this bench)

| Split | backend_id | Role |
|-------|------------|------|
| 0 | 2 | CPU |
| 1 | 0 | RPC triton (main layers) |
| 2 | 1 | Client CUDA0 gather (`l_out-*` RPC→local) |

Straggler: backend 0 (RPC) @ ~8.5 ms/tok in waterfall; split-2 client flush+compute adds client-side wait.

## Per-token timeline (split 2)

### Steady state (decode 100)

```
event_sync_slot     7407 us   <- copy-slot wait (dominant)
rpc_download_issue     6 us   <- deferred GET issued (async OK)
copy_async_ok          0 us
input_copy_slow     7420 us   <- wall timer; reject=unknown (misleading)
input_wait_copy     7492 us
rpc_flush_downloads  126 us   <- prefetch worked; small flush
graph_compute_async    7 us
split_total         7635 us
```

### Cold / large tensor (decode 1, first gen token)

```
rpc_flush_downloads 113050 us  <- 7.5 MB l_out-16 GET+H2D not overlapped
graph_compute_async 264225 us
split_total        378952 us
```

## Code path (verified)

| Phase | Location | Behavior |
|-------|----------|----------|
| `event_sync_slot` | `ggml_backend_sched_wait_copy_slot()` | Waits `sched->events[split][cur_copy]` before input drain |
| `rpc_download_issue` | `ggml_backend_sched_try_async_tensor_copy()` → `ggml_backend_rpc_try_download_tensor()` | Issues deferred GET; returns immediately |
| `copy_async_ok` | same, success branch | Marker only (`elapsed_us=0`) |
| `input_copy_slow` | end of input loop iteration | Fires when wall > 50 us; **`reject` stale** |
| `rpc_flush_downloads` | after input loop, non-RPC split | `ggml_backend_rpc_flush_pending_downloads_for_dst()` recv+H2D |
| B+15 prefetch | split 1 **end** (RPC) | `event_synchronize` then `prefetch_gather_rpc_inputs` for split 2 |

### Why `reject=unknown`

`g_sched_copy_reject` defaults to `"unknown"` in `try_async_tensor_copy()` and is **not** updated when `rpc_try_download_tensor` succeeds. `input_copy_slow` logs the stale value.

## Root cause (overlap)

Async defer **works**, but overlap fails because:

1. Copy-slot wait (~7 ms) runs **before** inputs can reuse buffers.
2. B+15 prefetch for split 2 runs at split 1 **end** after **full RPC event sync** — downloads cannot start until split 1 RPC graph completes.
3. First-token / large `l_out-16` (7.5 MB) misses overlap → 113 ms flush.

## Fix candidates (ordered)

| ID | Change | Expected effect |
|----|--------|-----------------|
| B+13a | Set `g_sched_copy_reject="rpc_download_defer"` on successful defer paths | Correct trace attribution |
| B+13b | Issue split-2 deferred GETs **before** `wait_copy_slot` when dst is local gather | Hide slot wait behind wire transfer — **implemented** (`rpc_gather_prefetch_early`) |
| B+13c | B+15: prefetch at split 1 **start** (already partial) + avoid double `event_synchronize` at split 1 end | More overlap window — **implemented** (`rpc_prefetch_end`, defer-only) |
| B+13d | Pipeline `event_sync_slot` with `event_wait` on producer only (not full slot) for gather splits | Cut ~7 ms steady-state wait — **implemented** (`event_wait_producer_slot`, defer gather) |

## Pass criteria (unchanged)

- `overlap_pct` delta >= 1% on canonical n=384, or
- `input_wait_copy` / `event_sync_slot` ms down >= 50% on split 2 with stable G

## Artifacts

- `telemetry/blocking-audit-c.json`
- `benches/path-b-plus/phase12c-blocking-audit.tsv`