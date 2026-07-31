# Phase 1.2C blocking audit (B+13 preliminary)

Staged D step C: existing traces only. Per-dir JSON: `telemetry/blocking-audit-c.json`.

## Results

| label | gen_tokens | overlap_pct | stall_ratio | input_wait_copy_ms | graph_compute_ms | sync_copy_fallback_ms | copy_async_ok_count | blocking_ms | drain_ms | copy_issue | copy_tensor_rpc | copy_peer_rpc | get_tensor_rpc | set_hash_rpc | event_record_rpc | local_sync_gap_ms | c_full | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b6-2gpu-f-triton-n384-romulus-native | 384 | 0.2 | 0.838 | 2805.9 | 306.7 | 0.0 | 2310 | 208.7 | 207.0 | 1155 | 0.0 | 0.0 | 66.4 | 0.0 | 140.6 | 2805.9 | yes | COPY_ASYNC_OK_ONLY,GET_TENSOR_GT_COPY,WAIT_DOMINATES_COMPUTE |

## Interpretation guide

- **NO_COPY_ISSUE_TRACE**: `GGML_RPC_TRACE` copy_issue lines absent; wire-level copy deferral not visible in these artifacts.
- **LOCAL_SYNC_FALLBACK_LIKELY**: pre-C-full heuristic; `input_wait_copy_ms` >> wire COPY ms.
- **SYNC_COPY_FALLBACK_MEASURED**: C-full `sync_copy_fallback` phase present (>50ms gen window) — B+13 local sync path proven.
- **B13_DOMINATES_INPUT_WAIT**: measured `sync_copy_fallback_ms` > 50% of `input_wait_copy_ms`.
- **COPY_ASYNC_OK_ONLY**: async copy path succeeded (markers only, no fallback rows).
- **C_FULL_NO_B13_PHASES**: RPC join present but no B+13 phase rows (check trace env + build SHA).
- **EVENT_RECORD_DOMINATES_DRAIN**: B+9/B+12 less likely to move overlap until EVENT path shortened.
- **SET_HASH_RPC_HEAVY**: weight relay still costs gen-window budget (B+4 cache check).
- **WAIT_DOMINATES_COMPUTE**: assembly line starved regardless of straggler ms/tok.

## Next: Phase 1.2 A+B

Per-token blocking waterfall + assembly-line Gantt (decode_id x split x cmd class).

