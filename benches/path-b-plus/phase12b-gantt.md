# Phase 1.2B assembly-line Gantt + HOL tail RTT

Staged D step B: `decode_id` x split x cmd class timeline. Per-dir JSON: `telemetry/gantt-b.json` (ASCII Gantt + HOL tails).

## Summary

| label | gen_tokens | overlap_pct | overlap_pairs | pipeline_gap_p50_ms | pipeline_gap_p95_ms | cross_backend_overlap_pct | rpc_rtt_p50_ms | rpc_rtt_p95_ms | rpc_rtt_p99_ms | hol_tail_count | hol_tail_ms | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b6-4gpu-g-triton-n384-romulus-native | 384 | 0.2 | 3210/1482250 | -4.71 | -4.3 | 0.2 | 0.36 | 5.96 | 6.47 | 5 | 1036.23 | LOW_OVERLAP,HOL_TAIL_RTT |
| b6-4gpu-g-triton-n384-romulus-native-no-dual-socket | 384 | 0.2 | 3292/1482250 | -4.58 | -4.3 | 0.2 | 0.44 | 5.99 | 6.26 | 9 | 170.52 | LOW_OVERLAP,HOL_TAIL_RTT |

## Interpretation

- **pipeline_gap_ms**: `token_start[d+1] - token_end[d]`; >=0 means no cross-token overlap (serial assembly).
- **cross_backend_overlap_pct**: fraction of cross-backend split pairs with temporal overlap (S5 metric).
- **hol_tail_count**: RPC `send_recv` events >= max(5ms, 10x backend p50) — head-of-line / tail RTT suspects.
- **HOL_TAIL_RTT**: p99 >> p50 on wire RTT; motivates B+11 dual-socket RPC.
- **B11_HOL_CANDIDATE**: 4-GPU gate with p99 RTT > 20ms (romulus ladder hypothesis).
- **SERIAL_PIPELINE**: positive median pipeline gap — tokens do not overlap on the assembly line.

## Status

Phase 1.2 A+B complete. Next work is protocol-level (B+11 dual-socket / proto bump), not more path-b-plus patches.

