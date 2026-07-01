# Phase 1.2B assembly-line Gantt + HOL tail RTT

Staged D step B: `decode_id` x split x cmd class timeline. Per-dir JSON: `telemetry/gantt-b.json` (ASCII Gantt + HOL tails).

## Summary

| label | gen_tokens | overlap_pct | overlap_pairs | pipeline_gap_p50_ms | pipeline_gap_p95_ms | cross_backend_overlap_pct | rpc_rtt_p50_ms | rpc_rtt_p95_ms | rpc_rtt_p99_ms | hol_tail_count | hol_tail_ms | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b6-2gpu-f-triton-n384-romulus-native | 384 | 0.2 | 886/444675 | -7.6 | -6.84 | 0.2 | 0.01 | 0.16 | 4.25 | 4 | 129.38 | LOW_OVERLAP,HOL_TAIL_RTT |
| b6-2gpu-f-triton-n384-romulus-native-no-get-defer | 0 | 0.3 | 0/0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | LOW_OVERLAP,SERIAL_PIPELINE |
| b6-2gpu-f-triton-guard-n128 | 128 | 0.9 | 459/49923 | -7.09 | -4.85 | 0.9 | 2.45 | 4.46 | 27.8 | 2 | 226.81 | LOW_OVERLAP,HOL_TAIL_RTT |
| b6-2gpu-f-triton-n384-no-partial | 384 | 0.2 | 886/444675 | -4.7 | -4.62 | 0.2 | 0.53 | 4.52 | 5.57 | 2 | 112.98 | LOW_OVERLAP,HOL_TAIL_RTT |
| b6-2gpu-f-triton-n384-remus-docker | 384 | 0.2 | 877/444675 | -7.09 | -5.36 | 0.2 | 1.39 | 4.36 | 5.15 | 2 | 106.43 | LOW_OVERLAP |
| b6-4gpu-g-n384-romulus-native | 384 | 0.2 | 2318/1482250 | -3.79 | -2.79 | 0.2 | 1.68 | 6.66 | 7.29 | 15 | 1968.4 | LOW_OVERLAP |

## Interpretation

- **pipeline_gap_ms**: `token_start[d+1] - token_end[d]`; >=0 means no cross-token overlap (serial assembly).
- **cross_backend_overlap_pct**: fraction of cross-backend split pairs with temporal overlap (S5 metric).
- **hol_tail_count**: RPC `send_recv` events >= max(5ms, 10x backend p50) — head-of-line / tail RTT suspects.
- **HOL_TAIL_RTT**: p99 >> p50 on wire RTT; motivates B+11 dual-socket RPC.
- **B11_HOL_CANDIDATE**: 4-GPU gate with p99 RTT > 20ms (romulus ladder hypothesis).
- **SERIAL_PIPELINE**: positive median pipeline gap — tokens do not overlap on the assembly line.

## Status

Phase 1.2 A+B complete. Next work is protocol-level (B+11 dual-socket / proto bump), not more path-b-plus patches.

