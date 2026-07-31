# Configuration — Path-B+ (rpc-multi-backend-pipeline-plus)

This document defines the **blessed configuration** for production use of Path-B-Plus multi-backend orchestration and the deprecation policy for superseded flags.

## Blessed Configuration (4-GPU reference)

Taken from the clean baseline `benches/path-b-plus/b6-4gpu-g-n384-romulus-native/env.txt` and verified production runs (e.g. `trace-g-4gpu-primary` family).

```bash
GGML_PIPELINE_PLUS=1
GGML_PIPELINE_BARRIER_PARTIAL=1
GGML_PIPELINE_BARRIER_PARTIAL_STRICT=1
GGML_RPC_EVENT_DEFER_BARRIER=1
GGML_RPC_GET_TENSOR_DEFER=1          # or equivalent B+12
GGML_RPC_MULTI_SOCKET_FLUSH=1
GGML_SCHED_MOE_ASYNC_COPY=1
```

Typical topology:
```
RPC=... (client + workers)
TS=25,12,25,38   # example 4-GPU split
```

This set is expected to be the default for 4-GPU (and analogous 2-GPU) Path-B+ deployments unless a specific bisect or experiment is being run.

See `IMPLEMENTATION.md` for the full current env flag table with status.

## Deprecated Flags (do NOT enable in production)

These were explored in the mitigation ladder but showed regression or null effect on the primary mission metric (`overlap_pct`). They will be removed in a later phase after successors are proven.

| Flag                            | Origin | Regression / Verdict          | Superseded by          | Planned removal |
|---------------------------------|--------|-------------------------------|------------------------|-----------------|
| `GGML_RPC_DUAL_SOCKET`          | B+11   | -9.1% G on 4-GPU              | ADR-005 (RDMA+coalescing) | Phase R3       |
| `GGML_RPC_HASH_DEFER`           | B+7f   | -1.9% G                       | ADR-005                | Phase R3       |
| `GGML_SCHED_WAVEFRONT_DISPATCH` | B+14   | NULL on M3 (global_3bk <1%)   | ADR-003 (adaptive depth) | Phase R3     |

When enabled they must emit a clear one-time deprecation warning (see ADR-013).

## Audit at Startup

```bash
GGML_CONFIG_AUDIT=1 ./build/bin/llama-server ...
```

Emits a one-time block listing every known Path-B+ flag, its current value, and deprecation status. Non-blocking / observational.

## Rollback to Legacy Path B

```bash
GGML_PIPELINE_PLUS=0
```

Restores pre-Plus full synchronization behavior per decode.

## Notes

- All new R1 additions (`LLAMA_PIPELINE_DEPTH2`, `GGML_CONFIG_AUDIT`) default OFF.
- Trace correlation (`trace_id`) is always on when tracing is enabled; consumers that do not care can ignore the field (`trace_id=0` is the legacy signal).
- The blessed set above is the regression gate for any change in this workstream.

See:
- `PLAN.md` (integrated phasing)
- `TRACKING.md` (current verdicts)
- `IMPLEMENTATION.md` (full flag table + status)
- ADR-011, ADR-013, ADR-007 for the hygiene changes that produced this policy.
