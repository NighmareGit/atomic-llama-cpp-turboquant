# Pipeline Barrier RPC Synchronization Stall

**Date:** 2026-07-19
**Severity:** PERFORMANCE (critical for multi-GPU RPC setups)
**Status:** Root cause identified, fix pending

---

## Symptom

Multi-GPU RPC inference stalls for 12ms at every pipeline barrier during graph reuse (token generation). Observed on 7900XTX (ROCm) + 3060Ti (RPC via Docker, local PCIe). 9B Q4_K_M achieves only 55.7 t/s vs ~260 t/s single-GPU capability.

## Root Cause

`ggml_backend_sched_pipeline_barrier()` in `ggml/src/ggml-backend.cpp` calls `ggml_backend_event_synchronize()` on the RPC backend's event. This does a **blocking socket read** via `rpc_finish_event_response()` → `recv_rpc_cmd_deferred()`, waiting for the RPC server (3060 Ti Docker) to finish computing.

The per-split input handling in `compute_splits()` already synchronizes RPC events before using input tensors. The pipeline barrier's RPC wait is **redundant**.

## Evidence (rocprofv3 kernel trace)

- GPU utilization: 15.5% during inference
- 123 stalls at ~12ms each = 1,465 ms lost (46.2% of wall time)
- Pattern: compute kernel (queue 3) → 12ms stall → copyBuffer (queue 2)
- Profiler data: `/tmp/rocprof-data-20260719-191812/`

## Affected Configurations

Any setup with PPLUS=1 + RPC backend + graph reuse:
- `GGML_PIPELINE_PLUS=1` (required for the stall to manifest)
- `--rpc <host:port>` (RPC backend in the backend list)
- `-sm layer` or `-sm row` with `-ts` (layer split across backends)
- `GGML_RPC_EVENT_DEFER_BARRIER=1` (doesn't help — event is deferred but the response wait is still blocking)

## Proposed Fix

In `ggml_backend_sched_pipeline_barrier()`, skip RPC backends in the event wait loop when `wf_cross` is enabled. The deferred RPC drain (lines 3452-3467) and per-split input event synchronization provide sufficient synchronization.

See `docs/research/d78-pipeline-barrier-rpc-sync-stall.md` for full analysis.
