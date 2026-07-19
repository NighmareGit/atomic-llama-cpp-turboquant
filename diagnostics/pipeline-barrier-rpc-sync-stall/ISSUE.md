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

## Fix Applied

Three changes to `ggml/src/ggml-backend.cpp`:
1. wf_cross defaults ON with PPLUS=1 (no need for GGML_SCHED_WAVEFRONT_CROSS env)
2. RPC backends skipped in pipeline barrier event loop (no blocking event_synchronize on RPC)
3. RPC drain blocked skipped in barrier when wf_cross (per-split wait_producer handles it)

**Barrier time reduced: 12,000us → 8us (1500x).** Throughput unchanged: 55.5 t/s.

## Profiler Follow-up (2026-07-19)

### sched_trace Per-Split Timing

With `GGML_SCHED_TRACE=1`, the split-level timing reveals:

- **RPC split (3060Ti, 25% layers): avg 11.76ms compute + 1.99ms idle**
- CPU setup split: avg 0.01ms
- Barrier: 8us (fixed)

The 11.76ms IS the 3060Ti's actual layer compute time — not protocol overhead.

### RPC Telemetry Gap

`collect_telemetry()` is only called from `graph_compute()` (initial graph), NOT from `graph_recompute()` (token generation). The `graph_recompute` path at line 3485 of ggml-rpc.cpp measures server wall time but never writes it to the telemetry JSONL.

**Fix:** Added `collect_telemetry()` + direct JSONL write to `graph_recompute()` and `graph_recompute_all()`. Requires Docker image rebuild (CUDA toolchain).

### Remaining Bottleneck

The 3060Ti's 11.76ms compute for its 25% layer share is the hard ceiling. With single-depth pipeline (copy-slot rotation disabled by garble fix), throughput cannot exceed ~83 t/s ideal (~55 t/s observed). Solving this requires either re-enabling safe copy-slot rotation or using GPipe.

See `docs/research/d78-pipeline-barrier-rpc-sync-stall.md` for full analysis.
