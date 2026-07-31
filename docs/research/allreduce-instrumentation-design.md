# Durable split + AllReduce instrumentation design

**Date:** 2026-07-17  
**Wayfinder ticket:** [Design durable split and AllReduce instrumentation](../../.scratch/allreduce-timing/issues/04-design-durable-instrumentation.md)  
**Status:** locked (grilling)  
**Implements in:** tickets 05 (split), 06 (AR)

---

## Goals

| ID | Requirement |
|----|-------------|
| A | Split timestamps on **both rails** (layer + tensor) for idle vs compute vs bubble |
| B | AllReduce per-call timing + provider visibility on **tensor rail** (all providers) |
| C | Durable product feature, not throwaway; reuse Path-B NDJSON / env patterns |
| D | Host `ggml_time_us` wall clock for v1 |
| E | gpipe-profiler / scripts **consume** traces; library owns emission |

---

## Locked decisions

| Topic | Decision |
|-------|----------|
| Export family | Extend existing **SCHED/PIPELINE NDJSON** style |
| Split richness | Four absolute timestamps + ids |
| AR frequency | **Per-call** NDJSON + **first-call / init** provider INFO |
| Env gating | Split under **`GGML_SCHED_TRACE`**; AR under dedicated **`GGML_ALLREDUCE_TRACE`** |
| Clock | Host **`ggml_time_us`** around the timed region |
| Butterfly | **Yes** — time meta fallback same as NCCL/internal |

---

## A — Split timestamps

### Enable

- `GGML_SCHED_TRACE=1` (existing) enables emission; new fields are **additive**.
- `GGML_SCHED_TRACE_FILE` (existing) for file append; else stderr / current default.
- When `GGML_PIPELINE_TRACE=1`, include `decode_id` on split lines (existing pipeline tagging).

### Schema (per split event)

NDJSON object, same line style as current sched-trace. Illustrative fields:

```json
{
  "event": "split_timing",
  "ts_us": 0,
  "backend_id": 0,
  "split_id": 0,
  "decode_id": -1,
  "t_split_start_us": 0,
  "t_compute_start_us": 0,
  "t_compute_end_us": 0,
  "t_event_record_us": 0,
  "idle_us": 0,
  "compute_us": 0
}
```

| Field | Meaning |
|-------|---------|
| `t_split_start_us` | Before launching this split / entering split region |
| `t_compute_start_us` | After inputs ready / event wait done (start of useful compute) |
| `t_compute_end_us` | After backend compute finishes for this split |
| `t_event_record_us` | When EVENT_RECORD (or equivalent completion signal) is issued |
| `idle_us` | Derived: `t_compute_start - t_split_start` (wait / bubble before compute) |
| `compute_us` | Derived: `t_compute_end - t_compute_start` |

Absolute stamps + derived durations: stamps for cross-trace align; durations for summaries.

### Insert points (implementation guidance for ticket 05)

Prefer the Path-B / sched path that already emits `GGML_SCHED_TRACE` lines in `ggml-backend.cpp` (and GPipe stage path if splits are filtered there).  

- Record `t_split_start` at split begin.  
- Record `t_compute_start` after input copy / event wait for this split.  
- Record `t_compute_end` after `graph_compute` (or equivalent) returns/sync for timing purposes consistent with existing sched-trace phases.  
- Record `t_event_record` when the completion event is recorded for downstream consumers.

Must work for:

- Layer multi-backend (T1/T3 Path-D)  
- Tensor meta multi-device (T2)  

Do **not** require tensor mode for split stamps.

### Overhead

Only when `GGML_SCHED_TRACE` is on (default off). Acceptable for baseline runs.

---

## B — AllReduce timing

### Enable

| Env | Default | Role |
|-----|---------|------|
| `GGML_ALLREDUCE_TRACE` | unset/0 | `1` = per-call NDJSON AR lines |
| `GGML_ALLREDUCE_TRACE_FILE` | unset | Append path; else stderr (mirror SCHED_TRACE_FILE pattern) |

Independent of `GGML_SCHED_TRACE` so AR matrix runs need not spam full sched traces.

### First-call / init visibility

- On successful specialized provider selection in `comm_init`, keep/enhance INFO (internal pipeline already logs).  
- On **first** `comm_allreduce` (or first per provider), emit one INFO or NDJSON `event=allreduce_provider` with `provider` in `{nccl,rccl,internal,butterfly}` and `n_gpus`.  
- `GGML_CUDA_ALLREDUCE` remains the force switch (inventory).

### Per-call schema

```json
{
  "event": "allreduce",
  "ts_us": 0,
  "provider": "nccl",
  "path": "specialized",
  "n_gpus": 2,
  "ne": 0,
  "nbytes": 0,
  "duration_us": 0,
  "decode_id": -1
}
```

| Field | Meaning |
|-------|---------|
| `provider` | `nccl` / `rccl` / `internal` / `butterfly` |
| `path` | `specialized` vs `fallback` (meta butterfly) |
| `ne`, `nbytes` | Element count and wire/host buffer size as known at call |
| `duration_us` | Host wall: `ggml_time_us` around the AR call region |
| `decode_id` | If pipeline trace context available; else -1 |

### Insert points (ticket 06)

1. **Specialized:** wrap `ggml_backend_cuda_comm_allreduce_tensor` / try_allreduce success path (`ggml-cuda.cu`) and internal `ggml_cuda_ar_allreduce` if needed for provider label.  
2. **Butterfly:** wrap `allreduce_fallback` in `ggml-backend-meta.cpp` (entire fallback for that subgraph boundary).  
3. Use one shared helper (e.g. `ggml_allreduce_trace_log(...)`) to avoid divergent formats.

### Overhead

Only when `GGML_ALLREDUCE_TRACE=1`. Per-call is required for TG small-message distribution.

---

## Consumer notes

- **Baselines (ticket 07):** enable `GGML_SCHED_TRACE` + file for T1/T3 stall; enable `GGML_ALLREDUCE_TRACE` for T2 provider matrix; optionally both on T2 for correlation.  
- **Scripts:** extend or add a thin parser alongside `pathb-hotpath-summary` style tools; gpipe-profiler may ingest later without blocking 05/06.  
- **Kernel profilers** (rocprofv3/nsys): still optional external; not a substitute for these host traces.

---

## Non-goals (this design)

- Device-event pure GPU time (v2 optional).  
- Sampling every Nth AR (rejected for v1).  
- Placement control plane / plan IR (sibling map).  
- Changing AllReduce algorithms (plan map only measures).

---

## Implementation checklist

| Ticket | Work |
|--------|------|
| 05 | Additive split_timing on SCHED_TRACE path; four stamps + ids; both rails — **done** |
| 06 | GGML_ALLREDUCE_TRACE(+_FILE); per-call NDJSON; provider INFO; specialized + butterfly — **done** |

ASCII-only comments; minimal surface; no new subsystem beyond env + helper + fields.
