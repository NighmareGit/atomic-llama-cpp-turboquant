# Future Expansions — hot-path observability and adaptive routing

**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Date:** 2026-07-01  
**Status:** C-full instrumentation **in progress**; sample API **shipped**; live tail + adaptive routing **deferred**

Navigation: [TRACKING.md](TRACKING.md) | [PLAN.md](PLAN.md) | [IMPLEMENTATION.md](IMPLEMENTATION.md) | [pathb-sync-site-audit.md](../../rpc-patch/docs/pathb-sync-site-audit.md)

This document preserves design threads that are **not** in the current B+11–B+13 execution path. Grill decision (2026-07-01): ship **C-full** now; park live operator APIs and adaptive routing here so we do not lose the thread.

---

## What we ship now (C-full)

Phase 1.2C proved that pre-C-full traces cannot distinguish local sync fallback from on-wire copy (`copy_issue=0`, `COPY_TENSOR RPC=0 ms`, yet `input_wait_copy_ms` dominates). C-full adds correlated sched + RPC rows and explicit B+13 phases.

| Layer | Mechanism | Code |
|-------|-----------|------|
| Sched context | Thread-local `(split_id, backend_id)` set for each split in `ggml_backend_sched_compute_splits` | `ggml_hotpath_trace_set_sched_ctx()` in `ggml/include/ggml-backend.h` |
| Decode correlation | `decode_id` from `ggml_pipeline_trace_set_decode_id()` | `ggml-backend.cpp` sched emit |
| B+13 visibility | `sync_copy_fallback` (elapsed_us) when async copy fails and local sync path runs; `copy_async_ok` (0 us marker) when async succeeds | `ggml-backend.cpp` ~1865–1887 |
| RPC join | `decode_id`, `split`, `backend` appended to every `rpc_trace_emit` and `copy_issue` row | `ggml-rpc.cpp` `rpc_trace_emit_hotpath_fields()` |

**Env (unchanged):**

```bash
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE=1
export GGML_PIPELINE_TRACE=1   # decode_id on sched + rpc rows
# optional file targets:
# GGML_SCHED_TRACE_FILE=...
# GGML_RPC_TRACE_FILE=...
```

**Immediate consumers:** `scripts/b6-gate-phase12c-blocking-audit.sh` (extend for `sync_copy_fallback` / `copy_async_ok`), Phase 1.2 A+B parsers, one re-bench on `b6-2gpu-f-triton-n384-romulus-native` to validate new fields.

---

## C-full hotpath schema v1

Single-line JSON per event. Optional fields are omitted when unknown (`-1` context cleared after each split).

### sched-trace.jsonl

| Field | Type | Notes |
|-------|------|-------|
| `ts_us` | int64 | steady_clock microseconds |
| `decode_id` | int32 | gen token index when `GGML_PIPELINE_TRACE=1` |
| `split` | int32 | graph split index |
| `backend` | int32 | scheduler backend id |
| `copy` | int32 | pipeline copy slot (0..3) |
| `phase` | string | see phase table below |
| `elapsed_us` | int64 | phase duration |

**Sched phases (v1):**

| phase | Meaning |
|-------|---------|
| `input_wait_copy` | Full input-drain window for split (wait + copy) |
| `sync_copy_fallback` | **C-full** B+13 local sync path only (`synchronize` + `tensor_copy`) |
| `copy_async_ok` | **C-full** async copy succeeded (marker, `elapsed_us=0`) |
| `graph_compute_async` | Split graph dispatch |
| `event_record` | Per-split event record |
| `split_total` | Wall time for entire split |

Example (post re-bench):

```json
{"ts_us":8731883864,"decode_id":1,"split":2,"backend":1,"copy":1,"phase":"sync_copy_fallback","elapsed_us":14200}
{"ts_us":8731883870,"decode_id":1,"split":2,"backend":1,"copy":1,"phase":"copy_async_ok","elapsed_us":0}
```

### rpc-trace.jsonl

| Field | Type | Notes |
|-------|------|-------|
| `ts_us` | int64 | |
| `fn` | string | e.g. `send_rpc_cmd`, `rpc_issue_copy_tensor` |
| `phase` | string | `send_only`, `send_recv`, `copy_issue`, ... |
| `cmd` | int | `RPC_CMD_*` enum |
| `bytes` | size_t | payload size when applicable |
| `blocking` | bool | |
| `elapsed_us` | int64 | RTT or 0 for fire-and-forget |
| `decode_id` | int32 | **C-full** from pipeline trace |
| `split` | int32 | **C-full** from sched context |
| `backend` | int32 | **C-full** from sched context |
| `src_ep`, `dst_ep`, `peer_copy`, `defer` | | `copy_issue` rows only |

**Join key:** `(decode_id, split, backend)` links RPC hot cmds to sched waterfall for per-token Gantt (Phase 1.2 A+B).

### pipeline-trace.jsonl

Existing barrier / decode markers; unchanged. Small enough to copy whole file in samples.

### Schema versioning

- **v1 (C-full):** fields above; no `schema_version` column in jsonl (infer from presence of `sync_copy_fallback` / correlated RPC `split`).
- **v2 (future):** optional `schema_version`, `endpoint_id`, `tensor_hash` on copy_issue; see Future proofing below.

---

## Sample API (shipped — storage downsampling)

**Not** the live tail API (below). This is post-run downsampling for git-friendly artifacts and cross-node sharing.

| Entry | Role |
|-------|------|
| `scripts/llama-pipeline-trace-sample.sh` | Bash implementation |
| `tools/llama-pipeline-profiler/pipeline-trace-sample.cpp` | Profiler-native (`--trace-sample N`) |
| [TELEMETRY.md](../../tools/llama-pipeline-profiler/TELEMETRY.md) | Operator docs |

**Usage:**

```bash
# default: every 10th row + keep all blocking RPC + all split_total
bash scripts/llama-pipeline-trace-sample.sh benches/path-b-plus/<label>/telemetry

# tighter sample for huge 4-GPU traces
TRACE_SAMPLE_EVERY=5 TRACE_SAMPLE_MAX=2000 \
  bash scripts/llama-pipeline-trace-sample.sh <telemetry_dir>
```

**Outputs:**

| File | Policy |
|------|--------|
| `sched-trace.sample.jsonl` | Keep `phase==split_total`; every Nth decode; stride sample |
| `rpc-trace.sample.jsonl` | Keep all `blocking==true`; stride sample |
| `pipeline-trace.sample.jsonl` | Full copy (small) |
| `sample-meta.json` | `{version, every, max_lines, files: {source_lines, sample_lines}}` |

Profiler profile mode runs sampling automatically (`--trace-sample 10` default). Existing bench dirs already contain `*.sample.jsonl` from the Phase 1.1 wave.

**C-full extension (when parsers land):** add `sync_copy_fallback`, `copy_async_ok`, and `copy_issue` to sched/RPC keep sets so samples preserve B+13 evidence without full jsonl.

---

## Future expansions (deferred)

### F1 — Live tail + top blockers API

**Problem:** Operator waits for full bench + parse to see stall shift. Need sub-second feedback during long n=384 runs.

**Proposed surface (sketch):**

```bash
# tail last K decodes, aggregate blockers
pathb-trace-tail.sh <telemetry_dir> [--window 32] [--refresh 1]

# machine-readable snapshot for dashboards
pathb-trace-top-blockers.sh <telemetry_dir> --json
```

**Example JSON response:**

```json
{
  "window_decodes": 32,
  "overlap_pct_est": 0.3,
  "top_blockers": [
    {"id": "7c", "phase": "sync_copy_fallback", "ms": 1420, "backend": 1, "split": 2},
    {"id": "7d", "phase": "EVENT_RECORD", "ms": 890, "backend": 1},
    {"id": "7e", "phase": "GET_TENSOR", "ms": 120, "backend": 1}
  ],
  "per_backend_ms_per_token": {"0": 2.1, "1": 8.4, "2": 0.3}
}
```

**Implementation notes:**

- Read append-only jsonl with file offset cursor (no rewrite).
- Incremental rollup keyed by `decode_id`; reuse WAIT_MAP from `pathb-hotpath-summary.sh`.
- Map `sync_copy_fallback` ms directly to blocker **7c** / B+13 (replaces `LOCAL_SYNC_FALLBACK_LIKELY` inference).
- Optional: Unix socket or stderr heartbeat from profiler child process.

**Depends on:** C-full re-bench; Phase 1.2 A+B rollup functions shared with offline parsers.

---

### F2 — Phase 1.2 A+B parsers (offline, next after C-full)

| Deliverable | Description |
|-------------|-------------|
| **A — per-token blocking waterfall** | For each `decode_id`: sched phases stacked, RPC blocking cmds interleaved by `ts_us` |
| **B — assembly-line Gantt** | Backend lanes x time; overlap intervals; export `assembly-gantt.json` |

Scripts (proposed names): `pathb-trace-waterfall.sh`, `pathb-trace-gantt.sh`. Feed comparison matrix and B+11–B+13 before/after proofs.

---

### F3 — Adaptive hot-lane routing

**Problem:** Static split -> backend map pins hot MoE/RPC splits to a slow node (triton straggler @ 5–8 ms/tok). Throughput ceiling may be topology assignment, not just sync bugs.

**Idea:** Use rolling per-`(split, backend)` latency from C-full + tail API to **steer** hot splits toward faster endpoints at runtime.

**Constraints (Path-B+ scope):**

- Requires sched assignment hook or pre-split backend override — invasive; likely **post-M3** or Path C-adjacent.
- Must not break weight buffer affinity or HELLO `peer_copy` topology contracts.
- RX6600 / weak workers stay excluded by policy guard, not runtime roulette.

**Sketch:**

```
rolling_ms[split][backend]  # EMA from split_total or graph_compute_async
on_assign(split):
  if rolling_ms[split][current] > 1.5 * min_peer(split):
    prefer fastest eligible backend with VRAM headroom
```

**Telemetry hook:** emit `routing_hint` rows into `pipeline-trace.jsonl` (v2 schema) when hint fires; compare G/overlap in A/B bench.

**Out of scope until:** B+13/B+12 fixes proven on canonical triton; M3 retry fails with static map.

---

### F4 — Blocker 7f registry (audit doc)

Add to [pathb-sync-site-audit.md](../../rpc-patch/docs/pathb-sync-site-audit.md):

| # | Blocker | Symptom | Fix tier |
|---|---------|---------|----------|
| 7f | Hot-path observability gap | `LOCAL_SYNC_FALLBACK_LIKELY` without `sync_copy_fallback` rows | **C-full** (shipped); tail API (F1) |

---

### F5 — Future proofing checklist

Design choices in C-full that avoid another trace break:

1. **Stable join keys** — `decode_id`, `split`, `backend`, `copy` present on both sched and RPC rows.
2. **Explicit silent paths** — any sync fallback must emit a named phase (`sync_copy_fallback`), not only aggregate `input_wait_copy`.
3. **Sample preservation** — extend `llama-pipeline-trace-sample.sh` keep lists when adding phases.
4. **Parser-first** — new phase names registered in `pathb-hotpath-summary.sh` WAIT_MAP before bench wave.
5. **Schema bump** — add `schema_version: 2` only when adding required fields; v1 parsers remain valid on old artifacts.
6. **Path C boundary** — `topology_class: client_split` in diagnose.json stays; server-side aggregation fields remain in `path_c_reserved` ([TELEMETRY.md](../../tools/llama-pipeline-profiler/TELEMETRY.md)).

---

## Decision log

| Date | Decision | Notes |
|------|----------|-------|
| 2026-07-01 | **C-full over C-min** | C-min (decode_id only) insufficient to prove B+13; grill Q3/Q4 |
| 2026-07-01 | Sample API stays post-run | Existing `llama-pipeline-trace-sample.sh`; live tail -> F1 |
| 2026-07-01 | Adaptive routing deferred | F3; document intent here, no implementation on this branch |
| 2026-07-01 | B+13 before B+12 before B+11 | Driven by Phase 1.2C audit; unchanged |

---

## References

- Phase 1.2C audit: `benches/path-b-plus/phase12c-blocking-audit.md`
- Parse tools: `rpc-patch/scripts/pathb-hotpath-summary.sh`, `pathb-rpc-trace-parse.sh`
- C-full code: `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`
- Sample: `scripts/llama-pipeline-trace-sample.sh`