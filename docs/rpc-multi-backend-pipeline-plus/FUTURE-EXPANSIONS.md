# Future Expansions — C-full trace sample API

**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Date:** 2026-07-01  
**Status:** Deferred (not blocking current B+11–B+13 work)

Navigation: [IMPLEMENTATION.md](IMPLEMENTATION.md) (C-full schema + active queue) | [TRACKING.md](TRACKING.md)

This file holds **one** deferred item: extending the post-run trace sample scripts for C-full phases. Everything else from the Phase 1.2 grill (C-full emit, parsers, re-bench, B+13 fix) lives in IMPLEMENTATION and TRACKING.

---

## Deferred: C-full sample keep lists

**Shipped today:** `scripts/llama-pipeline-trace-sample.sh` and profiler `--trace-sample N` downsample traces for git-friendly artifacts. Current policy:

| File | Keep policy |
|------|-------------|
| `sched-trace.sample.jsonl` | `phase==split_total`; every Nth `decode_id`; stride |
| `rpc-trace.sample.jsonl` | all `blocking==true`; stride |
| `pipeline-trace.sample.jsonl` | full copy |

**Gap:** C-full adds `sync_copy_fallback`, `copy_async_ok`, and correlated `copy_issue` rows. Pre-C-full samples drop B+13 evidence because those phases are not in the keep set.

**Proposed extension (both bash + `pipeline-trace-sample.cpp`):**

```python
# sched: add to keep_phase
{"split_total", "sync_copy_fallback", "copy_async_ok"}

# rpc: add to keep_phase (non-blocking copy_issue rows)
{"copy_issue"}
```

**Acceptance:** Re-sample a C-full bench dir; `sched-trace.sample.jsonl` contains at least one `sync_copy_fallback` or `copy_async_ok` row when the full trace does; `sample-meta.json` documents `c_full_keep: true`.

**Why deferred:** C-full emit landed first (`6dc504bce`); sample script parity can follow after instrumented re-bench validates the new phases. No gate decision depends on sampled artifacts.

---

## References

- Sample scripts: `scripts/llama-pipeline-trace-sample.sh`, `tools/llama-pipeline-profiler/pipeline-trace-sample.cpp`
- [TELEMETRY.md](../../tools/llama-pipeline-profiler/TELEMETRY.md)
- C-full schema + work queue: [IMPLEMENTATION.md](IMPLEMENTATION.md#c-full-hotpath-instrumentation)