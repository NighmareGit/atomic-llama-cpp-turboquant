# Path-B Plus Handover (draft)

## What changed

Path-B Plus completes Path B assembly-line scheduling for multi-RPC clusters.

### B+1 (shipped in branch)

- **P0:** `ggml_backend_sched_pipeline_barrier` -- rotates `cur_copy` on graph reuse; event-wait per slot instead of full sched sync.
- **P1:** Sampling/logits C API uses narrow backend sync when `GGML_PIPELINE_PLUS=1`.
- **P2:** RPC fire-and-forget sends no longer drain pending events/GETs; blocking ops still drain.
- **P3:** Sched trace includes `copy` field; parser reports assembly-line overlap.

## Env vars

| Var | Default | Effect |
|-----|---------|--------|
| `GGML_PIPELINE_PLUS` | `1` | Enable P0/P1 narrow sync. `0` = legacy Path B behavior |
| `GGML_SCHED_TRACE` | `0` | Per-split trace jsonl |
| `GGML_RPC_TRACE` | `0` | RPC client trace jsonl |

## Verify

1. Server log: `pipeline parallelism enabled`, `sched copies = 4`
2. `trace-summary.txt`: `assembly_overlap_count > 0` during GEN
3. G(t/s) vs `trace-f-3gpu` baseline (38 t/s)

## Fallback

`GGML_PIPELINE_PLUS=0` or rebuild without B+1 commits restores full `sched_synchronize` on graph reuse.