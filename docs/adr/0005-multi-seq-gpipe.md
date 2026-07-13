# ADR-0005: Multi-Sequence GPipe Scheduling Model

## Status

**Accepted** -- 2026-07-13. Scheduling model for Mode B multi-seq GPipe.

## Context

Mode A GPipe is single-sequence: one token at a time flows through pipeline stages, with MTP draft coupling. Mode B extends this to multiple sequences occupying different pipeline stages concurrently -- the "microbatch" model where different sequences are at different positions in the pipeline.

This ADR establishes the scheduling model for multi-sequence GPipe.

### Problem

Single-sequence GPipe limitations:
- Only one token in-flight per pipeline stage
- Pipeline stages are underutilized when one sequence is slow
- Server multi-slot (multiple concurrent sequences) cannot exploit pipeline overlap

Multi-sequence GPipe questions:
- How do multiple sequences share pipeline stages?
- How does KV cache isolation work across sequences in the pipeline?
- What prevents one slow sequence from blocking all others?

### Inputs

| Analysis | Source | Finding |
|----------|--------|---------|
| Server multi-slot behavior | `tools/server/server-context.cpp` | LRU + prompt-similarity slot selection; configurable `n_parallel` (default 4) |
| KV cache per-sequence | `src/llama-kv-cache.cpp` | Bitset-based cell ownership per `llama_seq_id`; full `seq_rm/add/cp/keep` lifecycle |
| Pipeline stage state | `src/llama-context.h:44-61` | `llama_gpipe_state` with `cur_stage`, `gpipe_events[n_stages]`, `stage_tokens` (unused) |
| MTP coupling | ADR-0002 | MTP draft/target are separate contexts; no additional multi-seq constraints |
| Event signaling risk | D5.4 Finding 3 | Single-event-per-stage creates ambiguity under multi-seq; double-buffering required |

### Options

#### Option A: Sequential multi-seq (no pipeline sharing)

- Multiple sequences served, but each gets its own pipeline turn
- No pipeline overlap between sequences
- Simplest; no KV isolation challenges
- **Does not improve pipeline utilization** -- fails the core goal of Slice 4

#### Option B: Interleaved multi-seq (pipeline sharing)

- Different sequences occupy different pipeline stages concurrently
- Sequence A at Stage 0 while Sequence B at Stage 1
- Requires per-sequence stage tracking
- KV cache must isolate sequences at different pipeline positions

#### Option C: Microbatch (batch-level pipeline)

- Multiple sequences processed as a microbatch at each stage
- Stage 0 processes batch of N sequences, then Stage 1 processes same batch
- Requires batch-aware stage dispatch
- Higher throughput but higher latency per sequence
- **Incompatible with existing split-per-graph dispatch model** -- would require rewriting the scheduler

## Decision

**Option B -- Interleaved multi-seq with stage-available scheduling.**

### Scheduling Model: Stage-Available

Each pipeline stage can accept work from any sequence whose next stage matches the stage's position. The dispatch loop iterates over stages and picks an available sequence:

```
for each stage s in [0, n_stages):
    if stage_tokens[s] is empty:
        find sequence seq where seq_stage[seq] == s
        dispatch seq at stage s
        stage_tokens[s] = seq
        seq_stage[seq] = (s + 1) % n_stages  // advance
```

This is more flexible than round-robin: if a sequence stalls (e.g., waiting for an RPC response), other sequences can fill the pipeline gaps.

### Event Signaling: Double-Buffered

Per D5.4 Finding 3, single-event-per-stage creates ambiguity under multi-seq. The solution is **double-buffered events**:

```cpp
ggml_backend_event_t gpipe_events[2][GGML_SCHED_MAX_STAGES];
int gpipe_event_bank = 0;  // toggle 0/1 per dispatch cycle
```

Each sequence uses alternating event banks (even/odd token index), preventing timestamp overwrite. For romulus dual-GPU (target: 2 concurrent sequences), 2 banks are sufficient. Extensible to `n_seq_max` banks if needed.

### KV Cache Isolation: Reuse Existing Model

The existing `llama_kv_cells` bitset-based isolation already handles per-sequence cell ownership. No changes needed:
- Each sequence writes to its own KV cells via `llama_kv_cache_seq_add`
- Different sequences at different stages cannot corrupt each other's cells
- The gather stage (last stage) writes KV per-sequence; per-backend serialization prevents races

### Per-Sequence Stage Tracking

Replace the single `cur_stage` int with per-sequence tracking:

```cpp
std::map<llama_seq_id, int> seq_stage;   // seq_id -> current stage position
std::vector<llama_seq_id> stage_tokens;  // stage -> owning seq_id (-1 = empty)
```

The `stage_tokens` vector (already declared in `llama_gpipe_state`) serves as the stage ownership map. `seq_stage` tracks each sequence's position independently.

### Backend Concurrency

Each backend can run one graph at a time (single device queue). Multi-seq does not change this:
- The client dispatches `GRAPH_COMPUTE` for each sequence at its stage
- The RPC server's compute queue serializes requests per-backend
- No server-side changes needed for basic multi-seq

## Consequences

### Positive

1. **Pipeline utilization improves.** Multiple sequences fill stages that would otherwise be idle (e.g., while waiting for KV write on another sequence).

2. **No KV cache changes needed.** Existing per-sequence cell isolation handles multi-seq correctly out of the box.

3. **Minimal event system change.** Double-buffered events require only 2x the event array (2 banks × 8 stages = 16 events max, negligible).

4. **Server dispatch unchanged.** The existing `enqueue_graph_compute()` handles sequential multi-seq without modification.

5. **Stage-available scheduling is fault-tolerant.** If one sequence stalls, others fill the gap. No head-of-line blocking across sequences.

6. **Scales with sequence count.** `seq_stage` map and `stage_tokens` vector scale linearly with active sequences.

### Negative

1. **Double-buffered event management adds dispatch overhead.** Each `record`/`wait` must use the correct event bank. Overhead is O(1) per call.

2. **Stage-available scheduling is non-deterministic.** Without priority, any free sequence can claim a stage. This could cause fairness issues under heavy load.

3. **`stage_tokens` is per-scheduler, not per-context.** Since the GPipe state machine runs in `llama_decode_gpipe_impl()` which receives a single `llama_context`, the stage map lives in `gpipe_state`. Multiple contexts sharing a scheduler would need separate tracking.

4. **Does not address batching.** Option C (microbatch) was rejected because it requires scheduler changes, but it would yield better throughput for many concurrent sequences. This is deferred to a future optimization phase.

### Neutral

1. **MTP coupling is orthogonal.** Draft sequences have their own contexts and schedulers. Multi-seq GPipe operates independently of MTP.

2. **Adaptive depth continues to work.** The adaptive timing logic tracks per-stage timing regardless of which sequence occupies a stage. Warmup counts may need adjustment (more warmup tokens across sequences), but the mechanism is unchanged.

## Alternatives Rejected

| Alternative | Reason Rejected |
|-------------|-----------------|
| **Option A: Sequential multi-seq** | Does not improve pipeline utilization; fails core Slice 4 goal |
| **Option C: Microbatch (batch-level)** | Requires rewriting the split-per-graph dispatch model in `ggml_backend_sched`; out of scope |
| **Round-robin scheduling** | Head-of-line blocking when a sequence stalls; less fault-tolerant than stage-available |
| **Per-sequence event arrays (n_seq × n_stages)** | Over-allocates for dual-GPU target; double-buffering is simpler and sufficient |
| **Ring-buffer events** | Complex lifecycle management; pool + ref-count overhead not justified for 2-sequence target |
| **Separate scheduler per sequence** | Each scheduler would own separate backends, defeating pipeline sharing across backends |

## References

- `docs/wayfinder/D6.1-multi-seq-requirements.md` -- Research findings informing this decision
- `docs/wayfinder/D5.4-prototype-findings.md` -- Finding 3: event ambiguity under multi-seq
- `docs/adr/0002-gpipe-kv-ordering.md` -- KV ordering model (per-sequence KV-ready release)
- `docs/adr/0003-adaptive-pipeline-depth.md` -- Adaptive depth (compatible with multi-seq)
- `src/llama-context.h:44-61` -- Current `llama_gpipe_state` struct
- `ggml/src/ggml-backend.cpp:2847-2905` -- Current `gpipe_events` API

## Domain Language Additions

| Term | Definition |
|------|------------|
| **Stage-available scheduling** | Any sequence can claim a free stage whose position matches the sequence's next stage; no fixed ordering between sequences |
| **Double-buffered events** | Two alternating event banks (0/1) prevent timestamp overwrite when multiple sequences share pipeline stages |
| **Stage ownership** | `stage_tokens[s] = seq_id` records which sequence currently occupies stage `s`; -1 means stage is free |
| **Pipeline fill** | Ratio of occupied stages to total stages; target is 1.0 (all stages busy) for maximum utilization |

---

*ADR-0005 accepted 2026-07-13. Proceed to D6.3 (spec section).*
