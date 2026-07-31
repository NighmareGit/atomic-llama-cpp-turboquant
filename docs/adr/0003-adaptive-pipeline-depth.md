# ADR-0003: Adaptive Pipeline Depth Model

## Status

**Accepted** — 2026-07-11 (Option C: Hybrid with topology-aware default, adaptive opt-in)

## Context

The current Mode A implementation uses a fixed 2-stage pipeline (Stage 0: compute, Stage 1: gather). As documented in D0.5 section 7, this does NOT improve throughput per token until Stage 0 is also split into sub-stages. For actual throughput gain, a deeper pipeline (n_stages > 2) is needed.

This ADR establishes the model for determining pipeline depth: static configuration vs runtime adaptive assignment based on backend timing.

### Problem

Fixed n_stages=2 has these limitations:
- Stage 0 (RPC compute) is the bottleneck at ~16.8 ms/tok
- Stage 1 (gather) is trivial at ~0.03 ms/tok
- Throughput cycle time = max(Stage0, Stage1) = ~16.8 ms/tok (no improvement)
- The straggler RPC backend (romulus 3060) blocks the entire pipeline

Splitting Stage 0 into per-backend sub-stages requires deciding:
1. How many stages? (static n_stages vs adaptive)
2. How to assign backends to stages? (fixed mapping vs load-aware)
3. How to handle stragglers? (skip, timeout, speculative dispatch)

### Inputs (from D5.1)

| Analysis | Source | Finding |
|----------|--------|---------|
| Per-backend timing | D2.2 perf traces, D4.1 sched-trace.jsonl | RPC0 (3060 Ti): 10.0 ms warm, ROCm0 (7900 XTX): 5.7 ms warm |
| Straggler pattern | D0.2 topology map, D5.1 analysis | RPC0 on romulus dual-GPU (10.0 ms vs 5.7 ms); RPC1 on production (4.0 ms vs 2.1-3.5 ms) |
| Load variance | D5.1 split timing analysis | 3.8x variation cold-to-warm; 1.8x variation backend-to-backend |
| Sub-stage boundaries | D5.1 analysis | embed (11 us), RPC0 (10 ms), ROCm0 (5.7 ms) — natural partition |
| Gain analysis | D5.1 section 6 | 3-stage captures 1.57x improvement; 5-stage promises 3.2x |

### Options

#### Option A: Static n_stages

- Configure `n_stages` via env var (e.g., `GGML_SCHED_GPIPE_STAGES=5`)
- Fixed mapping: embed -> stage 0, RPC0 -> stage 1, RPC1 -> stage 2, etc.
- Simple to implement and reason about
- **Rejected:** Cannot adapt to different topologies or model types; user burden

#### Option B: Fully adaptive depth

- Runtime determines optimal n_stages based on backend timing
- Load-aware assignment: faster backends get fewer layers, straggler isolated
- **Rejected for v1:** Requires timing history and convergence logic; premature complexity; no benefit over topology-aware default for known hardware

#### Option C: Hybrid (topology-aware default, adaptive opt-in) — ACCEPTED

- Default: `n_stages = min(n_backends + 1, GGML_SCHED_GPIPE_DEPTH, 8)`
  - `n_backends + 1` = embed sub-stage + per-backend stages + gather (shared last stage)
  - For dual-GPU (2 backends): n_stages = 3 (embed + RPC + ROCm+gather)
  - For single GPU (1 backend): n_stages = 2 (compute + gather, matches current)
- Opt-in: `GGML_SCHED_GPIPE_ADAPTIVE=1` enables timing-based refinement
- `GGML_SCHED_GPIPE_DEPTH=N` allows user override for experimentation
- Fallback to 2-stage if adaptive fails to converge

### Decision

**Option C: Hybrid (topology-aware default, adaptive opt-in).**

Rationale:
1. **Topology-aware default captures the common case.** On romulus dual-GPU (2 backends),
   n_stages=3 automatically gives the 1.57x benefit. No user configuration needed.
2. **Single-GPU preserves current behavior.** n_stages=2 when n_backends=1.
3. **User override exists for tuning.** `GGML_SCHED_GPIPE_DEPTH` lets power users
   experiment without code changes.
4. **Adaptive path is opt-in.** No stability risk for the default path. The adaptive
   mechanism is a separate implementation ticket (D5.6) that can be refined later.
5. **Backward compatible.** `GGML_SCHED_GPIPE=1` with no other flags matches current
   2-stage behavior on single-GPU.

### Stage Assignment Model

#### Static (default) assignment

```
n_stages = min(n_backends + 1, user_depth_override, LLAMA_GPIPE_MAX_STAGES)

Stage 0: embed (CPU, always)
Stage 1: backend[0] compute  (first RPC or local backend)
Stage 2: backend[1] compute  (second backend)
...
Stage N-1: backend[N-1] compute + gather + KV write (last backend, always)
```

The last stage always merges compute + gather + KV write because gather
is trivially cheap (<0.1 ms) and separating it adds pipeline overhead
without benefit (confirmed by D5.1 section 6).

#### Adaptive (opt-in) assignment

When `GGML_SCHED_GPIPE_ADAPTIVE=1`:
1. On first decode, measure per-split timing from sched trace telemetry
2. After 5 warms-up decodes, compute running mean per backend
3. If max_backend_time / min_backend_time < 1.3x: collapse to 2-stage (homogeneous)
4. If ratio >= 1.3x: assign each backend its own stage
5. If a backend exceeds 3x the mean: mark as straggler, reduce its layer share
6. Re-evaluate every 100 decodes or on topology change

#### Fallback

If adaptive mode produces n_stages outside [2, LLAMA_GPIPE_MAX_STAGES],
or if timing data is unavailable (telemetry disabled), fall back to the
topology-aware default. Log a warning: "GPipe adaptive depth unavailable,
using static n_stages=%d".

### Interaction with copy-slot pipeline

The existing Path-B+ copy-slot pipeline (n_copies=4) and the GPipe stage
pipeline (n_stages) are independent concepts:
- Copy slots manage intra-token buffer reuse across backends
- GPipe stages manage inter-token dispatch ordering

They coexist: each GPipe stage internally uses copy-slot parallelism.
No change to `pipeline_barrier()` or `n_copies`.

### Consequences

#### Positive
- **Automatic performance gain on heterogeneous hardware.** Dual-GPU setups
  get 1.57x cycle-time reduction without configuration.
- **No regression on single GPU.** n_stages=2 when n_backends=1.
- **Simple mental model.** One stage per backend (plus embed). Easy to reason about.
- **Extensible.** Adaptive mode is an opt-in refinement that can iterate independently.

#### Negative
- **Requires GGML_SCHED_GPIPE_DEPTH env var plumbing.** New env var in ggml-backend.
- **Adaptive mode adds complexity.** Timing tracking, convergence logic, fallback paths.
  Scoped to D5.6 (separate implementation ticket).
- **Pipeline depth limited by n_backends.** Cannot go deeper than backend count + 1
  without artificial sub-backend splitting (not needed for current hardware).

#### Neutral
- **Copy-slot interaction is already handled.** The existing GPipe event system
  (ggml_sched_gpipe_*) is independent of n_copies.
- **No change to GGML_SCHED_GPIPE semantics.** The env var still enables GPipe;
  depth is a separate concern.

### Validation

- D5.4 prototype validates sub-stage dispatch on romulus dual-GPU
- D5.5 implements static topology-aware default
- D5.6 implements adaptive opt-in
- D5.7 benchmarks against D2.2 baseline

---

*ADR-0003 accepted 2026-07-11. Decision: Option C (hybrid). Static topology-aware default with adaptive opt-in.*