# ADR-0003: Adaptive Pipeline Depth Model

## Status

**Pending** — To be decided during D5.2 (Design step of Deeper Pipelining phase).

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

### Inputs (to be gathered during D5.1)

| Analysis | Source | What we need |
|----------|--------|--------------|
| Per-backend timing | D2.2 performance traces, D4.1 baseline | Compute time per split per backend |
| Straggler pattern | D0.2 topology map, D2 traces | Which backend is consistently slowest |
| Load variance | D5.1 split timing analysis | How much backend timing varies by model/topology |

### Options

#### Option A: Static n_stages

- Configure `n_stages` via env var (e.g., `GGML_SCHED_GPIPE_STAGES=5`)
- Fixed mapping: embed -> stage 0, RPC0 -> stage 1, RPC1 -> stage 2, etc.
- Simple to implement and reason about
- Cannot adapt to different topologies or model types

#### Option B: Adaptive depth

- Runtime determines optimal n_stages based on backend timing
- Load-aware assignment: faster backends get fewer layers, straggler isolated
- More complex; requires timing history and convergence logic
- Can adapt to different topologies and model types

#### Option C: Hybrid (static default, adaptive opt-in)

- Default: static n_stages=2 (current behavior)
- Opt-in: `GGML_SCHED_GPIPE_ADAPTIVE=1` enables runtime adaptation
- Fallback to static if adaptive fails to converge
- Best of both worlds; more code to maintain

### Decision

**To be filled during D5.2 based on D5.1 research findings.**

### Consequences

**To be filled after decision.**

---

*ADR-0003 placeholder — decision pending D5.1 research*