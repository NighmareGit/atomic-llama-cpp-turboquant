# ADR-0005: Multi-Sequence GPipe Scheduling Model

## Status

**Pending** — To be decided during D6.2 (Design step of Mode B Microbatch phase).

## Context

Mode A GPipe is single-sequence: one token at a time flows through pipeline stages, with MTP draft coupling. Mode B extends this to multiple sequences occupying different pipeline stages concurrently — the "microbatch" model where different sequences are at different positions in the pipeline.

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

### Inputs (to be gathered during D6.1)

| Analysis | Source | What we need |
|----------|--------|--------------|
| Server multi-slot behavior | `llama-server` source | How multiple sequences are currently handled |
| KV cache per-sequence | `llama-kv-cache.cpp` | KV isolation model |
| Pipeline stage state | D5 implementation | How stage state machine tracks tokens |
| MTP coupling | ADR-0002 | How draft sequences interact with target |

### Options

#### Option A: Sequential multi-seq (no pipeline sharing)

- Multiple sequences served, but each gets its own pipeline turn
- No pipeline overlap between sequences
- Simplest; no KV isolation challenges
- Does not improve pipeline utilization

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

### Decision

**To be filled during D6.2 based on D6.1 research findings.**

### Consequences

**To be filled after decision.**

---

*ADR-0005 placeholder — decision pending D6.1 research*