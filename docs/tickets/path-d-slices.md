# Path D Vertical Slices — Grab-able Issues

**Branch:** Path-D-Gpipeline-Assembly-Line
**Date:** 2026-07-11
**Source:** `docs/wayfinder/IMPLEMENTATION-PLAN.md`
**Detail reference:** `docs/tickets/path-d-tickets.md`

These are **tracer-bullet slices** — each cuts through every layer (code, test,
profiler, docs) and delivers a verifiable outcome. Slices are strictly serial;
grab them in order.

## Lifecycle

Each slice moves through three states:

| State | Who changes it | What happens |
|-------|---------------|--------------|
| `ready-for-agent` | (initial) | Slice is available; next agent can grab it |
| `in-progress` | Agent grabbing the slice | Agent changes status and starts work |
| `complete` | Agent finishing the slice | Agent checks all boxes, fills completion footer |

**On completion**, the agent also updates:
- `docs/wayfinder/TRACKING.md` — mark the corresponding phase(s) complete
- `docs/tickets/path-d-tickets.md` — check off each detail ticket

This file is the **canonical slice tracker**. The completion footer stays as a
permanent record of what was delivered. No manual cleanup needed — the agent
leaves the artifact clean.

---

## Cross-Cutting Infrastructure Fixes

These are fixes that span all slices — no ticket dependency chain, done in parallel.

| Ticket | Status | Date | Notes |
|--------|--------|------|-------|
| C1.1 | ✅ complete | 2026-07-11 | Fix `-INFINITY` IEEE-754 portability in all CUDA kernels (common.cuh, softmax.cu, topk-moe.cu, cross-entropy-loss.cu). Prevents NaN/corruption on Blackwell sm_120 + MSVC/nvcc 12.9 |

---

## Slice 1: Fix RPC Event Bug + Performance Gate

**Status:** in-progress
**Blocked by:** None — D1.7 and D3.1 are complete
**Detail tickets:** D2.2, D2.3, D3.2, D3.3

### What to build

Fix the pre-existing RPC event drain crash in `ggml/src/ggml-rpc/ggml-rpc.cpp`
that blocks all performance testing. Then validate GPipe end-to-end: correctness
regression, performance targets, profiler acceptance, and documentation.

The crash occurs in the `graph_recompute` path of `rpc_backend_graph_compute`
(~line 2100). The server's event record handler calls `wait_compute_idle()`,
which blocks the command-processing thread while the client's
`drain_pending_event_response` waits for a response that never arrives. The bug
affects both GPipe ON and OFF — it is not a GPipe regression.

Fix directions to investigate:
- Why the server's event handler is not sending the response after `wait_compute_idle()`
- Whether to restructure the `graph_recompute` path to avoid deferred event record
- Whether to add a proper async event acknowledgment mechanism
- Whether to make the compute worker signal completion via the event response

After the fix, validate through all three gates:
1. D2.2 performance: `global_3bk_pct >= 25%` on 5-GPU, `overlap_pct >= 5%`
2. D3.2 profiler: `b6-gate-phase0-assembly-bounds.py` with GPipe ON
3. D3.3 docs: update README.md and create `docs/path-d-complete-report.md`

### Acceptance criteria

- [ ] RPC event drain crash fixed — second token decode succeeds
- [ ] Fix verified with GPipe OFF (no regression on Path-B+ baseline)
- [ ] Fix verified with GPipe ON (D2.1 unit tests still pass)
- [ ] D2.2: `global_3bk_pct >= 25%` on 5-GPU production config
- [ ] D2.2: `overlap_pct >= 5%` on canonical n=384
- [ ] D2.2: G (A1) >= 180 t/s on romulus-local
- [ ] D2.2: G (A8) >= 15 t/s on 5-GPU production
- [ ] D3.2: `b6-gate-phase0-assembly-bounds.py` runs successfully with GPipe=1
- [ ] D3.2: `global_3bk_pct`, `overlap_pct`, G captured without profiler crashes
- [ ] D3.3: README.md mentions GPipe feature
- [ ] D3.3: `docs/` updated with GPipe documentation
- [ ] D3.3: `docs/path-d-complete-report.md` created
- [ ] Safety check (`scripts/safety-check.sh`) passes before each resource-intensive step

### Blocked by

None — can start immediately.

### Completion

<!-- Agent: fill this section on completion -->
- **Completed:** (date)
- **Commit range:** (first..last)
- **Notes:** (any deviations, trade-offs, or open follow-ups)

---

## Slice 2: Path C — Server-Side Scheduling

**Status:** ready-for-agent
**Blocked by:** Slice 1
**Detail tickets:** D4.1, D4.2, D4.3, D4.4, D4.5, D4.6

### What to build

Implement server-side scheduling for co-located GPUs on triton (`:50054` +
`:50055`). This is a stepping stone toward deeper pipelining — it de-risks the
scheduling model on real hardware before splitting into per-backend sub-stages.

The slice follows a full design-to-production cycle:
1. Establish a read-only C1 baseline on triton (per-device splits, RTT, GPU util)
2. Produce ADR-004 deciding the server-side scheduling model
3. Spec the API contracts for Path C functions
4. Prototype the `GRAPH_COMPUTE_ALL` concept (throwaway)
5. Implement C2 server-side scheduler
6. Test: measure server GPU duty cycle improvement vs C1

Target: 2x server GPU duty cycle with no G regression.

### Acceptance criteria

- [ ] D4.1: Per-device RPC splits documented for triton `:50054`+`:50055`
- [ ] D4.1: RTT counts and server GPU utilization captured
- [ ] D4.1: Baseline artifact at `docs/wayfinder/D4.1-triton-baseline-analysis.md`
- [ ] D4.2: ADR-004 created at `docs/adr/0004-server-side-scheduling.md`
- [ ] D4.2: Decision documented; alternatives considered and rejected with rationale
- [ ] D4.3: Path C spec section added to `docs/path-d-spec.md`
- [ ] D4.3: API contracts for Path C functions defined
- [ ] D4.4: Prototype demonstrates `GRAPH_COMPUTE_ALL` feasibility
- [ ] D4.4: Key risks identified (or ruled out); findings documented for D4.5
- [ ] D4.5: Server-side scheduler implemented for co-located GPUs
- [ ] D4.6: Server GPU duty cycle improved vs C1 baseline
- [ ] D4.6: G non-regression confirmed
- [ ] D4.6: Results documented in TRACKING.md
- [ ] Safety check passes before each resource-intensive step

### Blocked by

Slice 1 (RPC event fix unlocks all performance testing).

### Completion

<!-- Agent: fill this section on completion -->
- **Completed:** (date)
- **Commit range:** (first..last)
- **Notes:** (any deviations, trade-offs, or open follow-ups)

---

## Slice 3: Deeper Pipelining — n_stages > 2

**Status:** ready-for-agent
**Blocked by:** Slice 2
**Detail tickets:** D5.1, D5.2, D5.3, D5.4, D5.5, D5.6, D5.7

### What to build

Extend the 2-stage GPipe pipeline to `n_stages > 2` with per-backend sub-stages.
This is the main throughput lever — splitting Stage 0 (compute) into per-backend
sub-stages so that fast backends are not blocked by slow ones (straggler
isolation). Add adaptive depth so `n_stages` can be configured at runtime based
on backend timing.

The slice follows the same design-to-production cycle:
1. Analyze per-backend split timing from D4 traces to identify sub-stage boundaries
2. Produce ADR-003 deciding static vs adaptive pipeline depth
3. Spec per-backend sub-stage API contracts
4. Prototype sub-stage dispatch (throwaway)
5. Implement: split Stage 0 into embed + RPC0 + RPC1 + RPC2 + RPC3 sub-stages
6. Implement dynamic stage assignment (adaptive depth)
7. Test: verify `global_3bk_pct` improves measurably vs 2-stage baseline

### Acceptance criteria

- [ ] D5.1: Per-backend timing extracted from D4 traces
- [ ] D5.1: Sub-stage boundaries identified (embed, RPC0, RPC1, RPC2, RPC3)
- [ ] D5.1: Output at `docs/wayfinder/D5.1-split-timing-analysis.md`
- [ ] D5.2: ADR-003 created at `docs/adr/0003-adaptive-pipeline-depth.md`
- [ ] D5.2: Decision on how `n_stages` is determined (static vs adaptive)
- [ ] D5.2: Interaction with existing copy-slot pipeline defined
- [ ] D5.3: Spec section for `n_stages > 2` added to `docs/path-d-spec.md`
- [ ] D5.3: Per-backend sub-stage API contracts defined
- [ ] D5.4: Prototype demonstrates sub-stage dispatch feasibility
- [ ] D5.4: Straggler impact assessed; findings documented for D5.5
- [ ] D5.5: Stage 0 split into embed + per-backend RPC sub-stages
- [ ] D5.5: Per-sub-stage event signaling implemented
- [ ] D5.5: Straggler isolation: fast backends not blocked by slow
- [ ] D5.6: `n_stages` configurable at runtime (adaptive depth)
- [ ] D5.6: Stage assignment adapts to backend timing
- [ ] D5.6: Fallback to static assignment if adaptive fails
- [ ] D5.7: `global_3bk_pct` improves measurably vs D2.2 (2-stage) baseline
- [ ] D5.7: Results documented in TRACKING.md with comparison table
- [ ] Safety check passes before each resource-intensive step

### Blocked by

Slice 2 (needs server-side scheduling baseline and D4 tracing data).

### Completion

<!-- Agent: fill this section on completion -->
- **Completed:** (date)
- **Commit range:** (first..last)
- **Notes:** (any deviations, trade-offs, or open follow-ups)

---

## Slice 4: Mode B — Multi-Seq Microbatch

**Status:** ready-for-agent
**Blocked by:** Slice 3
**Detail tickets:** D6.1, D6.2, D6.3, D6.4, D6.5, D6.6, D6.7

### What to build

Extend the GPipe stage state machine to support multiple concurrent sequences
(microbatch). Different sequences can occupy different pipeline stages
simultaneously, improving overall pipeline utilization.

The slice follows the design-to-production cycle:
1. Analyze multi-slot requirements and KV cache interaction with pipeline stages
2. Produce ADR-005 for multi-seq GPipe scheduling
3. Spec multi-seq API contracts
4. Prototype multi-seq token tracking (throwaway)
5. Extend the stage state machine to track multiple tokens across sequences
6. Implement server-side multi-slot pipeline dispatch
7. Test: verify correctness with concurrent multi-seq decode

### Acceptance criteria

- [ ] D6.1: Multi-slot requirements documented
- [ ] D6.1: KV cache interaction with pipeline stages assessed
- [ ] D6.1: Output at `docs/wayfinder/D6.1-multi-seq-requirements.md`
- [ ] D6.2: ADR-005 created at `docs/adr/0005-multi-seq-gpipe.md`
- [ ] D6.2: Decision on how multiple sequences occupy pipeline stages
- [ ] D6.2: KV cache isolation model defined
- [ ] D6.3: Spec section for multi-seq added to `docs/path-d-spec.md`
- [ ] D6.3: API contracts for multi-seq functions defined
- [ ] D6.4: Prototype demonstrates multi-seq tracking feasibility
- [ ] D6.4: KV cache conflicts identified (or ruled out)
- [ ] D6.5: Stage state machine tracks multiple tokens across sequences
- [ ] D6.5: Per-sequence stage state isolated
- [ ] D6.5: Event signaling extended for multi-seq
- [ ] D6.6: Server dispatches different sequences to different pipeline stages
- [ ] D6.6: Multi-slot pipeline utilization improved
- [ ] D6.6: G scales with sequence count
- [ ] D6.7: Multiple sequences decode correctly in pipeline
- [ ] D6.7: No KV cache corruption
- [ ] D6.7: Results documented in TRACKING.md
- [ ] Safety check passes before each resource-intensive step

### Blocked by

Slice 3 (needs `n_stages > 2` state machine as foundation for multi-seq dispatch).

### Completion

<!-- Agent: fill this section on completion -->
- **Completed:** (date)
- **Commit range:** (first..last)
- **Notes:** (any deviations, trade-offs, or open follow-ups)

---

## Slice 5: Advanced Optimization + Deprecation Cleanup

**Status:** ready-for-agent
**Blocked by:** Slice 4
**Detail tickets:** R3.1, R3.2, R3.3, R3.4, R3.5

### What to build

Finalize the adaptive depth model based on empirical data from D5/D6, then clean
up the configuration surface by deprecating flags that GPipe renders obsolete.

The slice covers:
1. Analyze adaptive depth behavior from D5 findings; identify refinements
2. Produce ADR-006 to finalize the adaptive depth model (or update ADR-003)
3. Add deprecation warnings for superseded flags: B+11 (`GGML_RPC_DUAL_SOCKET`),
   B+14 (`GGML_SCHED_WAVEFRONT_DISPATCH`), B+7f (`GGML_RPC_HASH_DEFER`)
4. Refine adaptive depth based on real timing data, handling edge cases
5. Full regression test: all Path-B+ benchmarks pass with GPipe ON and OFF

### Acceptance criteria

- [ ] R3.1: Adaptive depth behavior analyzed from D5/D6 findings
- [ ] R3.1: Refinement opportunities identified
- [ ] R3.1: Output at `docs/wayfinder/R3.1-adaptive-depth-analysis.md`
- [ ] R3.2: ADR-006 created (or ADR-003 updated) with final model
- [ ] R3.2: Edge cases and fallback behavior defined
- [ ] R3.3: B+11 (`GGML_RPC_DUAL_SOCKET`) emits deprecation warning
- [ ] R3.3: B+14 (`GGML_SCHED_WAVEFRONT_DISPATCH`) emits deprecation warning
- [ ] R3.3: B+7f (`GGML_RPC_HASH_DEFER`) emits deprecation warning
- [ ] R3.3: Warnings are one-time only (not spammy)
- [ ] R3.4: Adaptive depth tuned based on real timing data from D5/D6
- [ ] R3.4: Edge cases handled (single backend, straggler dominance)
- [ ] R3.4: Performance stable across model types
- [ ] R3.5: All Path-B+ benchmarks pass with GPipe ON
- [ ] R3.5: All Path-B+ benchmarks pass with GPipe OFF
- [ ] R3.5: Final TRACKING.md updated with completion status
- [ ] Safety check passes before each resource-intensive step

### Blocked by

Slice 4 (needs multi-seq empirical data for adaptive depth refinement).

### Completion

<!-- Agent: fill this section on completion -->
- **Completed:** (date)
- **Commit range:** (first..last)
- **Notes:** (any deviations, trade-offs, or open follow-ups)
