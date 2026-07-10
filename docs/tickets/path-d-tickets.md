# Path D Tickets — Work Breakdown

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-10  
**Source:** `docs/path-d-spec.md`

---

## Implementation Tickets

### D1.1 -- Add llama_gpipe_state Struct

**Type:** implementation  
**Blocks:** D1.2  
**Blocked by:** none

**Goal:** Add `llama_gpipe_state` struct to `llama_context` in `src/llama-context.h`.

**Acceptance Criteria:**
- [ ] Struct defined with n_stages, cur_stage, microbatch_size, enabled, stage_tokens
- [ ] Member added to `llama_context` struct
- [ ] Zero-cost initialization when GPipe OFF

**Implementation Notes:**
- File: `src/llama-context.h`
- Reference: D0.5 section 1.1

---

### D1.2 -- Implement llama_gpipe_enabled() Helper

**Type:** implementation  
**Blocks:** D1.3  
**Blocked by:** D1.1

**Goal:** Add `llama_gpipe_enabled()` static helper following the pattern of `llama_pipeline_plus_enabled()`.

**Acceptance Criteria:**
- [ ] Function checks `GGML_SCHED_GPIPE` env var
- [ ] Caches result in static variable
- [ ] Returns false when env var is 0 or unset

**Implementation Notes:**
- File: `src/llama-context.cpp`
- Reference: D0.5 section 1.1, lines 37-41 pattern

---

### D1.3 -- Add GGML_SCHED_GPIPE Env Var Handling

**Type:** implementation  
**Blocks:** D1.4  
**Blocked by:** D1.2

**Goal:** Add `GGML_SCHED_GPIPE` environment variable handling in decode entry point.

**Acceptance Criteria:**
- [ ] `llama_context_params` extended with `gpipe_enabled` bool
- [ ] Env var parsed in `llama_context` constructor
- [ ] Flag stored in `llama_cparams` or context state

**Implementation Notes:**
- File: `src/llama-cparams.h`, `src/llama-context.cpp`
- Reference: D0.5 section 6.1

---

### D1.4 -- Implement llama_decode_gpipe_impl() Skeleton

**Type:** implementation  
**Blocks:** D1.5  
**Blocked by:** D1.3

**Goal:** Create skeleton for `llama_decode_gpipe_impl()` with stage state machine structure.

**Acceptance Criteria:**
- [ ] Function signature matches spec (section 3.2)
- [ ] Stage 0/Stage 1 branching implemented
- [ ] Returns -1 when GPipe not enabled

**Implementation Notes:**
- File: `src/llama-context.cpp`
- Reference: D0.5 section 3.1

---

### D1.5 -- Implement ggml_sched_gpipe_init()

**Type:** implementation  
**Blocks:** D1.6  
**Blocked by:** D1.4

**Goal:** Initialize GPipe event state in scheduler.

**Acceptance Criteria:**
- [ ] Function allocates per-stage events
- [ ] Called from `llama_context::sched_reserve()` when GPipe enabled
- [ ] No-op when GPipe disabled

**Implementation Notes:**
- File: `ggml/src/ggml-backend.cpp`
- Reference: D0.5 section 4.1

---

### D1.6 -- Implement ggml_sched_gpipe_wait()

**Type:** implementation  
**Blocks:** D1.7  
**Blocked by:** D1.5

**Goal:** Implement per-stage event wait for producer-consumer signaling.

**Acceptance Criteria:**
- [ ] Waits on correct stage event
- [ ] Handles split_id < 0 (wait all)
- [ ] Integrates with existing `event_wait` pattern

**Implementation Notes:**
- File: `ggml/src/ggml-backend.cpp`
- Reference: D0.5 section 4.4

---

### D1.7 -- Implement Stage State Machine

**Type:** implementation  
**Blocks:** D2.1  
**Blocked by:** D1.6

**Goal:** Implement the full GPipe stage state machine with KV-ready release.

**Acceptance Criteria:**
- [ ] Stage 0: embed + RPC compute + event_record(compute_done)
- [ ] Stage 1: event_wait(compute_done) + gather + KV write + event_record(kv_ready)
- [ ] Stage advance logic correct
- [ ] MTP coupling support (mode A)

**Implementation Notes:**
- File: `src/llama-context.cpp`
- Reference: D0.5 section 3.2, ADR-0002 section 3

---

## Test Tickets

### D2.1 -- Correctness Tests

**Type:** test  
**Blocks:** Review  
**Blocked by:** D1.7

**Goal:** Verify GPipe produces correct outputs matching classic decode.

**Acceptance Criteria:**
- [ ] Logits hash smoke: T+1 logits match classic decode
- [ ] Multi-turn KV fill: 8149 tokens without corruption
- [ ] MoE expert routing: same outputs
- [ ] MTP coupling: draft + verify work together

**Test Requirements:**
- Test script: `scripts/test-path-d-correctness.sh`
- Compare GPipe ON vs OFF outputs

---

### D2.2 -- Performance Tests

**Type:** test  
**Blocks:** Review  
**Blocked by:** D2.1

**Goal:** Verify GPipe achieves overlap targets.

**Acceptance Criteria:**
- [ ] `global_3bk_pct >= 25%` on 5-GPU
- [ ] `overlap_pct >= 5%` on canonical n=384
- [ ] G (A1) >= 180 t/s on romulus-local
- [ ] G (A8) >= 15 t/s on 5-GPU prod

**Test Requirements:**
- Test script: `scripts/test-path-d-performance.sh`
- Profile with `GGML_SCHED_TRACE=1`, `GGML_RPC_TRACE=1`

---

### D2.3 -- Regression Tests

**Type:** test  
**Blocks:** Review  
**Blocked by:** D2.2

**Goal:** Verify no regressions on Path-B+ baseline.

**Acceptance Criteria:**
- [ ] Path-B+ benchmarks unchanged when GPipe OFF
- [ ] Single GPU: no performance change
- [ ] Dense models: 70B/72B load without OOM

**Test Requirements:**
- Test script: `scripts/test-path-d-regression.sh`
- Run against `b6-2gpu-f-romulus-local` baseline

---

## Production Hardening Tickets

### D3.1 -- Runbook Delta in PIPELINE.md

**Type:** task  
**Blocks:** D3.2  
**Blocked by:** D2.3

**Goal:** Update PIPELINE.md with GPipe-specific runbook instructions.

**Acceptance Criteria:**
- [ ] GPipe mode documented in PIPELINE.md
- [ ] Env var (`GGML_SCHED_GPIPE`) documented
- [ ] Startup/shutdown procedures updated

**Implementation Notes:**
- File: `PIPELINE.md`
- Reference: D0.5 section 6

---

### D3.2 -- Profiler Acceptance

**Type:** task  
**Blocks:** D3.3  
**Blocked by:** D3.1

**Goal:** Verify profiler acceptance criteria for GPipe mode.

**Acceptance Criteria:**
- [ ] `b6-gate-phase0-assembly-bounds.py` runs with GPipe ON
- [ ] Metrics captured: `global_3bk_pct`, `overlap_pct`, G
- [ ] No profiler crashes on GPipe mode

**Implementation Notes:**
- Script: `scripts/b6-gate-phase0-assembly-bounds.py`
- Reference: DESIGN-path-d-layer-pipeline.md section 5

---

### D3.3 -- README.md + Docs Update

**Type:** task  
**Blocks:** D4.1  
**Blocked by:** D3.2

**Goal:** Update README.md and docs/ with GPipe feature documentation.

**Acceptance Criteria:**
- [ ] README.md mentions GPipe feature
- [ ] `docs/` updated with GPipe documentation
- [ ] `docs/path-d-complete-report.md` created

**Implementation Notes:**
- Files: `README.md`, `docs/`
- Reference: DESIGN-path-d-layer-pipeline.md section 5

---

## Path C Stepping Stone Tickets

### D4.1 -- C1 Baseline: Triton Dual-GPU Analysis

**Type:** research  
**Blocks:** D4.2  
**Blocked by:** D3.3

**Goal:** Establish read-only Path C baseline on triton `:50054`+`:50055`.

**Acceptance Criteria:**
- [ ] Per-device RPC splits documented
- [ ] RTT count and server GPU util captured
- [ ] Baseline artifact in `docs/wayfinder/D4.1-triton-baseline-analysis.md`

**Implementation Notes:**
- Output: `docs/wayfinder/D4.1-triton-baseline-analysis.md`
- Reference: DESIGN-path-d-layer-pipeline.md section 5 (D1)

---

### D4.2 -- ADR-004: Server-Side Scheduling Model

**Type:** design  
**Blocks:** D4.3  
**Blocked by:** D4.1

**Goal:** Produce ADR for server-side scheduling on co-localized GPUs.

**Acceptance Criteria:**
- [ ] ADR-004 created in `docs/adr/`
- [ ] Decision: server-side sched model for Path C
- [ ] Alternatives considered and rejected with rationale

**Implementation Notes:**
- Output: `docs/adr/0004-server-side-scheduling.md`
- Skill: `/grill-with-docs`

---

### D4.3 -- Spec Section for Path C

**Type:** spec  
**Blocks:** D4.4  
**Blocked by:** D4.2

**Goal:** Add Path C spec section to `docs/path-d-spec.md`.

**Acceptance Criteria:**
- [ ] Spec section added for server-side scheduling
- [ ] API contracts for Path C functions defined
- [ ] Acceptance criteria for C2 defined

**Implementation Notes:**
- File: `docs/path-d-spec.md`
- Skill: `/to-spec`

---

### D4.4 -- Prototype: GRAPH_COMPUTE_ALL Concept

**Type:** prototype  
**Blocks:** D4.5  
**Blocked by:** D4.3

**Goal:** Throwaway code to validate server-side scheduling concept.

**Acceptance Criteria:**
- [ ] Prototype demonstrates `GRAPH_COMPUTE_ALL` feasibility
- [ ] Key risks identified (or ruled out)
- [ ] Findings documented for D4.5 implementation

**Implementation Notes:**
- Skill: `/prototype`
- Throwaway: keep findings, delete code

---

### D4.5 -- C2: Server-Side Sched Implementation

**Type:** implementation  
**Blocks:** D4.6  
**Blocked by:** D4.4

**Goal:** Implement server-side scheduler with `GRAPH_COMPUTE_ALL`.

**Acceptance Criteria:**
- [ ] Server-side sched implemented for co-localized GPUs
- [ ] Target: 2x server GPU duty cycle
- [ ] G non-regression vs C1 baseline

**Implementation Notes:**
- File: `ggml/src/ggml-rpc/ggml-rpc.cpp`
- Reference: rpc-patch/docs/rpc-path-c-plan.md

---

### D4.6 -- Test: Server GPU Duty Improvement

**Type:** test  
**Blocks:** D5.1  
**Blocked by:** D4.5

**Goal:** Verify server GPU duty improvement from C2.

**Acceptance Criteria:**
- [ ] Server GPU duty cycle improved vs C1 baseline
- [ ] G non-regression confirmed
- [ ] Results documented in TRACKING.md

**Implementation Notes:**
- Metrics: server GPU util, G (t/s), duty cycle

---

## Deeper Pipelining Tickets

### D5.1 -- Split Timing Analysis

**Type:** research  
**Blocks:** D5.2  
**Blocked by:** D4.6

**Goal:** Analyze per-backend split timing to identify sub-stage boundaries.

**Acceptance Criteria:**
- [ ] Per-backend timing extracted from C1/D2 traces
- [ ] Sub-stage boundaries identified (embed, RPC0, RPC1, RPC2, RPC3)
- [ ] Output: `docs/wayfinder/D5.1-split-timing-analysis.md`

**Implementation Notes:**
- Output: `docs/wayfinder/D5.1-split-timing-analysis.md`

---

### D5.2 -- ADR-003: Adaptive Pipeline Depth

**Type:** design  
**Blocks:** D5.3  
**Blocked by:** D5.1

**Goal:** Produce ADR for adaptive pipeline depth model.

**Acceptance Criteria:**
- [ ] ADR-003 created in `docs/adr/`
- [ ] Decision: how n_stages is determined (static vs adaptive)
- [ ] Interaction with existing copy-slot pipeline defined

**Implementation Notes:**
- Output: `docs/adr/0003-adaptive-pipeline-depth.md`
- Skill: `/grill-with-docs`

---

### D5.3 -- Spec Section for Deeper Pipelining

**Type:** spec  
**Blocks:** D5.4  
**Blocked by:** D5.2

**Goal:** Add deeper pipelining spec section.

**Acceptance Criteria:**
- [ ] Spec section for n_stages > 2
- [ ] Per-backend sub-stage API contracts
- [ ] Adaptive depth acceptance criteria

**Implementation Notes:**
- File: `docs/path-d-spec.md`

---

### D5.4 -- Prototype: Per-Backend Sub-Stages

**Type:** prototype  
**Blocks:** D5.5  
**Blocked by:** D5.3

**Goal:** Throwaway code to test per-backend sub-stage dispatch.

**Acceptance Criteria:**
- [ ] Prototype demonstrates sub-stage dispatch feasibility
- [ ] Straggler impact assessed
- [ ] Findings documented for D5.5

**Implementation Notes:**
- Skill: `/prototype`

---

### D5.5 -- Implement Per-Backend Sub-Stages

**Type:** implementation  
**Blocks:** D5.6  
**Blocked by:** D5.4

**Goal:** Split Stage 0 into per-backend sub-stages.

**Acceptance Criteria:**
- [ ] Stage 0 split into: embed, RPC0, RPC1, RPC2, RPC3
- [ ] Per-sub-stage event signaling
- [ ] Straggler isolation (fast backends not blocked by slow)

**Implementation Notes:**
- File: `src/llama-context.cpp`, `ggml/src/ggml-backend.cpp`
- Reference: D0.5 section 7

---

### D5.6 -- Dynamic Stage Assignment

**Type:** implementation  
**Blocks:** D5.7  
**Blocked by:** D5.5

**Goal:** Implement adaptive depth — dynamic stage assignment based on backend load.

**Acceptance Criteria:**
- [ ] n_stages configurable at runtime
- [ ] Stage assignment adapts to backend timing
- [ ] Fallback to static assignment if adaptive fails

**Implementation Notes:**
- File: `ggml/src/ggml-backend.cpp`
- Reference: ADR-003

---

### D5.7 -- Test: Deeper Pipeline Performance

**Type:** test  
**Blocks:** D6.1  
**Blocked by:** D5.6

**Goal:** Verify `global_3bk_pct >= 25%` with deeper pipeline.

**Acceptance Criteria:**
- [ ] `global_3bk_pct` improves measurably vs 2-stage
- [ ] Results documented in TRACKING.md
- [ ] Comparison with D2.2 baseline

**Implementation Notes:**
- Metrics: `global_3bk_pct`, `overlap_pct`, G

---

## Mode B Microbatch Tickets

### D6.1 -- Multi-Seq Requirements Analysis

**Type:** research  
**Blocks:** D6.2  
**Blocked by:** D5.7

**Goal:** Analyze server multi-slot requirements and KV cache interaction.

**Acceptance Criteria:**
- [ ] Multi-slot requirements documented
- [ ] KV cache interaction with pipeline stages assessed
- [ ] Output: `docs/wayfinder/D6.1-multi-seq-requirements.md`

**Implementation Notes:**
- Output: `docs/wayfinder/D6.1-multi-seq-requirements.md`

---

### D6.2 -- ADR-005: Multi-Seq GPipe Scheduling

**Type:** design  
**Blocks:** D6.3  
**Blocked by:** D6.1

**Goal:** Produce ADR for multi-seq GPipe scheduling model.

**Acceptance Criteria:**
- [ ] ADR-005 created in `docs/adr/`
- [ ] Decision: how multiple sequences occupy pipeline stages
- [ ] KV cache isolation model defined

**Implementation Notes:**
- Output: `docs/adr/0005-multi-seq-gpipe.md`

---

### D6.3 -- Spec Section for Multi-Seq

**Type:** spec  
**Blocks:** D6.4  
**Blocked by:** D6.2

**Goal:** Add multi-seq spec section.

**Acceptance Criteria:**
- [ ] Spec section for multi-seq GPipe
- [ ] API contracts for multi-seq functions
- [ ] Acceptance criteria defined

---

### D6.4 -- Prototype: Multi-Seq Token Tracking

**Type:** prototype  
**Blocks:** D6.5  
**Blocked by:** D6.3

**Goal:** Throwaway code to test multi-seq token tracking in pipeline.

**Acceptance Criteria:**
- [ ] Prototype demonstrates multi-seq tracking feasibility
- [ ] KV cache conflicts identified (or ruled out)
- [ ] Findings documented

**Implementation Notes:**
- Skill: `/prototype`

---

### D6.5 -- Extend State Machine for Multi-Seq

**Type:** implementation  
**Blocks:** D6.6  
**Blocked by:** D6.4

**Goal:** Extend stage state machine for multi-seq token tracking.

**Acceptance Criteria:**
- [ ] Stage state machine tracks multiple tokens across sequences
- [ ] Per-sequence stage state isolated
- [ ] Event signaling extended for multi-seq

**Implementation Notes:**
- File: `src/llama-context.cpp`

---

### D6.6 -- Server Multi-Slot Dispatch

**Type:** implementation  
**Blocks:** D6.7  
**Blocked by:** D6.5

**Goal:** Implement server-side multi-slot pipeline dispatch.

**Acceptance Criteria:**
- [ ] Server can dispatch different sequences to different stages
- [ ] Multi-slot pipeline utilization improved
- [ ] G scales with sequence count

**Implementation Notes:**
- File: `ggml/src/ggml-rpc/ggml-rpc.cpp`

---

### D6.7 -- Test: Concurrent Multi-Seq Decode

**Type:** test  
**Blocks:** R3.1  
**Blocked by:** D6.6

**Goal:** Verify correctness with concurrent multi-seq decode.

**Acceptance Criteria:**
- [ ] Multiple sequences decode correctly in pipeline
- [ ] No KV cache corruption
- [ ] Results documented in TRACKING.md

---

## Advanced Optimization Tickets

### R3.1 -- Adaptive Depth Analysis

**Type:** research  
**Blocks:** R3.2  
**Blocked by:** D6.7

**Goal:** Analyze D5 adaptive depth behavior; identify refinement opportunities.

**Acceptance Criteria:**
- [ ] Adaptive depth behavior analyzed from D5 findings
- [ ] Refinement opportunities identified
- [ ] Output: `docs/wayfinder/R3.1-adaptive-depth-analysis.md`

---

### R3.2 -- ADR-006: Finalize Adaptive Depth

**Type:** design  
**Blocks:** R3.3  
**Blocked by:** R3.1

**Goal:** Finalize adaptive depth model (or update ADR-003).

**Acceptance Criteria:**
- [ ] ADR-006 created (or ADR-003 updated)
- [ ] Final adaptive depth model documented
- [ ] Edge cases and fallback behavior defined

---

### R3.3 -- Deprecation Warnings

**Type:** implementation  
**Blocks:** R3.4  
**Blocked by:** R3.2

**Goal:** Add deprecation warnings for superseded flags.

**Acceptance Criteria:**
- [ ] B+11 (`GGML_RPC_DUAL_SOCKET`) emits deprecation warning
- [ ] B+14 (`GGML_SCHED_WAVEFRONT_DISPATCH`) emits deprecation warning
- [ ] B+7f (`GGML_RPC_HASH_DEFER`) emits deprecation warning
- [ ] Warnings are one-time (not spammy)

**Implementation Notes:**
- Reference: CONFIGURATION.md, ADR-013

---

### R3.4 -- Adaptive Depth Refinement

**Type:** implementation  
**Blocks:** R3.5  
**Blocked by:** R3.3

**Goal:** Refine adaptive depth based on D5 findings.

**Acceptance Criteria:**
- [ ] Adaptive depth tuned based on real timing data
- [ ] Edge cases handled (single backend, straggler dominance)
- [ ] Performance stable across model types

---

### R3.5 -- Test: No Regression

**Type:** test  
**Blocks:** —  
**Blocked by:** R3.4

**Goal:** Verify no regression with all optimizations enabled.

**Acceptance Criteria:**
- [ ] All Path-B+ benchmarks pass with GPipe ON
- [ ] All Path-B+ benchmarks pass with GPipe OFF
- [ ] Final TRACKING.md update with completion status

---

## Ticket Dependency Graph

```
D1.1 -> D1.2 -> D1.3 -> D1.4 -> D1.5 -> D1.6 -> D1.7 -> D2.1 -> D2.2 -> D2.3
                                                                 \
                                                                  D3.1 -> D3.2 -> D3.3
                                                                                   \
                                                                                    D4.1 -> D4.2 -> D4.3 -> D4.4 -> D4.5 -> D4.6
                                                                                                                      \
                                                                                                                       D5.1 -> D5.2 -> D5.3 -> D5.4 -> D5.5 -> D5.6 -> D5.7
                                                                                                                                                          \
                                                                                                                                                           D6.1 -> D6.2 -> D6.3 -> D6.4 -> D6.5 -> D6.6 -> D6.7
                                                                                                                                                                              \
                                                                                                                                                                               R3.1 -> R3.2 -> R3.3 -> R3.4 -> R3.5
```

---

## Notes

- All tickets are agent-ready: full context in spec + ADR documents
- Blocking edges ensure sequential implementation
- Test tickets verify each implementation ticket
- Milestone commits placed after each phase completion
- Safety check (`bash scripts/safety-check.sh`) before every resource-intensive operation
- RDMA explicitly deferred (not in this plan)

---

*Work breakdown extended — D2-R3 phases added*