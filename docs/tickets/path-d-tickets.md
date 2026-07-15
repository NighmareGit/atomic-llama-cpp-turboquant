# Path D Tickets — Work Breakdown

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-10  
**Source:** `docs/path-d-spec.md`

---

## Implementation Tickets

### C1.1 -- Fix -INFINITY IEEE-754 Portability

**Type:** bugfix  
**Blocks:** none  
**Blocked by:** none

**Goal:** Replace all CUDA kernel `-INFINITY` literals with IEEE-754 bit-cast helpers to prevent silent NaN/corruption on Blackwell (sm_120) and MSVC/nvcc 12.9 builds.

**Acceptance Criteria:**
- [x] `neg_inf_f32()` device helper and `neg_inf_f32_host()` host helper added in common.cuh
- [x] `block_reduce_policy<MAX>::sentinel()` fixed: `-INFINITY` -> `neg_inf_f32()`
- [x] softmax.cu: 7 `-INFINITY` -> `neg_inf_f32()`
- [x] topk-moe.cu: 9 `-INFINITY` -> `neg_inf_f32()`, 1 -> `neg_inf_f32_host()`
- [x] cross-entropy-loss.cu: 2 `-INFINITY` -> `neg_inf_f32()`
- [x] HIP build: clean compile, zero errors
- [x] Smoke test: 35B model 105.79 t/s, 5x concurrent stress — 0 failures, no NaN

**Implementation Notes:**
- Files: `ggml/src/ggml-cuda/common.cuh`, `softmax.cu`, `topk-moe.cu`, `cross-entropy-loss.cu`
- Reference: `docs/wayfinder/TRACKING.md` section C1

---

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

### D4.1 -- C1 Baseline: Romulus Dual-GPU Analysis

**Type:** research  
**Blocks:** D4.2  
**Blocked by:** D3.3

**Goal:** Establish read-only Path C baseline on romulus dual-GPU (7900 XTX client + 3060 Ti RPC server).

**Acceptance Criteria:**
- [x] Per-device RPC splits documented for 7900 XTX + 3060 Ti
- [x] RTT counts and GPU utilization captured (rocm-smi for AMD, nvidia-smi for NVIDIA)
- [x] Models loaded from `/mnt/models`
- [x] Baseline artifact in `docs/wayfinder/D4.1-romulus-baseline-analysis.md`

**Implementation Notes:**
- Hardware: romulus — AMD 7900 XTX (client, ROCm), NVIDIA 3060 Ti (RPC server, CUDA)
- Cluster (triton 5-GPU) deferred to later session
- Output: `docs/wayfinder/D4.1-romulus-baseline-analysis.md`
- Reference: DESIGN-path-d-layer-pipeline.md section 5 (D1)

---

### D4.2 -- ADR-004: Server-Side Scheduling Model

**Type:** design  
**Blocks:** D4.3  
**Blocked by:** D4.1

**Goal:** Produce ADR for server-side scheduling on co-localized GPUs.

**Acceptance Criteria:**
- [x] ADR-004 created in `docs/adr/`
- [x] Decision: server-side sched model for Path C
- [x] Alternatives considered and rejected with rationale

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
- [x] Spec section added for server-side scheduling
- [x] API contracts for Path C functions defined
- [x] Acceptance criteria for C2 defined

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
- [x] Prototype demonstrates `GRAPH_COMPUTE_ALL` feasibility
- [x] Key risks identified (or ruled out)
- [x] Findings documented for D4.5 implementation

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
- [x] Server-side sched implemented for co-localized GPUs
- [x] Target: 2x server GPU duty cycle
- [x] G non-regression vs C1 baseline

**Implementation Notes:**
- File: `ggml/src/ggml-rpc/ggml-rpc.cpp`
- Reference: rpc-patch/docs/rpc-path-c-plan.md

---

### D4.6 -- Test: Server GPU Duty Improvement

**Type:** test  
**Blocks:** D5.1  
**Blocked by:** D4.5  
**Status:** complete

**Goal:** Verify server GPU duty improvement from C2.

**Acceptance Criteria:**
- [x] Server GPU duty cycle improved vs C1 baseline (Path-B-Plus: tg32=81.7 t/s vs D4.6 baseline 62.0 t/s = +32%)
- [x] G non-regression confirmed (no regression vs D4.5 baseline)
- [x] Results documented in TRACKING.md
- [x] Results documented in D4.6-romulus-rpc-multidevice-test.md

**Additional finding:** draft-mtp self-speculation with n_max=2 adds +41.5% gen throughput (113.2 t/s) over no-spec, for a combined +82% over the D4.6 baseline.

**Data:** romulus dual-GPU (7900 XTX + 3060 Ti RPC), Qwen3.5-9B-MTP-Q4_K_M.gguf

---

## Deeper Pipelining Tickets

### D5.1 -- Split Timing Analysis

**Type:** research  
**Blocks:** D5.2  
**Blocked by:** D4.6

**Goal:** Analyze per-backend split timing to identify sub-stage boundaries.

**Acceptance Criteria:**
- [x] Per-backend timing extracted from C1/D2 traces
- [x] Sub-stage boundaries identified (embed, RPC0, RPC1, RPC2, RPC3)
- [x] Output: `docs/wayfinder/D5.1-split-timing-analysis.md`

**Implementation Notes:**
- Output: `docs/wayfinder/D5.1-split-timing-analysis.md`

---

### D5.2 -- ADR-003: Adaptive Pipeline Depth

**Type:** design  
**Blocks:** D5.3  
**Blocked by:** D5.1

**Goal:** Produce ADR for adaptive pipeline depth model.

**Acceptance Criteria:**
- [x] ADR-003 created in `docs/adr/`
- [x] Decision: how n_stages is determined (static vs adaptive)
- [x] Interaction with existing copy-slot pipeline defined

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
- [x] Spec section for n_stages > 2
- [x] Per-backend sub-stage API contracts
- [x] Adaptive depth acceptance criteria

**Implementation Notes:**
- File: `docs/path-d-spec.md`

---

### D5.4 -- Prototype: Per-Backend Sub-Stages

**Type:** prototype  
**Blocks:** D5.5  
**Blocked by:** D5.3

**Goal:** Throwaway code to test per-backend sub-stage dispatch.

**Acceptance Criteria:**
- [x] Prototype demonstrates sub-stage dispatch feasibility
- [x] Straggler impact assessed
- [x] Findings documented for D5.5

**Implementation Notes:**
- Skill: `/prototype`

---

### D5.5 -- Implement Per-Backend Sub-Stages

**Type:** implementation  
**Blocks:** D5.6  
**Blocked by:** D5.4

**Goal:** Split Stage 0 into per-backend sub-stages.

**Acceptance Criteria:**
- [x] Stage 0 split into: embed, RPC0, RPC1, RPC2, RPC3
- [x] Per-sub-stage event signaling
- [x] Straggler isolation (fast backends not blocked by slow)

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
- [x] n_stages configurable at runtime
- [x] Stage assignment adapts to backend timing
- [x] Fallback to static assignment if adaptive fails

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
- [x] Results documented in TRACKING.md
- [x] Comparison with D2.2 baseline

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

### D6.10 -- Fix GPipe Multi-Seq Inter-Stage Event Synchronization

**Type:** bugfix
**Blocks:** D6.7 (re-verify)
**Blocked by:** none

**Goal:** Replace the current `ggml_backend_sched_synchronize()` fallback with proper per-backend event-based gating for multi-seq stage dispatch. Currently all GPU backends are fully drained between stages, which serializes work that could otherwise overlap.

**Background:**
The multi-seq dispatch (`llama_decode_gpipe_multi_impl`) calls `ggml_backend_sched_graph_compute_async` multiple times within a single token decode, once per pipeline stage. Between stages, it must ensure stage N is fully complete before stage N+1 overwrites shared tensor buffers. The original design used `ggml_sched_gpipe_wait_seq()` for this, but it is a silent no-op because:
- The gather backend is always CPU (asserted in `ggml_backend_sched_new`)
- The CPU device interface has `event_new`/`event_record`/`event_wait`/`event_synchronize` all NULL
- `ggml_backend_event_new(cpu_device)` returns NULL
- All gpipe_events are NULL, so `ggml_sched_gpipe_wait_seq` returns immediately without any synchronization

The temporary fix (2026-07-13) calls `ggml_backend_sched_synchronize(sched)` between stages, which drains all backends unconditionally. This is correct but degrades GPU backends (ROCm/CUDA/Vulkan/RPC) from async event-wait to full device sync.

**Proposed Fix:**
Record gpipe events on a GPU backend (not CPU gather) so the events are valid. Use the last non-CPU backend as the event host. If no GPU backends exist, fall back to full sync. This restores event-based pipelining for GPU configurations.

**Implementation sketch (`ggml/src/ggml-backend.cpp`):**
```cpp
void ggml_sched_gpipe_init_multi(ggml_backend_sched_t sched, int n_stages, int n_seqs) {
    // Find a GPU backend to host events; fall back to CPU if none
    int event_host_bid = -1;
    for (int b = sched->n_backends - 1; b >= 0; b--) {
        if (sched->backends[b]->device->iface.event_new != NULL) {
            event_host_bid = b;
            break;
        }
    }
    sched->gpipe_event_host_bid = event_host_bid;  // new field
    ggml_backend_dev_t event_dev = event_host_bid >= 0
        ? sched->backends[event_host_bid]->device
        : sched->backends[sched->n_backends - 1]->device;
    // ... create events on event_dev instead of gather_backend->device
}
```

Then `ggml_sched_gpipe_wait_seq` and `ggml_sched_gpipe_record_seq` use `sched->backends[event_host_bid]` instead of `sched->backends[gather_bid]`. When `event_host_bid < 0` (no GPU backends), fall back to `ggml_backend_sched_synchronize`.

**Affected files:**
- `ggml/src/ggml-backend.cpp`: `ggml_sched_gpipe_init_multi`, `ggml_sched_gpipe_wait_seq`, `ggml_sched_gpipe_record_seq`
- `src/llama-context.cpp`: revert `ggml_backend_sched_synchronize` back to `ggml_sched_gpipe_wait_seq`

**Acceptance Criteria:**
- [ ] gpipe_events created on GPU backend (not CPU) when GPU backends are present
- [ ] `ggml_sched_gpipe_wait_seq` properly gates stages with GPU events
- [ ] CPU-only configs fall back to full sync (no regression)
- [ ] Multi-seq dispatch no longer drains all backends between stages
- [ ] `ggml_sched_gpipe_record_seq` no longer a no-op on GPU configs
- [ ] Romulus dual-GPU (RPC + ROCm + CPU): no crash, no full-device-sync between stages
- [ ] Backend event tests still pass (tests use CPU backends, fallback path)

**Implementation Notes:**
- Reference: `docs/path-d-spec.md` section 10.3 (multi-seq event protocol)
- The `ggml_backend_sched` struct needs a new `int gpipe_event_host_bid` field
- D6.8 (per-sequence events) and D6.9 (GRAPH_COMPUTE_STAGE) were implemented without tickets

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

## Profiler Architecture (D4.7–D4.14)

### Scope Boundary

| Scope | Tickets | Build Now? |
|-------|---------|------------|
| Profiler data collection + native binary v1 | D4.7-D4.10 | **Yes** — this sprint |
| Pareto Optimizer in `llama-server` | D4.11-D4.14 | **No** — plan + ticket only |

Path C introduces `GRAPH_COMPUTE_ALL` with a synchronous response path. These tickets
build a native C++ profiler (`llama-gpipe-profiler`, like `llama-bench`) that produces
execution heatmaps. The heatmap feeds a future Pareto optimizer (D4.11-D4.14, stored
but not built) that lives inside `llama-server`'s governor/dispatcher.

Design decisions captured in `docs/wayfinder/IMPLEMENTATION-PLAN.md` Profiler
Architecture section: standalone binary, KV cache profiling included, task-aware
(pp/tg), Python profiler stays alive during transition.

---

### D4.7 — Profiler Research: Binary Design + Heatmap Schema

**Type:** research
**Blocks:** D4.8
**Blocked by:** none (parallel to D4.1)

**Goal:** Survey existing infrastructure and design the profiler binary — CLI surface,
heatmap schema, and KV cache profiling scope.

**Acceptance Criteria:**
- [x] `llama-bench` source structure analyzed as pattern template
- [x] Client-side trace formats mapped: `sched-trace.jsonl`, `rpc-trace.jsonl`
- [x] Existing profiler pipeline mapped: `llama-pipeline-profiler`, `diagnose.json`, gate scripts
- [x] Binary CLI designed: `--model`, `--endpoints`, `--tasks pp,tg`, `--output`, `--repeat`, `--warmup`
- [x] Heatmap JSON schema drafted: per-layer timing, GPU util, KV cache timing, task stratification
- [x] KV cache scope decided: instrument `llama_kv_cache` directly or collect via RPC telemetry
- [x] Task-awareness: how to drive pp vs tg workloads through the profiler
- [x] Output: `docs/wayfinder/D4.7-profiler-research.md`

---

### D4.8 — Profiler Prototype: Server Collection + Thin Client

**Type:** prototype
**Blocks:** D4.9
**Blocked by:** D4.7

**Goal:** Throwaway prototype validating low-overhead server-side telemetry collection
with KV cache fields, plus a thin C client to validate the end-to-end pipeline.

**Acceptance Criteria:**
- [x] Server-side overhead measured: collection + formatting cost vs baseline
- [x] `rpc_msg_server_telemetry` wire format prototyped with all 6 fields (including KV)
- [x] KV cache read/write timing capture validated on server side
- [x] Thin C client connects to RPC endpoints, exercises `GRAPH_COMPUTE_ALL`, parses telemetry, writes raw JSON
- [x] End-to-end pipeline validated before full profiler binary build
- [x] Key risks identified (or ruled out)
- [x] Output: `docs/wayfinder/D4.8-profiler-prototype-findings.md`

**Implementation Notes:**
- Skill: `/prototype`
- Throwaway: keep findings, delete code

---

### D4.9 — Profiler ADR: Binary + Heatmap + Transition

**Type:** design
**Blocks:** D4.10
**Blocked by:** D4.8

**Goal:** Decide the profiler architecture — standalone binary design, heatmap format,
telemetry protocol, and Python-to-native transition plan.

**Acceptance Criteria:**
- [x] Binary design decided: standalone `llama-gpipe-profiler`, patterned after `llama-bench`
- [x] CLI surface decided: `--model`, `--endpoints`, `--tasks`, `--output`, `--repeat`, `--warmup`
- [x] CMake target location decided: `tools/llama-gpipe-profiler/`
- [x] Telemetry frame format and versioning strategy decided
- [x] Opt-in mechanism decided: `GGML_RPC_SERVER_TELEMETRY=0|1`
- [x] Heatmap JSON schema finalized with task stratification, KV cache fields
- [x] Client ingestion path decided: `server-telemetry.jsonl`
- [x] Transition plan: Python profiler stays alive; phased deprecation milestones defined
- [x] Future hook: forward-compatible schema for Pareto optimizer (D4.11-D4.14)
- [x] Output: `docs/adr/0004b-profiler-architecture.md` or section in `docs/adr/0004-server-side-scheduling.md`

---

### D4.10 — Profiler Implementation v1: Binary + Server Telemetry + Scripts

**Type:** implementation
**Blocks:** D5.1 (feeds split timing analysis), D4.11 (future Pareto)
**Blocked by:** D4.4 (prototype), D4.5 (C2 implementation), D4.9 (ADR)

**Goal:** Implement v1 of the profiler — server telemetry paths, native profiler binary,
existing script adaptation. Keep Python profiler working.

**Acceptance Criteria:**
- [x] Server: per-backend timing collected after `ggml_backend_sched_graph_compute()`
- [x] Server: KV cache read/write timing collected per slot
- [x] Server: `rpc_msg_server_telemetry` with 6 fields, appended to `GRAPH_COMPUTE_ALL` response **and** `GRAPH_COMPUTE` (single-device) response
- [x] Server: gated by `GGML_RPC_SERVER_TELEMETRY` env var
- [x] Client: telemetry frame parsed, written to `server-telemetry.jsonl`
- [x] Profiler binary: `llama-gpipe-profiler` CMake target in `tools/llama-gpipe-profiler/`
- [x] Profiler binary: CLI working (`--model`, `--endpoints`, `--tasks pp,tg`, `--output`, `--repeat`, `--warmup`)
- [x] Profiler binary: task orchestration drives real inference through RPC endpoints
- [x] Profiler binary: synthesizes task-stratified heatmap JSON
- [x] Scripts: `b6-gate-phase0-assembly-bounds.py` consumes server telemetry fields when present
- [x] Scripts: `diagnose.json` schema extended with all 6 telemetry fields (optional)
- [x] Scripts: existing `llama-pipeline-profiler` continues working; new fields are optional extensions
- [x] Files: `ggml/src/ggml-rpc/ggml-rpc.cpp` (server + client), `tools/llama-gpipe-profiler/`, `tools/llama-pipeline-profiler/`

---

### Telemetry Field Specification

| Field | Type | Source | Consumer |
|-------|------|--------|----------|
| `device_timings_us[]` | `uint64[]` | Scheduler after `graph_compute` per backend | D5.1 straggler ID, R3 depth tuning, Pareto optimizer |
| `layer_assignments[]` | `int32[]` | Split output (layer start per device) | D5.1 sub-stage boundary mapping, Pareto placement |
| `copy_times_us[]` | `uint64[]` | PCIe copy duration per peer pair | D5.1 copy vs compute attribution |
| `device_meta[]` | `struct {name, vram, backend, pcie}` | Backend init at startup | Trace context, hardware regression, Pareto env analysis |
| `kv_read_times_us[]` | `uint64[]` | KV cache read per slot | Pareto optimizer: hot KV page placement |
| `kv_write_times_us[]` | `uint64[]` | KV cache write per slot | Pareto optimizer: KV eviction cost modeling |

---

### Future: Server-Side Pareto Optimizer (D4.11–D4.14)

> **Planned + ticketed, NOT built in current sprint.**

The Pareto optimizer lives inside `llama-server`'s governor/dispatcher. It consumes
the heatmap JSON from `llama-gpipe-profiler` and applies the 80/20 rule: hot 20% of
layers placed on fast 20% of GPUs. Cold layers become a "holding tank" in VRAM or
system RAM. The optimizer runs automatically — the server governor analyzes its
environment on startup and decides placement without external orchestration.

---

### D4.11 — Pareto Optimizer Research: Governor Integration Points

**Type:** research
**Blocks:** D4.12
**Blocked by:** none (future sprint)

**Goal:** Design the Pareto optimizer integration into `llama-server` governor/dispatcher.

**Acceptance Criteria:**
- [ ] Governor integration points identified (startup, re-config, periodic re-profile)
- [ ] Placement algorithm designed: hot-layer identification, GPU ranking, assignment
- [ ] 80/20 rule formalized: what counts as "hot" and "fast"
- [ ] KV cache interaction model: hot KV pages placement alongside hot layers
- [ ] Adaptive re-profiling strategy: when to re-measure and re-place
- [ ] Output: `docs/wayfinder/D4.11-pareto-governor-research.md`

---

### D4.12 — Pareto Optimizer Prototype

**Type:** prototype
**Blocks:** D4.13
**Blocked by:** D4.11

**Goal:** Throwaway prototype validating server-side placement decisions from heatmap input.

**Acceptance Criteria:**
- [ ] Server reads heatmap JSON and computes Pareto placement
- [ ] 80/20 rule validated on real cluster with real heatmap data
- [ ] Hot-on-fast placement shows measurable improvement over static assignment
- [ ] Edge cases handled: single GPU, homogenous GPUs, missing heatmap
- [ ] Output: `docs/wayfinder/D4.12-pareto-prototype-findings.md`

---

### D4.13 — Pareto Optimizer ADR

**Type:** design
**Blocks:** D4.14
**Blocked by:** D4.12

**Goal:** Decide the Pareto optimizer placement model and transition from static config.

**Acceptance Criteria:**
- [ ] Placement model: hot/cold tier definitions, GPU ranking algorithm
- [ ] Configuration model: how the optimizer output integrates with existing split configs
- [ ] Fallback behavior: what happens when heatmap is unavailable or stale
- [ ] Transition plan: from manual `--rpc-split` to automatic Pareto placement
- [ ] Output: `docs/adr/0005-pareto-optimizer.md`

---

### D4.14 — Pareto Optimizer Implementation

**Type:** implementation
**Blocks:** none (terminal ticket)
**Blocked by:** D4.10 (profiler v1), D4.13 (ADR)

**Goal:** Build the Pareto optimizer into `llama-server` governor/dispatcher.

**Acceptance Criteria:**
- [ ] Server governor analyzes environment on startup (GPU count, PCIe topology, model arch)
- [ ] Server consumes heatmap JSON from `llama-gpipe-profiler`
- [ ] Hot 20% layers identified; fast 20% GPUs ranked
- [ ] Hot-on-fast placement computed and applied to layer split
- [ ] Cold layers placed in VRAM holding tank or system RAM
- [ ] KV cache hot pages placed alongside hot layers for locality
- [ ] Adaptive re-profiling: periodic or event-driven re-analysis
- [ ] Fallback to static `--rpc-split` when no heatmap available
- [ ] G non-regression vs static assignment; measurable improvement on heterogeneous clusters
- [ ] Files: `tools/server/`, `ggml/src/ggml-rpc/` (governor integration)

---

### Dx.1 — Document and mitigate FUSE/mmaps hard-link corruption (KL-INFRA-1)

**Type:** documentation / infrastructure hardening
**Blocks:** none (terminal ticket)
**Blocked by:** none (incident-driven)

**Goal:** Permanently document the FUSE/NTFS mmap hard-link corruption discovered 2026-07-14, and add a startup-time diagnostic to detect risky model files before inference begins.

**Background:** Model files on `/mnt/models` (fuseblk NTFS SMB share) with `st_nlink > 1` return silently corrupted data via `mmap(MAP_SHARED)`. The file bytes are correct (md5sum matches), but the FUSE page-fault handler delivers wrong pages for shared inodes. This produces garbage logits with no error indication — byte-identical files at different paths give different inference results. See `src/llama-mmap.cpp` comment block for details.

**Acceptance Criteria:**
- [ ] `--mlock` is documented as the primary workaround in `docs/wayfinder/MASTER-ORCHESTRATION-PLAN.md` (done — this ticket)
- [ ] Startup diagnostic: `llama_model_loader` checks `st_nlink` of model file on first load and emits a `LLAMA_LOG_WARN` if > 1 on a FUSE filesystem (statfs `f_type == 0x65735546` FUSE_SUPER_MAGIC)
- [ ] Warning message includes: detected hard-link count, filesystem type, and recommendation to use `--mlock` or copy to local storage
- [ ] The diagnostic is a warning only — does not block loading (false positives possible on non-broken FUSE implementations)
- [ ] Files: `src/llama-mmap.cpp`, `src/llama-model-loader.cpp`, `docs/wayfinder/MASTER-ORCHESTRATION-PLAN.md`

---

*Work breakdown extended - D2-R3 phases added*