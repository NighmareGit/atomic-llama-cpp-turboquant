# Master Orchestration Plan — Path D Full Workflow

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-10  
**Status:** Slice 1 + Slice 2 complete — RPC event fix, dual-GPU validation, Path C stepping stone + profiler v1 delivered. Ready for Slice 3 (D5 Deeper Pipelining).

---

## Workflow Overview

```
Investigation Phase (Wayfinder)
    ↓
D0.2: Research Agent → Split Topology Map
    ↓
D0.3: Grill Agent → KV Ordering ADR
    ↓
D0.4: Prototype Agent → Romulus Local Baseline
    ↓
D0.5: Task Agent → Implementation Seam Design
    ↓
Specification Phase (/to-spec)
    ↓
Work Breakdown Phase (/to-tickets)
    ↓
Implementation Loop (/implement + /code-review)
    ↓
Testing Phase (/research for test plan)
    ↓
Documentation Phase (/wayfinder or manual)
```

---

## Phase Completion Status

| Phase | Status | Output |
|-------|--------|--------|
| Investigation | ✅ complete | D0.2-D0.5 documents |
| Specification | ✅ complete | `docs/path-d-spec.md` |
| Work Breakdown | ✅ complete | `docs/tickets/path-d-tickets.md` |
| Implementation (D1) | ✅ complete | D1.1-D1.7 tickets |
| Testing (D2) | ✅ complete | D2.1-D2.3 — RPC event bug fixed, dual-GPU validated |
| Production Hardening (D3) | ✅ complete | D3.1-D3.3 — TRACKING.md updated |
| Path C Stepping Stone (D4) | ✅ complete | D4.1-D4.6 core + D4.7-D4.10 profiler v1 (romulus local). Completed 2026-07-11. |
| Pareto Optimizer (D4 ext.) | 📋 planned | D4.11-D4.14 — ticketed, NOT built this sprint |
| Deeper Pipelining (D5) | ⏳ pending | D5.1-D5.7 tickets |
| Mode B Microbatch (D6) | ⏳ pending | D6.1-D6.7 tickets |
| Advanced Optimization (R3) | ⏳ pending | R3.1-R3.5 tickets |

**Hardware:** Romulus local — AMD 7900 XTX (client, ROCm) + NVIDIA 3060 Ti (RPC server, CUDA). Models at `/mnt/models`. GPU telemetry via `rocm-smi` + `nvidia-smi`. Cluster (triton 5-GPU, remus) deferred to later sessions.

---

## Agent Deployment Order

### Investigation Phase (completed)

| Order | Ticket | Agent | Blocks | Blocked By |
|-------|--------|-------|--------|------------|
| 1 | D0.2 | `/research` | D0.3, D0.5 | none |
| 2 | D0.3 | `/grill-with-docs` | D0.5 | D0.2 |
| 3 | D0.4 | `/prototype` | none | none |
| 4 | D0.5 | `/implement` or `/grill-with-docs` | Specification | D0.2, D0.3 |

### Implementation Phase (complete)

| Order | Ticket | Skill | Test File |
|-------|--------|-------|-----------|
| 1 | D1.1 | `/implement` + `/tdd` | `tests/test-gpipe-state.cpp` |
| 2 | D1.2 | `/implement` + `/tdd` | `tests/test-gpipe-enabled.cpp` |
| 3 | D1.3 | `/implement` + `/tdd` | `tests/test-gpipe-env.cpp` |
| 4 | D1.4 | `/implement` + `/tdd` | `tests/test-gpipe-decode-skel.cpp` |
| 5 | D1.5 | `/implement` + `/tdd` | `tests/test-gpipe-init.cpp` |
| 6 | D1.6 | `/implement` + `/tdd` | `tests/test-gpipe-wait.cpp` |
| 7 | D1.7 | `/implement` + `/tdd` | `tests/test-gpipe-stage.cpp` |

**Parallel execution possible:** D1.1-D1.7 are sequential due to blocking edges.

### Testing Phase (complete)

| Order | Ticket | Skill |
|--------|--------|-------|
| 8 | D2.1 | `/research` + `/tdd` |
| 9 | D2.2 | `/research` + `/tdd` |
| 10 | D2.3 | `/research` |

### Production Hardening Phase

| Order | Ticket | Skill |
|--------|--------|-------|
| 11 | D3.1 | `/implement` |
| 12 | D3.2 | `/implement` |
| 13 | D3.3 | `/wayfinder` |

### Beyond Phases (D4-R3)

D1-D3 and D4 (core + profiler v1) are complete. Next: D5.

Each beyond-phase follows the workflow loop: research → design → spec → prototype → implement → test → review (max 3 loops).

| Phase | Tickets | Agent Plan |
|-------|---------|------------|
| D4 Path C + Profiler v1 | D4.1-D4.10 | `D4-PATH-C-AGENT-PLAN.md` (romulus local) |
| D4 Pareto Optimizer | D4.11-D4.14 | Stored in `docs/tickets/path-d-tickets.md` — future sprint |
| D5 Deeper Pipelining | D5.1-D5.7 | `D5-DEEPER-PIPELINE-AGENT-PLAN.md` |
| D6 Mode B Microbatch | D6.1-D6.7 | `D6-MODE-B-AGENT-PLAN.md` |
| R3 Advanced Optimization | R3.1-R3.5 | `R3-ADVANCED-OPT-AGENT-PLAN.md` |

---

## Agent Plans

All plans are in `docs/wayfinder/`:

| File | Purpose |
|------|---------|
| `PHASE-D-WORKFLOW.md` | Overall workflow definition |
| `D0.1-wayfinder-map.md` | Investigation map |
| `D0.2-RESEARCH-AGENT-PLAN.md` | Split topology research plan |
| `D0.3-GRILL-AGENT-PLAN.md` | KV ordering grilling plan |
| `D0.4-PROTOTYPE-AGENT-PLAN.md` | Romulus local prototype plan |
| `D0.5-TASK-AGENT-PLAN.md` | Implementation seam plan |
| `D1-TO-TICKETS-AGENT-PLAN.md` | Work breakdown plan |
| `D2-IMPLEMENTATION-AGENT-PLAN.md` | Implementation loop plan |
| `MASTER-ORCHESTRATION-PLAN.md` | This file |

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/path-d-spec.md` | Master specification |
| `docs/tickets/path-d-tickets.md` | Work breakdown with blocking edges |
| `docs/adr/0002-gpipe-kv-ordering.md` | Architecture decisions |
| `docs/wayfinder/D0.2-split-topology-map.md` | Split analysis |

---

## Implementation Handoff Pattern

For each D1.X ticket:

```
/handoff "Implement D1.X: <ticket title>"
→ New session

/implement
→ Drives /tdd:
  1. Create failing test
  2. Implement to pass
  3. Run test
  4. Call /code-review
→ If review passes: update TRACKING.md, proceed
→ If review fails: hand back to /implement
```

---

## Testing Phase Trigger

After D1.7 completes:

```
/research "Create test plan for Path D"
→ Input: Implementation
→ Output: docs/path-d-test-plan.md, scripts/test-path-d-e2e.sh
```

---

## Documentation Phase Trigger

After all tests pass:

```
/wayfinder "Update Path D documentation"
→ Update: README.md, docs/, --help features
→ Output: docs/path-d-complete-report.md
```

---

## Review Loop Summary

```
Implementer → Code Review → Tester → (if fail) → Review → Implementer
                    ↑                      ↓
                    ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

Each cycle produces:
- Code changes (if fix needed)
- Updated test assertions (if spec unclear)
- Updated design docs (if approach changed)

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Investigation | All tickets complete | 100% |
| Specification | Spec coverage | All goals captured |
| Work Breakdown | Agent-ready tickets | All with acceptance criteria |
| Implementation | Review pass rate | 100% |
| Testing | Test pass rate | 100% |
| Documentation | Docs updated | All lateral docs refreshed |

---

## Slice 2 Completion (2026-07-11)

Path C core (D4.1-D4.6) + Profiler v1 (D4.7-D4.10) delivered on romulus dual-GPU:
- **D4.1-D4.6**: GRAPH_COMPUTE_ALL scheduler, server-side telemetry on both single and multi-device paths, draft-mtp n_max=2 yielding 113.2 t/s gen (+82% vs baseline)
- **D4.7-D4.10**: `llama-gpipe-profiler` native C++ binary (17,928 bytes), 6-field telemetry incl. KV cache timing, heatmap synthesis, 3-model benchmark suite validated
- **Hot paths analysis**: `docs/hot-paths-analysis.md` — per-layer tensor/GPU deployment map with bottleneck characterization
- **Pareto Optimizer (D4.11-D4.14)**: Planned + ticketed, NOT built

---

## Next Action

**Ready for Slice 3 — D5 Deeper Pipelining.**

Grab Slice 3 from `docs/tickets/path-d-slices.md` and execute via `D5-DEEPER-PIPELINE-AGENT-PLAN.md`:

```
/handoff "Slice 3: Deeper Pipelining — n_stages > 2"
-> Read docs/tickets/path-d-slices.md Slice 3
-> Read docs/wayfinder/D5-DEEPER-PIPELINE-AGENT-PLAN.md
-> Execute D5.1-D5.7 tickets
```

---

*Orchestration plan complete — ready for implementation agent invocation*