# Master Orchestration Plan — Path D Full Workflow

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-13  
**Status:** Slice 1 + Slice 2 + Slice 3 + Slice 4 complete — RPC event fix, dual-GPU validation, Path C stepping stone + profiler v1 + deeper pipelining + multi-seq Mode B delivered. Known limitation D6.10 tracked for GPU event pipelining fix. Slice 6 planned — pipeline depth (n_copies > 1) + split overhead mitigation (2026-07-16).

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
| Deeper Pipelining (D5) | ✅ complete | D5.1-D5.7 — n_stages>2 implemented. Completed 2026-07-11. |
| Mode B Microbatch (D6) | ✅ complete | D6.1-D6.9 — multi-seq, per-seq events, GRAPH_COMPUTE_STAGE. Completed 2026-07-13. |
| GPU Event Pipelining Fix (D6.10) | 📋 planned | gpipe_events on GPU backend — ticketed, known limitation KL-D6.1 |
| Pipeline Depth + Split Overhead (D7) | 📋 planned | n_copies > 1 (1.4-1.8x TG) + 4 deferred strategies. Research complete: `docs/research/split-overhead-mitigation.md`. Slice 6 at `docs/tickets/path-d-slices.md`. |
| Advanced Optimization (R3) | ⏳ pending | R3.1-R3.5 tickets |

**Hardware:** Romulus local — AMD 7900 XTX (client, ROCm) + NVIDIA 3060 Ti (RPC server, CUDA). Models at `/mnt/models`. GPU telemetry via `rocm-smi` + `nvidia-smi`. Cluster (triton 5-GPU, remus) deferred to later sessions.

### Known Infrastructure Limitations

| Limitation | Impact | Workaround | Ticket |
|------------|--------|------------|--------|
| **FUSE/NTFS mmap hard-link corruption** (KL-INFRA-1) — Model files on `/mnt/models` (fuseblk NTFS SMB share) with hard-link count > 1 return silently corrupted data via `MAP_SHARED`. Byte-identical files at different paths produce different inference results. Diagnosed 2026-07-14. | Any model loaded from SMB share without `--mlock` may produce garbage output. | `--mlock` flag (forces full read + mlock into RAM), or copy model to a fresh inode (local ext4/xfs). `--no-mmap` alone does NOT fix it. | Dx.1 |

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
| D7 Pipeline Depth + Split Overhead | D7.1-D7.5 | `D7.0-pipeline-depth-research.md` |
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

## Slice 4 Completion (2026-07-13)

Multi-seq Mode B (D6.1-D6.9) delivered on romulus dual-GPU. Scope expanded beyond original D6.1-D6.7:
- **D6.1-D6.7**: Multi-seq requirements, ADR-005 (Option B: interleaved multi-seq with stage-available scheduling), spec, prototype, state machine extension, backend tests (6/6), integration tests (4/4), 10/10 GPipe suites pass.
- **D6.8**: Per-sequence GPipe events — migrated from double-buffered to per-seq event arrays.
- **D6.9**: GRAPH_COMPUTE_STAGE RPC command — server-side per-stage split filtering with telemetry.
- **D6.10**: GPU event pipelining fix — known limitation, ticketed for future.

## Slice 3 Completion (2026-07-11)

Deeper pipelining (D5.1-D5.7) delivered on romulus dual-GPU:
- **n_stages > 2**: Stage 0 split into per-backend sub-stages. Topology-aware default: n_stages = n_backends + 1.
- **Adaptive depth**: `GGML_SCHED_GPIPE_ADAPTIVE=1` with straggler detection. 20/20 assertions pass, 0 regressions.
- **D5.7**: On 2-GPU, n_stages=3 shows no throughput gain — existing copy-slot pipeline captures all overlap. Benefit expected on 3+ GPU.

## Slice 2 Completion (2026-07-11)

Path C core (D4.1-D4.6) + Profiler v1 (D4.7-D4.10) delivered on romulus dual-GPU:
- **D4.1-D4.6**: GRAPH_COMPUTE_ALL scheduler, server-side telemetry on both single and multi-device paths, draft-mtp n_max=2 yielding 113.2 t/s gen (+82% vs baseline)
- **D4.7-D4.10**: `llama-gpipe-profiler` native C++ binary (17,928 bytes), 6-field telemetry incl. KV cache timing, heatmap synthesis, 3-model benchmark suite validated
- **Hot paths analysis**: `docs/hot-paths-analysis.md` — per-layer tensor/GPU deployment map with bottleneck characterization
- **Pareto Optimizer (D4.11-D4.14)**: Planned + ticketed, NOT built

---

## Next Action

**Ready for Slice 6 — Pipeline Depth: n_copies > 1 + Split Overhead Mitigation.**

Grab Slice 6 from `docs/tickets/path-d-slices.md`:

```
/handoff "Slice 6: Pipeline Depth — n_copies > 1 + split overhead mitigation"
→ Read docs/tickets/path-d-slices.md Slice 6 section,
  docs/research/split-overhead-mitigation.md,
  docs/wayfinder/D7.0-pipeline-depth-research.md
→ Execute D7.1 (increase n_copies 1→2):
  1. /prototype — A/B test n_copies=1 vs 2 via llama-cli --parallel
  2. /code-review — review prototype diff
  3. /improve-codebase-architecture — assess default config
  4. Test run — validate TG > 130 t/s, no OOM
  5. /implement — bake into production config
→ D7.2-D7.5: document deferred strategies with activation conditions
→ On completion update path-d-slices.md completion footer and TRACKING.md D7 status
```

**After Slice 6: Slice 5 — R3 Advanced Optimization + Deprecation Cleanup.**

```
/handoff "Slice 5: Advanced Optimization — adaptive depth refinement + deprecation cleanup" → Read docs/tickets/path-d-slices.md Slice 5 section, docs/tickets/path-d-tickets.md R3.1-R3.5, docs/wayfinder/TRACKING.md, docs/wayfinder/D5.1-split-timing-analysis.md, docs/adr/0003-adaptive-pipeline-depth.md → Execute R3.1-R3.5 (analyze adaptive depth → ADR-006 → deprecation warnings for B+11/B+14/B+7f → refine adaptive depth → full regression test) → On completion update path-d-slices.md completion footer and TRACKING.md R3 status
```

---

*Orchestration plan complete — ready for implementation agent invocation*