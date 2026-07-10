# Path D Workflow — Layer Pipeline Assembly Line

## Overview

This document defines the complete workflow for implementing the Path-D-Gpipeline-Assembly-Line feature branch, extending from the Path-B+ foundation.

---

## Workflow Stages

### Stage 1: Investigation Map (Wayfinder)

Create a local investigation map that branches into research tickets. Each ticket produces **decisions, not deliverables** — clarifying unknowns before implementation.

**Entry point:** `/wayfinder`

**Output location:** `docs/wayfinder/D0.X-*.md` files

---

### Stage 2: Design Grilling (Grill-with-docs)

For each investigation ticket requiring design decisions, sharpen the questions through interview. Maintain domain language in `CONTEXT.md` and ADRs.

**Entry point:** `/grill-with-docs`

**Output location:** `docs/adr/0002-*.md` (architecture decisions), updates to `docs/agents/domain.md`

---

### Stage 3: Specification (To-spec)

Turn the investigation outcomes into a formal specification.

**Entry point:** `/to-spec`

**Output location:** `docs/path-d-spec.md`

---

### Stage 4: Work Breakdown (To-tickets)

Split the spec into agent-ready tickets with clear blocking edges.

**Entry point:** `/to-tickets`

**Output location:** `docs/tickets/path-d-tickets.md` (local ticket file)

Each ticket includes:
- Clear acceptance criteria
- Test requirements
- Blocking dependencies

---

### Stage 5: Implementation Loop

For each ticket in sequence:

```
/implement
→ Drives: /tdd internally (red-green-refactor per ticket)
→ On completion: /code-review (two-axis: Standards + Spec)
→ Result: Working code + passing tests
```

**Review loop:**
- If review finds issues: hand back to `/implement` with fix instructions
- If tests fail: hand to `/research` for test plan update, then back to `/implement`
- Loop continues until all checks pass

---

### Stage 6: End-to-End Testing

After all implementation tickets complete, validate the integrated work.

**Entry point:** `/research` (for test planning) or dedicated test agent

**Output location:** `docs/path-d-test-plan.md`, `scripts/path-d-e2e-test.sh`

**Test array includes:**
- Edge cases from the specification
- Performance targets (G, overlap, stall metrics)
- Integration with existing Path-B+ components

---

### Stage 7: Documentation Completion

Update all project documentation to reflect the completed work.

**Entry point:** `/wayfinder` or manual update

**Output location:** 
- `README.md` updates
- `docs/path-d-complete-report.md`
- `--help` feature documentation in relevant tools
- Updates to all lateral documentation files

---

## Document Structure

```
docs/
├── wayfinder/
│   ├── PHASE-D-WORKFLOW.md       # This file
│   ├── D0.2-split-topology-map.md
│   ├── D0.3-gpipe-kv-ordering.md
│   ├── D0.4-path-c-baseline.md
│   └── D0.5-deploy-tag.md
├── adr/
│   └── 0002-*.md               # Architecture decisions
├── agents/
│   └── domain.md               # Domain language (existing)
├── tickets/
│   └── path-d-tickets.md       # Work breakdown
├── path-d-spec.md              # Master specification
├── path-d-test-plan.md         # Test requirements
└── path-d-complete-report.md   # Final documentation
```

---

## Phase D0 Investigation Tickets

### D0.2 — 5-GPU Split Topology Map
- **Type:** research
- **Goal:** Map RPC compute splits per token across 5-GPU topology
- **Deliverable:** `docs/wayfinder/D0.2-split-topology-map.md`
- **Questions:**
  - What is the current split assignment per layer for A1/A8/A13 models?
  - What is the per-split compute time distribution (from bench traces)?
  - What are the RPC hop timing characteristics?
  - What blocking dependencies exist between splits?

### D0.3 — GPipe KV-ordering Grill
- **Type:** grilling
- **Goal:** Determine KV write ordering for layer pipeline
- **Deliverable:** ADR in `docs/adr/`
- **Questions:**
  - Can Layer N+1 compute overlap with Layer N's KV write?
  - What ordering constraints exist for MoE expert routing?
  - Does MTP coupling require layer-local h_prev persistence?
  - What is the KV-ordering model for GPipe-style pipeline?

### D0.4 — Path C C1 Triton Baseline
- **Type:** prototype
- **Goal:** Establish read-only Path C baseline on triton
- **Deliverable:** `docs/wayfinder/D0.4-path-c-baseline.md`
- **Tasks:**
  - Clone current state to isolated branch
  - Run `llama-server` on triton `:50054` + `:50055`
  - Capture baseline metrics (G, overlap, stall)

### D0.5 — Deploy Tag Creation
- **Type:** task
- **Goal:** Tag Path-B+ deploy before D branch
- **Deliverable:** Git tag + changelog entry
- **Tasks:**
  - Verify all B+ work is committed
  - Create git tag `path-b-plus-deploy-v1`
  - Update `CHANGELOG.md`

---

## Success Metrics

| Metric | Target | Source |
|--------|--------|--------|
| Investigation completeness | All D0.X tickets resolved | TRACKING.md |
| Specification coverage | 100% of goals captured | path-d-spec.md |
| Ticket readiness | All tickets agent-implementable | path-d-tickets.md |
| Implementation pass rate | All tickets pass /code-review | GitHub PRs or local commits |
| Test pass rate | All tests in test array pass | scripts/path-d-e2e-test.sh |
| Documentation completeness | All docs updated | path-d-complete-report.md |

---

## Lateral Documentation References

- `docs/rpc-multi-backend-pipeline-plus/MISSION.md` — Original mission
- `docs/rpc-multi-backend-pipeline-plus/PLAN.md` — Phase structure
- `docs/rpc-multi-backend-pipeline-plus/TRACKING.md` — Current state
- `docs/rpc-multi-backend-pipeline-plus/DESIGN-path-d-layer-pipeline.md` — Design document
- `rpc-patch/patch/HANDOVER-*.md` — Session handovers
- `MTP.md`, `NEXTN.md`, `PIPELINE.md` — Ancestry documentation

---

## Universal Phase Workflow Loop

Every phase beyond D2 follows the same iterative loop:

```
safety-check → research → design → spec → prototype → implement → test → review
      ↑                                                              │
      └─────────────────── (loop back on failure) ───────────────────┘

MAX 3 LOOPS — if not converged after 3 iterations:
→ Mark as FAILED SPIKE
→ Document findings + blockers
→ Commit + push
→ Continue to next phase (no human intervention)

On success:
→ /wayfinder to update all docs
→ Milestone commit (self-describing) + push to GitHub
→ Update TRACKING.md
→ Handoff to next phase
```

## Stage 8: Beyond Mode A — Path C Stepping Stone (D4)

**Entry point:** `docs/wayfinder/D4-PATH-C-AGENT-PLAN.md`  
**Goal:** Prove server-side scheduling on triton dual-GPU  
**Workflow loop:** research → design → spec → prototype → implement → test → review

## Stage 9: Deeper Pipelining (D5)

**Entry point:** `docs/wayfinder/D5-DEEPER-PIPELINE-AGENT-PLAN.md`  
**Goal:** n_stages > 2, per-backend sub-stages, adaptive depth  
**Workflow loop:** research → design → spec → prototype → implement → test → review

## Stage 10: Mode B Microbatch (D6)

**Entry point:** `docs/wayfinder/D6-MODE-B-AGENT-PLAN.md`  
**Goal:** Multi-seq pipeline sharing  
**Workflow loop:** research → design → spec → prototype → implement → test → review

## Stage 11: Advanced Optimization (R3)

**Entry point:** `docs/wayfinder/R3-ADVANCED-OPT-AGENT-PLAN.md`  
**Goal:** Adaptive depth refinement + deprecation cleanup  
**Workflow loop:** research → design → implement → test → review

## Operational Safety (All Stages)

| Rule | How |
|------|-----|
| Safety check before every resource-intensive operation | `bash scripts/safety-check.sh` |
| Never run multiple llama-server instances | One at a time; verify with `nvidia-smi` |
| Never run llama-cli | Can OOM machine and kill session |
| Use existing scripts for VRAM pre-calculation | `pathb-rpc-vram-preflight.sh` |
| Models in `/mnt/models` or `~/models` | Do not download elsewhere |
| Disk space on `/` | `df -h /` before docker builds; abort if < 20 GB |
| Clean up docker after test builds | `docker system prune -f` |
| Do not kill romulus | This machine hosts the coding session |
| Commit + push after every ticket | Small, frequent commits for backup |
| Milestone commit after each phase | Self-describing commit + push to GitHub |

---

*Prepared for `/wayfinder` invocation — 2026-07-10*