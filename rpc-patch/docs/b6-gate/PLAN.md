# B+6 overlap gate plan (Path-B-Plus)

Navigation: [TRACKING.md](TRACKING.md) | [PERFORMANCE-RULES.md](PERFORMANCE-RULES.md) | [rpc-multi-backend-pipeline-plus/](../../../docs/rpc-multi-backend-pipeline-plus/) | [rpc-path-b-plus-overview.md](../rpc-path-b-plus-overview.md) | [pathb-sync-site-audit.md](../pathb-sync-site-audit.md)

**Branch:** Path-B-Event-Support-Pipeline-Plus  
**Goal (original):** Pass B+6 (`overlap_pct >= 5%`) on 2-GPU F without Path C.  
**Outcome (2026-07-01):** **STRUCTURAL CEILING** -- M1/M3 not reached; Phase 4 **CLOSED FAIL**. Overlap mission **reopens** under split-mission review (see below).  
**Tracking:** [TRACKING.md](TRACKING.md)

Profiler tooling (R1-R5) is **done**; `llama-pipeline-profiler` is measurement-only.

---

## Final measured state

| Topology | overlap_pct | G_tps | Primary blocker | gate_b6 |
|----------|-------------|-------|-----------------|---------|
| 2-GPU remus `b6-2gpu-f` n=384 | 0.2% | 75.6 | straggler + drain | FAIL |
| 2-GPU triton `b6-2gpu-f-triton` n=384 | 0.2-0.3% | 128-187 | pipelining depth | FAIL |
| 4-GPU JUPITER `b6-4gpu-g` n=384 | 0.1% | 77.2 | drain 50s | FAIL |
| 4-GPU triton `b6-4gpu-g-triton` n=384 | 0.1-0.2% | 63-80 | drain 3-6s | FAIL |
| 5-GPU `b6-5gpu-g` n=2048 multiturn | 0.0% | 60.1 | EVENT_RECORD 29s + HASH 11s | FAIL |

Best overlap peek: **0.9%** (`b6-2gpu-f-triton-guard-n128`); **0.7%** (4-GPU ts-grid G2 @ n=128). M1 (1%) not reached at n=384.

**Stop rule (triggered):** M1 not reached after two+ B+7 fixes on 2-GPU F; extended bisect ladder B+8--B+13 NULL on overlap; 5-GPU confirms drain-bound ceiling. **No Path C** on this branch.

---

## Split missions (forward)

| Mission | Goal | Status | Next |
|---------|------|--------|------|
| **A -- Production ship** | Stable multi-GPU inference, max G_tps | **ACTIVE** | Tip `a2d63acf1`; triton docker; relay + HASH_DEFER opt-in |
| **B -- Overlap gate (B+6)** | `overlap_pct >= 5%` Path B only | **CLOSED FAIL** | Reopen via grill: new hypotheses only |
| **C -- Ops / deploy** | Windows CUDA portable, triton :50054 | **MAINTENANCE** | Phase A artifacts (A1-A11 DONE) |

Production cluster (5-GPU Linux): label `b6-5gpu-g-prod` / `scripts/b6-gate-5gpu-production-env.sh`

| Setting | Value |
|---------|-------|
| `--rpc` | `192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054,192.168.8.23:50055` |
| `-ts` | `20,10,30,10,30` |
| `GGML_PIPELINE_PLUS` | `1` |
| `GGML_RPC_DUAL_SOCKET` | `1` (5-GPU only; OFF on 2/4-GPU) |
| `GGML_RPC_HASH_DEFER` | `0` default; opt-in `B6_5GPU_HASH_DEFER=1` |

---

## Reopen overlap (post-close)

Phase 4 is **exhausted** for Path-B-only levers. Reopening overlap does **not** resume Phase 0-3 profiler matrix or Phase A A/B. It requires a **new hypothesis** documented before code:

| Hypothesis class | Examples | Preconditions |
|------------------|----------|---------------|
| Scheduler model change | Parallel split dispatch, copy-slot semantics | Design doc + grill PASS |
| Transport | B+11 dual-socket at scale, RDMA | Bisect on 5-GPU n=128+ |
| Architecture (out of branch scope) | Path C unified rpc-server | Explicit branch decision |

Grill session owns prioritization and acceptance criteria for the reopened hunt.

---

## Milestones (final)

| ID | overlap_pct | stall_ratio | Status |
|----|-------------|-------------|--------|
| Baseline (2-GPU) | 0.6% | 0.68 | DONE |
| Spike ref (triton) | 0.2-0.3% | 0.87-0.92 | DONE |
| M1 | >= 1.0% | < 0.80 | **FAIL** |
| M2 | >= 2.5% | < 0.60 | **FAIL** |
| M3 (B+6 PASS) | >= 5.0% | < 0.50 | **FAIL** |

---

## Phases (final status)

### Phase A -- Windows CUDA build merge + triton spike prep

**DONE** (A1-A11). **Gate ROI: SKIP** -- worker swap improves G_tps and drain, not overlap.

### Phase 0 -- Profiler matrix

**DONE** -- `b6-2gpu-f`, `b6-2gpu-f-triton`, `b6-4gpu-g`, `b6-4gpu-g-triton`, `b6-2gpu-f-plus0`.

### Phase 1 -- Stall ledger + A/B verdict

**DONE** -- MIXED; drain-primary at 4-GPU and 5-GPU scale.

### Phase 2 -- B+7 RPC fixes + 5-GPU Phase B

**DONE** -- relay (`0d7b1edb1`), HASH_DEFER (`1b21c05b3`), EVENT-before-GET (`a2d63acf1`). See TRACKING Phase B table.

### Phase 3 -- `-ts` sweep

**DONE** -- best G2 0.7% @ n=128; M1 not reached @ n=384.

### Phase 4 -- Gate close

**CLOSED FAIL** (2026-07-01):

| Step | Status | Evidence |
|------|--------|----------|
| P4-1 M3 on 2-GPU F | **FAIL** | All gate rows < 1% @ n=384 |
| P4-2 Overview milestones | **DONE** | `rpc-path-b-plus-overview.md`, mission TRACKING |
| P4-3 GATES / PIPELINE SS9 | **DONE** | Structural ceiling note; no metric shift to PASS |

### Phase 5-6 -- Diagnosis + routing

**DONE** -- D3 drain-bound + D1 worker partial; extended by 5-GPU confirm.

### Phase B -- 5-GPU cluster (was deferred in original plan)

**DONE** 2026-07-01 -- `b6-5gpu-g` triton docker; overlap 0%; relay stable.

---

## Optional backlog (post-Phase 4)

Executed or adjudicated 2026-07-01. See TRACKING **Backlog** section for bench dirs.

| ID | Item | Verdict |
|----|------|---------|
| BL-1 | EVENT_RECORD / dual-socket (B+11) on 5-GPU | Run n128 bisect; see TRACKING |
| BL-2 | 3060 docker straggler (`backend1`) ts shift | Run n128 ts experiment; see TRACKING |
| BL-3 | B+16 CUDA `leaf_55` MoE | **REJECT** (mission TRACKING); split-slot wait hurts G |

---

## Path C stop line

No `rpc-server -d CUDA0,CUDA1` aggregation, no parallel split loop in `ggml_backend_sched_compute_splits`.

## Key paths

- Measurement: `tools/llama-pipeline-profiler/`, `scripts/b6-gate-run-remote.sh`
- Regression: `benches/path-b-plus/regression.jsonl`
- RPC hot path: `ggml/src/ggml-rpc/ggml-rpc.cpp`
- Gates: `tools/llama-pipeline-profiler/GATES.md`
- Mission root: [docs/rpc-multi-backend-pipeline-plus/](../../../docs/rpc-multi-backend-pipeline-plus/)