# DESIGN: B+14 parallel assembly-line dispatch (overlap reopen)

| Field | Value |
|-------|-------|
| **Date** | 2026-07-01 |
| **Status** | Hypothesis — grill in progress |
| **Branch** | Path-B-Event-Support-Pipeline-Plus |
| **Prereq** | Tier A baseline frozen (14/16); B+8–B+13 ladder NULL on overlap |

Navigation: [PLAN.md](PLAN.md) | [b6-gate/PLAN.md](../../rpc-patch/docs/b6-gate/PLAN.md) | [DESIGN-b13-input-wait-audit.md](DESIGN-b13-input-wait-audit.md)

---

## 1. Problem statement (evidence-backed)

Two independent ceilings block M3 (`overlap_pct >= 5%`):

### Ceiling A — intra-token serial split dispatch

`ggml_backend_sched_compute_splits()` runs `for (split_id = 0; split_id < n_splits; split_id++)` **serially**. Trace audit on 3-GPU A1 reference (n=384):

| Metric | Value | Implication |
|--------|-------|-------------|
| `serial_dispatch_pct` | **91.4%** | Wall time with 0–1 backends active |
| `timeline_multi_pct` | 8.6% | Only tail overlap between adjacent splits |
| Max concurrent backends | **2** (never 3+) | 3-GPU chain underutilized per token |
| `assembly_overlap` | 0.2% | Pair-start metric flat across GPU-class skew |

Tier A matrix (14 archetypes, `b6-3gpu-g-triton`, production TS): **all** rows show overlap ~0.2%, serial_dispatch 80–95%. Overlap ceiling is **architecture-class**, not Qwen3.6-specific.

### Ceiling B — cross-token pipeline shallow vs straggler

`pipeline_parallel` + `n_copies=4` rotates copy slots at `pipeline_barrier` between **decode tokens**, not splits. Hotpath note: *"splits within one token remain serial; overlap is cross-token pipeline."*

Yet `overlap_pct` stays 0.2% because:

1. Straggler split (RPC backend) sets per-token wall (~8–12 ms/tok on 3-GPU).
2. Assembly overlap counts **pair starts** across copy slots, not concurrent backend-hours.
3. B+8–B+13 optimized **wait/defer/flush inside the serial loop** — NULL on overlap (confirmed on A1 reference this session).

### What is NOT the bottleneck (closed hypotheses)

| Hypothesis | Evidence |
|------------|----------|
| `sync_copy_fallback` | 0 ms; `copy_async_ok`=2310 (B+13 audit) |
| Partial `pipeline_barrier` | B+8 NULL |
| EVENT defer / MoE async / 4-GPU flush | B+9, B+10, B+7a NULL |
| GPU class skew | +23% G, overlap unchanged |
| Dual-socket transport | B+11 NULL; hurts G on 4-GPU |

Dominant RPC cost classes remain: `blocking_events`, `SET_TENSOR_HASH`, `drain_flush`, split-2 gather `event_sync_slot` + deferred GET flush (B+13 design).

---

## 2. Goal (reopen overlap mission)

**Primary:** Achieve **full assembly-line flow** — multiple backends doing useful work concurrently for the majority of each decode token wall, not only cross-token tail overlap.

**Measurable targets (tiered):**

| Gate | Metric | Baseline | Target |
|------|--------|----------|--------|
| G1 (smoke) | `timeline_multi_pct` | 8.6% | **>= 25%** |
| G2 (M1) | `overlap_pct` @ n=384 | 0.2% | **>= 1.0%** |
| G3 (M3) | `overlap_pct` @ n=384 | 0.2% | **>= 5.0%** |
| G4 (serial) | `serial_dispatch_pct` | 91.4% | **<= 70%** |
| G5 (perf) | `G_tps` vs Tier A baseline | — | no regression > 5% on PASS rows |
| G6 (breadth) | Tier A archetypes | 14 PASS | >= 12/14 still PASS @ G5 |

Reference bench: **A1** `Qwen3.6-35B-A3B` on `b6-3gpu-g-triton`, n=384, production flags (dual ON). Confirm on **A6** (Gemma MoE), **A8** (Llama70B skew), **A4** (dense).

---

## 3. Proposed model: B+14 "wavefront assembly line"

**Core idea:** Separate **split dispatch** from **split completion**. Today each split blocks the loop until `graph_compute` + input drain finish. B+14 launches downstream splits when **producer events** fire, not when the full upstream split returns.

### 3.1 Split graph (typical 3-GPU MoE decode)

Per token (~3 splits, ~385 `split_total` rows / 384 gen tokens):

```
split 0: CPU embed / small ops
split 1: RPC (dominant compute, straggler)
split 2: local gather (RPC->CUDA0 l_out-*)
```

Layer dependencies forbid split 2 **compute** before split 1 **produces** `l_out` for that layer. Parallelism window:

- **Window W1:** split 1 `graph_compute_async` (RPC) **overlapped with** split 2 **input pipeline** (prefetch GET + slot wait) for token T, while split 0 of token T+1 or split 1 of T-1 still draining.
- **Window W2:** multiple RPC endpoints (4–6 GPU) — split 1 on backend A overlapped with split 1 on backend B for **different layer ranges** (requires finer split granularity or Path C server scheduler — out of scope here).

B+14 scope is **Window W1** on existing layer-split graphs without Path C.

### 3.2 Mechanism (three PRs)

| PR | Name | Change | Expected effect |
|----|------|--------|-----------------|
| **B+14a** | Async split handoff | Return from split loop iteration after `graph_compute_async` + `event_record`; defer input drain blocking to consumer split via existing event graph | Cuts serial tail wait; raises `timeline_multi_pct` |
| **B+14b** | Wavefront input start | Start split K input prefetch + producer `event_wait` as soon as split K-1 records completion event (not after full drain flush + slot sync) | Overlaps RPC wire with local slot waits (extends B+13b/d) |
| **B+14c** | Pipeline depth guard | Env `GGML_SCHED_PIPELINE_DEPTH` (default 4, max `n_copies`); track in-flight decode wavefront per backend; block only on oldest required event | Prevents unbounded buffer growth; enables 3+ backend concurrency |

**Flag:** `GGML_SCHED_WAVEFRONT_DISPATCH=1` (default off until bisect PASS).

**Code locus:** `ggml_backend_sched_compute_splits()` in `ggml/src/ggml-backend.cpp`; event edges already exist for `n_copies` pipeline.

### 3.3 What B+14 is NOT

- Not another `pipeline_barrier` mask tweak (B+8 exhausted).
- Not Path C unified rpc-server multi-GPU scheduler.
- Not increasing `n_copies` alone (already 4; overlap still 0.2%).
- Not topology/TS changes (retired per GPU-skew experiment).

---

## 4. Acceptance protocol (grill-owned)

1. **Read-only baseline:** `serial-overlap-audit.json` on all Tier A PASS dirs.
2. **Spike arm:** B+14a only on A1 -> G1/G4 must move before b/c.
3. **Bisect:** `GGML_SCHED_WAVEFRONT_DISPATCH=0` vs `1` on A1, A6, A8.
4. **Matrix delta:** Re-run Tier A PASS rows; append `model-archetype-baseline-b14.jsonl`.
5. **placement-broken:** A2/A3/A11 excluded from M3; document separately.

**Fail closed:** If G1 not met on A1 spike, stop before Tier A matrix. Do not stack B+14c depth without G1.

---

## 5. Risks

| Risk | Mitigation |
|------|------------|
| KV / graph reuse correctness | Gate on graph reuse path in `llama-context.cpp`; bisect with `graph_reuse_disable` |
| VRAM from deeper wavefront | Cap `GGML_SCHED_PIPELINE_DEPTH`; monitor per-backend peak in preflight |
| MoE `MUL_MAT_ID` expert routing | Extra validation on A6, A12; compare logits hash smoke |
| RPC event ordering | Keep B+9 defer ON; extend barrier drain for wavefront |
| HIP async gaps | Reuse B+14 HIP postfix branch pattern if needed |

---

## 6. Expected outcome distribution (honest)

| Outcome | Probability | Action |
|---------|-------------|--------|
| G1 met, M1 met, M3 miss | **40%** | Ship G wins; document partial ceiling |
| G1 met, M3 met on MoE only | **25%** | Tier A shows archetype-specific; production rules per class |
| G1 not met | **25%** | Close overlap reopen; Mission A only |
| Regression / correctness bug | **10%** | Revert; postmortem |

Best-case **full assembly line** does not require all 6 backends at 100% — it requires `serial_dispatch_pct` drop and `timeline_multi_pct` rise **with stable G** across dense + MoE + skew archetypes.

---

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Tier A baseline | `benches/path-b-plus/model-archetype-baseline.jsonl` |
| Serial audit script | `scripts/b6-gate-overlap-serial-audit.py` |
| Placement broken | `rpc-patch/docs/b6-gate/placement-broken.tsv` |
| B+13 root cause | [DESIGN-b13-input-wait-audit.md](DESIGN-b13-input-wait-audit.md) |

---

## 8. Phase 0 results (20260701-175805)

Read-only bounds from Tier A `sched-trace.jsonl` (`scripts/b6-gate-phase0-assembly-bounds.py`).

### Key finding: cluster is not filled

| Metric | A1 (ref) | A8 (70B) | A13 (72B) | A3 (4-GPU) | MiniMax 6-GPU |
|--------|----------|----------|-----------|------------|---------------|
| `per_decode multi_pct` | 9.1% | 16.9% | 16.4% | 11.3% | 15.8% |
| `global multi_pct` | 13.7% | 22.9% | 22.3% | 16.9% | 13.9% |
| **`global_3bk pct`** | **0.97%** | **0.34%** | **0.37%** | **0.16%** | **0.86%** |
| `intra_max_p50` (W1 ceiling) | 30.5% | 27.5% | 28.0% | 41.8% | 27.6% |
| `cross_ideal` (W2 ceiling) | 45.6% | 39.0% | 39.1% | **58.8%** | 25.4% |
| `overlap_pct` (pair metric) | 0.2% | 0.2% | 0.2% | 0.1% | 0.2% |

**Interpretation:**

- **Third GPU (and beyond) is idle >99% of concurrent wall** on every topology — validates near-full-cluster requirement for big models.
- **Global multi exceeds per-decode multi on every PASS row** (+4–6 pp) — cross-token overlap exists but is throttled; W2 is the primary gap.
- **W1-only cannot fill the cluster:** intra theoretical max ~30% multi within a token, but measured per_decode multi ~9–17%; even perfect W1 leaves 2+ backends idle.
- **Cross-token ideal 40–59%** on 3–4 GPU — work package must target W2+depth to approach this, not incremental W1.
- **Pair `overlap_pct` stays 0.2%** while duration-weighted global multi is 14–23% — optimization must use global timeline metrics (L3), not pair metric alone.

Artifact: `benches/path-b-plus/b6-phase0-assembly-bounds-20260701-175805/`

### Work package (locked for Phase 1 spike)

| Priority | Item | Rationale |
|----------|------|-----------|
| P0 | W2 cross-decode wavefront + depth guard | `global_3bk` ~0% on all rows |
| P1 | W1 async split handoff | Required for gather/RPC; insufficient alone |
| P2 | L4 finer splits (4–6 GPU) | A3 4-GPU intra_max 42% vs 30% on 3-GPU |
| P3 | L1 HASH defer on RPC path | Blocking_events / hash dominate B+13 audit |
| P4 | Factorial A0–A3 on **A1 + A8 + A13** n=64 | Big-model validation before n=384 |

**Explicit non-goal:** W1-only PR.

## 9. Grill log

- Q1: W1-first vs W2 — **resolved:** bundle W1+W2; Phase 0 proves W1-only insufficient for cluster fill.
- Q2: Phase 0 before code — **agreed and executed.**