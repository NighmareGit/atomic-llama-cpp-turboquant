# DESIGN: Path D — layer pipeline (GPipe-style assembly line)

| Field | Value |
|-------|-------|
| **Date** | 2026-07-01 |
| **Status** | **Phase D0** — design + grill gate (no code yet) |
| **Prereq** | Path-B+ Phase 1c **complete**; M3 overlap **closed**; structural ceiling documented |
| **Successor branch** | `Path-D-Layer-Pipeline` (proposed; branch from Path-B+ deploy tag) |
| **Parent mission** | [MISSION.md](MISSION.md) | [PLAN.md](PLAN.md) Phase D |

Navigation: [DESIGN-b14-parallel-assembly-line.md](DESIGN-b14-parallel-assembly-line.md) | [rpc-path-c-plan.md](../../rpc-patch/docs/rpc-path-c-plan.md) | [TRACKING.md](TRACKING.md)
**Current state (factual):** [CURRENT-STATE-pipeline-flow.md](CURRENT-STATE-pipeline-flow.md) — what actually runs today (serial split dispatch, not GPipe)

---

## 1. Problem statement (grill-backed)

### 1.1 Goal (operator mental model)

**Assembly-line saturation:** when RPC worker 0 finishes its layer stage for token `T`, worker 1 consumes the activation for `T` while worker 0 concurrently starts token `T+1`. Under saturation, throughput should scale with each GPU added to the chain.

This is **Window W2** from B+14: concurrent RPC layer compute across backends, not only cross-token tail overlap.

### 1.2 What Path-B+ achieved vs what it cannot

| Delivered (ship Path-B+) | Not delivered (requires Path D / Path C) |
|--------------------------|------------------------------------------|
| Stable G on 2–5 GPU topologies | Linear scaling with RPC node count |
| All GPUs carry layer weight (L4 equal-safe TS) | `global_3bk_pct` >= 25% (3+ backends concurrent) |
| Cross-token overlap: RPC(T+1) vs local tail(T) | RPC0(T+1) concurrent with RPC1(T) |
| B+13 G/stall wins; deploy runbooks | M3 `overlap_pct >= 5%` |

Phase 0 evidence (`b6-phase0-assembly-bounds-20260701-175805`): `global_3bk` < 1% on every topology while `global_multi` is 14–23%. Third+ GPU idle >99% of concurrent wall.

### 1.3 Terminology (what moves between GPUs)

There is no discrete **workpackage** or **token** handed RPC-to-RPC. Units:

| Level | Name | Description |
|-------|------|-------------|
| Decode step | `llama_decode()` | One forward pass per generated token |
| Graph | `ggml_cgraph` | Full forward graph for that token |
| Split | `ggml_backend_sched_split` | Subgraph assigned to one backend — the real "hop" |
| Handoff | Intermediate **activations** | Hidden states (e.g. `l_out-*`), via `GRAPH_RECOMPUTE` + `GET_TENSOR` / relay |

Typical MoE 3-GPU decode per token:

```text
split 0: CPU embed
split 1: RPC (dominant layer compute — straggler)
split 2: local gather (RPC -> client CUDA)
```

`llama-server` does **not** orchestrate hops between RPC workers. The **client** `ggml_backend_sched` (romulus) drives every split. Each `rpc-server` is a passive compute endpoint.

---

## 2. Why the assembly-line model fails on Path-B+

### 2.1 Ceiling A — serial split dispatch (within one token)

`ggml_backend_sched_compute_splits()` runs `for (split_id = 0; split_id < n_splits; split_id++)` serially. Each iteration completes input wait, flush, `graph_compute_async`, `event_record`, `split_total` before advancing.

Code locus: `ggml/src/ggml-backend.cpp`.

### 2.2 Ceiling B — autoregressive sample gate (across tokens)

Token `T+1` does not exist as a decode job until token `T` is **fully** forward-passed **and sampled**.

```text
RPC0(T) done  -->  RPC1(T) still required  -->  local gather(T)  -->  sample  -->  T+1 known
```

When RPC0 finishes mid-token, RPC1 still owes layers for `T`. The desired overlap RPC0(T+1) || RPC1(T) is **illegal** for single-sequence autoregressive decode without layer-wise pipelining across tokens (GPipe) or known-ahead tokens (speculation).

### 2.3 What Path B actually pipelines

Path B (`n_copies=4`, `pipeline_barrier`) achieves:

```text
Token N:   [RPC stages done]  [local GPU tail still running]
Token N+1:                    [RPC head starts]
```

Not: RPC0(T+1) while RPC1(T). On typical topology (one dominant RPC compute split + gather splits), when T+1 RPC starts, no RPC is still busy on T.

B+14 wavefront (W1+W2) tried to loosen the serial loop. Result: `global_3bk` < 1%; default `B6_5GPU_WAVEFRONT=0`.

### 2.4 Blocker summary

| Blocker | Fixable on Path-B+? | Path D approach |
|---------|---------------------|-----------------|
| Serial split for-loop | B+14 tried; NULL | GPipe sched: decouple dispatch from completion |
| T+1 unknown until T sampled | No (fundamental) | GPipe across layers + speculative/MTP, or multi-seq |
| Client-side RPC hops (TCP RTT per split) | Marginal (B+11–B+13) | Path C for co-located server GPUs |
| One straggler RPC split per token | L4 helps placement, not concurrency | Finer splits + concurrent stage dispatch |

---

## 3. Mission fork decision (locked 2026-07-01)

### 3.1 Freeze Path-B+ (production line)

- **Do not delete** `Path-B-Event-Support-Pipeline-Plus` — deploy-ready (Phase 1c).
- **Stop** overlap hunt (B+8–B+16, wavefront) unless new trace-backed hypothesis.
- **Tag** before branching: e.g. `path-b-plus-deploy-20260701`.
- Prod defaults: `b6-gate-5gpu-production-env.sh` (dual ON, wavefront OFF, hash defer OFF).

### 3.2 Open Path D (new branch, same repo)

- Proposed branch: `Path-D-Layer-Pipeline` from Path-B+ deploy tag.
- **Same git repo** (`atomic-llama-cpp-turboquant`) — not a separate fork (upstream merges stay tractable).
- New doc root section: this file + PLAN Phase D.

### 3.3 Path C as stepping stone (not either/or)

| Route | Where parallelism lives | Best for | Track |
|-------|-------------------------|----------|-------|
| **Path C** | Server `rpc-server` internal `ggml_backend_sched` | triton 3090+3070 (co-located GPUs, local PCIe) | **D1** — smaller, localized proof |
| **Path D / GPipe** | Client sched — multiple tokens in-flight at layer granularity | Cross-host RPC chain (romulus/remus/triton) | **D2** — full cluster saturation |

**Recommended order:** Path C on triton first (C1/C2 in [rpc-path-c-plan.md](../../rpc-patch/docs/rpc-path-c-plan.md)), then GPipe client sched. Path C does not replace GPipe for 5-GPU cross-machine; it de-risks scheduling on one node.

---

## 4. Path D architecture options

### 4.1 GPipe modes (must pick in D0 grill)

| Mode | Mechanism | Single-seq 70B? | Notes |
|------|-----------|-----------------|-------|
| **A — Layer pipeline across tokens** | T+1 enters stage 0 while T at stage 1 | Needs MTP/NextN draft or known token | Matches operator mental model |
| **B — Microbatch / multi-seq** | Different sequences at different positions | Server multi-slot | Later server feature |
| **C — Head/tail only** | T+1 RPC vs T local tail | Yes (Path B today) | Does not scale with RPC count |

**D0 recommendation:** primary **A** (MTP/NextN-coupled GPipe for single-seq); secondary **B** backlog.

### 4.2 Flag-gated integration (upstream compatibility)

```text
GGML_PIPELINE_PLUS=1     # existing Path-B+ production (unchanged)
GGML_SCHED_GPIPE=0       # new; 0 = classic Path-B+ decode path
```

| Touch surface | Policy |
|---------------|--------|
| `ggml/src/ggml-backend.cpp` | GPipe state in `ggml_sched_gpipe_*` helpers — expected merge conflict zone |
| `src/llama-context.cpp` | **One seam:** decode entry selects pipelined vs classic; do not fork KV logic |
| Model arch / parsers / templates | **Do not touch** — upstream model updates flow through |
| `ggml/src/ggml-rpc/` | Path C server sched on D1 sub-track only |
| `rpc-patch/`, `scripts/b6-gate-*` | Fork collateral (existing pattern) |

### 4.3 Upstream merge policy

Reuse [TURBOQUANT_UPSTREAM_MERGE.md](../../TURBOQUANT_UPSTREAM_MERGE.md) playbook:

1. **Merge** (not rebase) `upstream/master` into fork line every 4–8 weeks or before a target model release.
2. After merge: `b6-gate-validate-rpc-matrix.sh` + one 70B equal-safe load smoke.
3. GPipe default OFF — upstream behavior unchanged when flag unset.
4. Do not plan ggml-org PR until 2 topologies PASS + proto versioned.

---

## 5. Phase D execution plan

### D0 — Design + read-only bounds (current)

| Step | Work | Gate | Status |
|------|------|------|--------|
| D0.1 | This design doc + PLAN/TRACKING cross-links | Review complete | **in progress** |
| D0.2 | Split topology map on 5-GPU prod (RPC compute splits per token) | Table in TRACKING | pending |
| D0.3 | GPipe KV-ordering grill (layer write vs stage release) | Grill log s7 | **next: grill or Path C kickoff** |
| D0.4 | Path C C1 baseline on triton `:50054`+`:50055` | RTT + util CSV | pending |
| D0.5 | Paper prototype: in-flight depth, copy-slot layout | Design s6 acceptance draft | pending |

**Workload gate (recommended):** develop on **A1 MoE** (fast iteration, MTP/NextN); production prove on **A8/A13 70B** dense.

### D1 — Path C triton stepping stone

Per [rpc-path-c-plan.md](../../rpc-patch/docs/rpc-path-c-plan.md):

| Step | Work | Gate |
|------|------|------|
| C1 | Baseline: per-device RPC splits, RTT count, server GPU util | Artifact dir |
| C2 | `GRAPH_COMPUTE_ALL` + server-side sched | 2x server GPU duty cycle; G non-regression |

### D2 — GPipe client scheduler

| Step | Work | Gate |
|------|------|------|
| D2.1 | `GGML_SCHED_GPIPE=1` skeleton; default OFF | Build green; Path-B+ bisect unchanged |
| D2.2 | MTP-overlapped single-seq layer pipeline | `global_3bk` movement; logits hash smoke PASS |
| D2.3 | 5-GPU prod matrix A1 + A8/A13 | G within 5% of Path-B+ tag; `global_3bk` >= 25% target |

### D3 — Production hardening

- Runbook delta in [PIPELINE.md](../../PIPELINE.md)
- Preflight unchanged (`pathb-rpc-vram-preflight.sh`)
- Profiler acceptance via `b6-gate-phase0-assembly-bounds.py`

---

## 6. Acceptance metrics (Path D)

| Gate | Metric | Path-B+ baseline | Path D target |
|------|--------|------------------|---------------|
| G0 | Deploy regression | Path-B+ tag G | No regression > 5% with GPipe OFF |
| G1 | Cluster fill | `global_3bk_pct` < 1% | **>= 25%** @ n=384 5-GPU |
| G2 | Serial dispatch | `serial_dispatch_pct` ~91% | **<= 50%** (stretch) |
| G3 | Throughput scaling | Sub-linear add-GPU | Measurable G gain when adding RPC stage (same model class) |
| G4 | Correctness | logits hash / gen completes | PASS all gate archetypes |

Primary gate: **G1 (`global_3bk_pct`)** — not pair `overlap_pct` (see DESIGN-b14 s10).

---

## 7. Decision fork (next session)

Choose one entry path after D0.1 lands:

| Option | Entry | When |
|--------|-------|------|
| **Grill** | D0.3 KV-ordering + GPipe mode A semantics | Before any D2 code |
| **Path C first** | D1 C1 baseline on triton | Smaller blast radius; proves server-side assembly line |
| **Parallel** | C1 read-only while grilling D0.3 | **Recommended** — no code conflict |

---

## 8. Commands (reference)

```bash
# Tag Path-B+ before branching
git tag path-b-plus-deploy-20260701  # example; use current deploy SHA

# Phase 0 bounds on any bench dir
python3 scripts/b6-gate-phase0-assembly-bounds.py <out>/telemetry

# Path C C1 baseline (read-only, triton dual-GPU rpc-server)
# See rpc-path-c-plan.md Phase C1

# Deploy (unchanged on Path-B+)
bash scripts/b6-gate-5gpu-deploy.sh
bash scripts/pathb-rpc-vram-preflight.sh --preset b6-5gpu-g-prod \
  --gguf /mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf \
  --ts-mode equal --phase load
```

---

## 9. Grill log

| ID | Question | Status | Resolution |
|----|----------|--------|------------|
| Q1 | Primary saturation metric? | **resolved** | `global_3bk_pct` (not pair `overlap_pct`) |
| Q2 | W1 vs W2 bottleneck? | **resolved** | W2 (within-token concurrent RPC) |
| Q3 | RPC0(T+1) \|\| RPC1(T) on single-seq? | **resolved** | Blocked by sample gate; needs GPipe mode A or Path C server-local |
| Q4 | Primary D0 workload gate? | **proposed** | **C** — A1 MoE dev, A8/A13 70B prod prove |
| Q5 | Path C before GPipe? | **proposed** | **Yes** — triton D1 stepping stone |
| Q6 | Grill vs Path C kickoff? | **open** | See s7 — recommend parallel C1 + grill D0.3 |

---

## 10. References

- Structural ceiling: [TRACKING.md](TRACKING.md) s Structural ceiling
- B+14 Phase 0: [DESIGN-b14-parallel-assembly-line.md](DESIGN-b14-parallel-assembly-line.md) s8
- Path C bridge (superseded by this doc): DESIGN-b14 s11
- Path C implementation plan: [rpc-path-c-plan.md](../../rpc-patch/docs/rpc-path-c-plan.md)
- Upstream merges: [TURBOQUANT_UPSTREAM_MERGE.md](../../TURBOQUANT_UPSTREAM_MERGE.md)
- Pipeline layers: [PIPELINE.md](../../PIPELINE.md)
- Session handover: [HANDOVER-SESSION-2026-07-01-phase-d.md](../../rpc-patch/patch/HANDOVER-SESSION-2026-07-01-phase-d.md)