# Handover: Phase D0 design + path-forward decision fork (2026-07-01)

**Purpose:** Continue after grill session on assembly-line saturation. Path-B+ frozen for deploy; Phase D opened. **Next session must pick an entry path** (grill / Path C C1 / parallel).  
**Branch:** `Path-B-Event-Support-Pipeline-Plus` (docs only this commit; D implementation on proposed `Path-D-Layer-Pipeline`)  
**Mission root:** [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/)  
**Prior handover:** [HANDOVER-SESSION-2026-07-01.md](HANDOVER-SESSION-2026-07-01.md)

| Location | Path | Notes |
|----------|------|-------|
| **romulus (primary client)** | `hunter@192.168.8.108` -> `~/atomic-llama-cpp-turboquant` | 5-GPU prod client |
| **remus** | `hunter@192.168.8.176` | RPC0 5060 `:50051` |
| **triton** | `hunter@192.168.8.23` | RPC2 3090 `:50054`, RPC3 3070 `:50055` |
| **dev / edit** | `/home/hunter/projects/atomic-llama-cpp-turboquant/atomic-llama-cpp-turboquant` | this repo |

---

## Session summary (done)

### Grill conclusions (locked)

1. **Handoff unit** is **activations per split** (`l_out-*`), not tokens or workpackages. Client `ggml_backend_sched` orchestrates; `rpc-server` is passive per hop.
2. **Saturation metric:** `global_3bk_pct` >= 25% (not pair `overlap_pct` ~0.2%).
3. **Operator goal:** RPC0(T+1) concurrent with RPC1(T) under assembly-line saturation.
4. **Why Path-B+ cannot deliver:** serial `compute_splits` loop (Ceiling A) + autoregressive **sample gate** (T+1 unknown until T fully decoded + sampled) on single-seq decode.
5. **What Path B actually overlaps:** RPC(T+1) vs **local tail**(T) — not RPC chain concurrency on same token.
6. **Phase 0 evidence:** `global_3bk` < 1% all topologies; B+8-B+16 + wavefront NULL.

### Documentation shipped

| Doc | Role |
|-----|------|
| [DESIGN-path-d-layer-pipeline.md](../../docs/rpc-multi-backend-pipeline-plus/DESIGN-path-d-layer-pipeline.md) | **Phase D0 master design** — GPipe + Path C stepping stone |
| [PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) | Phase D added; Phase 3 superseded |
| [TRACKING.md](../../docs/rpc-multi-backend-pipeline-plus/TRACKING.md) | V6 section + decision log |
| [MISSION.md](../../docs/rpc-multi-backend-pipeline-plus/MISSION.md) | Current focus -> Phase D0 |
| [DESIGN-b14 s11](../../docs/rpc-multi-backend-pipeline-plus/DESIGN-b14-parallel-assembly-line.md) | Path C bridge criteria **met** |
| [rpc-path-c-plan.md](../docs/rpc-path-c-plan.md) | Path C = Phase **D1** |

### Path-B+ production state (unchanged code)

- Phase 1c **complete** — deploy-ready.
- Prod defaults: wavefront OFF, hash defer OFF, dual-socket ON (`b6-gate-5gpu-production-env.sh`).
- 5-GPU 70B+ equal-safe L4 **PASS**; preflight with `--ts-mode equal --phase load`.
- M3 overlap hunt **closed** on this branch.

```bash
# Deploy (Path-B+)
bash scripts/b6-gate-5gpu-deploy.sh
bash scripts/pathb-rpc-vram-preflight.sh --preset b6-5gpu-g-prod \
  --gguf /mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf \
  --ts-mode equal --phase load
```

---

## Decision fork (pick one next session)

Documented in DESIGN-path-d s7 and TRACKING V6.

| Option | First action | Pros | Cons |
|--------|--------------|------|------|
| **A — Grill** | D0.3 GPipe KV-ordering + mode A (MTP-coupled) semantics | Locks D2 design before code | No util proof until later |
| **B — Path C first** | D1 C1 read-only baseline on triton `:50054`+`:50055` | Smaller blast radius; server-local PCIe | Does not fix cross-host 5-GPU alone |
| **C — Parallel** | C1 baseline **while** grilling D0.3 | **Recommended** — de-risk both tracks | Two threads to track |

### Open grill item (Q6)

**Path C before GPipe?** — Proposed **yes** (triton D1 stepping stone). Confirm or override at session start.

### Workload gate (proposed, Q4)

- **Develop:** A1 MoE (Qwen3.6-35B-A3B) — fast iteration, MTP/NextN available.
- **Prod prove:** A8/A13 70B dense — equal-safe load + `global_3bk` + G regression guard.

---

## Phase D0 checklist (remaining)

| ID | Task | Owner | Status |
|----|------|-------|--------|
| D0.1 | DESIGN-path-d doc + PLAN/TRACKING cross-links | this session | **done** |
| D0.2 | 5-GPU split topology map (how many RPC **compute** splits per token) | next | pending |
| D0.3 | GPipe KV-ordering grill | next | pending |
| D0.4 | Path C C1 triton baseline (RTT, server GPU util) | next | pending |
| D0.5 | Tag Path-B+ deploy SHA before `Path-D-Layer-Pipeline` branch | next | pending |

### Before any D2 code

1. Tag: `git tag path-b-plus-deploy-YYYYMMDD <deploy-sha>`
2. Branch: `git checkout -b Path-D-Layer-Pipeline`
3. Flag plan: `GGML_SCHED_GPIPE=0` default (Path-B+ behavior preserved)

---

## Path map (quick reference)

```text
Path-B-Event-Support-Pipeline-Plus  -->  PRODUCTION (frozen overlap hunt)
        |
        +-- tag path-b-plus-deploy-*
        |
        +-- Path-D-Layer-Pipeline (proposed)
              |
              +-- D1 Path C (triton 3090+3070 server sched)  [rpc-path-c-plan C1/C2]
              |
              +-- D2 GPipe client sched (GGML_SCHED_GPIPE)   [cross-host saturation]
              |
              +-- D3 runbook + upstream merge cadence        [TURBOQUANT_UPSTREAM_MERGE.md]
```

**Acceptance (Path D):** `global_3bk_pct` >= 25% @ n=384 5-GPU; G within 5% of Path-B+ tag with GPipe OFF; logits/gen smoke PASS.

---

## Commands for next session

```bash
# Phase 0 bounds on existing bench
python3 scripts/b6-gate-phase0-assembly-bounds.py benches/path-b-plus/<run>/telemetry

# Path C C1 (read-only) — see rpc-path-c-plan.md Phase C1
# triton dual-GPU rpc-server; measure RTT/tok + nvidia-smi util

# Grill entry: read DESIGN-path-d s9 grill log Q6; one question at a time
```

---

## Backlog: SYNC vs Plus comparison chart

Before or in parallel with Phase D0:

- Matrix: [path-b-plus-vs-sync-comparison.md](../../docs/rpc-multi-backend-pipeline-plus/BENCHMARKS/path-b-plus-vs-sync-comparison.md)
- Run: `DRY_RUN=1 bash scripts/b6-gate-sync-vs-plus-comparison.sh` (36 cells)
- **Both workloads:** `n384` + `n2048-mt` (multi-turn hard prompt, 2048 gen)
- Pin SYNC SHA (`feature/turboquant-kv-cache`); rebuild `build-rocm-docker-sync` + rpc-servers
- Suggested order: n384 full matrix first, then `B6_COMPARE_WORKLOADS=n2048-mt` overnight
- MTP/NextN rows (M2/M4) need `llama-server` bench — manual per chart s6.3

---

## Not in this commit

- Bench artifact dirs under `benches/path-b-plus/b6-*-b14*` (untracked; keep local or commit separately).
- `docs/cuda-windows-5070ti/.../blocking-audit-c.json` (untracked).
- No `GGML_SCHED_GPIPE` or Path C code — docs only.

---

## Suggested opening for next session

1. Read this handover + [DESIGN-path-d-layer-pipeline.md](../../docs/rpc-multi-backend-pipeline-plus/DESIGN-path-d-layer-pipeline.md) s7.
2. **Decide:** A / B / C from decision fork table above.
3. If **C (parallel):** run D0.2 split map from latest 5-GPU trace; schedule triton C1; start grill Q1 on KV layer ordering.
4. If **grill only:** "Grill Phase D0.3 — GPipe mode A + MTP coupling."
5. If **Path C only:** "Execute Path C C1 baseline on triton."

---

**Last updated:** 2026-07-01  
**Assisted-by:** Grok (session handover)