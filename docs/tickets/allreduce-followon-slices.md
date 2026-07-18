# AllReduce follow-on + Placement slices — Grab-able Issues

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-17  
**Source (AllReduce):** `docs/wayfinder/ALLREDUCE-OPTIMIZATION-PLAN.md`  
**Source (Placement):** `.scratch/placement-control-plane/map.md` + research under `docs/research/placement-*.md`  
**Baselines:** `docs/research/allreduce-multi-topology-baselines.md`  
**Wayfinder maps:** `.scratch/allreduce-timing/` (complete), `.scratch/placement-control-plane/` (design tickets 04–07 still open)

These are **tracer-bullet slices** — each delivers a verifiable outcome. Work the **frontier**: any `ready-for-agent` slice whose blockers are complete.

## Lifecycle

| State | Who changes it | What happens |
|-------|----------------|--------------|
| `ready-for-agent` | (initial) | Slice is available; next agent can grab it |
| `in-progress` | Agent grabbing the slice | Agent changes status and starts work |
| `complete` | Agent finishing the slice | Agent checks all boxes, fills completion footer |
| `blocked` | Human or agent | Waiting on hardware, design lock, or another slice |

**On completion**, also update:

- `docs/wayfinder/TRACKING.md` if Path-D phases apply  
- Baseline / plan docs when numbers or defaults change  

**Tracker note:** GitHub issues are disabled on the primary remote; this markdown file is the canonical slice tracker (local).

---

## Track overview

| Track | Goal | Start |
|-------|------|--------|
| **A — AllReduce follow-on** | Ops, measurement debt, G1 layer residual; gated AR micro-opt | F1, F2 ready now |
| **B — Placement control plane** | Capacity → plan IR → apply → heat (lever 1) | P1 ready after design gate note below |

**Out of scope for this file (do not open):**

- Distributed AllReduce over RPC  
- Global heterogeneous client `-sm tensor`  
- Hybrid TP-unit **implementation** until placement designs lock Shape A (see P5)  
- Layer-2 matmul / kernel-anvil (Path-D Slice 7)  

---

## Design gate (placement)

Placement wayfinder still has open grilling tickets (04 capacity, 05 plan IR, 06 TP-unit, 07 lock plan).  

**Rule for agents:**

- **P1–P3 may start** using inventories + provisional decisions recorded in each slice (prefer existing RPC free/total + HELLO; layer-range plan only).  
- If a design choice conflicts with a later locked `PLACEMENT-CONTROL-PLANE-PLAN.md`, **prefer the locked plan** and amend the slice.  
- Completing placement wayfinder tickets 04–07 in parallel is encouraged; not a hard blocker for P1 MVP if the slice stays minimal.

---

# Track A — AllReduce follow-on

## Slice F1: Dual-CUDA AR ops defaults and NCCL notes

**Status:** ready-for-agent  
**Blocked by:** None  
**Plan rank:** 3  

### What to build

Document and wire **fleet defaults** so dual-CUDA tensor experiments and any production TP island use **`GGML_CUDA_ALLREDUCE=internal`** unless a re-bench on large models says otherwise. Document NCCL install on Ubuntu/CUDA hosts, that host AR timers for NCCL are async-enqueue (rank by **TG/PP**), and point at T2 baseline artifacts.

### Acceptance criteria

- [ ] Ops/docs state: default dual-CUDA AR provider = **internal**; butterfly last resort  
- [ ] NCCL: install packages (`libnccl2`/`libnccl-dev` or equivalent), rebuild with `GGML_CUDA_NCCL=ON`, verify `Found NCCL` / link  
- [ ] Explicit: do **not** rank providers by `GGML_ALLREDUCE_TRACE` duration alone for NCCL  
- [ ] Links to `docs/wayfinder/ALLREDUCE-OPTIMIZATION-PLAN.md` and T2 baseline sections  
- [ ] No kernel/algorithm change required  

### Blocked by

None — can start immediately.

---

## Slice F2: Release re-baseline T0 / T1 / T3 layer TG and PP

**Status:** ready-for-agent  
**Blocked by:** None  
**Plan rank:** measurement debt  

### What to build

Re-run the layer-rail recipes from the wayfinder baselines on a **Release** client (not debug/asserts), same topologies:

- T0: single ROCm  
- T1: ROCm + local CUDA RPC  
- T3a: ROCm + remus RPC  
- T3b (optional): ROCm + local + remus  

Capture TG and PP (and `split_timing` if cheap). Update the baselines document with a **Release** table so absolute t/s can be compared to Path-D lore.

### Acceptance criteria

- [ ] Release build identified and used for all rows  
- [ ] T0 + T1 + T3a numbers recorded (TG + PP, multi-run)  
- [ ] `docs/research/allreduce-multi-topology-baselines.md` updated with Release section  
- [ ] Artifacts under `benches/allreduce-baselines/<label>/`  
- [ ] Note debug vs release gap if both exist  

### Blocked by

None — can start immediately (needs cluster/RPC up).

---

## Slice F3: T2 large-model tensor matrix (Gemma-4 and/or Qwen3.6-35B)

**Status:** ready-for-agent  
**Blocked by:** F1 recommended (NCCL docs); not hard-blocked  
**Plan rank:** 4  

### What to build

On triton dual-CUDA (3090+3070), Release CUDA build **with NCCL**, run layer control + tensor × `{internal, nccl, none}` for at least one map-scale model (Gemma-4 dense and/or Qwen3.6-35B as VRAM allows). Use existing `scripts/allreduce-t2-baseline.sh` (`-dev CUDA0/CUDA1`). Update provider ranking if TG order changes vs 0.8B smoke.

### Acceptance criteria

- [ ] Model(s) named and present on triton  
- [ ] Layer TG/PP + tensor×3 providers TG/PP recorded  
- [ ] `allreduce_provider` confirms real `nccl` when requested  
- [ ] Baselines doc updated; ranking note vs plan §3.2  
- [ ] If tensor still loses to layer/single: **do not** open AR micro-opt (F6 stays blocked)  

### Blocked by

None hard — F1 recommended first.

---

## Slice F4: Layer residual pass (G1) using split_timing

**Status:** ready-for-agent  
**Blocked by:** F2 (need Release baseline to claim improvement)  
**Plan rank:** 2  

### What to build

Using Release T1 (and T3 if available) + `event=split_timing`, identify the dominant residual (idle vs compute, which backend). Apply **one** concrete Path-D/ops fix or config change (event wait, RPC drain, tensor-split ratios, Plus knobs — not placement IR). Re-measure TG and PP; require non-regression on the other metric within noise.

### Acceptance criteria

- [ ] Before/after Release TG (and PP) tables  
- [ ] split_timing summary before/after (idle/compute p50 on heavy backend)  
- [ ] Single primary change documented (what and why)  
- [ ] No TG or PP regression beyond stated noise bar  
- [ ] Notes in baselines or Path-D tracking  

### Blocked by

- Slice F2  

---

## Slice F6: (Gated) AR micro-optimization

**Status:** blocked  
**Blocked by:** F3 — only if F3 shows tensor TG competitive with layer/single **and** AR-bound  
**Plan rank:** 5  

### What to build

Only after F3: targeted AR path improvements (thresholds, wire format knobs, etc.) under plan §8 acceptance (TG+PP non-regress, provider logged, T2 release).

### Acceptance criteria

- [ ] Written gate: F3 evidence that AR is the bottleneck  
- [ ] T2 release before/after TG+PP  
- [ ] Provider identity logged  
- [ ] Plan §8 checklist satisfied  

### Blocked by

- Slice F3 (and explicit human go after reading F3 results)  

---

# Track B — Placement control plane (implementation)

## Slice P1: Capacity discovery (local + RPC usable VRAM)

**Status:** done (2026-07-17) — client P0 on `path-g-gpipeline-assembly-line`  
**Blocked by:** None  
**Plan rank:** 1 / P0  
**Map issue:** [08-discover-inventory-e2e](../../.scratch/placement-control-plane/issues/08-discover-inventory-e2e.md)  
**Design / ops:** `docs/research/placement-capacity-discovery-design.md` §10  

### What was built

End-to-end path so the client can **discover** each backend's memory and identity without guessing `-ts`:

- Local GPU and RPC backends: **total / free** plus client **`table-v1` usable** (`usable_weight_mib`).  
- Stable **backend_id** (`local:…` / `local:pci:…`, `rpc://host:port#i`).  
- Opt-in dump: `--placement-discover` / `--placement-inventory PATH` (env `LLAMA_ARG_PLACEMENT_*`).  
- Works against **legacy** cluster `rpc-server` via N× `GET_DEVICE_MEMORY` (`rpc_fetch_mode=legacy_n_get_device_memory`). Batch capacity opcode still open.

### Acceptance criteria

- [x] Dump shows ≥2 backends on a dual Path-D config (e.g. ROCm + Triton RPC)  
- [x] Each row has total, free, usable (or documented formula), backend_id, kind  
- [x] Docs for operators: how to run the dump (design §10 + plan P0)  
- [x] No requirement for TP-unit yet  
- [x] Unit tests: `tests/test-placement-capacity.cpp`; live smoke on romulus+triton  

### Blocked by

None — complete for P0 client.

---

## Slice P2: Plan IR load + deterministic layer-range apply

**Status:** ready-for-agent  
**Blocked by:** P1  
**Plan rank:** 1 / P1  

### What to build

Versioned **placement plan** (JSON or equivalent) that assigns **layer ranges** to `backend_id`s. Client loads plan at model load and applies so weights land as specified. Mis-assignment is **fail-loud** (error), not silent partial fit. Escape hatches: keep `-ts` / overrides for debug only.

Demo: plan pins early layers to RPC and late layers to ROCm (or reverse); verify via logs or buffer backend names; run a short bench.

### Acceptance criteria

- [ ] Documented plan schema (version field, backends[], layer assignments)  
- [ ] Load via CLI flag or config path  
- [ ] Apply is deterministic and fail-loud on conflict/OOM-at-load when possible  
- [ ] Demo recipe on T1 or T3 topology  
- [ ] `-ts` not required for the demo path  

### Blocked by

- Slice P1  

---

## Slice P3: Heat-informed plan generation (hot-on-fast / cold-on-slow)

**Status:** ready-for-agent  
**Blocked by:** P2  
**Plan rank:** 1 / P2  

### What to build

Two-phase workflow:

1. Probe or offline heatmap / node timings → **per-layer heat** (not device-stub `layers[]` alone; use rollup from node timings if needed).  
2. Generate a plan IR that places hot layers on faster backends and cold layers on slower/high-VRAM fillers, respecting capacity from P1.  

Serve loads the generated plan (P2 apply path).

### Acceptance criteria

- [ ] Input: heatmap or node_timings artifact documented  
- [ ] Output: valid plan IR consumable by P2  
- [ ] At least one demo where hot layers prefer the faster device  
- [ ] Docs for probe → generate → serve  
- [ ] No continuous replan every token (load-time only)  

### Blocked by

- Slice P2  

---

## Slice P4: Prefactor / heat schema cleanup (if needed for P3)

**Status:** ready-for-agent  
**Blocked by:** None (can parallel P1)  
**Plan rank:** prerequisite for honest P3  

### What to build

If device-stub `layers[]` still misleads: productize **layer rollup from node_timings** (or document the single supported heat input) so P3 does not invent a second heatmap schema. Align with ADR 0004b / gpipe-profiler where possible.

### Acceptance criteria

- [ ] Clear contract: what file/fields P3 consumes  
- [ ] Either fix stub layers[] or explicitly deprecate it for placement  
- [ ] Example artifact checked into benches or docs/research  

### Blocked by

None — can start immediately; **should complete before or with P3**.

---

## Slice P5: RPC TP-unit (Shape A) — gated

**Status:** blocked  
**Blocked by:** P2; placement wayfinder ticket 06 (Shape A decision); prefer plan lock 07  
**Plan rank:** P3 in placement map  

### What to build

Homogeneous equal-VRAM multi-GPU inside one RPC process, advertised as **one** logical device with internal tensor+AR; client layer-rails across logical devices. Only after architecture decision is locked.

### Acceptance criteria

- [ ] Written Shape A decision (wayfinder 06 or plan)  
- [ ] One logical RPC device with ~sum weight capacity semantics documented  
- [ ] Internal AR uses internal provider default (AllReduce plan)  
- [ ] Client layer plan can target the logical backend_id  
- [ ] Equal-VRAM constraint enforced or documented  

### Blocked by

- Slice P2  
- Placement design: RPC TP-unit architecture decision  
- Human go after designs  

---

## Frontier (start here)

| Slice | Status | Notes |
|-------|--------|--------|
| **F1** | ready-for-agent | Ops docs |
| **F2** | ready-for-agent | Release baselines |
| **P1** | ready-for-agent | Capacity discovery |
| **P4** | ready-for-agent | Heat contract (parallel) |
| F3 | ready-for-agent | After models on triton |
| F4 | blocked by F2 | |
| P2 | blocked by P1 | |
| P3 | **complete** (2026-07-18) | Heat-aware plan generate |
| F6 | blocked by F3 + human go | |
| P3 | complete | Heat-aware plan generate (issue 13) |
| F6 | blocked by F3 + human go | |
| P5 | blocked by designs + P2 | |

**Agent instruction:** Grab one frontier slice, set `in-progress`, complete acceptance boxes, set `complete`, stop. Do not start F6 or P5 without gates.

---

## Completion footer template

```text
### Completion
- Date:
- Agent/session:
- Key commits:
- Verification:
- Notes:
```
