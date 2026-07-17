# Placement control plane — architecture and phased plan (locked)

**Date:** 2026-07-17  
**Map:** `.scratch/placement-control-plane/map.md`  
**Ticket:** [Lock PLACEMENT-CONTROL-PLANE-PLAN.md phases](../../.scratch/placement-control-plane/issues/07-lock-placement-plan-doc.md)  
**Status:** **LOCKED** (wayfinding complete; implementation of P0–P4 is follow-on work)

This document is the **destination** of the placement control plane wayfinder map. It freezes architecture decisions and a phased delivery plan. It does **not** implement the subsystem.

---

## 1. Executive summary

| Question | Locked answer |
|----------|----------------|
| Primary problem? | Hand `-ts` / weak `--fit` / flaky `-ot` cannot run heterogeneous multi-RPC fleets (incl. **8 GB** cards) as first-class VRAM. |
| Primary UX when add-on on? | **Discover → plan IR → native apply** (optional heat-aware generate). |
| Default when add-on off? | **Unchanged** classic knobs (full back-compat). |
| Cross-backend rail? | **Layer** (and Path-D GPipe/RPC). No global HIP+RPC client `-sm tensor` as pooling. |
| Multi-GPU on one RPC node? | **Shape B first** (visible devices / Path C); **Shape A TP-unit** later for equal-VRAM N=2 + specialized AR, opt-in. |
| Map done when? | This file locked + ticket 07 resolved — **not** when code ships. |

**Bottom line:** Build an **opt-in placement control plane** that plans from **usable capacity** and deploys **deterministic layer-range plans** (with tensor overrides), then add heat and TP-units in later phases. Keep every GPU, including 8 GB, in the pool.

---

## 2. Evidence and design base

| Asset | Role |
|-------|------|
| `docs/research/placement-knobs-inventory.md` | Why current knobs fail |
| `docs/research/placement-heatmap-inputs-inventory.md` | Heatmap stub vs real `node_timings` heat |
| `docs/research/placement-rpc-capacity-inventory.md` | free/total today; gaps |
| `docs/research/placement-capacity-discovery-design.md` | Capacity protocol (ticket 04) |
| `docs/research/placement-plan-ir-apply-design.md` | Plan IR + native apply (ticket 05) |
| `docs/research/placement-rpc-tp-unit-architecture.md` | Shape A/B sequencing (ticket 06) |
| `docs/research/allreduce-topology-feasibility.md` | Rails T0–T5; AR only same-process |
| `docs/wayfinder/ALLREDUCE-OPTIMIZATION-PLAN.md` | Sibling: residual gap G1 owns placement |
| ADR 0004 / 0004b | Path C multi-device; profiler/heatmap schema |

---

## 3. Architecture (locked)

### 3.1 Standing rules

1. **Add-on, not replacement** — placement CLI/env off ⇒ classic behavior only.
2. **8 GB non-negotiable** — multi-record capacity and layer plans keep small cards as cold fillers or small ranges; never strand them for a tensor-only 24 GB pair.
3. **Layer rail** across heterogeneous backends; tensor/AR only inside eligible same-process islands (or future TP-unit).
4. **Fail-loud** by default on discover/apply mismatches; optional degrade switches off by default.
5. **One apply story** — plan IR → native apply. Heat/Pareto are **generators**, not a second silent governor.

### 3.2 Capacity discovery (P0 design)

| Item | Lock |
|------|------|
| Usable | Hybrid: backend **static pads** + client **`table` / `auto`** reserves |
| RPC transport | **Batch per endpoint** → N capacity records |
| Identity | `rpc://host:port#i`, `local:…` + optional aliases; dynamic discover |
| Refresh | Before plan + before apply; TTL; no per-token |
| Partial fail | Default refuse incomplete configured topology |
| perf_class | Optional; missing ⇒ unknown |
| Enable | Explicit placement CLI + optional env |

Detail: `docs/research/placement-capacity-discovery-design.md`.

### 3.3 Plan IR + apply (P1 design)

| Item | Lock |
|------|------|
| Artifact | Versioned **JSON**; `--placement PATH` |
| Core | Full **layer partition**: contiguous ranges → `backend_id` \| `cpu` |
| Apply | **Native** into load (`dev_layer` / bufts); not `-ts` as authority |
| Knobs when on | Plan wins; `-ts`/`-fit`/`-ot` ignored with **warning** |
| Overrides | `overrides[]` with **full apply in architecture** (delivered in P1) |
| Heat section | Inline per-layer; `status` full\|partial\|none; apply never requires heat |
| split_mode | In plan; default `layer`; refuse illegal global tensor |
| Failure | Fail-loud; optional auto-CPU-offload / warn-degrade |

Detail: `docs/research/placement-plan-ir-apply-design.md`.

### 3.4 Heat (P2 design intent)

| Item | Lock |
|------|------|
| Prerequisite | Productize **`layer_rollup` from `node_timings`**; stop publishing device-stub as `layers[]` |
| Generator default | **TG primary** for hot/cold; PP optional second plan |
| Policy sketch | Hot layers → faster backends; cold → slower / 8 GB as VRAM fillers within usable budgets |
| Missing heat | Capacity-only plans remain valid (`heat.status=none`) |

Detail: `docs/research/placement-heatmap-inputs-inventory.md`.

### 3.5 RPC multi-GPU / TP-unit (P3 design)

| Item | Lock |
|------|------|
| Sequencing | **B first**, **A target** (seq C) |
| Shape B | N × `rpc_device`; GRAPH_COMPUTE_ALL when enabled; mixed VRAM forever B |
| Shape A | `kind=rpc_tp_unit`; logical id + `members[]`; internal tensor + **specialized AR** |
| A gates | Equal VRAM, same family, **N=2**, AR OK, **opt-in** |
| Usable (A) | Sum(member usable) − tp pad; not single-GPU semantics |

Detail: `docs/research/placement-rpc-tp-unit-architecture.md`.

---

## 4. Phased implementation plan

### Phase overview

| Phase | Name | Depends on | Outcome |
|-------|------|------------|---------|
| **P0** | Capacity discover | — | Usable capacity records for local + RPC |
| **P1** | Plan IR + native apply | P0 | Load from plan JSON; overrides work; add-on CLI |
| **P2** | Heat import + generator | P1 + heat rollup | Trustworthy heat; hot/cold plans |
| **P3** | Shape A TP-unit | P1 (P2 optional) | Eligible equal-VRAM pairs as logical device |
| **P4** | Polish / extensions | P1+ | Stages, MoE heat, UX, tuning |

Implementation of a phase may be multiple PRs; acceptance is per phase below.

---

### P0 — Capacity discovery

**Goal:** Client can build a capacity inventory without trusting raw free MiB alone.

**Status (2026-07-17):** **Client P0 shipped** on branch `path-g-gpipeline-assembly-line` (issue 08). Design detail + ops notes: `docs/research/placement-capacity-discovery-design.md` §10.

**Deliverables**

- Capacity record schema (`schema_version`, `backend_id`, `kind`, total/free, `static_pads`, caps, `reported_at`, optional metadata/perf_class).
- RPC batch endpoint capacity query (legacy fallback to `GET_DEVICE_MEMORY`).
- Local fill of same record shape.
- Client `table` reserves + optional `auto` projection mode → `usable_weight_mib`.
- Discover CLI/env gated (add-on); dump JSON inventory for ops.

**Shipped surface**

| Item | Notes |
|------|--------|
| Code | `common/placement-capacity.{h,cpp}`; CLI in `common/arg.cpp` |
| CLI | `--placement-discover`, `--placement-inventory PATH` (+ `LLAMA_ARG_PLACEMENT_*`) |
| RPC wire | `rpc_fetch_mode=legacy_n_get_device_memory` (N× `GET_DEVICE_MEMORY`); batch opcode later |
| Tests | `tests/test-placement-capacity.cpp` |
| Smoke | Romulus ROCm + Triton `:50054`/`:50055` (legacy docker rpc-server) |

**Acceptance**

- [x] Romulus-style topology (local ROCm + 1+ RPC) produces stable `backend_id`s across restart if endpoint list unchanged.
- [x] Multi-device / multi-record RPC: N records with `rpc://host:port#i` (P0: N× memory query per endpoint; dual single-GPU endpoints on Triton verified; same-process `-d A,B` uses same expand path).
- [x] `usable_weight` < free when pads/reserves non-zero; formula version `table-v1` in dump.
- [x] Missing endpoint ⇒ error list; default no "full topology OK" flag (`topology_complete=false`, exit 2).
- [x] Classic server without placement flags: **zero** discover side effects.

**Still open under P0 umbrella / follow-on**

- True batch capacity opcode + HELLO cap (server pads / names / perf_class).
- `reserve_mode=auto` (currently falls back to table).

**Non-goals for P0:** plan apply, heat, TP-unit kind.

---

### P1 — Plan IR + native apply

**Goal:** Deterministic deploy from a versioned plan file.

**Status (2026-07-17):** **Layer-range + tensor override apply shipped** (issues 09–10). Design: `docs/research/placement-plan-ir-apply-design.md` §12.

**Deliverables**

- Plan JSON schema (assignments, overrides, split_mode, capacity snapshot fields, heat section may be empty).
- Validator: full layer partition, no double-count backends, illegal tensor refuse.
- Native apply into model load path.
- `--placement PATH` (+ env mirror); generate helper for capacity-only packer (greedy by usable_weight is enough).
- Plan-wins warnings for classic knobs.
- Fail-loud apply re-discover; optional degrade flags stubbed or implemented behind off-default switches.

**Shipped surface (issues 09–10)**

| Item | Notes |
|------|--------|
| Code | `common/placement-plan.*`; `layer_devices` + plan `tensor_buft_overrides` |
| CLI | `--placement PATH` / `LLAMA_ARG_PLACEMENT` |
| Overrides | Applied; **override wins** for matched tensors; CLI `-ot` ignored |
| Tests | `tests/test-placement-plan.cpp` |

**Acceptance**

- [x] Same plan + same topology → same layer→device assignment (log or debug dump).
- [x] Plan placing a minority of layers on 8 GB RPC and majority on 24 GB local loads without OOM on smoke model (romulus docker RPC + ROCm; Qwen3.5-0.8B).
- [x] Conflicting `-ts` does not change assignment (warning only).
- [x] Tensor `overrides[]` (e.g. expert/embedding patterns → cpu) applied as specified.
- [x] Missing `backend_id` at apply ⇒ refuse load.
- [x] Placement off ⇒ classic path for fixed argv without `--placement`.

**Non-goals for P1:** heat-aware packing quality, Shape A, perfect reserve numerics.

---

### P2 — Heat synthesis + heat-aware generate

**Goal:** Two-phase probe → serve using real per-layer heat.

**Deliverables**

- `llama-gpipe-profiler` (or shared lib): `layer_rollup` from `node_timings`; honest `heat.status`.
- Generator: TG-primary hot-on-fast / cold-on-slow within usable budgets; write plan with inline heat.
- Document probe baseline (capacity plan or equal pack) and re-probe triggers (topology/model/quant change).

**Acceptance**

- [ ] Heatmap `tasks.tg.layers.length == n_layer` (or explicit partial) on a multi-backend run with telemetry.
- [ ] Generated plan differs from pure capacity pack on heterogeneous speed (when heat full); still respects usable caps.
- [ ] Apply of heat-generated plan does not require live profiler.
- [ ] Stub/device-count “layers” never labeled `heat.status=full`.

**Non-goals for P2:** MoE `experts_active`, continuous replan, Shape A.

---

### P3 — RPC TP-unit (Shape A)

**Goal:** Eligible equal-VRAM dual-GPU RPC endpoints can appear as one logical backend with internal tensor+AR.

**Deliverables**

- Discover `kind=rpc_tp_unit`, logical id, `members[]`, sum usable − tp pad.
- Server internal TP + specialized AR gate; opt-in; refuse A if AR missing.
- Plan IR / native apply bind ranges to logical id; double-count guard vs physical ids.
- Shape B remains default and only mode for mixed VRAM.

**Acceptance**

- [ ] Equal-VRAM N=2 endpoint can load under A with specialized AR active (log/metrics).
- [ ] Mixed 24+8 cannot enable A.
- [ ] Butterfly-only cannot enable A.
- [ ] Outer plan still layer-rails logical unit with local HIP (T4-style), no client meta across RPC.
- [ ] Opt-out returns to Shape B without breaking P1 plans that use physical ids.

**Non-goals for P3:** N>2 TP-units, cross-node AR, replacing Path C B path.

---

### P4 — Polish and extensions

**Goal:** Operational depth without new core architecture.

**Candidates (priority order flexible)**

- GPipe **stage** view mapped to layer ranges (if needed for Path-D ops).
- MoE / SSM / MTP heat models; dual PP+TG scores.
- Reserve table numeric tuning from production; static pad catalog on servers.
- Override vs range conflict matrix polish; better CLI names.
- Optional degrade UX hardening; alias auto-rebind policies (still fail-loud default).
- Align D4.11–D4.14 Pareto work as **generators** of plan IR only.

**Acceptance:** per-item as pulled into a sprint; none block calling the control plane “architecturally complete” after P1–P3.

---

## 5. Explicit non-goals (this plan)

| Non-goal | Owner / note |
|----------|----------------|
| AllReduce kernel / provider speedups | AllReduce map / `ALLREDUCE-OPTIMIZATION-PLAN.md` |
| Global heterogeneous client `-sm tensor` | Rejected pooling strategy |
| Distributed AR over RPC as first expansion | Out of scope unless destination redrawn |
| Continuous replan every request | Replan on load / explicit admin only |
| Replacing classic knobs when placement off | Back-compat |
| Layer-2 matmul/kernel optimization | Other efforts |
| Shipping all phases to close the wayfinder map | Map closes on **this document** |

---

## 6. Cross-links and ownership

| Concern | Owner |
|---------|--------|
| Where weights/layers live; usable VRAM; plan deploy; heat place; TP-unit packaging | **This plan / placement map** |
| AR providers, tensor-rail science, residual AR gap | AllReduce map |
| Path-D GPipe/RPC overlap performance | Path-D / pipeline-plus docs |
| GRAPH_COMPUTE_ALL weighted multi-device (Shape B) | ADR 0004 + this plan P0–P1 |

---

## 7. Suggested implementation order (post-map)

1. P0 capacity record + RPC batch + client usable (table first, auto second).  
2. P1 schema + validator + native apply + `--placement` + capacity packer.  
3. Heat rollup fix in profiler → P2 generator.  
4. P3 TP-unit when dual equal-VRAM CUDA RPC is a product priority.  
5. P4 as needed.

Optional: ADR for TP-unit before P3 coding (not required by this lock).

---

## 8. Decisions index (ticket 07 grilling)

1. Phase ladder **A** (P0…P4 as above; overrides in P1 not deferred architecture).  
2. P2 heat ranking default: **TG primary**, PP optional.  
3. Map complete when **this file is locked** and ticket 07 resolves.

---

## 9. Revision

| Version | Date | Note |
|---------|------|------|
| 1 | 2026-07-17 | Initial lock from wayfinder tickets 01–07 |
