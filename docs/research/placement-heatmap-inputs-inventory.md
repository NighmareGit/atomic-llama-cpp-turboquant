# Profiler and heatmap inputs for placement — inventory

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line  
**Wayfinder ticket:** [Inventory profiler and heatmap inputs for placement](../../.scratch/placement-control-plane/issues/02-inventory-profiler-heatmap-inputs.md)  
**Scope:** code + docs + on-disk sample artifacts; no new profiler runs.  
**Related:** ADR 0004b, D4.7 research, D4.10 telemetry fields, D4.11–D4.14 Pareto tickets (planned), `docs/rpc-multi-backend-pipeline-plus/TELEMETRY.md`

---

## Summary

| Source | Status | Usable for hot-on-fast / cold-on-slow today? |
|--------|--------|-----------------------------------------------|
| `llama-gpipe-profiler` + `heatmap.json` schema v1 | **Shipped** | **Partial** — wall PP/TG + GPU metadata yes; **true per-layer heat no** in practice |
| `rpc_msg_server_telemetry` (6 fields) | **Shipped** wire + JSONL | **Device-level** timings; `layer_assignments` is a **stub** (identity per device), not layer→GPU map |
| `node_timings` (sched-trace / server-telemetry) | **Shipped** when telemetry-trace on | **Best raw signal** for layer rollup (`l_out-N.*` / `blk.N.*`); **not** synthesized into heatmap by gpipe-profiler today |
| `layer_rollup` / `op_categories` in TELEMETRY.md | **Documented only** | Not emitted by `write_heatmap_json` |
| `llama-pipeline-profiler` + `diagnose.json` | **Shipped** (B+ gates) | Overlap/straggler/gates — not a placement plan input |
| Pareto governor D4.11–D4.14 | **Ticketed, not built** | Intended consumer of heatmap; no server apply path yet |
| `experts_active` (MoE) | **Designed in D4.7, not implemented** | Missing for MoE-aware heat |

**Bottom line:** the **probe-run half** of two-phase placement has tools and a schema shell; the **heat signal required for layer-range plan IR is not yet trustworthy**. Two-phase workflow is blocked on (1) real per-layer (or per-stage) heat synthesis, (2) a consume/apply path into plan IR, (3) PP vs TG / MoE policy choices — not on inventing a greenfield profiler binary.

---

## 1. Artifact catalog

### 1.1 `llama-gpipe-profiler` (primary placement-oriented tool)

| Item | Detail |
|------|--------|
| Path | `tools/llama-gpipe-profiler/` |
| ADR | `docs/adr/0004b-profiler-architecture.md` (Accepted) |
| Role | Drive **pp** and **tg** tasks; optional RPC + server telemetry; emit **task-stratified** `heatmap.json` |
| CLI (placement-relevant) | `-m`, `--rpc`, `-ts`, `--tasks pp,tg`, `-p`/`-n`/`-r`, `--warmup`, `-o` heatmap, `--out-dir`, `--trace`, `--server-telemetry`, `-ngl`, `-sm`, `-ctk`/`-ctv` |
| Does **not** | Apply placement; replan server; emit plan IR; rank backends by perf_class beyond crude straggler |

**Heatmap schema v1** (ADR + writer in `llama-gpipe-profiler.cpp` `write_heatmap_json`):

```
schema_version, generated_at, git_sha
model: { path, n_layers, param_count_b, ctx_size }
tasks.pp | tasks.tg: {
  n_prompt_tokens | n_gen_tokens, wall_ms, tps,
  layers: [ { idx, ms, gpu_id } ],   // intended per-layer
  kv?: { read_ms, write_ms, evict_count },
  kv_source?: "estimated",
  draft_*? (MTP)
}
gpu_metadata: [ { id, name, backend, vram_total_mib, pci_*, source, compute_*? } ]
summary: { straggler_gpu, kv_source?, gate_*? }
```

Additive-only evolution promised; unknown keys ignored by consumers.

### 1.2 Server telemetry frame (`rpc_msg_server_telemetry`)

| Field | Type | Intended consumer use | Actual fill quality (code) |
|-------|------|----------------------|----------------------------|
| `device_timings_us[]` | per device | Straggler / GPU cost | Filled from per-device compute when collect runs |
| `layer_assignments[]` | per device slots | Layer→GPU map for Pareto | **Stub:** `layer_assignments[i] = i` — "split boundary info not exposed" (`ggml-rpc.cpp` `collect_telemetry`) |
| `copy_times_us[]` | peer pairs | Copy vs compute | **Zero** — not tracked yet |
| `device_meta[]` | name, vram, backend, pcie | Hardware context | Startup snapshot |
| `kv_read_times_us[]` / `kv_write_times_us[]` | per slot | Hot KV pages | Often **empty** (`n_slots = 0`; "server has no direct kv cache context") |

Wire path: gated by `GGML_RPC_SERVER_TELEMETRY` / `--telemetry` / capability `RPC_CAP_SERVER_TELEMETRY`; appended on `GRAPH_COMPUTE` and `GRAPH_COMPUTE_ALL` responses; client writes `server-telemetry.jsonl`.

**Critical mapping bug/limitation for placement:** gpipe-profiler treats `layer_assignments[i]` as **layer index i → gpu_id**, but the array length is **n_devices**, not `n_layers`. On-disk heatmaps that are "non-empty" typically show **1–2 layer entries** matching device count (e.g. 64-layer model, `layers: [{idx:0,...}]` only). That is **not** usable hot-layer ranking.

Sources: `ggml/src/ggml-rpc/ggml-rpc.cpp` (~748–759, ~3688–3723), `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp` (~924–938).

### 1.3 Per-node timings (`node_timings`)

| Channel | When | Content |
|---------|------|---------|
| `sched-trace.jsonl` | `GGML_SCHED_TRACE` / `--telemetry-trace` | Per-split backend phases; optional `node_timings: [{name, us}]` with `l_out-N.op` / `blk.N.op` |
| `server-telemetry.jsonl` event `node_timings` | Server telemetry + per-node callback | Same naming; optional aggregated avg/min/max/count |

**Documented** rollup (`TELEMETRY.md`):

- `layers[]` per tensor name  
- `layer_rollup[]` per transformer layer index (sum of nodes)  
- `op_categories[]` attn/ffn/norm/other  

**Implemented in gpipe-profiler heatmap writer:** **no** `layer_rollup` / `op_categories` emission. Client path only aggregates **per-backend** min/avg/max from sched-trace for `gpu_metadata.compute_*` and straggler heuristics (`parse_sched_trace_for_client_timing`).

So the **highest-fidelity heat source for placement already exists as raw JSONL**, but the **placement-facing product (heatmap.json layers[]) does not consume it correctly**.

### 1.4 `llama-pipeline-profiler`

| Item | Detail |
|------|--------|
| Path | `tools/llama-pipeline-profiler/` |
| Role | Path B+/B+6 gates, diagnose, overlap metrics, RPC preflight |
| Outputs | `sched-trace.jsonl`, `diagnose.json`, regression rows — **not** schema-v1 heatmap as primary |
| Relation | Sibling tool; ADR 0004b explicitly keeps it; gpipe-profiler copies helpers rather than sharing a lib |

Useful for **probe infrastructure patterns** (RPC register, trace env, multi-rep), not as the plan IR heat contract.

### 1.5 Unified telemetry CLI

`docs/rpc-multi-backend-pipeline-plus/TELEMETRY.md` — `--telemetry`, `--telemetry-trace`, sample interval, aggregate mode — applies to `llama-server`, `rpc-server`, `llama-gpipe-profiler`. Env fallbacks still work (`GGML_RPC_SERVER_TELEMETRY`, `GGML_SCHED_TRACE`, …).

### 1.6 Pareto / governor path (planned)

| Ticket | Intent | Status |
|--------|--------|--------|
| D4.11 research | Governor integration, 80/20 hot/fast formalization | Open / future sprint |
| D4.12 prototype | Server reads heatmap, validates hot-on-fast | Blocked on D4.11 |
| D4.13 ADR 0005 | Placement model + fallback | Blocked on D4.12 |
| D4.14 implement | Apply to layer split in llama-server | Blocked on D4.10 (done) + D4.13 |

Intended policy sketch (tickets): hot ~20% layers on fast ~20% GPUs; cold as VRAM filler / RAM; fallback to static `--rpc-split` / `-ts`. **No code path** from heatmap → model load params today.

This map's destination (**plan IR + apply**) should **absorb** that consumer role rather than assuming a separate server-only Pareto subsystem — reuse heatmap as **input**, decide apply ownership in design tickets.

### 1.7 On-disk samples (this tree)

- Many `heatmap.json` under `profiler-out/` and `benches/path-b-plus/`: schema present, **`layers: []` or device-count stub entries**.
- Wall `tps` / `wall_ms` for pp and tg are populated and trustworthy for **task-level** comparison.
- `gpu_metadata` often lists RPC devices (`source: rpc_server`) plus client ROCm (`source: client`) with optional `compute_avg_us`.

---

## 2. What can feed hot-on-fast / cold-on-slow **today**

| Signal | Granularity | Quality | Placement use |
|--------|-------------|---------|---------------|
| `tasks.tg.tps` / `tasks.pp.tps` | Whole config | Good | Compare candidate static plans offline; not layer pins |
| `summary.straggler_gpu` | Device | Weak (max device time) | Hint which backend is slow under **current** `-ts` |
| `gpu_metadata.vram_total_mib` + name/backend | Device | Good for inventory | Capacity-adjacent; not heat |
| `gpu_metadata.compute_*` (client sched-trace) | Device | Medium | Crude perf_class proxy for **local** backends |
| `tasks.*.layers[]` as written | Broken / device-stub | **Do not trust** for layer ranking | Blocker |
| Raw `node_timings` names | Op / layer | High if collected | Offline scripts can roll up; not productized |
| KV fields | Slot | Usually empty | Not ready for hot-KV co-placement |
| `experts_active` | Layer×token | Missing | MoE expert placement N/A |

**Circular placement problem:** probe runs use a **current** `-ts` / device order. Measured layer times are **conditioned on that placement** (RPC latency, pipeline bubbles, quant). Heat for "intrinsic layer cost" vs "cost on this backend" are different quantities; plan IR design must choose which heat model (absolute layer cost estimate vs relative under a baseline plan).

---

## 3. PP vs TG and MoE caveats (do not solve here)

| Topic | Fact | Placement implication |
|-------|------|------------------------|
| **PP vs TG** | Separate heatmap tasks; PP batch-parallel + KV write heavy; TG serial + KV read / compute bound (D4.7 §6) | Serve plan may optimize **TG** first (latency) with PP-aware capacity; or store **two heat maps** and pick by workload |
| **Same layers, different heat** | A layer slow in TG may not dominate PP | Single scalar `ms` per layer is insufficient if one plan serves both phases |
| **MoE routing** | TG expert subset varies per token; PP sees aggregate (D4.7) | Without `experts_active`, treat MoE layers as uniform expert cost or use weight size only |
| **SSM / hybrid / MTP** | MTP draft stats optional on TG heatmap; SSM not special-cased | Cold/hot may need graph-type tags later; out of probe v1 detail |
| **Quant / FA interaction** | Smaller quant can move more layers onto slow 8 GB (placement knobs inventory F3 / D7.15) | Heatmaps are **not portable across quant** without re-probe |
| **CUDA graph** | Per-node timing skipped in pure graph replay (TELEMETRY.md) | TG heat may under-instrument; sample non-graph or capture phase |

---

## 4. Gaps blocking two-phase **probe → serve**

Ordered by dependency for the placement control plane destination.

### G1 — No trustworthy per-layer heat in `heatmap.json`

- Stub `layer_assignments` + writer that treats device array as layers.
- Documented `layer_rollup` from `node_timings` not implemented in heatmap synthesis.
- **Need:** rollup pipeline (client and/or server) → `tasks.{pp,tg}.layers[n_layers]` with real `ms` (and optional `gpu_id` = where it **ran** under probe plan).

### G2 — No plan-IR consumer / apply path

- D4.11–D4.14 not built; server does not read heatmap at load.
- Manual re-entry is still `-ts` proportions (see knobs inventory).
- **Need:** heat → ranked layer list → named backend assignment → versioned plan → deterministic apply (design tickets 05+).

### G3 — Probe placement circularity

- Must load **some** split to measure; bad baseline biases heat.
- **Need:** policy for baseline probe plan (e.g. capacity-proportional layer plan from P0/P1 without heat; then refine) and optional second probe.

### G4 — Backend identity mismatch

- Heatmap `gpu_id` / `gpu_metadata.id` are **run-local integers**, not stable backend names (RPC endpoint, ROCm0, TP-unit).
- Multi-RPC samples show duplicate `CUDA0` names across endpoints.
- **Need:** stable backend ids in heatmap and plan IR (ties to capacity discovery ticket).

### G5 — Cross-process heat incomplete

- Local client layers: sched-trace backend timing.
- RPC layers: server telemetry / node_timings.
- Merge logic for a **global** layer vector is incomplete (id numbering bugs visible in samples: server ids 5–8 + client 0).

### G6 — KV / copy / expert fields hollow

- Blocks advanced co-placement (hot KV with hot layers, copy-aware cuts).
- v1 layer-only plan can proceed without them if G1 fixed; keep as fog for later phases.

### G7 — Operational workflow not productized

- No first-class "probe mode" in `llama-server` that writes heatmap then reloads with plan.
- No staleness rules (when re-probe: model/quant/ctx/topology change).
- Map destination allows replan on load / explicit admin only — still needs a defined artifact handoff.

### G8 — PP/TG policy undecided

- Not a code gap alone: design must choose which task's heat drives the serve plan (or dual plans).

---

## 5. What two-phase can reuse without greenfield

| Phase | Reuse |
|-------|--------|
| **Probe drive** | `llama-gpipe-profiler` CLI + `--tasks pp,tg` + `--trace --server-telemetry` |
| **Raw capture** | `server-telemetry.jsonl`, `sched-trace.jsonl` (`node_timings`) |
| **Schema shell** | `heatmap.json` schema_version 1 — extend, do not replace |
| **Gate tooling** | pipeline-profiler diagnose for overlap health of a candidate plan |
| **Policy sketch** | D4.11 80/20 / hot-on-fast language — re-home under plan IR, not only server governor |
| **Avoid** | New parallel heatmap format; depending on stub `layers[]` as-is; assuming Pareto tickets ship before plan IR |

**Minimum engineering to unblock heat import (P2 hint, not this ticket's job):**

1. Implement `layer_rollup` from `node_timings` (client+server merge) into `tasks.*.layers[]` with length `n_layers`.  
2. Stop publishing device-stub rows as `layers[]` (or rename to `device_timings[]` so consumers cannot confuse them).  
3. Add stable `backend_id` on gpu_metadata and on layer run location.  
4. Document probe baseline + which task (tg/pp) is authoritative for serve.

---

## 6. Implications for later map tickets

| Ticket | Feed from this inventory |
|--------|---------------------------|
| [Design plan IR and deterministic apply](../../.scratch/placement-control-plane/issues/05-design-plan-ir-and-apply.md) | Heat input shape: prefer `layers[idx].ms` per task; plan may store `heat_ref` path + schema_version; apply must not require live profiler |
| [Design capacity discovery](../../.scratch/placement-control-plane/issues/04-design-capacity-discovery.md) | `gpu_metadata` is not capacity protocol; only hints (vram_total, name). perf_class may start from probe `compute_*` / straggler |
| [Lock PLACEMENT-CONTROL-PLANE-PLAN.md](../../.scratch/placement-control-plane/issues/07-lock-placement-plan-doc.md) | P2 = heatmap import **after** G1-quality heat; P0/P1 can ship capacity+layer plan without heat |
| Pareto D4.11–D4.14 | Treat as **overlapping intent**; this map owns the locked plan — avoid two competing apply stories |

---

## 7. Code / doc index

| Topic | Path |
|-------|------|
| ADR heatmap + CLI | `docs/adr/0004b-profiler-architecture.md` |
| D4.7 design space | `docs/wayfinder/D4.7-profiler-research.md` |
| Telemetry CLI + node_timings + documented rollup | `docs/rpc-multi-backend-pipeline-plus/TELEMETRY.md` |
| Profiler binary | `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp` |
| Pipeline profiler | `tools/llama-pipeline-profiler/` |
| Server telemetry struct + stub assignments | `ggml/src/ggml-rpc/ggml-rpc.cpp` |
| Path-D Pareto tickets | `docs/tickets/path-d-tickets.md` (D4.11–D4.14) |
| Sample heatmaps | `profiler-out/**/heatmap.json`, `benches/path-b-plus/**/heatmap.json` |

---

## 8. Bottom line for the map

**Inputs exist** (profiler binary, schema, RPC telemetry, sched traces, planned Pareto story).  
**Placement-grade heat does not** — until `layers[]` means one row per model layer from real timings (almost certainly `node_timings` rollup), two-phase hot-on-fast remains a design target fed by **wall metrics and operator `-ts`**, not an automatic pipeline.

Plan IR design should:

1. Accept an **optional** heatmap reference with schema_version and task key (`tg` default).  
2. Define behavior when heat is **missing/stale/stub** (capacity-only plan).  
3. Not block P0/P1 on Pareto implementation.  
4. Treat fixing heatmap synthesis as a **prerequisite task for P2**, possibly a small follow-on ticket once plan IR fields are known — not as fog forever.
