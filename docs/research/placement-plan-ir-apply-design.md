# Plan IR and deterministic apply — design notes

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line (impl: `path-g-gpipeline-assembly-line`)  
**Wayfinder ticket:** [Design plan IR and deterministic apply semantics](../../.scratch/placement-control-plane/issues/05-design-plan-ir-and-apply.md)  
**Status:** design locked (HITL grilling 2026-07-17); **P1 layer-range apply shipped** (issue 09; overrides = issue 10)  
**Inputs:** capacity discovery design, knobs inventory, heatmap inputs inventory  
**Feeds:** TP-unit ticket 06, plan doc ticket 07

---

## Standing rules (carried)

- **Add-on, not replacement** — classic `-ts` / `-fit` / `-ot` / `-ngl` / `-sm` / `--rpc` unchanged when placement is off.
- **Layer rail** for multi-backend / multi-RPC; no global heterogeneous client tensor as pooling strategy.
- **Capacity** uses stable `backend_id` (`rpc://host:port#i`, `local:…`) from discover; apply re-discovers and **fail-loud** if ids missing.

---

## Goals

1. Versioned, inspectable **plan artifact** as source of truth for placement when the add-on is on.
2. **Deterministic apply** into model load (not “best effort proportions”).
3. Layer-range primary assignment; **full tensor overrides** supported in v1 apply when present.
4. Heat scores **inline** in the plan when available; plans still valid with null heat (capacity-only).
5. Escape hatches remain for non-placement runs; when a plan is active, knobs do not silently win.

---

## 1. Primary assignment unit

**v1 core:** contiguous **layer ranges → `backend_id`** (or explicit `cpu`).

Example:

```json
"assignments": [
  { "layer_start": 0,  "layer_end": 12, "backend_id": "rpc://127.0.0.1:50051#0" },
  { "layer_start": 12, "layer_end": 64, "backend_id": "local:ROCm0" }
]
```

- Half-open ranges `[start, end)` recommended.
- **Full partition:** every layer in `[0, n_layer)` appears in exactly one range (or cpu). No gaps, no overlaps.
- Output / special layers: plan states assignment explicitly; engine defaults (e.g. embeddings on CPU) must not contradict the plan without validation error.
- Proportions (`-ts` style) may be used by a **generator** only; they are **not** the plan of record.

---

## 2. Apply path: **native** into load

When `--placement PATH` (or env) is active:

1. Parse + validate plan (schema, coverage, backends).
2. Re-run capacity discover; match `backend_id`s; check usable weight vs assigned layers (per reserve mode in snapshot or plan).
3. **Native apply:** set per-layer device / buffer placement for `load_tensors` (and related paths) **from the plan**, not by first writing `-ts` as source of truth.
4. Apply **tensor overrides** from the plan (v1 required capability — see §5).
5. Load model; on violation → **refuse** (default).

**Not v1 primary path:** compile-only to `-ts` (may still **dump** equivalent debug argv for operators, but load authority is the plan map).

**Implementation note (for later):** extend or bypass the proportion cut in `llama_model_base::load_tensors` so `dev_layer[il]` comes from the plan’s range table; device list order must resolve `backend_id` → `ggml_backend_dev_t`.

---

## 3. Conflict with classic knobs

When a placement plan is **active**:

| Knob | Behavior |
|------|----------|
| `-ts`, `-fit`, conflicting `-ngl` | **Ignored**; **warning** logged (plan wins) |
| `-ot` on CLI | **Ignored** with warning unless future explicit `--placement-allow-cli-ot` (default off) — plan’s own `overrides` are the channel |
| `-sm` | Plan’s `split_mode` wins (see §6); warn if CLI differs |
| `--rpc` / `-dev` | Still establish topology; must be **superset** of plan backends or discover fails |

When placement is **off:** all classic knobs behave as today.

---

## 4. Heat in the plan IR

**Inline per-layer heat** section in every plan document:

```json
"heat": {
  "status": "full" | "partial" | "none",
  "task": "tg" | "pp" | null,
  "schema_version": 1,
  "source": "path or omit",
  "layers": [
    { "idx": 0, "ms": 1.2, "rank": 0 },
    ...
  ]
}
```

| `heat.status` | Meaning |
|---------------|---------|
| `full` | One score per model layer (trustworthy rollup) |
| `partial` | Some layers only |
| `none` / empty `layers` | Capacity-only or manual plan; **still valid** |

- **Apply never requires heat** — only assignments + validation.
- Generators: capacity packer; or hot-on-fast / cold-on-slow when `status=full` (and backend perf known).
- Today’s stub heatmap must not be written as `full`; generators set `none` or `partial` until layer_rollup exists.

---

## 5. Tensor overrides (v1 apply required)

Plan includes:

```json
"overrides": [
  { "match": "regex or tensor name", "backend_id": "cpu" | "rpc://…#0" | "local:…" }
]
```

- **v1 apply must implement** override application (buffer/device selection for matched tensors), not schema-only.
- Empty `overrides: []` is valid.
- Overrides must not break layer ownership invariants without validation error (document conflict rules at implement: e.g. expert tensors to CPU while layer stays on GPU is allowed; moving a full layer’s weights to another GPU via override while range says otherwise → error or explicit “override wins for those tensors” flag — **recommend: override wins for matched tensors, validation warns**).

This is stronger than the old map phasing hint “P4 only”; user lock for this ticket: **full override apply in v1**.

---

## 6. Split mode in the plan

```json
"split_mode": "layer"
```

- Default for multi-backend plans: **`layer`**.
- **`tensor`** only if the plan’s backend set is a legal same-process tensor group (not HIP+RPC fleet pooling).
- Apply **refuses** plans that request illegal global tensor over mixed RPC/HIP.
- CLI `-sm` does not override the plan when placement is on (warn if different).

---

## 7. Failure policy

### Default: **fail-loud**

Refuse to load / refuse to apply when:

- Schema invalid; layer coverage incomplete (gap/overlap)
- `backend_id` missing on re-discover
- Assigned weights exceed `usable_weight` for a backend (under declared reserve mode)
- Illegal `split_mode` for topology
- Override cannot be bound to a known backend

No silent fallback to `--fit` or proportional `-ts`.

### Optional switches (off by default)

| Switch | Behavior |
|--------|----------|
| Auto CPU-offload | Overflow layers/tensors that do not fit → CPU; must **emit amended plan** or structured log of what moved (no silent mystery) |
| Warn-and-degrade | Attempt classic fit/ts-like recovery after warning; **not** deterministic; for bring-up only |

Mid-load OOM after successful validation remains a hard process failure (pre-checks reduce likelihood).

---

## 8. Artifact and CLI

| Item | Decision |
|------|----------|
| Format | **Versioned JSON** file (`schema_version`) |
| Apply | `--placement PATH` (name finalizable at implement) |
| Generate | `--placement-generate` / discover+plan tool writing the same JSON |
| Env | Optional mirror of path / enable (capacity ticket) |
| Diffable | Yes — ops review in git |

### Sketch top-level plan document

```json
{
  "schema_version": 1,
  "created_at": "...",
  "model": { "path": "...", "n_layer": 64 },
  "split_mode": "layer",
  "capacity_snapshot_at": "...",
  "reserve_mode": "table",
  "reserve_model_version": 1,
  "backends": [ { "backend_id": "...", "usable_weight_mib": 7000, ... } ],
  "assignments": [ { "layer_start": 0, "layer_end": 12, "backend_id": "..." } ],
  "overrides": [],
  "heat": { "status": "none", "task": null, "layers": [] }
}
```

---

## 9. Determinism checklist

| Property | Rule |
|----------|------|
| Same plan + same discover ids + same model | Same `dev_layer[]` / override bindings |
| Device list order | Irrelevant if resolve is by `backend_id` |
| Free memory drift | Re-check at apply; refuse if insufficient |
| Heat | Does not change apply of a fixed assignment section |

---

## 10. Relationship to Pareto / D4.11–D4.14

One apply story: **plan IR → native apply**. Heat-aware placement is a **generator** that writes assignments (+ inline heat). No second silent governor path that bypasses the plan file in v1. Server-side Pareto tickets should target producing or consuming this IR, not a parallel split mechanism.

---

## 11. Phasing vs this lock

Map phasing hint was P1 layer IR, P4 tensor overrides. **This ticket locks v1 apply for both layer ranges and tensor overrides.** Implementation may still stage delivery, but the **architecture** requires override apply as part of plan support—not a future schema surprise.

Capacity discover (ticket 04) remains P0 dependency for generated plans; manual plans can still list `backend_id`s if topology is known.

---

## 12. Implementation status (issue 09)

**Code:** `common/placement-plan.{h,cpp}`; `llama_model_params.layer_devices` / `n_layer_devices` in `include/llama.h`; apply in `src/llama-model.cpp` `load_tensors`; wire-up in `common/common.cpp` (`common_placement_prepare`).

| Item | Status |
|------|--------|
| Plan JSON parse + schema_version | Done |
| Full partition validate (gap/overlap) | Done |
| `split_mode=layer` default; refuse illegal tensor on mixed local+RPC | Done |
| Empty `overrides` / `heat.status=none` | Done |
| Non-empty `overrides` apply (issue 10) | Done - maps to `tensor_buft_overrides` |
| Re-discover at apply; missing `backend_id` fail-loud | Done |
| Plan wins: warn and ignore `-ts` / `--fit` / CLI `-sm` | Done |
| Native per-layer map (not proportion `-ts` authority) | Done via `layer_devices` |
| CLI | `--placement PATH` / `LLAMA_ARG_PLACEMENT` |
| Tests | `tests/test-placement-plan.cpp` |

### Operator example

```bash
# plan: minority layers on 8 GB-class RPC, rest on local
./build/bin/llama-cli -m model.gguf \
  --placement /path/to/plan.json \
  --rpc 127.0.0.1:50051 \
  -fit off -c 128 -n 4 -p "Hi" --single-turn -no-cnv -v
```

Logs: `using explicit layer_devices map`, `layer N assigned to device X (plan)`.

Smoke (2026-07-17): Qwen3.5-0.8B (25 layers) — layers 0-2 on `rpc://127.0.0.1:50051#0` (romulus docker 8 GB), 3-24 on `local:pci:…` ROCm; also CPU+local plan. Triton dual-RPC load hit legacy server crash (unrelated proto/image); local RPC path OK.

### Issue 11 — capacity packer

| Item | Behavior |
|------|----------|
| CLI | `--placement-generate PATH` (+ env); optional `--placement-generate-only` (write + exit) |
| Input | Live discover inventory + `llama_model_n_layer_all` from `--model` |
| Pack | Proportional layers by `usable_weight_mib` (deterministic: usable desc, then `backend_id`); each usable backend gets ≥1 layer when `n_layer` allows |
| Output | Plan JSON with full partition, `heat.status=none`, empty `overrides`, backends snapshot |
| Load | Same process sets `--placement` to generated path unless generate-only |

```bash
# write plan only
./build/bin/llama-cli -m model.gguf --rpc host:port \
  --placement-generate /tmp/plan.json --placement-generate-only

# generate + load in one command
./build/bin/llama-cli -m model.gguf --rpc host:port \
  --placement-generate /tmp/plan.json -c 256 -n 8 -p "Hi" --single-turn --no-warmup
```

### Issue 10 — tensor overrides

| Item | Behavior |
|------|----------|
| Apply | Each `overrides[]` entry: `match` (regex) + `backend_id` → `llama_model_tensor_buft_override` |
| Empty `overrides` | Still valid |
| Conflict policy | **Override wins** for matched tensor names; layer ranges still apply to unmatched tensors (warn at prepare) |
| Unknown override `backend_id` | Fail-loud at apply re-discover |
| CLI `-ot` | Ignored with warning when `--placement` active |
| CPU target | Uses `ggml_backend_cpu_buffer_type()` (same path as classic `-ot …=CPU`; mmap may use host buft — prefer `--no-mmap` for pure CPU) |

Example:

```json
"overrides": [
  { "match": "token_embd", "backend_id": "cpu" },
  { "match": "blk\\..*\\.ffn_.*_exps", "backend_id": "cpu" }
]
```

---

## Decisions index (grilling)

1. Primary unit: contiguous layer ranges → backend_id  
2. Apply: **native** into load_tensors (not compile-to-ts as authority)  
3. Plan wins over -ts/-fit/-ot (warn)  
4. Heat: **inline** per-layer; null/none allowed (capacity-only valid)  
5. Failure: **fail-loud** default; optional auto-CPU-offload / warn-degrade switches  
6. Artifact: versioned JSON + `--placement PATH`  
7. Coverage: full partition of layers (no implicit gaps)  
8. `split_mode` in plan; default layer; refuse illegal tensor  
9. **overrides[]** with **v1 full apply required**  
