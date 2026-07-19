# Capacity discovery protocol — design notes

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line (impl: `path-g-gpipeline-assembly-line`)  
**Wayfinder ticket:** [Design capacity discovery protocol](../../.scratch/placement-control-plane/issues/04-design-capacity-discovery.md)  
**Status:** design locked (HITL grilling 2026-07-17); **P0 client discover shipped** (issue 08)  
**Inputs:** `docs/research/placement-rpc-capacity-inventory.md`, `docs/research/placement-knobs-inventory.md`  
**Feeds:** plan IR / apply (ticket 05 / issue 09+), RPC TP-unit (ticket 06 / issue 14), plan doc (ticket 07)

---

## Standing rule: add-on, not replacement

The placement control plane (capacity → plan IR → apply → optional heat) is an **opt-in add-on**. Existing `--rpc`, `-ts`, `-fit`, `-ot`, `-ngl`, `-sm`, device lists keep **full backwards compatibility**. Default process behavior without placement entrypoints is unchanged.

---

## Goals

1. Know **usable** weight capacity and optional **device class** before planning — not raw free MiB alone.
2. Support local GPUs, ordinary RPC devices, multi-device endpoints; leave room for future TP-unit kind (ticket 06).
3. Keep **8 GB cards** plannable via transparent reserves and multi-record capacity (never silent strand or silent overfill).
4. Survive cluster churn with **dynamic discover** and minimal manual maintenance.

Non-goals for this design: implementing wire opcodes, continuous per-request replan, replacing `--fit` as the default path.

---

## 1. Who computes what (hybrid usable)

| Source | Reports | Does not |
|--------|---------|----------|
| **Backend** (local helper or RPC) | `total_mib`, `free_mib`, **static pads** it can justify without model/ctx knowledge | llama ctx / n_parallel / FA / KV math |
| **Client planner** | `client_reserves` and **`usable_weight_mib`** | Trust a single opaque server `usable` as the only number |

```
usable_weight_mib ≈ free_mib − sum(static_pads) − client_reserves(...)
```

(Exact arithmetic and clamping documented with `reserve_model_version`; never negative usable without error.)

**Static pads (examples, not exhaustive):** fragmentation, fixed RPC/runtime overhead, server-known process reserve. May be zero in early servers; fields still present.

**Client reserves — two modes:**

| Mode | Name | Behavior |
|------|------|----------|
| Default | **`table`** | Versioned parameter table: ctx, n_parallel, kv types, fa on/off, graph pad, per-kind floors (esp. small VRAM). Fast, inspectable. |
| Auto | **`auto`** | Fit-like / no_alloc **projection** for this model+context onto the backend; higher fidelity; costlier. On failure → fall back to `table` + warning unless strict flag requires auto. |

Snapshot must record `reserve_mode` and `reserve_model_version` (and projection inputs when `auto`).

---

## 2. Capacity record (v1 schema sketch)

Logical record (JSON-ish; wire encoding TBD at implement):

### Required

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | int | Start at 1; additive evolution |
| `backend_id` | string | Canonical identity (see §3) |
| `kind` | enum | v1: `local_gpu` \| `rpc_device` (future: `rpc_tp_unit`, …) |
| `total_mib` | uint | |
| `free_mib` | uint | Snapshot at `reported_at` |
| `static_pads` | object | Named MiB pads; sum subtracted before client reserves |
| `caps` | object | e.g. `multi_device`, `telemetry`, proto hints |
| `reported_at` | timestamp | |

### Optional (fill when known)

| Field | Notes |
|-------|-------|
| `display_name` | GPU product or endpoint label |
| `endpoint` | RPC host:port |
| `device_index` | Index on endpoint |
| `backend_family` | CUDA / HIP / … |
| `pci` / `device_id` | When available |
| `perf_class` | Score or enum; **missing ⇒ unknown** (not an error) |

### Client-only (after discover + load knobs)

| Field | Notes |
|-------|-------|
| `client_reserves_mib` | From table or auto |
| `usable_weight_mib` | Derived |
| `reserve_mode` | `table` \| `auto` |
| `reserve_model_version` | string/int |

**perf_class policy:** servers **should** report when implemented; old servers omit → unknown. v1 packing uses usable VRAM; speed ranking uses reported class, heatmap/probe, or operator tiers — **never invent from free MiB**.

**TP-unit:** do not add `rpc_tp_unit` to discover until ticket 06 locks Shape A/B. Multi-device endpoints remain **N × `rpc_device`** records from one batch response.

---

## 3. Identity

### Canonical `backend_id` (required)

| Kind | Form |
|------|------|
| Local | `local:<devname>` and/or `local:pci:<busid>` when PCI known |
| RPC | `rpc://<host>:<port>#<device_index>` |

Derived from **live discover**, not from `RPC0` process order.

### Optional aliases (ops UX)

Config map: `alias → backend_id` (e.g. `rpc-8gb` → `rpc://192.168.1.10:50051#0`).

- **Not required** for cluster restart recovery.
- Plans store **canonical ids** (may also echo aliases for humans).
- Missing alias target: drop/fail that alias; planner still works from discover.

### Apply matching

Re-discover at apply; match by `backend_id`. **Fail-loud** if missing — no silent remap to “current RPC0.”

---

## 4. Wire / composition shape

### Planner view

Uniform list: `CapacityRecord[]` for every backend in scope.

### RPC fetch (batch per endpoint)

One call per endpoint (name TBD, e.g. `GET_ENDPOINT_CAPACITY` / extended device-info):

```
{ endpoint, devices: [ { device_index, total, free, static_pads, display_name, caps, perf_class?, ... } ] }
```

Client expands to N records with `backend_id = rpc://endpoint#i`.

Single-GPU server: `devices.length == 1`.

**Legacy fallback:** only old `GET_DEVICE_MEMORY` → synthesize record with empty/zero static pads + warning.

### Local fetch

Same record shape; no RPC; static pads from local table/helper.

### Not used for capacity body

- HELLO stuffing (caps only remain on HELLO).
- Out-of-band file as the **primary** path (optional ops later).

---

## 5. Refresh rules

| Event | Action |
|-------|--------|
| Before **generating** a plan | Discover (or refresh if past TTL) |
| Before **apply** | Discover again; fail-loud on missing ids / insufficient free vs plan |
| Per token / per request | **No** |
| Operator force | Refresh flag on placement CLI |

- Default TTL: **30–60s** for interactive re-plan (exact default at implement).
- Plan stores `capacity_snapshot_at` + topology hash (set of backend_ids).
- Topology change or expired snapshot → refuse apply until re-plan.
- Free **down** below planned weights → refuse / re-plan; free **up** → OK.

Aligns with map non-goal: no continuous automatic replan every request.

---

## 6. Partial failure

| Situation | Default | Optional |
|-----------|---------|----------|
| Configured `--rpc` / topology incomplete (endpoint down) | **Refuse** full-cluster plan; list `discover_errors[]` | `--placement-partial` (or config): plan over reachable set only |
| Single device within endpoint missing | Treat as endpoint incomplete unless partial | |

Never silently shrink or remap backend_ids.

---

## 7. Enablement (add-on)

| Mechanism | Role |
|-----------|------|
| **Explicit placement CLI** | Primary: e.g. `--placement`, `--placement-plan`, `--placement-discover` (final names with plan IR ticket) |
| **Optional env mirror** | Same opt-in for scripts/labs |
| Classic `llama-server` / cli with only `-ts`/`-fit`/`--rpc` | **No** capacity discover side effects |

---

## 8. Equal-VRAM dual vs mixed 24+8

Unchanged from inventory + this design:

| Topology | Discover | Plan implication |
|----------|----------|------------------|
| Equal-VRAM multi-device one endpoint | N records, similar totals | Ticket 06 may introduce TP-unit; v1 = N `rpc_device` |
| Mixed 24+8 one endpoint | N records, different totals | Multi-record layer/hybrid; **not** one logical device by default |
| Multi-node | One batch per endpoint | Named `backend_id`s across nodes |

---

## 9. Relationship to existing fit / knobs

| When placement add-on **off** | Unchanged fit, ts, ot, ngl |
| When **on** | Discover + plan IR + apply become primary; `-ts`/`-fit`/`-ot` remain **escape hatches** / debug (plan IR ticket details apply compile) |

Capacity discover **reuses** free/total building blocks; **does not** delete `GET_DEVICE_MEMORY` or `--fit`.

---

## 10. Implementation status (P0)

**Issue:** [08-discover-inventory-e2e](../../.scratch/placement-control-plane/issues/08-discover-inventory-e2e.md) (resolved)  
**Code:** `common/placement-capacity.h`, `common/placement-capacity.cpp`  
**Tests:** `tests/test-placement-capacity.cpp`

| Item | Status |
|------|--------|
| Capacity record schema v1 + JSON dump | Done |
| Stable `backend_id` (`local:…` / `local:pci:…`, `rpc://host:port#i`) | Done |
| Local GPU fill + static pad floor (process reserve) | Done |
| Client `table` reserves (`reserve_model_version=table-v1`) → `usable_weight_mib` | Done |
| Opt-in CLI/env; classic argv unchanged | Done |
| Incomplete configured `--rpc` → `discover_errors[]`, `topology_complete=false`, exit 2 | Done |
| Legacy RPC free/total only → pads empty/zero + warning | Done |
| RPC multi-device: expand to N records per endpoint | Done (see wire path below) |
| Batch capacity wire opcode + HELLO cap | **Not yet** (legacy path only) |
| `reserve_mode=auto` projection | Stub: falls back to `table` + warning |
| Plan IR snapshot / apply | Later issues (09+) |

### CLI / env (locked for P0)

| Flag | Env | Behavior |
|------|-----|----------|
| `--placement-discover` | `LLAMA_ARG_PLACEMENT_DISCOVER` | Run discover, print JSON (or write inventory), exit |
| `--placement-inventory PATH` | `LLAMA_ARG_PLACEMENT_INVENTORY` | Write inventory JSON to PATH (implies discover) |

No model path required for discover-only. Exit **0** if `topology_complete`, **2** if any configured endpoint missing/unreachable.

Example (romulus + Triton dual RPC):

```bash
./build/bin/llama-cli --placement-discover \
  --rpc 192.168.8.23:50054,192.168.8.23:50055 \
  --placement-inventory /tmp/placement-inv.json
```

Smoke (2026-07-17): local ROCm 7900 XTX + Triton legacy docker `rpc-server` (3090 `:50054`, 3070 `:50055`) → 3 records, stable ids, `usable < free`, `topology_complete=true`. Servers were **existing** cluster images (not a new batch-capacity build).

### RPC wire path (P0)

Inventory field `rpc_fetch_mode`:

| Value | Meaning |
|-------|---------|
| `legacy_n_get_device_memory` | **Current default.** Per endpoint: registered devices + N× `GET_DEVICE_MEMORY` (same free/total as today). Documented multi-device expand. |
| (future) batch opcode | One response body expanded to N records; HELLO cap bit; optional static pads / display_name / perf_class from server |

P0 does **not** require a new `rpc-server` binary for discover against current cluster workers.

### `table-v1` reserves (client)

```
usable_weight_mib = max(0, free_mib - sum(static_pads) - client_reserves_mib)
```

`client_reserves_mib` (table-v1): graph pad 256 MiB + coarse ctx/parallel KV pad (+ FA scale) + **256 MiB small-VRAM floor** when `total_mib < 10240` (8 GB class). Local process pad default 64 MiB. Not a full fit projection.

### Source map

| Concern | Location |
|---------|----------|
| Types, pure helpers, live discover | `common/placement-capacity.*` |
| CLI gate + dump-and-exit | `common/arg.cpp` (`placement_discover` params in `common/common.h`) |
| Export `ggml_backend_rpc_get_device_memory` via reg | `ggml/src/ggml-rpc/ggml-rpc.cpp` |

---

## 11. Open items (other tickets / fog)

- Wire batch capacity opcode number, packing, HELLO cap (server + client).
- Numeric tuning of table pads / KV estimate (or replace with `auto` via fit/no_alloc).
- `rpc_tp_unit` record shape (ticket 06 / issue 14).
- How plan IR embeds capacity snapshot (ticket 05 / issues 09–11).
- Whether `auto` projection shares code with `common_fit_params` or a thinner no_alloc path.

---

## Decisions index (grilling)

1. Hybrid usable (server static pads + client formula)  
2. Uniform records + RPC batch-per-endpoint  
3. Add-on / full back-compat  
4. `backend_id` transport-address + optional aliases; dynamic discover; fail-loud  
5. v1 field set core+optional  
6. Refresh at plan/apply boundaries + TTL  
7. Client reserves: `table` + `auto` projection  
8. Partial fail: default refuse incomplete topology  
9. `perf_class` optional / unknown if missing  
10. Enable: explicit CLI + optional env  
