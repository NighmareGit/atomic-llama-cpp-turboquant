# RPC TP-unit architecture — design notes

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line  
**Wayfinder ticket:** [RPC TP-unit architecture decision (Shape A vs B)](../../.scratch/placement-control-plane/issues/06-rpc-tp-unit-architecture.md)  
**Status:** locked by HITL grilling (session 2026-07-17)  
**Inputs:** capacity discovery design, plan IR design, topology feasibility (T2/T4/T5), spin-off context  
**Feeds:** plan doc ticket 07; capacity `kind=rpc_tp_unit` when P3

---

## Standing rules (carried)

- Placement control plane is an **add-on**; classic multi-device RPC remains valid.
- Outer fleet pooling is **layer rail** across logical backends; no global HIP+RPC client tensor.
- **8 GB cards** stay in the pool — never stranded by collapsing mixed VRAM into a fake single device.
- Specialized AllReduce only on **same-process same-backend** islands (topology T2 / future T4).

---

## Decision summary

| Topic | Lock |
|-------|------|
| Target architecture | **Shape A** (TP-unit) for eligible equal-VRAM multi-GPU RPC |
| Sequencing | **C:** Shape **B first** (visible multi-device / Path C); **A later (P3)** when gates met |
| Mixed 24+8 | **B only** — never Shape A |
| A eligibility | Strict: equal VRAM, same family, **N=2** first, specialized AR, opt-in |
| Discover / plan IR | One logical `backend_id` + **members[]**; layer ranges bind to logical id |
| A capacity | **Sum** of member `usable_weight` − tp workspace pad; documented caveats |
| B vs A coexistence | B forever for non-eligible and as opt-out on eligible nodes |
| AR | Specialized AR **required** to enable A; else stay B |

---

## Shape definitions

### Shape B — visible multi-device (v1 / default multi-GPU RPC)

- Client discover: N × `kind=rpc_device` for `rpc-server -d GPU0,GPU1,…`.
- Plan IR: layer ranges (and overrides) to each physical `rpc://host:port#i`.
- Runtime: existing paths — per-device GRAPH_COMPUTE and/or **GRAPH_COMPUTE_ALL** (Path C) when caps/env allow; weights follow buffers.
- No hidden tensor world on the client for that endpoint.
- **This is the multi-GPU RPC story for P0–P2 and for all non-eligible endpoints forever.**

### Shape A — RPC TP-unit (target, P3+)

- Client discover: **one** `kind=rpc_tp_unit` logical backend, plus `members[]` (physical devices) for debug/heat.
- `backend_id` form (sketch): `rpc-tp://host:port` (exact string at implement).
- Inside the rpc-server process: **tensor split + specialized AllReduce** across members (same-process CUDA meta / NCCL or internal for n=2).
- Client outer plan: **layer** ranges onto the logical unit as if it were one fat device among others (7900, other RPC, …).
- Weight capacity advertised ≈ **sum of member usable** minus TP workspace pad — **not** “identical to one GPU with sum(V) VRAM” for activations/KV.

---

## Sequencing (lock C)

```
P0 capacity (B-shaped multi-device records)
P1 plan IR + native apply (ranges → rpc_device | local)
P2 heat-aware generate (optional)
P3 Shape A: discover kind + server internal TP + plan binds to rpc_tp_unit
```

- Placement plane **does not block** on A engineering.
- P3 is architecture-committed, not “maybe someday” — but **not** day-1 code for map close (map closes on plan doc).
- Implementing A is out of scope for resolving this ticket.

---

## Shape A eligibility (strict)

An endpoint may advertise `tp_unit_eligible` / offer A only if **all** hold:

| Gate | Rule |
|------|------|
| VRAM class | **Equal** member `total` within a small tolerance (implement constant); reject mixed 24+8 |
| Backend family | Homogeneous (e.g. all CUDA on that server) |
| Count | **N = 2** for first A generation (internal AR is 2-device today; NCCL may allow more later under a new decision) |
| AllReduce | **Specialized** AR init succeeds (NCCL and/or CUDA internal); butterfly-only ⇒ **do not enable A** |
| Opt-in | Server flag / config / discover field; client or plan must **choose** A (no silent hide of devices) |
| Process | Single rpc-server process owns both devices |

If any gate fails → Shape B only.

---

## Discover and plan IR integration

### Capacity record (P3 extension of ticket 04)

```text
kind: rpc_tp_unit
backend_id: rpc-tp://host:port
members: [ { backend_id: rpc://host:port#0, ... }, { ...#1 } ]
usable_weight_mib: sum(member usable) - tp_workspace_pad
static_pads: include tp_workspace / AR scratch as documented
tp_unit_eligible: true  // on physical endpoint view when gates pass
```

- While A not opted in: same hardware still appears as N × `rpc_device` (B).
- Optional: discover returns **both** views (physical + eligible logical); plan picks one mode per endpoint — recommend **one active view per endpoint** per plan to avoid double-counting VRAM.

### Plan IR (ticket 05)

- Assignments use the **logical** `backend_id` for A.
- Native apply resolves logical id → TP-unit backend registration (server-side multi-GPU tensor device), not N client RPC devices for that endpoint.
- Heat: prefer rollup attributed to logical unit; members[] available for debug.

### Double-count rule

A plan must not assign layers to **both** `rpc-tp://host:port` and `rpc://host:port#0` for the same physical GPUs.

---

## Usable capacity semantics (A)

```
usable_weight_unit ≈ Σ usable_weight(member_i) − tp_workspace_pad
```

Document for operators:

- Tensor split increases **weight** capacity roughly with N; **KV / activations / workspace** do not scale the same way.
- Packer uses `usable_weight_unit` for **how many layers** the unit can hold, not as a promise of single-GPU latency.
- Never advertise raw `sum(free)` without static pads + client reserves (hybrid usable from ticket 04 still applies per member before sum).

---

## AllReduce policy (A)

| Outcome | Action |
|---------|--------|
| NCCL or internal AR OK | Allow A |
| Only butterfly / comm_ctx null | **Refuse A**; remain B; log reason |
| Operator forces `ALLREDUCE=none` | A disabled |

Aligns with AllReduce map: AR science on T2-like islands; A is the productized form of “TP inside RPC process” (topology T4).

---

## Coexistence matrix

| Endpoint | Mode |
|----------|------|
| Single GPU RPC | `rpc_device` only |
| Multi-GPU mixed VRAM | Shape **B** only |
| Multi-GPU equal VRAM, not opted in | Shape **B** (default) |
| Multi-GPU equal VRAM, opted in, AR OK | Shape **A** available |
| Multi-GPU equal VRAM, AR fail | Shape **B** only |

Path C GRAPH_COMPUTE_ALL remains the B multi-device accelerator; A is a different packaging (logical device + internal tensor), not a rename of GRAPH_COMPUTE_ALL.

---

## Non-goals

- Cross-RPC or HIP+CUDA client tensor world
- Auto Shape A on mixed VRAM to “use all memory”
- Replacing placement plan IR with server-only hidden partition
- Implementing A in this wayfinding session

---

## ADR offer

Sequencing (B first, A target) is **reversible** enough that a formal ADR is optional. Offer ADR if implementers want a durable `docs/adr/` record before P3 coding; not required to close this ticket.

---

## Decisions index (grilling)

1. **C:** Shape A target; Shape B first for v1 multi-GPU RPC  
2. Strict A gates: equal VRAM, same family, N=2, specialized AR, opt-in  
3. Discover/plan: logical `backend_id` + `members[]`  
4. B remains for non-eligible and as opt-out  
5. Usable ≈ sum(member usable) − tp pad, with caveats  
6. Specialized AR required to enable A  

---

## Implementation notes (issue 14, 2026-07-19)

**Module seam (client control plane):**

| Symbol | Role |
|--------|------|
| `placement_tp_unit_eval_gates` | Pure gates: N=2, equal VRAM (+/-512 MiB), same family, specialized AR, opt-in |
| `placement_inventory_apply_tp_units` | Annotate or collapse inventory; default Shape B |
| `placement_make_rpc_tp_backend_id` | `rpc-tp://host:port` |
| `placement_plan_ids_tp_double_count` | Refuse plan using both logical + physical ids for same endpoint |

**CLI (opt-in add-on):**

- `--placement-tp-unit` / `LLAMA_ARG_PLACEMENT_TP_UNIT` - request Shape A collapse
- `--placement-tp-ar-ok` / `LLAMA_PLACEMENT_TP_AR_OK` - operator asserts specialized AR is available (no automatic NCCL probe yet; butterfly-only must not set this)

**Usable weight:** `sum(member usable_weight_mib) - PLACEMENT_TP_WORKSPACE_PAD_MIB` (default 512). Documented on the logical record `warnings[]`.

**Apply:** `rpc-tp://host:port` resolves to the first registered RPC device for that endpoint (device 0). Outer plan stays **layer** rail (local HIP + logical unit). Client does not open a meta tensor world across RPC.

**Runtime honesty:** Full internal tensor-split + specialized AR packaging inside `rpc-server` remains server-side work; the control plane refuses A without AR assertion and never enables A on mixed VRAM. Path C multi-device on the endpoint remains the B accelerator when A is off.

**AR gate interpretation:** AC "specialized AR init OK" is implemented as a required operator assertion flag until automatic AR capability probe exists. Without the flag, Shape A collapse is refused with a clear reason.
