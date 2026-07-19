# RPC and local capacity signals — inventory

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line  
**Wayfinder ticket:** [Inventory RPC capacity and multi-device signals](../../.scratch/placement-control-plane/issues/03-inventory-rpc-capacity-signals.md)  
**Scope:** code + docs inventory; no live cluster queries.  
**Related:** ADR 0004 (GRAPH_COMPUTE_ALL), knobs inventory, heatmap inventory, topology feasibility.

---

## Summary

| Signal | Local GPU | Ordinary RPC GPU | Multi-device RPC endpoint (`-d CUDA0,CUDA1`) | Future TP-unit logical device |
|--------|-----------|------------------|---------------------------------------------|-------------------------------|
| free / total bytes | `ggml_backend_dev_memory` | `RPC_CMD_GET_DEVICE_MEMORY` → same API | One query **per** server device index | **Missing** (no logical sum/advertise) |
| device name | backend name (`ROCm0`, `CUDA0`) | Synthetic `RPC0`, `RPC1`, … (global counter) | Same; **N client devices** per endpoint | Missing |
| GPU model / PCI | description + `device_id` (PCI bus) on CUDA/HIP | description = **endpoint string only**; no PCI | Same; telemetry may have names, props path does not | Missing |
| device type | real enum | **Hardcoded GPU** (TODO: not from server) | Same | Missing kind field |
| multi-device cap | N/A | single device typical | HELLO `RPC_CAP_MULTI_DEVICE` + env; `DEVICE_COUNT` | Would need new kind + caps |
| usable VRAM / reserves | **No** | **No** | **No** | **No** |
| perf_class / speed | **No** (fit uses free MiB only) | **No** (`RPC_WEIGHT_SPEED_RATIO` constant unused in live SET_TENSOR policy) | Comment-level 64/36 sketch only | Missing |

**Bottom line (inventory-time):** the client can already list backends and read **raw free/total** for local and each RPC device index. That is enough to drive today's `-ts` / `--fit` fill heuristics and is **not** enough alone for a full capacity protocol (server pads, perf_class, TP-unit).

**Update (2026-07-17, P0 client):** opt-in **placement discover** builds versioned capacity records with stable `backend_id`, client `table-v1` `usable_weight_mib`, and JSON dump (`--placement-discover`). RPC still uses legacy N× `GET_DEVICE_MEMORY` (no batch capacity opcode yet). See `docs/research/placement-capacity-discovery-design.md` §10.

---

## 1. Common device property surface

`ggml_backend_dev_props` (`ggml/include/ggml-backend.h`):

| Field | Meaning |
|-------|---------|
| `name` | Short id (`CUDA0`, `ROCm0`, `RPC3`) |
| `description` | Human string (GPU product name locally; **endpoint** for RPC) |
| `memory_free` / `memory_total` | Bytes |
| `type` | CPU / GPU / iGPU / … |
| `device_id` | PCI bus id when known (local CUDA); often null for RPC |
| `caps` | async, host_buffer, buffer_from_host_ptr, events |

Accessors: `ggml_backend_dev_memory`, `ggml_backend_dev_get_props`, `ggml_backend_dev_name`, …

**CLI:** `--list-devices` prints name + free/total MiB for every enumerated device after `--rpc` registration (`common/arg.cpp`).

**Consumers today:** `llama_prepare_model_devices` log line; default `-ts` free-memory split; `common_fit_params` margins; model load debug.

---

## 2. Local backends

| Backend | Memory source | Notes |
|---------|---------------|-------|
| CUDA / HIP (ggml-cuda) | `cudaMemGetInfo` (device-scoped) | PCI `device_id` filled when available |
| CPU | host free/total | Fit uses as fallback when non-GPU reports 0/0 |
| Other accel | often 0/0 | Fit may skip GPU-typed 0/0 or map to host |

**What local already gives a planner:** free/total snapshot, stable-ish name within a process, PCI id for topology scripts.

**What it does not:** usable budget after KV/FA/slots/graph; long-lived reservation; cross-process contention; relative TFLOPS/perf_class.

---

## 3. RPC protocol signals

### 3.1 Registration path

```
--rpc host:port[,host:port...]
  -> ggml_backend_rpc_add_server(endpoint) per endpoint
  -> RPC_CMD_DEVICE_COUNT
  -> for i in 0..count-1: create ggml_backend_dev (RPC{global_id}, desc=endpoint, device=i)
```

Sources: `common/arg.cpp` `add_rpc_devices`; `ggml_backend_rpc_add_server` (`ggml-rpc.cpp` ~4561–4597).

**Naming:**

- Client device **name:** `RPC0`, `RPC1`, … via process-global `g_rpc_dev_id` (order of `add_server` + per-endpoint device index).
- **Description:** endpoint string only (`192.168.1.10:50051`), not GPU model.
- Same endpoint with 2 server GPUs → **two** client devices, same description, consecutive RPC names.

**Implication:** names are **not stable across process restarts if `--rpc` order changes**. Capacity protocol needs endpoint+device_index (or server-reported UUID) as identity, not bare `RPC3`.

### 3.2 Memory query

| Wire | Request | Response |
|------|---------|----------|
| `RPC_CMD_GET_DEVICE_MEMORY` | `uint32_t device` (index into server `backends[]`) | `free_mem`, `total_mem` (uint64) |

Server: `ggml_backend_dev_memory` on that backend's local device (`rpc_server::get_device_memory`).

Client: `ggml_backend_rpc_device_get_memory` → same numbers surface through `ggml_backend_dev_memory` / props.

**API:** `ggml_backend_rpc_get_device_memory(endpoint, device, free, total)` (`ggml-rpc.h`).

**Failure / zero:** connect fail returns 0/0; fit then refuses to place on GPU-typed 0/0.

### 3.3 Device count

| Wire | Response |
|------|----------|
| `RPC_CMD_DEVICE_COUNT` | `device_count` = `backends.size()` on server |

Used at add_server and when initializing multi-device backend context (`n_devices_on_endpoint`).

### 3.4 HELLO / capabilities

HELLO exchanges proto major/minor/patch and `conn_caps[]`:

| Cap bit | Meaning | When set on server |
|---------|---------|-------------------|
| `RPC_CAP_TRACE_ID` | EVENT_RECORD 20B | transport default |
| `RPC_CAP_MULTI_DEVICE` | Path C multi-GPU / GRAPH_COMPUTE_ALL | only if `rpc_multidevice_env_enabled()` |
| `RPC_CAP_SERVER_TELEMETRY` | telemetry frame on compute responses | telemetry env/CLI |

**HELLO does not carry memory, GPU names, or topology.**

### 3.5 RPC device props gaps (client iface)

| Prop | Implementation |
|------|----------------|
| type | **Always** `GGML_BACKEND_DEVICE_TYPE_GPU` with TODO "obtain from server" |
| device_id | **Not set** (null) |
| supports_op | **Always true** (TODO: remote query) |
| host_buffer | false |
| events | true (RPC event path) |

So the planner cannot distinguish RPC CPU fallback, iGPU, or real GPU class via props alone.

### 3.6 Telemetry meta (side channel, not capacity API)

`rpc_telemetry_device_meta` at server startup: name, `vram_mib` (= total/1MiB), backend_type, pcie_gen/width (**left 0** — "not exposed via ggml api").

Emitted only when server telemetry enabled; not a first-class discover-before-plan call. Heatmap inventory already covers consumption quality.

---

## 4. Multi-device endpoints and GRAPH_COMPUTE_ALL

### 4.1 Server process shape

`rpc-server -d CUDA0,CUDA1` (or default all non-CPU devices) starts one process with `n_devices` backends (`tools/rpc/rpc-server.cpp` `get_devices` + `ggml_backend_rpc_start_server`).

Client view after `--rpc that-endpoint`:

- `DEVICE_COUNT` → 2  
- Two `ggml_backend_dev_t` entries (e.g. `RPC0`, `RPC1`)  
- Each has its own free/total via GET_DEVICE_MEMORY(0) and (1)

### 4.2 GRAPH_COMPUTE_ALL (Path C / ADR 0004 B+)

| Condition (client graph_compute) | Behavior |
|----------------------------------|----------|
| `rpc_multidevice_env_enabled()` AND HELLO multi-device cap AND `n_devices_on_endpoint > 1` | Serialize full graph; `RPC_CMD_GRAPH_COMPUTE_ALL` (or recompute); server `create_multi_device_sched` |
| Else | Per-device `GRAPH_COMPUTE` as usual |

Server advertises multi-device **only when env/flag enables it** — having two GPUs is not enough by itself.

**Weight placement:** ADR and comments describe client weighted SET_TENSOR by speed ratio (e.g. 3090/3070 1.75× → ~64/36 layers). Code defines `RPC_WEIGHT_SPEED_RATIO = 1.75f` and **commented** split formulas; there is **no** general capacity-driven weighted placer wired as the primary path — operators still use client `-ts` across the **visible** RPC devices (or equal splits). Scheduler follows **where weights live** (buffer types), which remains the real control surface.

### 4.3 Equal-VRAM dual-GPU vs mixed 24+8 on one node

| Node shape | Capacity signal today | Placement control plane stance |
|------------|----------------------|--------------------------------|
| **Equal VRAM** e.g. 2×24 or 2×8 | Two free/total nearly equal; no "homogeneous pair" flag | **Shape A TP-unit candidate:** could later advertise **one** logical device with ~summed weight capacity + internal tensor+AR. **Not present** as a backend kind. Shape B: keep two devices + GRAPH_COMPUTE_ALL / layer weights. |
| **Mixed VRAM** e.g. 3090 24 + 3070 8 (or 24+8 generally) | Two very different totals | **Must not** collapse to one TP-unit without explicit policy. Layer-weighted / separate logical devices keep 8 GB in pool without pretending single-GPU semantics. ADR 0004 B+ is this world. |
| **Two processes** (separate ports per GPU) | Two endpoints, one device each | Client layer-rails across endpoints; no server-side ALL; free/total still per endpoint |

**Detection gap:** nothing reports "these two devices share a process / PCIe domain / equal VRAM class." Client can **infer** same endpoint + similar total, but that is heuristic, not protocol.

---

## 5. What the client knows at planning time (checklist)

After `--rpc` + optional `-dev`, before/during model load:

| Known | How |
|-------|-----|
| Endpoint list | CLI |
| Devices per endpoint | `DEVICE_COUNT` at add_server |
| free/total per RPC device | `GET_DEVICE_MEMORY` via `ggml_backend_dev_memory` |
| free/total local | local backend memory |
| Multi-device capable? | HELLO cap after connect (and env on both sides) |
| Proto version / peer_copy / dual socket | HELLO / logs |
| Device order for `-ts` | RPC devices first, then local GPUs (`llama_prepare_model_devices`) |

| Not known (or not trustworthy) | Why it matters for capacity protocol |
|--------------------------------|--------------------------------------|
| **Usable** weight MiB | free ≠ residual after KV, FA workspace, slots, fragmentation, RPC overhead |
| Reserved margins as first-class fields | only client `--fit-target` flat MiB |
| GPU model / SM / arch on RPC props path | cannot rank without telemetry or new query |
| Relative speed / perf_class | hardcoded ratio comment only |
| PCIe link between co-located GPUs | pcie fields zero |
| Whether multi-GPU endpoint is homogeneous | needed for TP-unit eligibility |
| Logical "TP-unit" capacity | would be ~sum weights, not sum free, with workspace caveats |
| Concurrent clients / stolen free | snapshot races |
| Refresh policy | no subscription; free changes after load |
| Op support matrix remote | always-true stub |
| Stable backend UUID | RPC index renumbers |

---

## 6. Gaps for a capacity-discovery protocol

Sized for ordinary RPC GPUs **and** future TP-unit logical device.

### G1 — Usable capacity model

Protocol must return more than free/total, e.g.:

- `total_mib`, `free_mib` (diagnostic)
- `usable_weight_mib` (or formula inputs: free − reserves)
- explicit **reserves**: KV, graph/compute, slots, system, fragmentation pad
- optional `ctx` / `n_parallel` assumptions used for the estimate

Without this, plan IR repeats fit's free-MiB mistakes (knobs inventory).

### G2 — Device identity and metadata

Per logical backend:

- stable `backend_id` (e.g. `rpc://host:port#0` or server UUID)
- `name`, `description` (real GPU product string)
- `backend_family` (CUDA/HIP/…)
- `device_index_on_endpoint`
- optional PCI / NUMA

### G3 — Backend kind

Enum (sketch for design ticket, not locked):

| Kind | Meaning |
|------|---------|
| `local_gpu` | Client-process GPU |
| `rpc_device` | One GPU behind one RPC endpoint index |
| `rpc_multidevice` | Endpoint with N>1, devices still visible (Shape B) |
| `rpc_tp_unit` | **Future:** one logical device, internal multi-GPU (Shape A) |

Today only the first two exist as first-class client devices; multi-device is N× `rpc_device` + optional ALL compute path.

### G4 — Topology / co-location

- Same endpoint ⇒ same process (known)
- Equal-VRAM class flag or per-device totals for client to decide TP-unit eligibility
- Interconnect hint (PCIe gen/width, P2P) for ALL vs TP vs separate endpoints

### G5 — Performance class

- Static table, microbench, or server-reported relative score
- Required for hot-on-fast **and** for non-equal multi-GPU weight bias without magic 1.75 constant

### G6 — Capability bundle

Single discover response should include:

- proto + caps (multi-device, telemetry, …)
- `n_devices`, per-device capacity records
- whether GRAPH_COMPUTE_ALL is live
- whether endpoint is willing to act as TP-unit (future)

### G7 — Refresh and lifecycle

- When to re-query (before plan, before apply, on failure)
- How free changes after weights loaded (plan must use **pre-load** usable estimate, not post-load free)

### G8 — TP-unit advertisement (future)

For Shape A, server or control plane must expose **one** logical device:

- `kind=rpc_tp_unit`
- `member_devices[]` or hidden
- `usable_weight_mib ≈ f(sum totals)` with documented non-equivalence to single-GPU
- `internal_split_mode=tensor` + AR provider constraints
- eligibility: equal VRAM only (map preference), N=2 first, etc.

Ordinary discovery today **cannot** represent this without a new message or synthetic device.

---

## 7. Equal-VRAM vs mixed: capacity protocol implications

| Scenario | Discover should say | Plan implication |
|----------|--------------------|------------------|
| 2× equal VRAM, multi-device env on | Two `rpc_device` **or** optional `rpc_tp_unit` candidate flag | Ticket 06 chooses Shape A vs B; capacity fields must support both views |
| 24+8 same process | Two devices, different totals; **not** TP-unit eligible by default | Layer split / weighted weights; keep 8 GB as cold filler |
| 24 local + 8 RPC | local_gpu + rpc_device | Cross-backend layer rail only (topology T1) |
| Multi-node multi-RPC | N rpc_device | Named backends; no global free pool without plan |

---

## 8. What to reuse vs invent

| Reuse | Role in capacity protocol |
|-------|---------------------------|
| `RPC_CMD_GET_DEVICE_MEMORY` / `DEVICE_COUNT` | Building blocks for free/total |
| HELLO `conn_caps` | Extensible capability bits |
| `ggml_backend_dev_*` on client | Uniform walk of local+RPC after registration |
| Telemetry `device_meta` | Prototype for richer name/vram fields (fix wire, not only post-compute) |
| ADR 0004 multi-device model | Shape B apply path; weights follow buffers |

| Invent (design ticket territory) | Why |
|----------------------------------|-----|
| Usable + reserves schema | free MiB insufficient |
| Discover RPC opcode or control-plane HTTP/file | Need pre-plan bundle without loading model |
| Stable backend_id | RPC0 renumber footgun |
| `backend_kind` + TP-unit record | Destination item 4 |
| perf_class | hot/cold and weighted multi-GPU |
| Homogeneous-pair eligibility | Shape A gate |

Prefer **extending** HELLO or a dedicated `RPC_CMD_GET_DEVICE_INFO` / capacity message over scraping telemetry JSONL for planning.

---

## 9. Code / doc index

| Topic | Path |
|-------|------|
| Props / memory API | `ggml/include/ggml-backend.h` |
| RPC public API | `ggml/include/ggml-rpc.h` |
| GET_DEVICE_MEMORY, DEVICE_COUNT, add_server, caps | `ggml/src/ggml-rpc/ggml-rpc.cpp` |
| Cap bits | `ggml/src/ggml-rpc/transport.h` |
| rpc-server `-d` | `tools/rpc/rpc-server.cpp` |
| Client `--rpc` / `--list-devices` | `common/arg.cpp` |
| Device order | `src/llama.cpp` `llama_prepare_model_devices` |
| Fit free/total use | `common/fit.cpp` |
| Path C decision | `docs/adr/0004-server-side-scheduling.md` |
| Topology rails | `docs/research/allreduce-topology-feasibility.md` |

---

## 10. Bottom line for the map

**Have:** free/total per local and per RPC device index; device count per endpoint; multi-device capability bit; client device list with RPC-first ordering.

**Missing for destination "capacity discovery":** usable VRAM + reserves, stable ids and real GPU metadata on the props path, perf_class, explicit backend kinds, homogeneous multi-GPU / TP-unit advertisement, and a single pre-plan discover response that is not "run fit and hope."

**Equal-VRAM dual-GPU** can be *inferred* today as two similar totals on one endpoint; **protocol should make eligibility explicit** before Shape A. **Mixed 24+8** must remain multi-record capacity so 8 GB cards stay plannable without false single-device pooling.

Feeds directly: [Design capacity discovery protocol](../../.scratch/placement-control-plane/issues/04-design-capacity-discovery.md) (now unblocked with ticket 01) and [RPC TP-unit architecture decision](../../.scratch/placement-control-plane/issues/06-rpc-tp-unit-architecture.md) (blocked on 04).
