# Placement knobs and failure modes inventory

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line  
**Wayfinder ticket:** [Inventory placement knobs and failure modes](../../.scratch/placement-control-plane/issues/01-inventory-placement-knobs.md)  
**Scope:** code + docs inventory only; no measurements.  
**Sibling map context:** `.scratch/placement-control-plane/assets/CONTEXT-FROM-ALLREDUCE-SPINOFF.md`

---

## Summary

Today's multi-backend placement is a **stack of load-time knobs** that assign layers/tensors by **device-list index and free-MiB proportions**. There is no versioned plan, no capacity protocol beyond a free/total snapshot, and no heat-aware replan. That is adequate for homogeneous same-process multi-GPU; it is **inadequate** as primary UX for heterogeneous multi-RPC fleets (HIP client + CUDA RPC + 8 GB cards + multi-node).

| Knob | CLI | What it actually controls | Primary or override today |
|------|-----|---------------------------|---------------------------|
| Device set / order | `--rpc`, `-dev` / `--device`, `--list-devices` | Which backends exist and **index order** for every other knob | Topology (primary) |
| Split rail | `-sm` / `--split-mode` | `none` / `layer` / `row` / `tensor` | Rail choice (primary) |
| Layer/tensor shares | `-ts` / `--tensor-split` | Normalized proportions → contiguous layer slices (layer) or tensor segments (tensor/row) | **Primary assignment** |
| GPU layer budget | `-ngl` / `--gpu-layers` | How many layers from the **end** of the stack stay on GPU; rest CPU | Offload ceiling |
| Main GPU | `-mg` / `--main-gpu` | Single device in `none`; intermediate/KV hints in `row` | Narrow |
| Auto-fit | `-fit` / `--fit`, `-fitt`, `-fitc`, `-fitp` | Fills **unset** `ngl`/`ts`/buft overrides from projected free memory | Soft auto (default on) |
| Tensor buft override | `-ot` / `--override-tensor` (ticket `-op` = this) | Regex → buffer type; first match wins | Escape hatch / fit internal |
| MoE CPU helpers | `-cmoe`, `-ncmoe` | Syntactic sugar for expert-weight overrides to CPU | Escape hatch |
| RPC multi-device | `--rpc-multidevice` / env | Path C `GRAPH_COMPUTE_ALL` on multi-GPU RPC endpoint | Server-side sched; weights still client |

**Plan IR must replace as primary UX:** proportion-only `-ts`, free-MiB `--fit`, and ad-hoc `-ot` for bulk placement.  
**Plan IR should keep as escape hatches / debug:** `-ot`, `-ngl` ceiling, `-dev`/`--rpc` topology, `-sm` rail, optional `--fit` when no plan file, MoE CPU helpers.

---

## 1. Device list and ordering

### CLI

- `--rpc SERVERS` — register RPC endpoints as backend devices (`common/arg.cpp` `add_rpc_devices`).
- `-dev` / `--device` — comma-separated device names (or `none`); null-terminated device array into model params.
- `--list-devices` — print name + free/total memory and exit.

### Default selection (`llama_prepare_model_devices` in `src/llama.cpp`)

When `params.devices` is null (no `-dev`):

1. Enumerate backends; skip CPU/ACCEL for the GPU list.
2. Collect **RPC** devices separately from local discrete GPUs; dedupe local GPUs by `device_id`.
3. **Insert RPC servers first**, then discrete GPUs, then iGPU only if no discrete GPU (RPC alone does not suppress iGPU).
4. Rationale in code: RPC first "to minimize network transfers" (early pipeline stages on remote → activations flow toward local).

When `-dev` is set, order is **exactly** the user list (null-terminated). Tensor mode with explicit devices builds one **meta** device over that list.

`LLAMA_SPLIT_MODE_NONE` keeps only `main_gpu` index into the prepared list.

### Implications

- **`-ts` indices follow this order**, not "fastest first" or "largest VRAM first".
- Path-D recipes that assume `RPC0, ROCm0` match default RPC-first ordering; reordering with `-dev` silently reinterprets every `-ts` component.
- Free memory printed at prepare time is a **snapshot**, not reserved usable capacity.

Sources: `src/llama.cpp` (~125-275), `common/arg.cpp` (`parse_device_list`, `add_rpc_devices`, `--list-devices`).

---

## 2. Split modes (`-sm`)

| Mode | Enum | Behavior | Fit | Notes |
|------|------|----------|-----|-------|
| `none` | `LLAMA_SPLIT_MODE_NONE` | One GPU (`-mg`) | yes | |
| `layer` | `LLAMA_SPLIT_MODE_LAYER` | Pipeline: contiguous layer ranges + KV with those layers | yes | **Default**; Path-D / multi-RPC production rail |
| `row` | `LLAMA_SPLIT_MODE_ROW` | Deprecated row-split dense weights | fit cannot change weight alloc | Prefer `tensor` |
| `tensor` | `LLAMA_SPLIT_MODE_TENSOR` | Meta device; tensor + KV split; AllReduce between subgraphs | **not implemented** | Experimental; architecture allow-list; needs FA + non-quant KV |

Docs: `docs/multi-gpu.md`. Topology constraints for HIP+RPC vs same-process CUDA: `docs/research/allreduce-topology-feasibility.md`.

**Heterogeneous fleet rule (already locked elsewhere):** global client `-sm tensor` over mixed HIP + RPC is **not** the pooling strategy. Layer rail owns cross-backend VRAM aggregation; tensor/AR only on same-process same-backend islands (or future RPC TP-unit).

---

## 3. Tensor split (`-ts` / `--tensor-split`)

### Parsing

- Comma-separated floats into `params.tensor_split[i]` for `i < llama_max_devices()` (`common/arg.cpp`).
- Values are **proportions**, not hard layer indices. `3,1` ≡ `75,25` after normalize.

### Layer-mode assignment (`llama_model_base::load_tensors` in `src/llama-model.cpp`)

1. If all split components are 0 → **default split = free memory** per device (`ggml_backend_dev_memory`). If free/total are 0, fall back to host memory numbers.
2. Else copy user `-ts` into `splits[]`.
3. Cumulative sum, then normalize to `[0,1]` cut points.
4. `n_gpu_layers` selects a suffix of the layer stack for GPU (`i_gpu_start = max(n_layer_all+1 - n_gpu_layers, 0)`).
5. Layer `il` maps to device index via  
   `upper_bound(splits, (il - i_gpu_start) / act_gpu_layers)`.
6. Input embeddings stay on CPU; output layer uses the same mapping at `il == n_layer_all`.

So `-ts` defines **contiguous proportional slices** of the offloaded layer range in **device-list order**. It does **not**:

- name devices (`RPC0=...`),
- pin absolute layer IDs (`blk.12-blk.20`),
- distinguish weight VRAM vs KV vs compute workspace,
- prefer fast devices for "hot" layers.

### Tensor / row mode

- `-ts` feeds split buffer types / meta tensor segment boundaries (`make_gpu_buft_list`, meta split state).
- Even split if omitted in tensor mode (docs); layer mode free-memory default does not apply the same way once meta is built.

### Operational recipes (cluster docs)

| Pattern | Example | Intent |
|---------|---------|--------|
| Bias compute to fast local GPU | `-ts 10,90` with RPC-first devices | ~10% layers on 8 GB RPC, ~90% on 24 GB ROCm |
| Multi-RPC VRAM props | `-ts 36,24,24,16` | 4-device proportional |
| 72B survival | `-ts 10,90` + heavy CPU offload | Avoid putting majority share on 8 GB RPC |

Sources: `src/llama-model.cpp` (~1240-1298), `docs/multi-gpu.md`, `PIPELINE.md`, `rpc-patch/docs/RPC_PATH_AB_OPTIMIZATION_REPORT.md`, `docs/hot-paths-analysis.md`.

---

## 4. GPU layers (`-ngl`)

| Value | Meaning |
|-------|---------|
| `auto` / default `-1` | Fit may choose; model path treats negative as "all layers" when not fit-overridden (`n_gpu_layers()` → `n_layer_all+1`) |
| `all` / `-2` | Force full offload budget |
| integer `N` | Offload last `N` layers (plus output handling); earlier layers CPU |

Interacts with `-ts`: only the offloaded suffix is split across GPUs; CPU layers are outside the proportion game.

**Fit constraint:** if user already set `n_gpu_layers` away from default, fit **refuses** to change it (`common/fit.cpp`).

---

## 5. Fit (`--fit` family)

### Flags

| Flag | Role |
|------|------|
| `-fit on\|off` | Default **on**. Run `common_fit_params` before `llama_model_load_from_file` (`common/common.cpp` init). |
| `-fitt` / `--fit-target` | Per-device free margin in MiB (default **1024 MiB** each); single value broadcasts. |
| `-fitc` / `--fit-ctx` | Minimum context fit is allowed to leave when shrinking ctx. |
| `-fitp` / `--fit-print` | Print projected memory breakdown. |
| `llama-fit-params` app | Standalone fit helper binary. |

### Algorithm (high level, `common/fit.cpp`)

1. **Hard abort** if `split_mode == TENSOR`.
2. Project per-device model+context+compute via no_alloc load path (`common_get_device_memory_data`).
3. Compare projected use to **free** memory minus margins.
4. If already OK → no changes.
5. Else try reduce context, then assign **layers per device** (`ngl_per_device`), writing:
   - `n_gpu_layers` = sum of per-device layers,
   - `tensor_split[id] = n_layer` counts (still interpreted as proportions after normalize),
   - `tensor_buft_overrides` for MoE **partial-layer overflow** (regex patterns → overflow buft / CPU).
6. Only mutates params that still equal **defaults** (except ctx when 0 / fit-owned). User-set `tensor_split`, `n_gpu_layers`, or existing `tensor_buft_overrides` → exception / abort path.

### Why fit is weak for this fleet

| Gap | Detail |
|-----|--------|
| Free ≠ usable | No explicit KV/FA/slot/graph/TP workspace reserve model beyond flat MiB margin |
| No performance class | Will fill a slow 8 GB card if free MiB allows — can **hurt** TG (see D7.15) |
| No heat | Cannot hot-on-fast / cold-on-slow |
| No-ops when knobs set | Production recipes set `-ts` + `-ngl` → fit often logs and changes nothing |
| Not for tensor rail | Tensor mode must be hand-sized |
| Opaque overrides | Emits regex buft overrides operators rarely inspect |
| Snapshot race | Free memory at fit time ≠ free at steady-state multi-slot server |

With `-ts` + `-ngl` already set, docs explicitly note fit often makes **no changes** (`RPC_PATH_AB_OPTIMIZATION_REPORT.md`).

---

## 6. Override tensor (`-ot`) — ticket name `-op`

There is **no** primary CLI `-op` for placement. The ticket's `-op` maps to:

- **`-ot` / `--override-tensor`** — tensor buffer-type overrides (placement escape hatch), and/or
- incidental **op offload** toggles (host tensor ops), which are not a placement plan.

### Mechanism

- CLI: `pattern=BufferType,...` parsed in `parse_tensor_buffer_overrides` (`common/arg.cpp`).
- Buffer type names come from `ggml_backend_buft_name` of each device buffer type (must match exactly; unknown type prints available list and throws).
- Stored as `llama_model_tensor_buft_override { pattern, buft }` null-terminated.
- At load (`llama_model_loader`): for each tensor name, **first regex match** wins; CPU override goes through extra CPU buft selection; warns if mmap + CPU override.

### Related sugar

- `-cmoe` / `--cpu-moe` — all expert FFN weights → CPU (prebuilt override).
- `-ncmoe N` / `--n-cpu-moe` — experts for first N blocks → CPU.
- Fit generates additional overflow patterns when a device cannot hold full layers (MoE fractioning).

### Why overrides feel flaky

| Failure mode | Why |
|--------------|-----|
| Wrong buffer type string | Names are backend-specific; typos fail hard or surprise |
| Regex footguns | `std::regex_search`; over-broad patterns move unintended tensors |
| First match only | Order matters; later rules never apply |
| No layer-range IR | Operators encode policy as opaque regex lists |
| Interaction with fit | User `-ot` blocks fit from installing its own overrides |
| mmap + CPU | Silent perf trap without `--no-mmap` |
| Does not reorder pipeline stages | Only changes **weight buffer type**; sched still follows buffers — easy to create illegal/slow graphs if patterns fight `-ts` |

---

## 7. Layer rail vs tensor rail (interaction matrix)

| Concern | Layer (`-sm layer`) | Tensor (`-sm tensor`) |
|---------|---------------------|------------------------|
| Primary share knob | `-ts` → layer cut points | `-ts` → tensor segments on meta device |
| Multi-RPC Path-D | Supported (activations copy) | Default device scan can pull HIP+RPC into one meta group — **no specialized AR**; treat as N/A for fleet pooling |
| Fit | Supported | Hard error |
| `-ot` | Works; used by fit for MoE | Works but no substitute for plan |
| 8 GB card use | Via small `-ts` share or free-mem default | Only if inside a same-backend meta group; not via cross-RPC tensor |
| KV placement | With layer owner | Split across meta members (non-quant KV required) |

Path C multi-GPU **inside one RPC process** (`GRAPH_COMPUTE_ALL` + weighted SET_TENSOR) is a **third** path: server sched follows weight buffers; client still chooses weights. That is not the same as client `-sm tensor` across RPC. See ADR 0004 / `docs/path-d-spec.md`.

---

## 8. Concrete failure modes (heterogeneous multi-RPC)

Documented or code-evident; severity for **placement control plane** motivation.

| ID | Failure mode | Mechanism | Evidence / locus |
|----|--------------|-----------|------------------|
| F1 | **Index/order footgun** | `-ts` component `i` is device `i` after RPC-first default or `-dev` reorder | `llama_prepare_model_devices`; silent wrong bias |
| F2 | **Proportion ≠ intention** | Operators think in layer counts or GiB; engine normalizes ratios over offloaded suffix only | `load_tensors` upper_bound math |
| F3 | **Free-MiB default overfills slow GPU** | Omitted `-ts` uses free memory; smaller quant → more layers on 8 GB straggler → TG collapse | D7.15: Q4_K_M vs Q6_K −46% TG; root cause placement, not kernel |
| F4 | **Fit disabled by success path** | Setting `-ts`/`-ngl` (normal ops) freezes fit | `fit.cpp` default checks; Path B report |
| F5 | **Fit OOM/under-RPC** | 72B with wrong ts put majority on 8 GB RPC; workarounds `-ngl 0 --fit on` or hand ts; fit+ngl0 can show ROCm+CPU only (RPC not engaged) | `rpc-path-b-tracking.md`, multi-node remus notes |
| F6 | **Margin-only capacity** | `--fit-target` is flat MiB, not KV/FA/slots/graph | `fit_params_target` default 1 GiB |
| F7 | **No inspectable plan** | Assignment lives in argv + debug logs (`layer N assigned to device X`); no versioned artifact to reapply | load debug logs only |
| F8 | **Override soup** | Bulk policy via `-ot` regex; conflicts with fit; hard to review in runbooks | loader regex loop |
| F9 | **Quant change invalidates split** | Same `-ts` after quant/model change moves absolute bytes per device | D7.15; hot-paths 30/70 MoE notes |
| F10 | **Tensor rail misuse** | `-sm tensor` across HIP+RPC or multi-node | topology feasibility T1/T3 N/A |
| F11 | **Manual matrix explosion** | Each topology × model × quant × ctx × parallel needs hand-tuned ts | Path B+/D gate scripts (`-ts` sweeps) |
| F12 | **CPU offload cliff** | Dense 72B needs ~8 GB+ CPU weights; cli manual ngl + RPC CUDA graphs can crash; server+fit path differs | pathb-72b notes |
| F13 | **No heat feedback** | Profiler/heatmap exist as research (ADR 0004b, D4.11+) but do not drive `-ts` | tickets D4.11–D4.14 still open |
| F14 | **Multi-endpoint opacity** | Multiple `--rpc` hosts: only proportional slices; no first-class "backend kind" or perf_class | device list is flat |

---

## 9. What a successor plan IR must replace vs keep

### Replace as **primary** UX

| Today | Why replace |
|-------|-------------|
| Hand `-ts` proportions | Non-inspectable, order-dependent, not layer-absolute, not heat-aware |
| Default free-mem split | Optimizes fill, not latency; strands policy for 8 GB cards |
| `--fit` as the auto story | Default-only mutation, free-MiB margins, no perf/heat, no tensor mode |
| Regex `-ot` for bulk placement | Unreviewable policy language |

### Keep as **escape hatches / building blocks**

| Mechanism | Role under plan IR |
|-----------|-------------------|
| `--rpc` / `-dev` / device registry | Topology discovery input; plan names backends, does not invent them |
| `-sm layer` (and future backend-local tensor) | Rail selection; plan chooses layer-first for cross-backend |
| `-ngl` / CPU residual | Explicit overflow policy when plan cannot place all weights |
| `-ot` / `-cmoe` / `-ncmoe` | Debug and rare expert pins; plan may *compile* to overrides for apply v1 |
| `--fit` | Emergency / bootstrap when no plan file; never sole production path |
| `--fit-target`-like margins | Evolve into capacity **reserves** fields (KV, graph, slots) |
| Path C weighted SET_TENSOR / GRAPH_COMPUTE_ALL | Apply path for multi-device RPC endpoints; plan emits weights per logical device |
| Load-time layer assignment engine | Apply backend: plan IR → deterministic `dev_layer[]` (or equivalent), fail-loud if mismatch |

### Minimum semantics the plan IR needs (inputs to later tickets)

From knob gaps alone (not a full IR design):

1. **Named backends** (stable ids), not bare indices.
2. **Absolute layer ranges** (or ordered stages) per backend, not only ratios.
3. **Capacity object** richer than free MiB (usable weights vs KV vs workspace; perf_class).
4. **Deterministic apply** with validation (refuse / fail-loud vs silent proportion drift).
5. **Override channel** for tensor-level exceptions without abandoning the plan.
6. **Version field** so probe-run heatmaps and serve plans are auditable.

---

## 10. Code / doc map (quick index)

| Area | Path |
|------|------|
| CLI knobs | `common/arg.cpp` (`--rpc`, `-dev`, `-ot`, `-ngl`, `-sm`, `-ts`, `-fit*`, `-cmoe`) |
| Params defaults | `common/common.h` (`tensor_split`, `fit_params*`, `split_mode`, devices) |
| Fit implementation | `common/fit.cpp`, `common/fit.h` |
| Init calls fit then load | `common/common.cpp` `common_init_result` |
| Device prepare | `src/llama.cpp` `llama_prepare_model_devices` |
| Layer assignment | `src/llama-model.cpp` `load_tensors` |
| Override apply | `src/llama-model-loader.cpp` tensor buft override loop |
| Multi-GPU user doc | `docs/multi-gpu.md` |
| Path-D / RPC ops | `PIPELINE.md`, `docs/hot-paths-analysis.md`, `rpc-patch/docs/*` |
| Topology rails | `docs/research/allreduce-topology-feasibility.md` |
| Path C placement | `docs/adr/0004-server-side-scheduling.md`, `docs/path-d-spec.md` |
| Heatmap future | `docs/adr/0004b-profiler-architecture.md`, path-d tickets D4.11+ |

---

## 11. Bottom line for the map

Existing knobs can **express** a static layer split on a known device order, and fit can **guess** a fill when the operator leaves knobs unset. They cannot **reliably**:

- query usable capacity across local + multi-RPC,
- deploy a reviewable deterministic plan,
- keep 8 GB cards in the pool without becoming the latency bottleneck,
- or consume profiler heat for hot-on-fast / cold-on-slow.

That is exactly the destination gap for the placement control plane: **capacity → plan IR → apply**, with today's flags demoted to topology, rail, and escape hatches.
