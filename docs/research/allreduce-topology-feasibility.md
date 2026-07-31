# Topology feasibility: both rails + AllReduce

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line  
**Wayfinder ticket:** [Topology feasibility for AllReduce](.scratch/allreduce-timing/issues/03-topology-feasibility-allreduce.md)  
**Depends on:** `docs/research/allreduce-providers-inventory.md`, `docs/research/allreduce-tensor-mode-model-allow-list.md`  
**Scope:** code + cluster layout; no live runs.

---

## Definitions

| Term | Meaning |
|------|---------|
| **Layer rail** | `--split-mode layer` (and Path-D GPipe / RPC orchestration). Weights/layers assigned across backends; cross-backend **activation copies**, not meta AllReduce. |
| **Tensor rail** | `--split-mode tensor`. Builds a **meta** device over multiple simple devices; subgraph boundaries use **AllReduce** (NCCL/RCCL/internal/butterfly). |
| **Specialized AR** | CUDA/HIP `comm_allreduce` (NCCL/RCCL/internal). Requires every simple backend in the meta group to pass `ggml_backend_is_cuda` (HIP shares this GUID when built as ggml-cuda/HIP). |
| **Butterfly only** | Meta `allreduce_fallback` via `tensor_copy_async` + ADD. No NCCL/internal. |
| **N/A (tensor)** | Do not run tensor matrix here for diligence; not a failed experiment. |

### Code gates (recap)

1. **Tensor device group** (`llama_prepare_model_devices`):  
   - Explicit `params.devices`: those devices only.  
   - Default: **every non-CPU device** (local GPU **and** RPC) into one meta group.  
2. **Specialized AR** (`ggml_backend_cuda_comm_init`): all backends `ggml_backend_is_cuda`; else `comm_ctx = nullptr` → butterfly every time.  
3. **Internal AR**: CUDA-only body, **exactly 2** devices, sm70+. HIP stubs out.  
4. **RPC**: separate backend reg; **no** `comm_init`; not part of a CUDA NCCL world. Cross-node AR does not exist.

---

## Feasibility matrix

| Topology | Host / roles | Layer rail | Tensor rail | Specialized AR | Covering means for baselines |
|----------|--------------|------------|-------------|----------------|------------------------------|
| **T0** Single GPU main | romulus 7900 XTX only | N/A (or trivial) | N/A | N/A | Peak reference: `-sm none` / one device TG+PP |
| **T1** Main dual mixed | 7900 XTX (HIP) + 3060 Ti as **Docker RPC** (typical Path-D) | **Required** | **N/A** for specialized AR | **No** | Layer + Path-D best config; split timestamps; residual stall. Do not force `-sm tensor` across HIP+RPC. |
| **T1b** Main dual same-process (if 3060 local CUDA + HIP in one process) | Rare / nonstandard | **Required** if used | **Fragile** | **No** specialized (mixed regs); butterfly only if meta builds | Optional curiosity; not first-class AR matrix. Prefer T1 operational shape. |
| **T2** RPC node dual CUDA | 3090 + 3070 same process (client or standalone on that box) | **Required** | **Required** | **Yes** NCCL and/or internal (2-GPU) | Full both-rails matrix + provider A/B. **Primary AR science topology.** |
| **T3** Multi-node Path-D | Client romulus HIP + RPC workers (3060 and/or remus/triton) | **Required** | **N/A** end-to-end | **No** cross-RPC AR | Layer/Path-D multi-backend TG+PP + stalls. Tensor not an end-to-end rail. |
| **T4** Hybrid (planned) | 3090+3070 as TP **unit** + pipeline stages to 7900 (and careful 3060) | **Required** (outer pipeline) | **Per-unit only** on 3090+3070 | AR only **inside** the CUDA pair | After T1–T3 baselines; design later. Not a day-1 baseline row. |
| **T5** Server multi-GPU RPC (`rpc-server -d CUDA0,CUDA1`, GRAPH_COMPUTE_ALL) | triton/remus-style endpoint | Server-side multi-GPU **layer-like** sched (Path C path) | **Not** client `-sm tensor` | Client still does not run meta AR across RPC | Optional later; do not confuse with tensor-rail AR matrix. Measure as layer/RPC variant if at all. |

---

## Topology details

### T1 — Main mixed (Path-D production shape)

```
Client process: ROCm/HIP 7900 XTX
RPC process:    CUDA 3060 Ti (docker, often 127.0.0.1:50051)
```

| Rail | Status | Why |
|------|--------|-----|
| Layer | **Yes** | Documented Path-D / GPipe; layer splits + RPC copies. |
| Tensor default | **Dangerous** | Default device scan puts HIP + RPC into one meta group → no specialized AR; butterfly / broken expectations. |
| Tensor intentional | **N/A** | No same-family multi-GPU group for NCCL/internal. |

**Baseline rows:** single-GPU 7900; layer dual Path-D best; instrumentation on layer path. **No** AR provider matrix on T1.

### T2 — RPC node 3090 + 3070 (clean TP)

```
One process, two CUDA devices (same machine)
```

| Rail | Status | Why |
|------|--------|-----|
| Layer | **Yes** | `-sm layer -ts ...` on this host alone. |
| Tensor | **Yes** | Meta over CUDA0+CUDA1; models allow-listed. |
| AR providers | **Yes** | `GGML_CUDA_ALLREDUCE=nccl|internal|none` (butterfly). Internal needs n=2 (satisfied). |

**How to run tensor cleanly:**

- Prefer **explicit devices** limited to the two CUDA GPUs (avoid accidentally pulling RPC if any).  
- FA on, `-ctk f16 -ctv f16`.  
- Confirm NCCL linked in that build at baseline time.

**Baseline rows:** single-GPU each card optional; layer dual; tensor x {internal, nccl, none}; AR us + TG/PP.

### T3 — Multi-node Path-D

```
Client: romulus HIP (+ maybe local RPC 3060)
Workers: remus / triton CUDA rpc-servers
```

| Rail | Status | Why |
|------|--------|-----|
| Layer | **Yes** | Primary multi-backend production path. |
| Tensor end-to-end | **N/A** | No NCCL/AllReduce across RPC. |
| Per-node tensor | Only if a **worker process** runs local dual-CUDA tensor (unusual for passive rpc-server) | Not the client tensor rail. |

**Baseline rows:** best multi-node layer/Path-D config; stall metrics. Tensor matrix stays on T2.

### T4 — Hybrid (fog until after baselines)

Concept from frontier report: treat 3090+3070 as one strong **TP unit**, pipeline remaining work to 7900.

| Piece | Rail |
|-------|------|
| Inside 3090+3070 | Tensor + specialized AR (like T2) |
| Across nodes / to 7900 | Layer / RPC pipeline (like T3) |

**Not** a single `-sm tensor` over all devices. Baseline only after T1–T3 and plan go/no-go.

### T5 — GRAPH_COMPUTE_ALL (do not conflate)

Server-side multi-GPU on one RPC endpoint is **not** the same as client `-sm tensor` AllReduce. Optional layer/RPC experiment; exclude from AR provider matrix unless a later ticket redefines scope.

---

## Default `-sm tensor` pitfall

`llama_prepare_model_devices` with `split_mode == TENSOR` and **no** explicit device list:

- Skips CPU only.  
- Includes **all** GPUs and **all** RPC devices into one meta group.

On a Path-D client that has loaded RPC backends, **naive** `-sm tensor` is the wrong experiment for AllReduce. Baselines must use an explicit same-family device set (T2: two CUDA IDs only).

---

## What "covering" each topology means (baseline ticket checklist)

| Topology | Must measure | Must not require |
|----------|--------------|------------------|
| T0 / single-GPU refs | TG+PP peak | AR |
| T1 main Path-D | Layer TG+PP, split idle/compute/bubble | Tensor AR matrix |
| T2 3090+3070 | Layer TG+PP; tensor TG+PP x providers; AR timing when shipped | Multi-node |
| T3 multi-node | Layer Path-D TG+PP + stalls | End-to-end tensor |
| T4 hybrid | Only after plan go | Day-1 baseline |

Equal-weight topologies = equal diligence under **applicable** rails, not equal number of tensor rows everywhere.

---

## Implications for later tickets

| Ticket | Implication |
|--------|-------------|
| Design / ship split timestamps | Must work on **layer** (T1/T3) and on **tensor meta** (T2). |
| Ship AR timing | Instrument CUDA comm path; validate on **T2**. |
| Baselines (07) | Use this matrix as the row definition. |
| Lock plan (08) | Residual gap: T1/T3 orchestration vs T2 AR; hybrid T4 only if T2 wins enough vs layer on that pair. |

---

## Key sources

| Source | Use |
|--------|-----|
| `src/llama.cpp` `llama_prepare_model_devices` | Tensor device group construction |
| `ggml/src/ggml-backend-meta.cpp` | Meta AR dispatch + butterfly |
| `ggml/src/ggml-cuda/ggml-cuda.cu` `comm_init` | Specialized AR gate |
| `docs/research/allreduce-providers-inventory.md` | Provider table |
| `rpc-patch/patch/CLUSTER-NODE-LAYOUT.md` | romulus / remus / triton roles |
| `docs/rpc-multi-backend-pipeline-plus/CURRENT-STATE-pipeline-flow.md` | Layer/RPC production flow |
| `docs/D4.6-romulus-rpc-multidevice-test.md` | Main = HIP client + CUDA RPC 3060 |
