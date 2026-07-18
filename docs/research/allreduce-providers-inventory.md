# AllReduce providers and knobs inventory

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line  
**Wayfinder ticket:** [Inventory AllReduce providers and knobs](../../.scratch/allreduce-timing/issues/01-inventory-allreduce-providers.md)  
**Scope:** code + docs inventory only; no measurements.

---

## Summary

| Provider | Where | Devices | Backend family | When selected |
|----------|-------|---------|----------------|---------------|
| **NCCL** | `ggml-cuda.cu` `comm_allreduce_nccl` | 2+ | CUDA (Linux default) | `GGML_USE_NCCL` build + init success; env `GGML_CUDA_ALLREDUCE=nccl` or Linux default |
| **RCCL** | same NCCL code path under HIP | 2+ | HIP when `GGML_HIP_RCCL=ON` | Defines `GGML_USE_NCCL`; same env/init chain |
| **Internal AR** | `allreduce.cu` pipeline | **exactly 2** | **CUDA only** (HIP/MUSA stub returns nullptr) | After NCCL fail or env `internal`; needs sm70+ |
| **Butterfly fallback** | `ggml-backend-meta.cpp` `allreduce_fallback` | 2+ | any backends that can `tensor_copy_async` + ADD | When `comm_ctx` null, `try_allreduce` returns false, or env `none` |

Tensor-mode inference uses the **meta backend**. Between subgraphs it prefers backend-specific `comm_allreduce`; on failure it runs the generic butterfly (copy + ADD).

---

## Call path (meta backend)

```
ggml_backend_meta_context ctor
  -> ggml_backend_comm_init(simple_backends[])   // proc from first device's reg
  -> stores comm_ctx + ggml_backend_comm_allreduce_tensor

ggml_backend_meta_graph_compute
  for each subgraph i:
    graph_compute_async on every simple backend
    if n_backends > 1 && not last subgraph:
      if comm_ctx:
        backend_allreduce_success = comm_allreduce(comm_ctx, last_nodes[])
      if !success:
        allreduce_fallback(i)   // butterfly via copy_async + GGML_OP_ADD
```

Sources: `ggml/src/ggml-backend-meta.cpp` (~1616-1646, ~2069-2214).

**Proc symbols** (CUDA reg only, `ggml-cuda.cu` ~5990-6003):

- `ggml_backend_comm_init`
- `ggml_backend_comm_free`
- `ggml_backend_comm_allreduce_tensor`

If first device's backend reg has no `comm_init`, or init returns nullptr, meta has no specialized path and always butterflies.

**CUDA comm_init gate:** every backend in the group must be CUDA (`ggml_backend_is_cuda`). Mixed CUDA+HIP or CUDA+RPC does not form a CUDA comm group.

---

## Provider 1: NCCL (CUDA) / RCCL (HIP)

### Build

| Flag | Default | Effect |
|------|---------|--------|
| `GGML_CUDA_NCCL` | **ON** | `find_package(NCCL)`; if found, `GGML_USE_NCCL` + link NCCL |
| `GGML_HIP_RCCL` | **OFF** | `find_package(rccl)`; defines `GGML_USE_NCCL` (same API); HIP notes RCCL not universally beneficial |

If NCCL package missing with `GGML_CUDA_NCCL=ON`: CMake warning, no `GGML_USE_NCCL`, runtime falls through to internal then butterfly.

### Init selection (`ggml_backend_cuda_comm_init`)

```
env GGML_CUDA_ALLREDUCE?
  unset + Linux     -> try NCCL -> on fail internal -> on fail butterfly-stub
  unset + non-Linux -> try internal -> on fail butterfly-stub
  "nccl"            -> try NCCL chain
  "internal"        -> try internal chain
  "none"            -> try_allreduce = butterfly stub (always returns false)
  other             -> warn; butterfly stub
```

NCCL init: `ncclCommInitAll` over device ids. Failure clears comms and falls through to internal.

### Per-call NCCL behavior

- Element count 0: no-op success.
- Contiguous F32 tensors; inactive shards (`!GGML_TENSOR_FLAG_COMPUTE`) zeroed first.
- **Small** (FP32 wire):  
  - `n_backends <= 2 && ne < 32768`  
  - or `n_backends == 3 && ne < 131072`  
  - or `n_backends >= 4 && ne < 262144`  
  Heuristic comment: tuned for RTX 4090s on PCIe 4.0 x16.
- **Large:** F32->BF16, NCCL BF16 sum, BF16->F32 (bandwidth path).
- Grouped with `ncclGroupStart/End` across devices, each on that device's CUDA stream.

### Doc drift

`docs/multi-gpu.md` says there is no runtime flag for NCCL. **Code has** `GGML_CUDA_ALLREDUCE`. Prefer the env var as source of truth; docs should be updated in a later ticket if desired (not this inventory's job to edit docs beyond this note).

---

## Provider 2: Internal AllReduce (CUDA, 2-GPU)

### Constraints

| Constraint | Detail |
|------------|--------|
| n_devices | **Must be 2**; else init returns nullptr |
| Platform | **CUDA only**. Under `GGML_USE_HIP` / `GGML_USE_MUSA`, stubs return nullptr / false |
| CC | sm70+ (Volta+) for `__nanosleep` in chunked kernel |
| Types | F32, F16, BF16; contiguous; 16-byte aligned data; size multiple of 16 bytes |
| Transport | Pinned host staging over PCIe (no NVLink required); designed for consumer multi-GPU |

### Sub-strategies (per call, by wire size)

1. **Chunked kernel** (small / latency path, default when wire bytes **&lt; copy_threshold**, default 1 MB):  
   - Same compute stream as caller (no AR-stream hop).  
   - Kernel stages T_wire through mapped pinned host, busy-waits peer arrival token, sums in-place.  
   - 8 blocks x 256 threads; pool of 2 slots.  
   - Host staging ring: 1 MB per GPU per slot (`GGML_CUDA_AR_MAX_BYTES`).

2. **Copy-engine** (large / bandwidth path, wire bytes **&gt;= copy_threshold**):  
   - Dedicated non-blocking AR streams + events.  
   - D2H / H2D chunks via copy engine, then device ADD.  
   - Staging up to 32 MB per GPU; larger tensors outer-chunked.  
   - Chunk size: env override or `clamp(nbytes/4, 512KB, 2MB)`.

### BF16 round-trip (internal)

Default `GGML_CUDA_AR_BF16_THRESHOLD=1` means **any non-zero F32** uses BF16 on the wire (matches NCCL large-path idea). Set to `0` to disable; set larger to keep small F32 on F32 wire.

### Internal env knobs

| Env | Default | Meaning |
|-----|---------|---------|
| `GGML_CUDA_AR_COPY_THRESHOLD` | 1048576 (1 MB) | Wire-byte size to switch chunked kernel -> copy-engine; `0` forces always-kernel |
| `GGML_CUDA_AR_COPY_CHUNK_BYTES` | 0 (heuristic) | Fixed CE chunk size; min clamp 256 KB |
| `GGML_CUDA_AR_BF16_THRESHOLD` | 1 | F32 byte threshold for BF16 wire; 0 disables |

### Startup log

On success:  
`initialized AllReduce pipeline: 2 GPUs, ... KB chunked kernel staging + ... MB copy-engine staging per GPU`

---

## Provider 3: Butterfly fallback (meta)

- Used when specialized AR declines the call.
- Zeros inactive slices, then recursive butterfly: `tensor_copy_async` peer -> tmp, `GGML_OP_ADD` into dst.
- Handles non-power-of-2 device counts (fold excess, then butterfly, copy back).
- Works across backends that support async copy (including theoretically heterogeneous pairs, at higher cost).
- **No dedicated timing hooks** today; pure graph ops.

`try_allreduce_butterfly` in CUDA comm always returns **false**, so env `none` forces this meta path every call.

---

## MoE delay AllReduce (meta scheduling, not a provider)

`ggml-backend-meta.cpp` can **delay** the AllReduce insertion for MoE patterns (ADD_ID + MUL chain with mirrored/partial splits) to reduce I/O. This changes *when* reduction happens in the graph, not *which* provider runs. Relevant for later "scheduling/overlap" levers; inventory only notes it exists (~1833+).

---

## Runtime visibility today

| Signal | Present? |
|--------|----------|
| Init WARN if NCCL missing / init fail | Yes |
| Init INFO for internal pipeline sizes | Yes |
| Per-call AR duration_us / size / provider | **No** (ticket 06) |
| First-call "using provider X" one-liner | **No** (only init logs) |
| Env to force provider | **Yes** `GGML_CUDA_ALLREDUCE` |

Related but separate: `GGML_CUDA_P2P` (docs/multi-gpu.md) for peer memcpy; not the AR provider switch.

---

## Implications for this effort's topologies

| Topology | Specialized AR likely? | Notes |
|----------|------------------------|-------|
| RPC node 3090+3070 same-process CUDA | **Yes** | Best NCCL vs internal A/B; both real |
| Main 7900 XTX only | N/A | No multi-device AR |
| Main 7900 + 3060 Ti (HIP + CUDA/RPC) | **Unlikely one comm group** | Mixed backends; meta butterfly or no TP |
| Multi-node RPC workers | **No cross-RPC NCCL** | AR is in-process meta/CUDA; RPC is separate backend |
| Dual HIP multi-GPU (if ever) | RCCL if built; **no** internal AR | Internal stubs under HIP |

Detailed feasibility is owned by ticket **Topology feasibility for AllReduce**; this inventory only states the code gates.

---

## Instrumentation insert points (for later design)

| Site | File | Why |
|------|------|-----|
| After provider chosen in `comm_init` | `ggml-cuda.cu` | Log provider name once |
| Wrap `try_allreduce` / NCCL / internal | `ggml-cuda.cu` / `allreduce.cu` | `ggml_time_us`, size, provider |
| Meta fallback entry | `ggml-backend-meta.cpp` | Mark butterfly path |
| Split compute boundaries | scheduler / Path-D path | Split idle vs compute (separate from AR) |

---

## Key source files

| File | Role |
|------|------|
| `ggml/src/ggml-backend-meta.cpp` | Meta compute + butterfly fallback + MoE delay |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | comm_init chain, NCCL, dispatch, proc table |
| `ggml/src/ggml-cuda/allreduce.cu` | Internal 2-GPU pipeline (CUDA) |
| `ggml/src/ggml-cuda/allreduce.cuh` | Public AR pipeline API |
| `ggml/include/ggml-backend.h` | `comm_init` / `comm_allreduce_tensor` typedefs |
| `ggml/src/ggml-cuda/CMakeLists.txt` | `GGML_CUDA_NCCL` |
| `ggml/src/ggml-hip/CMakeLists.txt` | `GGML_HIP_RCCL` -> `GGML_USE_NCCL` |
| `docs/multi-gpu.md` | User-facing multi-GPU notes (partially stale on runtime env) |

---

## Open questions for later tickets (not answered here)

- Measured TG/PP cost of each provider on 3090+3070 (baseline ticket).
- Whether builds on cluster actually have NCCL/RCCL linked.
- Durable log schema (design instrumentation ticket).
