# Analysis: KV Cache Deployment in Vanilla llama.cpp (RPC Multi-Backend)

Source files checked: `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`, `src/llama-context.cpp`, `src/llama-model.cpp`, `docs/multi-gpu.md`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`

---

## 1. KV Cache Device Assignment

**Rule: each layer's KV cache tensors go to the same device as that layer's weights.**

| Step | File | Lines | Mechanism |
|---|---|---|---|
| Layer-to-device map | `src/llama-model.cpp` | 1334-1337 | `pimpl->dev_layer[il]` populated by `get_layer_buft_list(il)` — proportional split, explicit placement plan, or CPU fallback |
| KV cache queries it | `src/llama-kv-cache.cpp` | 282-287 | `model.dev_layer(il)` → `ggml_backend_dev_buffer_type(dev)` |
| KV tensor created | `src/llama-kv-cache.cpp` | 395-399 | `ggml_new_tensor_3d(ctx, type, ...)` in per-buft context |
| Buffer allocated | `src/llama-kv-cache.cpp` | 452-469 | `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)` |

The KV cache constructor iterates `hparams.n_layer_all`, calls `model.dev_layer(il)` for each layer, picks the right backend buffer type, creates the K/V tensors in that device's context, then allocates them.

---

## 2. KV Cache Allocation: Contiguous Per Device, Not Interleaved

**Layout: one `ggml_backend_buffer` per device, containing *all* K and V tensors for layers assigned to that device.**

```
Device A buffer:   [cache_k_l0, cache_v_l0, cache_k_l2, cache_v_l2, ...]
Device B buffer:   [cache_k_l1, cache_v_l1, cache_k_l3, cache_v_l3, ...]
```

| Property | Detail |
|---|---|
| Grouping | `ctx_for_buft` (line 181) creates one `ggml_context` per unique `ggml_backend_buffer_type_t` |
| Tensors per ctx | All layers mapping to the same device share one context |
| Allocation | Single `ggml_backend_alloc_ctx_tensors_from_buft()` call per device |
| Stream views | Per-stream K/V `ggml_view_2d` slices from the parent 3D tensor (line 404-407) — same buffer, same device |
| CPU fallback | If `offload=false`, all KV tensors go to `ggml_backend_cpu_buffer_type()` (line 281) |
| `no_alloc` mode | Dummy buffer assigned; real allocation deferred to the backend scheduler (line 455-458) |

No interleaving: tensors for different devices live in different buffers. The RPC backend sends per-buffer serialized content (ggml-rpc.cpp alloc_buffer/set_tensor messages).

---

## 3. Pipeline Parallelism and KV Cache Accessibility

**With `--split-mode layer` (default multi-GPU), each pipeline stage reads/writes only its own device-local KV cache. No cross-stage KV cache access.**

| Aspect | Detail |
|---|---|
| Layer split | Each GPU holds a contiguous slice of layers. `dev_layer[il]` maps layer → owning GPU. |
| KV cache writes | Happen via `ggml_set_rows` ops embedded in the graph of the last split ("gather split"). The gather split is the final backend in topological order (per ADR-0002). |
| KV cache reads | Attention ops read KV tensors on the same device that hosts them, because attention subgraphs are assigned to the backend owning those layers. |
| Cross-stage constraint | A stage never needs KV data from another stage — each stage's layers are self-contained. |
| Pipeline barrier | `ggml_backend_sched_pipeline_barrier()` (llama-context.cpp:1633) synchronizes event signaling across stages, but only to order computation, not to move KV data. |
| `no_alloc` KV | If `hparams.no_alloc`, KV tensors carry dummy buffers and the scheduler allocates them lazily on the correct backend during `alloc_graph`. |

**GPipe extension**: The `llama_pipeline_plus_enabled()` path (lines 33-37 + llama-context.cpp:636-652) adds explicit stage signaling (`ggml_backend_sched_signal_gpipe_stage`), but the KV cache isolation model does not change — each stage still owns its layers' KV.

---

## 4. Non-Contiguous Layer-to-Device Mapping: No KV Cache Constraints

**Vanilla llama.cpp has zero constraints on non-contiguous layer-device mapping with respect to KV cache.**

| Property | Finding |
|---|---|
| `params.layer_devices[]` | Explicit per-layer device array (placement plan) — maps layer *l* to any device, including non-contiguous order (llama-model.cpp:1294-1312) |
| KV cache follows | `model.dev_layer(il)` returns whatever device the placement plan assigned (llama-kv-cache.cpp:282-287) |
| Per-device grouping | KV tensors are still grouped by device, regardless of which layers they are — contiguous or not, they land in the right buffer |
| `SPLIT_MODE_TENSOR` | Different path: KV cache uses meta-device abstraction ("fatbin"), split across GPUs via NCCL/RCCL. Requires Flash Attention and non-quantized KV types. Not relevant to `layer` mode. |

**Implication for pipeline+ GPipe**: Since `dev_layer[il]` supports arbitrary per-layer device mapping, the KV cache allocation already handles the case where layer *l*'s KV lives on a different device than layer *l+1*'s KV. Each device gets a buffer with only its assigned layers' K/V tensors. No code changes needed.

---

## Summary

| Question | Answer |
|---|---|
| KV cache same device as weights? | Yes. `model.dev_layer(il)` determines both. |
| Allocation contiguous? | **Per device** — one buffer per backend, all that device's layers' K/V tensors contiguously within it. Not interleaved across devices. |
| Pipeline parallel KV access? | Each stage accesses its own device-local KV only. No cross-stage KV data movement. |
| Non-contiguous layer mapping constraints? | None. KV cache follows `dev_layer` which is fully flexible (placement plan or proportional split). |
| Tensor split mode KV? | Different path: meta-device with NCCL reductions. Requires FA, no quantized KV. |
| RPC-specific KV handling? | KV tensors are just buffers to RPC — `alloc_buffer`, `set_tensor`, `get_tensor` messages in `ggml-rpc.cpp`. No special KV treatment. |
