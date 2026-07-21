# KV Cache Analysis: Path-B-Plus vs Vanilla

## Comparison Table

| Aspect | Vanilla (current workspace) | Path-B-Plus (/tmp/path-b-plus) | Impact |
|--------|---------------------------|-------------------------------|--------|
| **Placement plan** | `use_layer_map` + `params.layer_devices[]` for arbitrary per-layer device assignment (`llama-model.cpp:1286-1310`) | **Missing** — no `use_layer_map`; only proportional `splits[]` contiguous block assignment (`llama-model.cpp:1270-1282`) | Path-B-Plus cannot accept interleaved/non-contiguous layer-to-device maps |
| **KV allocation backend** | `model.dev_layer(il)` → `ggml_backend_dev_buffer_type(dev)` — same as Path-B-Plus | `model.dev_layer(il)` → `ggml_backend_dev_buffer_type(dev)` (`llama-kv-cache.cpp:284-286`) | Identical; both allocate per-layer KV on the weight device |
| **Buffer grouping** | `ctx_for_buft` lambda groups by buffer type, all per-layer K/V tensors in one buffer per device | Same `ctx_for_buft` pattern (`llama-kv-cache.cpp:180-200`) | Identical; single buffer per device for KV regardless of mapping |
| **TurboQuant KV** | **Not present** | Full support: Turbo3_0/4_0/2_0 + WHT rotation + InnerQ (`llama-kv-cache.cpp:410-490`) | Path-B-Plus only; 2-4x KV compression |
| **Layer-adaptive KV** | **Not present** | 7 modes (`TURBO_LAYER_ADAPTIVE`), default mode 7 for turbo2-V: boundary layers q8_0, rest turbo2 (`llama-kv-cache.cpp:314-368`) | Path-B-Plus only; quality-preserving compression |
| **Auto-asymmetric K/V** | **Not present** | GQA ratio >= 6: K auto-upgraded turbo→q8_0 (`llama-kv-cache.cpp:142-165`) | Path-B-Plus only; PPL protection for high-GQA models |
| **Attention rotation** | **Not present** | `attn_rot_k`/`attn_rot_v` per-side env overrides; OFF by default (`llama-kv-cache.cpp:517-578`) | Path-B-Plus only; optional PPL improvement |

## Detailed Findings

### 1. KV Cache Allocation for Multi-Backend (RPC)

Both versions allocate per-layer K/V tensors via the same pattern:

```
llama-kv-cache.cpp:282: auto * dev = model.dev_layer(il);
llama-kv-cache.cpp:284: buft = ggml_backend_dev_buffer_type(dev);
```

Each layer's `cache_k_l{il}` and `cache_v_l{il}` are created as `ggml_new_tensor_3d(ctx, type, n_embd_gqa, kv_size, n_stream)` and grouped by buffer type into a single backend buffer per device.

The `n_stream` parameter (unified mode) collapses all sequences into one stream (`n_stream=1`) vs `n_stream=n_seq_max` for per-sequence streams. Both versions support this identically.

### 2. Layer-to-Device Mapping — Critical Difference

**Vanilla** (`load_tensors` in `llama-model.cpp:1286-1310`) supports explicit placement plans:

```cpp
const bool use_layer_map = params.layer_devices != nullptr && params.n_layer_devices > 0;
if (use_layer_map && il < params.n_layer_devices) {
    ggml_backend_dev_t mapped = params.layer_devices[il];
    // NULL -> CPU, otherwise use the specified device
    return {mapped, &pimpl->gpu_buft_list.at(mapped)};
}
```

This enables arbitrary interleaved/non-contiguous assignment like `[GPU0, GPU1, GPU0, GPU1, ...]`.

**Path-B-Plus** (`llama-model.cpp:1270-1282`) only supports proportional splits:

```cpp
if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
    return {cpu_dev, &pimpl->cpu_buft_list};
}
const int layer_gpu = std::upper_bound(splits.begin(), ...) - splits.begin();
return {devices.at(layer_gpu).dev, ...};
```

This produces **contiguous blocks** — layers are assigned in-order by proportional VRAM, e.g. `[CPU×N, GPU0×M, GPU1×K, ...]`.

**Note**: The vanilla code is at `/home/hunter/projects/path-d-gpipeline-assembly-line/src/llama-model.cpp`. The Path-B-Plus code at `/tmp/path-b-plus/src/llama-model.cpp` is *behind* and has not merged this placement plan feature.

### 3. Path-B-Plus KV Cache Optimizations

Path-B-Plus has extensive TurboQuant KV compression not present in vanilla:

| Optimization | Detail | Env/Config |
|---|---|---|
| **WHT rotation** | 128x128 Walsh-Hadamard matrices pre-loaded; zero-padding for head_dim ∤ 128 | Built-in for turbo types |
| **InnerQ equalization** | Per-channel scale_inv (128 channels) for Q/V | Built-in for turbo types |
| **Layer-adaptive** | Boundary layers get higher precision (q8_0) while middle layers use turbo2 | `TURBO_LAYER_ADAPTIVE` (0-7) |
| **Auto-asymmetric** | GQA ≥ 6: K→q8_0, V stays turbo | `TURBO_AUTO_ASYMMETRIC` |
| **Attn rotation** | Optional K/V-side rotation for PPL improvement | `LLAMA_ATTN_ROT_K_OVERRIDE`, `LLAMA_ATTN_ROT_V_OVERRIDE` |

### 4. Interleaved Mapping — Path-B-Plus Impact

Since Path-B-Plus does **not** support `use_layer_map` (explicit per-layer device assignment), **interleaved (non-contiguous) mapping is not possible** on this version. The question of whether it would break anything is moot — it simply isn't implemented.

To answer hypothetically: if the placement plan code were backported to Path-B-Plus, the KV cache allocation (`ctx_for_buft` grouping by buffer type) would handle it correctly because KV tensors are created per-layer and grouped by device buffer type, not by contiguity. No assumptions about sequential device assignment exist in the KV cache code.

### 5. Pipeline Parallel — RPC Transport Enhancements

Path-B-Plus has substantial RPC transport improvements over vanilla:

| Feature | Vanilla | Path-B-Plus |
|---|---|---|
| `RPC_CMD_EVENT_RECORD` | **Not present** | Event notification with TCP ordering (`ggml-rpc.cpp:648-655`) |
| `RPC_CMD_COPY_TENSOR_PEER` | **Not present** | GPU direct cross-endpoint copy |
| `RPC_CMD_CHANNEL_BIND` | **Not present** | Dual-socket cmd/response split |
| RDMA transport | **Not present** | InfiniBand verbs (RoCE v1/v2) in `transport.cpp` |
| Pipelined SET_TENSOR | **Not present** | Deferred hash + batch flush |
| `GGML_PIPELINE_PLUS` | **Not present** | Multi-backend seq ordering, deferred everything |

The `ggml_pipeline_plus_enabled()` guard controls all deferred operations (events, GET_TENSOR, hash) and defaults to **enabled** (`=1`) in Path-B-Plus RPC. Vanilla has none of these optimizations.

## Key Takeaway for Cluster Ops

The primary concern for Path-D is that **Path-B-Plus is missing the `use_layer_map` placement plan** (lines 1286-1310 of vanilla `llama-model.cpp`). This means:

1. Non-contiguous/interleaved layer-to-device mapping is **unsupported**
2. The `params.layer_devices[]` field is ignored if set
3. KV cache landing devices are always contiguous proportional splits

If upstream llama.cpp adopts `use_layer_map` as standard (it's already merged), Path-B-Plus will need to cherry-pick these changes. The KV cache code itself is compatible — no changes needed there.
