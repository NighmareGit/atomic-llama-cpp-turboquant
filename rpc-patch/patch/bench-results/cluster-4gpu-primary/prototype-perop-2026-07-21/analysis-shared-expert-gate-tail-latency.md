# Analysis: `shared_expert_gate` tail latency root cause

## Root cause

The 50-145ms first-call spike on `shared_expert_gate` is **lazy CUDA resource initialization** (streams, cuBLAS handles, and GPU memory pool) that fires on the first `ggml_mul_mat` operation dispatched to each GPU. The CUDA backend defers all resource creation to first use, and `shared_expert_gate` (as a `ggml_mul_mat` on quantized weights) is the MUL_MAT that triggers this initialization on the remote RPC server. There is **no backend-wide warmup pass**.

## What `shared_expert_gate` actually is

`shared_expert_gate` is NOT a custom CUDA kernel. It is a name assigned by the model builder:

```
// src/models/qwen35moe.cpp:549-550
ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
cb(shared_gate, "shared_expert_gate", il);
```

`build_lora_mm` (`src/llama-graph.cpp:1085`) wraps a `ggml_mul_mat` (matrix-vector multiply) on the shared expert gate weight matrix. For Qwen3.6-35B-A3B the gate weight is a tiny `[n_shared_experts, n_embd]` matrix (typically 2-8 rows). The `cb()` call stamps the name `shared_expert_gate` on the tensor for telemetry -- it is the name the per-node timing records.

The `ggml_sigmoid` on the result is tracked separately as `shared_expert_gate_sigmoid` (a trivial element-wise op).

## Full call path

```
ggml_backend_rpc_graph_compute                  (ggml-rpc.cpp:2458)
  -> serialize + send_rpc_cmd                    (ggml-rpc.cpp:2647)
     Server: rpc_server::graph_compute           (ggml-rpc.cpp:3514)
       [sampled path] -> compute_graph_per_node  (ggml-rpc.cpp:2397)
         ggml_backend_graph_compute_async         (ggml-rpc.cpp:2408)
           ggml_backend_cuda_graph_compute        (ggml-cuda.cu:4858)
             for each node:
               ggml_cuda_compute_forward           (ggml-cuda.cu:3101)
                 case GGML_OP_MUL_MAT:
                   ggml_cuda_mul_mat                (ggml-cuda.cu:3294)
                     ggml_cuda_op_mul_mat            (ggml-cuda.cu:2059)
                       [LAZY: pool alloc, stream get, cublas get]
                       -> ggml_cuda_op_mul_mat_vec_q (mmvq.cu:1459)
         ggml_backend_synchronize                   (ggml-rpc.cpp:2409)
```

## Cold-start overhead sources

Every per-op path through `ggml_cuda_op_mul_mat` allocates 3+ GPU buffers from the pool (lines 2184-2232):

```cpp
// ggml-cuda.cu:2184-2232
dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), ...);   // cudaMalloc
dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ...); // cudaMalloc
dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), ...); // cudaMalloc (quantized)
dev[id].dst_dd   = dev[id].dst_dd_alloc.alloc(ctx.pool(id), ...);   // cudaMalloc
```

The pool (`ggml_cuda_pool_leg`, `ggml-cuda.cu:373`) is **cold** on first use per GPU. It calls `cudaMalloc` for each buffer. While individual `cudaMalloc` calls are fast (microseconds), the **first CUDA kernel launch** on a device can trigger driver-level setup that takes tens of milliseconds.

Additionally, these resources are lazily created (first call per GPU):

| Resource | Lazy init at | Cost |
|----------|------------|------|
| CUDA stream | `cudaStreamCreateWithFlags` at `common.cuh:1486` | ~50 us |
| cuBLAS handle | `cublasCreate` at `common.cuh:1498` | 5-30 ms (first time per device) |
| Memory pool | `ggml_cuda_pool_leg` ctor + first `cudaMalloc` at `ggml-cuda.cu:449` | 1-50 ms (driver heap expansion) |
| CUDA graph warmup | 2nd call trigers graph capture at `ggml-cuda.cu:4877` | ~10-50 ms (full graph capture) |

The `cudaStreamNonBlocking` property (line 1486) does not affect first-call cost.

## Why `shared_expert_gate` bears the brunt

In the per-layer graph, `shared_expert_gate` is typically the **first matrix-vector multiply** operation on the 3060 Ti/3070 RPC backend (other ops before it in the graph are metadata ops like VIEW/RESHAPE/PERMUTE that are no-ops in CUDA -- `ggml-cuda.cu:3323-3328`):

```cpp
// ggml-cuda.cu:3323-3328  (no-op cases in compute_forward)
case GGML_OP_NONE:
case GGML_OP_RESHAPE:
case GGML_OP_VIEW:
case GGML_OP_PERMUTE:
case GGML_OP_TRANSPOSE:
        break;  // NO resource init fires
```

The first MUL_MAT therefore pays the cumulative cost of lazy stream creation + cuBLAS handle creation + first cold pool alloc + first kernel launch.

The spike scales with GPU capability: 3060 Ti (GA106, 4864 cores) = 144.9ms max, 3070 (GA104) = 45.3ms, 3090 (GA102) = 53.2ms. The three-fold difference between 3060 Ti and 3090 matches the GPU performance ratio and is consistent with first-kernel driver overhead.

## Warmup gap

**There is no backend-wide warmup pass.** The CUDA backend has:

1. **No explicit warmup call** -- zero `warmup`/`prewarm`/`pre_warm` strings exist anywhere in `ggml/src/ggml-cuda/`.

2. **CUDA graph warmup** (`ggml-cuda.cu:4879`) is NOT a kernel warmup -- it is a 2-pass graph-capture mechanism that adds additional overhead on top of normal compute, does not pre-initialize streams/cuBLAS/pools.

3. **RPC server startup** (`ggml_backend_rpc_start_server`, `ggml-rpc.cpp:4591`) calls `ggml_backend_dev_init` for each device, which creates a `ggml_backend_cuda_context` but does NOT call any CUDA APIs. The first CUDA API call during `cudaDeviceGetPCIBusId` at registry init (line 6070) creates the CUDA driver context, but per-process resources like cuBLAS handles and streams are NOT created.

## Proposed fix

### Option A (Recommended): Dummy MUL_MAT warmup in RPC server startup

After initializing each CUDA backend in `ggml_backend_rpc_start_server` (`ggml-rpc.cpp:4591-4625`), issue a trivial dummy compute on each device to pre-initialize lazy resources. Add something like:

```cpp
// In ggml_backend_rpc_start_server, after backend creation (line 4615):
for (auto backend : backends) {
    // Warmup: force lazy stream + cuBLAS handle + pool creation
    // on the CUDA backend via a trivial MUL_MAT
    ggml_backend_warmup(backend);
}
```

In the CUDA backend, implement `ggml_backend_warmup`:

```cpp
// In ggml-cuda.cu:
void ggml_backend_cuda_warmup(ggml_backend_t backend) {
    // Force lazy initialization of streams, cuBLAS handles, and pools
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_cuda_set_device(ctx->device);
    
    // Trigger stream creation for all stream slots
    for (int s = 0; s < GGML_CUDA_MAX_STREAMS; s++) {
        ctx->curr_stream_no = s;
        ctx->stream();          // lazy cudaStreamCreateWithFlags
    }
    ctx->curr_stream_no = 0;
    
    // Trigger cuBLAS handle creation
    ctx->cublas_handle();       // lazy cublasCreate
    
    // Trigger pool creation via a tiny allocation+free
    // (pool already initialized after the alloc+free cycle)
    auto * pool_ptr = ctx->pool().alloc(256, &actual_size);
    ctx->pool().free(pool_ptr, actual_size);
}
```

### Option B (Lower effort): Eager init in RPC server

Replace lazy initialization at `ggml_backend_cuda_context` with eager init in the constructor:

```cpp
// common.cuh:1473 - constructors
explicit ggml_backend_cuda_context(int device) :
    device(device),
    name(GGML_CUDA_NAME + std::to_string(device)) {
    ggml_cuda_set_device(device);
    // Pre-create all streams
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
        for (int s = 0; s < GGML_CUDA_MAX_STREAMS; s++) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i][s], cudaStreamNonBlocking));
        }
    }
    // Pre-create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&cublas_handles[device]));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
}
```

This moves the ~30ms of init cost to backend creation time (during server startup) instead of first compute.

## Expected impact

- **Eliminates the 50-145ms first-call spike entirely** for shared_expert_gate and every other first-seen operation.

- The **24.2% of backend time** attributed to `shared_expert_gate` on the 3060 Ti would drop to ~0.5-1% (the true compute cost of a tiny MUL_MAT, ~15-30us per call, times ~150 calls per trace).

- For all backends combined: the first-call spike accounts for roughly 144+53+45 = ~242ms of one-time overhead per server startup, which inflates per-op percentages for the entire trace. With warmup, this vanishes.

- **No steady-state throughput change** -- this only affects the first graph compute on each GPU after server startup or after long idle periods (pool eviction cycles).
