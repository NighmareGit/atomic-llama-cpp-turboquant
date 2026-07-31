# Path A: Protocol Optimization — Tracking

**Overall:** IN PROGRESS
**Build:** PASS
**Current Phase:** A2 — Pipelined get_tensor
**Phase Status:** COMPLETE

## Implementation Log

### 2026-06-24 — Phase A1: Batch SET_TENSOR calls
- **Done:** Implemented thread-local batch buffer, flush mechanism, server batch handler
- **Files:** `ggml/src/ggml-rpc/ggml-rpc.cpp`, `include/ggml-rpc.h`
- **Blockers:** none
- **Debug:** Fixed `send_data` forward declaration issue (function used before definition). Fixed `ctx->sock` → `sock` in get_tensor and cpy_tensor to use local variable after flush. Smoke test passed: server v4.1.0 accepts both individual SET_TENSOR (backward compat) and batch SET_TENSOR_BATCH.
- **Next:** Phase A2 — Async get_tensor

### 2026-06-24 — Phase A2: Pipelined get_tensor
- **Directory:** `build-a2/` (separate from `build-work/` for Phase A1)
- **Done:** Implemented pipelined get_tensor_async: send request immediately, defer receive to synchronize
- **Files:** `ggml/src/ggml-rpc/ggml-rpc.cpp` (in build-a2/)
- **Key design:**
  - `ggml_backend_rpc_get_tensor_async`: backend-level entry point, delegates to buffer-level
  - `ggml_backend_rpc_buffer_get_tensor_async`: sends GET_TENSOR request immediately, queues receive
  - `flush_pending_get_tensor()`: receives all pending responses in order
  - Added flush calls in `get_tensor`, `cpy_tensor`, `graph_compute` before blocking operations
- **Issue found:** Initial implementation deferred BOTH send and receive — caused 86% regression (44 vs 265 t/s) because ggml expects data available immediately after `get_tensor_async`. Fixed by sending immediately and only deferring the receive.
- **Result:** Pipelined version performs identically to Phase A1 (P=33 G=22 with 20 layers). No measurable improvement for 9B model — get_tensor calls are infrequent during inference (main overhead is SET_TENSOR during model load, which Phase A1 already handles).
- **Docker image:** `llama-rpc-cuda-a2`

## Issues

1. **Protocol version mismatch** — Patched client (v4.1.0) initially sent `RPC_CMD_SET_TENSOR_BATCH` to vanilla server (v4.0.0) which rejected it with "Unknown command: 17". Fixed by adding `server_supports_batch` flag to `socket_t` and checking it before batching.

2. **Phase A2 get_tensor_async semantics** — ggml's `get_tensor_async` is NOT truly async; it means "skip the preceding synchronize, but data must be available immediately". Initial implementation queued both send and receive, breaking this contract. Fixed by sending immediately and only pipelining the receive.

## Benchmarks

### A/B Test Results (0.8B Q4_K_M, RTX 3060 Ti)
| Metric | Vanilla (v4.0.0) | Patched (v4.1.0) | Delta |
|--------|------------------|------------------|-------|
| Prompt | 1018.9 t/s | 1010.7 t/s | -0.8% |
| Generation | 244.0 t/s | 230.5 t/s | -5.5% |

Note: Small model shows minimal difference. Batch optimization expected to benefit larger models with more SET_TENSOR calls per decode step.

### 1:1 Server Benchmark (12B Q4_K_M, 15 GPU layers, RTX 3060 Ti)
| Metric | Vanilla Stack | Patched Stack | Delta |
|--------|---------------|---------------|-------|
| Latency (avg) | 9078ms | 9053ms | 0% |
| Throughput | 10.9-11.3 t/s | 10.9-11.3 t/s | 0% |

Note: Full stack test (llama-server + rpc-server). Performance identical at 15 GPU layers. Batch optimization expected to benefit more with higher GPU layer counts and larger models.

### Cross-GPU RPC Matrix (Qwen3.5-9B-Q5_K_M, -ngl 99 -ctk q4_0 -ctv q4_0)

**Config A: AMD llama-server (client) ↔ NVIDIA rpc-server (worker)**
| Variant | Prompt (t/s) | Gen (t/s) | Delta |
|---------|---------------|------------|-------|
| Vanilla (v4.0.0) | 306.0 | 68.1 | — |
| Patched A1+A2 (v4.1.0) | 317.0 | 68.4 | +3.6% P, +0.4% G |

**Config B: NVIDIA llama-server (client) ↔ AMD rpc-server (worker)**
| Variant | Prompt (t/s) | Gen (t/s) | Delta |
|---------|---------------|------------|-------|
| Vanilla (v4.0.0) | 283.2 | 60.6 | — |
| Patched A1+A2 (v4.1.0) | 284.2 | 60.5 | +0.4% P, -0.2% G |

Note: Config A outperforms Config B because the llama-server's GPU (AMD 7900 XTX, 24GB) has more VRAM for KV cache and compute buffers. The patched version shows modest improvement in Config A (~3.6% prompt) but not in Config B. Generation speed is limited by the RPC worker's tensor operations, not by the SET_TENSOR/GET_TENSOR overhead.

### Phase A2 (20 layers, single GPU 3060 Ti)
| Config | Prompt (t/s) | Gen (t/s) |
|--------|---------------|------------|
| Phase A1 (batch SET_TENSOR only) | 33.0 | 21.7 |
| Phase A2 (batch SET_TENSOR + pipelined get_tensor) | 33.0 | 21.7 |
Note: Identical performance — get_tensor pipelining provides no measurable benefit for 9B model at 20 layers.

### Full GPU Offload Tests (with -ctk q4_0 -ctv q4_0)
| Model | Size | Layers | VRAM | Prompt (t/s) | Gen (t/s) |
|-------|------|--------|------|---------------|------------|
| Qwen3.5-4B-Q4_K_M | 2.7G | 34/34 | 3836 MiB | 535 | 93 |
| Qwen3.5-9B-Q5_K_M | 6.4G | 33/33 | 7040 MiB | 349 | 54 |
| smollm3-3b-q4_k_m | 1.8G | 37/37 | 2584 MiB | 143 | 143 |

Note: Full GPU offload with cache quantization dramatically improves performance. Phase A1 and A2 show identical results — SET_TENSOR batching is the main bottleneck during model load, not get_tensor during inference. Cache quantization (-ctk q4_0 -ctv q4_0) is critical for fitting larger models on 8GB VRAM.

## Files Modified

| File | Change |
|------|--------|
| `include/ggml-rpc.h` | `RPC_PROTO_MINOR_VERSION` 0→1 |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `RPC_CMD_SET_TENSOR_BATCH` to enum (17) |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `set_tensor_batch_t`, `tls_set_batch`, `RPC_SET_TENSOR_BATCH_MAX_SIZE` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `set_tensor_batch_append()`, `flush_set_tensor_batch()` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Modify `ggml_backend_rpc_buffer_set_tensor()` to append to batch |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add flush points in `get_tensor`, `cpy_tensor`, `graph_compute` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `RPC_CMD_SET_TENSOR_BATCH` case in server `rpc_serve_client()` |

### Phase A2 Files Modified (in build-a2/)
| File | Change |
|------|--------|
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `rpc_pending_get_tensor` struct and `tls_pending_get_tensor` queue |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `ggml_backend_rpc_buffer_get_tensor_async()` — sends immediately, queues receive |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `ggml_backend_rpc_get_tensor_async()` — backend-level entry point |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `flush_pending_get_tensor()` — receives all pending responses |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Update `ggml_backend_rpc_synchronize()` to call `flush_pending_get_tensor()` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Update interface: `get_tensor_async = ggml_backend_rpc_get_tensor_async` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add flush calls in `get_tensor`, `cpy_tensor`, `graph_compute` |
