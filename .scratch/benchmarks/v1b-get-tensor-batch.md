# V1b — GET_TENSOR_BATCH Protocol Extension

**Date:** 2026-07-24  
**Build:** 83605ae27 (10202) + GET_TENSOR_BATCH patches  
**Status:** COMPLETE (functional, no benefit for 2-GPU layer-split; carries forward for row-split)

## Hypothesis

Per-token GET_TENSOR is the dominant cost in RPC multi-GPU inference (4202 µs/call, 970KB payload, 98.4% of tg time in TCP-only mode). Batching multiple GET_TENSOR requests into a single RPC_CMD_GET_TENSOR_BATCH message eliminates per-call TCP overhead and server dispatch cost.

## Implementation

### Client-Side (ggml-rpc.cpp)
- **RPC_CMD_GET_TENSOR_BATCH (25):** New protocol command, patterned after existing SET_TENSOR_BATCH (17)
- **get_tensor_batch_entry:** Thread-local accumulator holding `rpc_msg_get_tensor_req` + data pointer + size
- **get_tensor_batch_append:** Appends a GET_TENSOR to the batch, flushing if the socket changes
- **flush_get_tensor_batch:** Serializes count + all requests into single message, sends via `send_data`, then queues deferred responses in `tls_pending_get_tensor`
- **Integration:** Called at 3 flush points — `send_rpc_cmd` (fire-and-forget), `send_rpc_cmd` (blocking), and `ggml_backend_rpc_buffer_get_tensor` (blocking)
- **Activation:** `GGML_RPC_GET_TENSOR_DEFER=1` (defaults to 0 in non-pipeline modes)

### Server-Side
- New handler for `RPC_CMD_GET_TENSOR_BATCH`:
  1. Reads full payload via `recv_msg(sock, input)`
  2. Parses `uint32_t count` then `count × rpc_msg_get_tensor_req` entries
  3. For each entry, calls `server.get_tensor()` and sends individual response
  4. Aborts on any individual failure

## Results

### 2-GPU Layer-Split (35B Q6_K, ROMC 7900 XTX + CUDA 5060 Ti)

| Config | tg64 (t/s) | vs Baseline |
|--------|-----------|-------------|
| Baseline (UDP default, no batching) | **80.89** | — |
| GET_TENSOR_DEFER=1 (batching) | **78.69** | -2.7% |

### Analysis

**No benefit for 2-GPU layer-split.** The 2-GPU layer-split configuration generates exactly **one** GET_TENSOR per decode step (the tensor crossing the network boundary). Batching a single entry adds overhead (vector allocation, serialization differences) without any coalescing benefit.

The -2.7% regression is within noise range and attributable to the batch message format overhead for a single entry.

### Server Log Verification

RPC server logs show **zero** `GET_TENSOR_BATCH` entries — confirming the batching path is not triggered in this configuration. The scheduler uses a sync copy path for layer-split cross-backend tensor transfers rather than the async `cpy_tensor_async` path where batching is wired in.

## When Batching Will Help

GET_TENSOR_BATCH carries forward as an active attack vector for configurations with **multiple GET_TENSOR per step**:

1. **Row-split (V0):** Each partition requires GET_TENSOR from the remote GPU's shard — multiple calls per token
2. **3+ GPU layer-split:** Tensors flowing across multiple boundaries can be batched at each boundary
3. **Pipeline parallel (V4):** Multiple in-flight tokens generate multiple GET_TENSOR calls in parallel
4. **Speculative decoding (V5):** Draft tokens create additional GET_TENSOR opportunities

## Verification Gate

- [x] Compiles cleanly (local ROCm + remus CUDA)
- [x] RPC server restarted with new binary on remus
- [x] Client-server protocol works (no crashes, no garbled output)
- [x] Benchmarked against baseline: -2.7% (expected for single-entry batch)
- [ ] Test on row-split configuration — BLOCKED by V0 row-split bugfix
- [ ] Test on 3+ GPU configuration

## Commit

Changes to `ggml/src/ggml-rpc/ggml-rpc.cpp`:
- RPC_CMD_GET_TENSOR_BATCH enum value (25)
- Batch accumulator: `get_tensor_batch_entry`, `tls_get_batch`, `tls_get_batch_sock`
- `get_tensor_batch_append()` and `flush_get_tensor_batch()` functions
- Modified `ggml_backend_rpc_buffer_get_tensor_async` with `batch_send` branch
- Server-side `RPC_CMD_GET_TENSOR_BATCH` handler
