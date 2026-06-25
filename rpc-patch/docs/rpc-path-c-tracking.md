# Path C: Multi-GPU RPC Aggregation — Tracking

## Status: PLANNING COMPLETE

### Build: N/A  |  Test: N/A  |  Benchmark: N/A

### Current Phase: C1 (handover doc written)
### Phase Status: NOT STARTED

---

## Implementation Log

*(No entries yet)*

---

## Issues Log

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|

---

## Benchmark Results

| Run | Settings | Baseline tok/s | Modified tok/s | Delta |
|-----|----------|----------------|----------------|-------|

---

## Success Criteria

- [ ] **RTTs per token reduced**: from 7-22 to ~3-5 (one SET_TENSOR batch, one GRAPH_COMPUTE_ALL, one GET_TENSOR)
- [ ] **Correct results**: output matches per-device compute bit-exact (compare token sequence)
- [ ] **Throughput improvement**: measurable gain in cross-GPU RPC scenarios (single server, multiple GPUs)
- [ ] **Backward compatible**: old clients work with new server and vice versa
- [ ] **No regression**: single-GPU RPC still works at ±2% of baseline

---

## Phase Checklist

### Phase C1: Baseline Measurement (no code changes)

- [ ] RPC server started with 2 GPUs (`rpc-server -d "CUDA0,CUDA1" -p 50052`)
- [ ] Client CPU-only, `--rpc SERVER:50052,SERVER:50052 -ngl 99`
- [ ] tok/s measured for 512-token generation
- [ ] `n_splits` per token recorded (`GGML_SCHED_DEBUG=1`)
- [ ] RTT count per token estimated (2 × n_splits + 2)
- [ ] Server GPU utilization recorded (`nvidia-smi dmon`)
- [ ] Number of compute commands per token verified (one per device → 2 per token baseline)

### Phase C2: Server Internal Scheduler

- [ ] `GGML_RPC_MAX_DEVICES` defined (8) in `ggml-rpc.h`
- [ ] `RPC_CMD_GRAPH_COMPUTE_ALL` enum value added (19 or 20, before `RPC_CMD_COUNT`)
- [ ] `RPC_CMD_GRAPH_RECOMPUTE_ALL` enum value added (next after COMPUTE_ALL)
- [ ] `rpc_msg_graph_compute_all_req` struct defined (`n_devices`, `devices[8]`, followed by serialized graph)
- [ ] `rpc_msg_graph_compute_all_rsp` struct defined (`result`, `output_device`)
- [ ] `rpc_msg_graph_recompute_all_req` struct defined (`n_devices`, `devices[8]`)
- [ ] `rpc_msg_graph_recompute_all_rsp` struct defined (`result`)
- [ ] `rpc_server::stored_multi_graph` struct added (graph cache, `buffer`, `graph`, `n_devices`, `devices`)
- [ ] `rpc_server::graph_compute_all` implemented (deserialize + create scheduler + reserve + compute)
- [ ] `rpc_server::graph_recompute_all` implemented (reuse cached graph + re-compute)
- [ ] `RPC_CMD_GRAPH_COMPUTE_ALL` dispatch case added in `rpc_serve_client`
- [ ] `RPC_CMD_GRAPH_RECOMPUTE_ALL` dispatch case added in `rpc_serve_client`
- [ ] Server-side scheduler creation succeeds with 2+ backends
- [ ] Server sends response (`result` + `output_device`) after compute

### Phase C3: Client Combined Graph Submission

- [ ] `ggml_backend_rpc_context` extended with `n_devices_on_endpoint`, `is_multi_device_capable`, `multi_device_compute_sent`, `multi_device_output_device`
- [ ] `socket_t` has `server_supports_multi_device` flag (set from HELLO response)
- [ ] `ggml_backend_rpc_init` queries device count and sets context fields
- [ ] `graph_compute_all_devices` client helper implemented (serializes full graph, sends GRAPH_COMPUTE_ALL or GRAPH_RECOMPUTE_ALL)
- [ ] `ggml_backend_rpc_graph_compute` modified: `n_devices > 1 && multi_device_capable` → device 0 sends combined, others skip
- [ ] Combined graph caching works via `rpc_ctx->gc` (reuse from device 0's context)
- [ ] Only ONE compute command per token (not two)

### Phase C4: Combined Output Retrieval

- [ ] `multi_device_output_device` stored from `GRAPH_COMPUTE_ALL` response
- [ ] `get_tensor` reads from the correct server GPU (output device, not per-device buffer)
- [ ] Output tensor correctness verified against vanilla (non-RPC) run
- [ ] `GRAPH_RECOMPUTE_ALL` correctly handles output device (same as first compute)
- [ ] Test: output from device 0 vs device 1 both produce correct results

### Phase C5: HELLO Negotiation

- [ ] `RPC_CAP_MULTI_DEVICE` macro defined (bit 0)
- [ ] `rpc_msg_hello_rsp` `padding` field renamed to `server_caps` (same offset, same struct size — backward compatible)
- [ ] `rpc_server::hello` sets `RPC_CAP_MULTI_DEVICE` flag when `backends.size() > 1`
- [ ] `negotiate_hello` parses `server_caps` → sets `sock->server_supports_multi_device`
- [ ] `RPC_PROTO_MINOR_VERSION` bumped to 3
- [ ] Old server + new client: fallback to per-device (server_caps = 0)
- [ ] Old client + new server: server_caps byte at padding offset is ignored
- [ ] New server + new client: multi-device mode activated
