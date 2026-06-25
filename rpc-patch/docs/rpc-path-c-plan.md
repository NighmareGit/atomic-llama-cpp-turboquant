# Path C: Multi-GPU RPC Server Aggregation

## Why Separate from Path B

Path B (events + pipeline parallelism) enables the scheduler to overlap RPC compute with local GPU compute **across tokens**. But it does NOT eliminate the fundamental sequentialization WITHIN a single token: the RPC server's graph compute and the local GPU's graph compute still execute one after the other for any given token.

Path C takes a different approach: **eliminate the split boundary entirely** by making the RPC server handle ALL GPUs internally. The client sends one graph, the server distributes it across its local GPUs using the ggml scheduler. This:

1. **Eliminates cross-machine split boundaries** — one RPC call per token instead of multiple per-device calls
2. **Server-side scheduling** — the server can overlap GPU work using local CUDA events (fast PCIe)
3. **Reduces RTT count** — from 7-22 RTTs per token to ~3-5 (one SET_TENSOR batch, one GRAPH_COMPUTE, one GET_TENSOR)

## Architecture

### Current Flow (per token)

```
Client                                    RPC Server
  |-- SET_TENSOR (embeddings) ----------->|
  |-- GRAPH_RECOMPUTE (device 0) -------->|  layers 0-19 on GPU 0
  |<-- (fire-and-forget, no response) ----|
  |                                        | GPU 0 computes layers 0-19
  |-- COPY_TENSOR (hidden state) -------->|  GPU 0 → GPU 1 (server-local PCIe)
  |<-- (copy done) -----------------------|
  |-- GRAPH_RECOMPUTE (device 1) -------->|  layers 20-39 on GPU 1
  |<-- (fire-and-forget) -----------------|
  |                                        | GPU 1 computes layers 20-39
  |-- GET_TENSOR (output) --------------->|
  |<-- (output data) --------------------|
  |-- [local GPU compute layers...]       |
  |                                        |
  TOTAL: 3 RTTs + 3 compute steps (if 2 RPC devices)
```

### Target Flow (single combined compute)

```
Client                                    RPC Server
  |-- SET_TENSOR_BATCH ------------------>|
  |-- GRAPH_COMPUTE_ALL ----------------->|  ALL layers on ALL GPUs
  |<-- (compute done + output data) ------|  server does internal multi-GPU scheduling
  |                                        |
  |                                        | Server's scheduler:
  |                                        |   Split 0: layers 0-19 on GPU 0
  |                                        |   Split 1: layers 20-39 on GPU 1
  |                                        |   GPU copy (server-local, PCIe)
  |                                        |   Compute completes
  |                                        |
  TOTAL: 1 RTT + 1 (combined) compute step
```

### Key Difference

With per-device RPC (current):
- Client's scheduler creates splits between RPC devices
- Client waits for each split sequentially  
- Tensor copies between RPC devices go through the client (`RPC_CMD_COPY_TENSOR`)

With combined RPC (Path C):
- Client sends one graph for ALL server GPUs
- Server's internal scheduler distributes and executes
- Tensor copies between server GPUs are local (CUDA device copies, PCIe P2P)
- Client gets ONE response with the final output

## Implementation Phases

### Phase C1: Baseline Measurement

**Goal:** Measure current multi-GPU RPC performance as baseline.

**Setup:**
- RPC server with 2 GPUs exposed as separate devices
- Client CPU-only, `--rpc SERVER:50052 -ngl 99`
- Measure: tok/s, RTT count per token, server GPU utilization

**Expected baseline:**
- Multiple splits (one per server GPU)
- Sequential execution on server GPUs
- Client-side copy overhead for cross-device transfers

No code changes in this phase. Just benchmark.

---

### Phase C2: RPC Server Internal Scheduler

**Goal:** Make the RPC server capable of running a graph across multiple GPUs using the ggml scheduler INTERNALLY.

**Target files:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp` — server-side graph_compute
- `tools/rpc/rpc-server.cpp` — possibly modify server initialization

#### Step 1: Add server-side scheduler creation

In `rpc_server` class, add a helper to create a scheduler for a subset of devices:

```cpp
class rpc_server {
    // ... existing members ...
    
    // Create a ggml scheduler from a list of device indices
    // The scheduler distributes work across the specified GPU devices + CPU
    ggml_backend_sched_t create_multi_device_sched(
        const uint32_t * devices, uint32_t n_devices,
        const ggml_cgraph * graph) {
        
        std::vector<ggml_backend_t> dev_backends;
        std::vector<ggml_backend_buffer_type_t> dev_bufts;
        
        for (uint32_t i = 0; i < n_devices; i++) {
            uint32_t dev_id = devices[i];
            if (dev_id >= backends.size()) return nullptr;
            dev_backends.push_back(backends[dev_id]);
            dev_bufts.push_back(ggml_backend_get_default_buffer_type(backends[dev_id]));
        }
        
        // Add CPU backend as the last (fallback) backend
        // The CPU backend is always available on the server
        ggml_backend_t cpu_backend = get_cpu_backend();
        dev_backends.push_back(cpu_backend);
        dev_bufts.push_back(ggml_backend_cpu_buffer_type());
        
        size_t graph_size = graph ? graph->n_nodes : 4096;
        
        // Create scheduler WITHOUT pipeline parallelism server-side
        // (that's handled by the client-side scheduler)
        return ggml_backend_sched_new(
            dev_backends.data(), dev_bufts.data(),
            dev_backends.size(), graph_size,
            false,  // no pipeline parallelism server-side
            false   // no op offload
        );
    }
    
private:
    ggml_backend_t get_cpu_backend() {
        static ggml_backend_t cpu_backend = nullptr;
        if (!cpu_backend) {
            cpu_backend = ggml_backend_init_by_name("CPU", nullptr);
        }
        return cpu_backend;
    }
};
```

#### Step 2: Add new RPC command `RPC_CMD_GRAPH_COMPUTE_ALL`

```cpp
enum rpc_cmd {
    // ... existing values ...
    RPC_CMD_GRAPH_COMPUTE_ALL = 18,  // NEW
    RPC_CMD_COUNT,                   // now 19
};

struct rpc_msg_graph_compute_all_req {
    uint32_t n_devices;
    uint32_t devices[GGML_RPC_MAX_DEVICES];  // max 8 devices
    // Followed by serialized graph (same format as GRAPH_COMPUTE)
    // But the graph is the FULL graph (all nodes for all devices),
    // not the subgraph for one device
};

struct rpc_msg_graph_compute_all_rsp {
    uint32_t result;  // 0 = success
};

// And for recompute:
struct rpc_msg_graph_recompute_all_req {
    uint32_t n_devices;
    uint32_t devices[GGML_RPC_MAX_DEVICES];
    uint64_t graph_hash;  // identifies which cached graph to recompute
};

struct rpc_msg_graph_recompute_all_rsp {
    uint32_t result;
};
```

#### Step 3: Server-side handler for `GRAPH_COMPUTE_ALL`

```cpp
// Extend stored_graph to support multi-device:
struct stored_multi_graph {
    uint64_t hash;
    std::vector<uint8_t> buffer;
    ggml_cgraph * graph;
    uint32_t n_devices;
    uint32_t devices[GGML_RPC_MAX_DEVICES];
};

std::vector<stored_multi_graph> stored_multi_graphs;

case RPC_CMD_GRAPH_COMPUTE_ALL: {
    std::vector<uint8_t> input;
    if (!recv_msg(sockfd, input)) return;
    
    // Parse header
    const uint8_t * src = input.data();
    if (input.size() < sizeof(uint32_t)) return;
    uint32_t n_devices;
    memcpy(&n_devices, src, sizeof(n_devices));
    src += sizeof(n_devices);
    
    if (n_devices == 0 || n_devices > GGML_RPC_MAX_DEVICES) return;
    if (input.size() < sizeof(uint32_t) + n_devices * sizeof(uint32_t)) return;
    
    uint32_t devices[GGML_RPC_MAX_DEVICES];
    memcpy(devices, src, n_devices * sizeof(uint32_t));
    src += n_devices * sizeof(uint32_t);
    
    // Remaining data is the serialized graph
    size_t graph_data_size = input.size() - (src - input.data());
    
    // Deserialize the graph
    // (reuse existing deserialization logic from graph_compute)
    // ... create ggml_cgraph from src ...
    
    // Create scheduler for these devices
    auto sched = server.create_multi_device_sched(devices, n_devices, graph);
    if (!sched) return;
    
    // Allocate and compute
    if (!ggml_backend_sched_reserve(sched, graph)) {
        GGML_LOG_ERROR("[%s] failed to reserve graph for multi-device compute\n", __func__);
        ggml_backend_sched_free(sched);
        return;
    }
    
    ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    
    // Cache the graph for recompute
    // ... store in stored_multi_graphs ...
    
    ggml_backend_sched_free(sched);
    
    // Send response
    rpc_msg_graph_compute_all_rsp response = { (status == GGML_STATUS_SUCCESS) ? 0 : 1 };
    if (!send_msg(sockfd, &response, sizeof(response))) return;
    break;
}
```

#### Step 4: Graph serialization for combined compute

The client must serialize the FULL graph (not just a per-device subgraph). The format is:

```
| n_devices (4) | device_ids (4 * n_devices) | serialized_graph_data |
```

Where `serialized_graph_data` is in the same format as existing `GRAPH_COMPUTE` (device + n_nodes + nodes + n_tensors + tensors), but with device=0 (ignored for ALL variant).

**Design decision:** The serialized graph format from the existing code includes a `device` field. For `GRAPH_COMPUTE_ALL`, this field is ignored (the server uses the device list from the header instead).

#### Step 5: Weight tensor handling

All model weights are already on the server (loaded via SET_TENSOR during initialization). The `GRAPH_COMPUTE_ALL` graph references the same weight tensors. Server-side scheduler handles allocation/computation using the existing GPU memory.

The key challenge: **the server-side scheduler must allocate intermediate buffers.** The `ggml_backend_sched_reserve` call handles this. But the client must NOT have already allocated these buffers — which it hasn't, because the client sends the RAW graph, not pre-allocated one.

**This means the client must serialize the COMPUTATION GRAPH (operations), not the pre-allocated memory graph.** The current `ggml_backend_rpc_graph_compute` already does this — it serializes the ggml_cgraph (operations + tensor shapes), and the server deserializes and allocates.

#### Verification
- [ ] Server can create an internal scheduler for multiple devices
- [ ] `GRAPH_COMPUTE_ALL` handler deserializes the graph correctly
- [ ] Server-side scheduler runs the graph across multiple GPUs
- [ ] Results match per-device graph compute

---

### Phase C3: Client-Side Combined Graph Submission

**Goal:** Modify the client to detect when multiple RPC backends share the same endpoint and submit a combined graph.

**Target file:** `ggml/src/ggml-rpc/ggml-rpc.cpp`

#### Step 1: Track endpoint device count

During initialization (`ggml_backend_rpc_add_server`), the number of devices per endpoint is known. Store this information:

```cpp
struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    graph_cache gc;
    uint32_t    n_devices_on_endpoint;  // NEW: total devices at this endpoint
    bool        is_multi_device_capable; // NEW: server supports GRAPH_COMPUTE_ALL
};
```

Set these during `ggml_backend_rpc_init`:

```cpp
ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    // ... existing ...
    auto reg = ggml_backend_rpc_add_server(endpoint);
    auto * reg_ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ctx->n_devices_on_endpoint = reg_ctx ? reg_ctx->devices.size() : 1;
    ctx->is_multi_device_capable = (ctx->n_devices_on_endpoint > 1);
    // ...
}
```

#### Step 2: Modify graph_compute to use ALL variant

When the RPC backend's `graph_compute` is called, and:
- The backend has `n_devices_on_endpoint > 1`
- The server supports `GRAPH_COMPUTE_ALL` (detected during HELLO handshake)

Then instead of serializing for a single device, serialize for ALL devices on the endpoint.

**The challenge:** `ggml_backend_rpc_graph_compute` is called once per split (per device). Each split has a subgraph for that device. To combine them, we need access to ALL splits' subgraphs.

**Approach:** 

1. Detect consecutive same-endpoint splits in the scheduler
2. Buffer subgraphs for all same-endpoint splits
3. Send as one `GRAPH_COMPUTE_ALL` when the last same-endpoint split is encountered

**Simplification for Phase C3:** Don't combine splits in the scheduler. Instead:

1. Each RPC split sends `GRAPH_COMPUTE_ALL` with ALL devices (not just its own device)
2. The server's internal scheduler distributes the full graph across ALL devices
3. Only the FIRST RPC split's graph_compute actually sends the command
4. Subsequent RPC splits' graph_compute become no-ops (the work was already done)

This wastes a bit of work (the full graph is sent each time), but is much simpler to implement.

```cpp
static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    auto sock = get_socket(rpc_ctx->endpoint);
    flush_pending_get_tensor();
    flush_set_tensor_batch(sock);
    
    // Determine if we should use multi-device compute
    if (rpc_ctx->n_devices_on_endpoint > 1 && rpc_ctx->is_multi_device_capable) {
        // Check if this is the first RPC device on this endpoint
        // If the multi-device compute was already sent by another RPC split, skip
        if (rpc_ctx->device == 0) {
            // Send GRAPH_COMPUTE_ALL with all devices
            return graph_compute_all(backend, cgraph, sock, rpc_ctx);
        } else {
            // This split's work was already done by the first split's combined compute
            // Just skip the compute call
            return GGML_STATUS_SUCCESS;
        }
    }
    
    // Original single-device path
    bool reuse = rpc_ctx->gc.is_cached(cgraph);
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    } else {
        rpc_ctx->gc.add(cgraph);
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}
```

#### Step 3: The graph_compute_all helper

```cpp
static enum ggml_status graph_compute_all(ggml_backend_t backend, ggml_cgraph * cgraph,
    const std::shared_ptr<socket_t> & sock, ggml_backend_rpc_context * rpc_ctx) {
    
    // Build the combined request
    // Format: | n_devices (4) | device_ids (4 * n) | serialized_full_graph |
    
    // For the first invocation, serialize the graph as if it's for device 0
    // (the server ignores the device field in ALL mode)
    std::vector<uint8_t> graph_data;
    serialize_graph(0, cgraph, graph_data);
    
    // Prepend the device list
    uint32_t n_devices = rpc_ctx->n_devices_on_endpoint;
    size_t header_size = sizeof(uint32_t) + n_devices * sizeof(uint32_t);
    std::vector<uint8_t> input(header_size + graph_data.size());
    
    uint8_t * p = input.data();
    memcpy(p, &n_devices, sizeof(n_devices)); p += sizeof(n_devices);
    for (uint32_t i = 0; i < n_devices; i++) {
        uint32_t dev_id = i;
        memcpy(p, &dev_id, sizeof(dev_id)); p += sizeof(dev_id);
    }
    memcpy(p, graph_data.data(), graph_data.size());
    
    // Check graph cache
    // For multi-device, we cache based on first device's graph content
    // (the full graph is the same for all devices)
    bool reuse = rpc_ctx->gc.is_cached(cgraph);
    if (reuse) {
        rpc_msg_graph_recompute_all_req req;
        req.n_devices = n_devices;
        memcpy(req.devices, input.data() + sizeof(uint32_t), n_devices * sizeof(uint32_t));
        req.graph_hash = 0;  // TODO: use actual hash
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE_ALL, &req, sizeof(req));
        RPC_STATUS_ASSERT(status);
    } else {
        rpc_ctx->gc.add(cgraph);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE_ALL, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    
    return GGML_STATUS_SUCCESS;
}
```

#### Verification
- [ ] Client detects same-endpoint devices
- [ ] Only the first RPC split sends `GRAPH_COMPUTE_ALL`
- [ ] Subsequent RPC splits skip their compute (work already done)
- [ ] Results match per-device compute

---

### Phase C4: Combined Output Retrieval

**Goal:** After `GRAPH_COMPUTE_ALL`, the server has the final output in its GPU memory. The client must read it back with one `GET_TENSOR` call instead of multiple per-device reads.

**Target file:** `ggml/src/ggml-rpc/ggml-rpc.cpp`

#### Step 1: Track which device has the output

After multi-device compute, the output tensor is on whichever GPU the scheduler placed it (typically the last GPU that ran the output layer). The client needs to know which device to read from.

**Approach:** The server returns the output device in the response:

```cpp
struct rpc_msg_graph_compute_all_rsp {
    uint32_t result;
    uint32_t output_device;  // device index where output tensor lives
};
```

Or simpler: always read from device 0, and have the server copy the output to device 0 if needed. Actually, the simplest approach: the client reads the output tensor from whichever device it was originally assigned to (the scheduler decides this). For RPC, the output is typically on the LAST GPU in the device list.

**Simplest approach:** The client reads from the last device:

```cpp
// In ggml_backend_rpc_buffer_get_tensor:
// If multi-device mode is active, the output tensor is on the last GPU
// Just read from whatever tensor the client has (its buffer knows which device)
// No changes needed — the existing GET_TENSOR path reads from the correct device
```

#### Step 2: Server-side cache for combined graphs

Add a cache on the server for combined graphs so subsequent recompute calls are fast:

```cpp
// In rpc_server class:
std::unordered_map<uint64_t, stored_multi_graph> multi_graph_cache;

bool graph_compute_all(const std::vector<uint8_t> & input) {
    // ... parse and compute ...
    
    // Cache the graph for recompute
    uint64_t hash = hash_graph(graph);
    multi_graph_cache[hash] = { hash, buffer, graph, n_devices, devices };
    
    return true;
}

bool graph_recompute_all(const rpc_msg_graph_recompute_all_req & request) {
    auto it = multi_graph_cache.find(request.graph_hash);
    if (it == multi_graph_cache.end()) return false;
    
    auto & cached = it->second;
    auto sched = create_multi_device_sched(cached.devices, cached.n_devices, cached.graph);
    // ... compute without re-allocating ...
    ggml_backend_sched_graph_compute(sched, cached.graph);
    ggml_backend_sched_free(sched);
    
    return true;
}
```

#### Verification
- [ ] Output tensor readback works correctly
- [ ] Multi-device graph cache works for recompute
- [ ] `RPC_CMD_GRAPH_RECOMPUTE_ALL` is handled correctly

---

### Phase C5: HELLO Negotiation for Multi-Device Capability

**Goal:** The client needs to know if the server supports `GRAPH_COMPUTE_ALL` before using it.

**Target file:** `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`

#### Step 1: Protocol version bump

```cpp
#define RPC_PROTO_MAJOR_VERSION 0
#define RPC_PROTO_MINOR_VERSION 3  // was 2
#define RPC_PROTO_PATCH_VERSION 0
```

Or add a capability flag to the HELLO handshake:

```cpp
// In rpc_msg_hello_rsp or a separate caps exchange:
struct rpc_msg_hello_rsp {
    // ... existing fields ...
    uint8_t server_caps;  // NEW: bit 0 = supports GRAPH_COMPUTE_ALL
};

#define RPC_CAP_MULTI_DEVICE (1 << 0)
```

#### Step 2: Client-side capability check

```cpp
// In negotiate_hello or during initialization:
if (response.server_caps & RPC_CAP_MULTI_DEVICE) {
    sock->server_supports_multi_device = true;
}
```

#### Step 3: Fallback

If the server doesn't support multi-device, fall back to per-device GRAPH_COMPUTE. This is the existing behavior.

#### Verification
- [ ] Old server + new client: works via fallback
- [ ] New server + old client: old client sends per-device commands, server responds normally
- [ ] New server + new client: multi-device mode used

---

### Success Criteria for Path C

- **RTTs per token reduced**: from 7-22 to ~3-5 (one SET_TENSOR batch, one GRAPH_COMPUTE_ALL, one GET_TENSOR)
- **Correct results**: output matches per-device compute bit-exact
- **Throughput improvement**: measurable gain in cross-GPU RPC scenarios
- **Backward compatible**: old clients work with new server and vice versa

---

## Red-Teaming / Debugging for Both Paths

### Common Issues

1. **Memory corruption from deferred response reads**
   - Symptom: garbled tensor data, assertion failures
   - Fix: ensure ALL pending responses are drained before any socket read
   - Test: `valgrind --tool=memcheck ./llama-cli --rpc ...`

2. **Deadlock from mismatched request/response ordering**
   - Symptom: client hangs waiting for response that never comes
   - Fix: add socket timeout (set `SO_RCVTIMEO`), log and abort on timeout
   - Test: use `strace -e trace=network` to trace all send/recv calls

3. **Event response consumed by wrong function**
   - Symptom: `recv_data` returns data from the wrong command
   - Fix: drain events before any non-event recv
   - Test: `GGML_RPC_DEBUG=1` logs every command with its expected response size

4. **Scheduler asserts on event functions**
   - `ggml_backend_event_record` asserts `backend->iface.event_record != NULL` (line 540)
   - `ggml_backend_event_wait` asserts `backend->iface.event_wait != NULL` (line 555)
   - Fix: ensure these are non-NULL in `ggml_backend_rpc_interface`

5. **Scheduler allocates too much memory with n_copies=4**
   - Symptom: OOM on RPC server or client
   - Fix: reduce `GGML_SCHED_MAX_COPIES` (default 4) via CMake or env var

### Debugging Protocol

1. Set `GGML_RPC_DEBUG=1` and `GGML_SCHED_DEBUG=1`
2. Log every RPC command send/recv with sizes
3. Log every event_record/event_wait call with event ID
4. Log every split boundary and copy operation
5. On failure, replay with `strace -e trace=network` to capture raw I/O

### Rollback Plan

If a phase causes regression:
1. Revert the specific phase's changes
2. Add a runtime flag to disable the feature (e.g., environment variable)
3. Document the failure mode and share with the team
