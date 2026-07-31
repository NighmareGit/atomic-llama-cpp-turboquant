# Path B: Event-Based Pipeline Parallelism for RPC Backend

## Why Path A1+A2 Failed to Improve Generation

**Benchmark result:** A1 (batch SET_TENSOR) gave +3.6% prompt, 0% generation. A2 (pipelined GET_TENSOR) gave 0% improvement.

**Root cause:** The generation bottleneck is **GRAPH_COMPUTE** (actual GPU compute on the RPC worker), not protocol overhead. The RPC worker computes its layers, sends the result back, then the client computes its layers — all **sequentially**. Protocol optimizations cannot fix this because the compute itself is the bottleneck — neither sending SET_TENSOR batches nor pipelining GET_TENSOR responses reduces the time spent waiting for `ggml_backend_graph_compute` to finish on the remote server.

**Path B fixes this at the architecture level** by implementing **event-based pipeline parallelism**. This allows the scheduler to overlap different tokens' compute across backends using double-buffered copy slots (n_copies > 1). While one token's GPU split runs on the local GPU, the next token's RPC split can already begin on the RPC server.

## Architecture

### How Pipeline Parallelism Works (with CUDA)

```
Without pipeline (n_copies=1):
  Token N:   [RPC_copy][RPC_compute]         [GPU_copy][GPU_compute]
  Token N+1:                                              [RPC_copy][RPC_compute]...
  → No overlap. 2 tokens = 2 × (RPC + GPU) time.

With pipeline (n_copies=2, events working):
  Token N:   [RPC_C0][RPC_COMP0]              [GPU_C0][GPU_COMP0]
  Token N+1:                     [RPC_C1][RPC_COMP1]           [GPU_C1][GPU_COMP1]
                                ^-- can start while GPU_COMP0 is still running!
  → Overlap: RPC work for Token N+1 starts BEFORE GPU finishes Token N.
```

The overlap is possible because:
1. Copy slot 0 is used by token N; copy slot 1 by token N+1
2. `event_record` on (RPC, slot 0) signals "RPC compute for slot 0 is done"
3. `event_wait` on (RPC, slot 1) at the start of token N+1 checks: "was slot 1 used by token N-1?" Yes, and token N-1 finished long ago, so slot 1 is available immediately
4. Token N+1's RPC copy+compute can start while token N's GPU compute is still running

### Requirements for RPC Backend

To enable this, the RPC backend needs:
1. **`event_new`, `event_free`, `event_synchronize`** on `ggml_backend_device_i` — device-level event management
2. **`event_record`, `event_wait`** on `ggml_backend_i` — stream-level event sync
3. **`caps.events = true`** so the pipeline detection in `llama-context.cpp:369` passes
4. **`caps.async = true`** (already partially supported by A2's `get_tensor_async`)

### How RPC Events Work Over TCP

RPC events use **TCP ordering** rather than CUDA events:

1. After `graph_compute_async` sends `RPC_CMD_GRAPH_RECOMPUTE` (fire-and-forget), the server is computing.
2. `event_record` sends `RPC_CMD_EVENT_RECORD` to the server. Due to TCP ordering, the server processes this AFTER the GRAPH_RECOMPUTE completes. The server replies immediately with a success response.
3. The event is "complete" when the EVENT_RECORD response arrives at the client.
4. `event_wait` or `event_synchronize` reads the pending EVENT_RECORD response (blocking if not yet arrived).

## Current State

The RPC backend currently has:
- `caps.async = false` (in `ggml_backend_rpc_device_get_props`, line 2727)
- `caps.events = false` (line 2730)
- `event_record = NULL` (line 1583)
- `event_wait = NULL` (line 1584)
- `event_new/free/synchronize = NULL` on device interface (lines 2779-2781)
- `synchronize` is a no-op that flushes pending GET_TENSOR (A2 change, line 1498-1501)
- `get_tensor_async` is populated (A2 change, line 1573)

## Implementation Phases

### Phase B1: Add Event Commands to RPC Protocol

**Goal:** Define and implement `RPC_CMD_EVENT_RECORD` and `RPC_CMD_EVENT_WAIT` on both client and server.

**Target file:** `ggml/src/ggml-rpc/ggml-rpc.cpp` + `ggml/include/ggml-rpc.h`

#### Step 1: Add enum values

In `ggml/include/ggml-rpc.h` (or at the top of `ggml-rpc.cpp` where the enum is):

```cpp
enum rpc_cmd {
    RPC_CMD_GRAPH_COMPUTE = 12,   // existing
    RPC_CMD_HELLO = 14,           // existing (must be 14)
    RPC_CMD_DEVICE_COUNT = 15,    // existing
    RPC_CMD_GRAPH_RECOMPUTE = 16, // existing
    RPC_CMD_SET_TENSOR_BATCH = 17, // existing (A1)
    RPC_CMD_GRAPH_COMPUTE_V2 = 18, // NEW: graph compute + implicit event
    RPC_CMD_COUNT,                 // was 18, now 19
};
```

**Design choice:** Instead of sending separate `EVENT_RECORD` after `GRAPH_RECOMPUTE` (which requires an extra RTT), add a **new** `GRAPH_COMPUTE_V2` command that:
1. Performs the graph compute
2. Sends back a response when the compute is DONE (vs the fire-and-forget of V1)

This eliminates one RTT: the response to `GRAPH_COMPUTE_V2` IS the event notification.

```cpp
struct rpc_msg_graph_compute_v2_req {
    uint32_t device;
    uint64_t event_id;   // client-generated event ID
    // Followed by serialized graph data (same format as GRAPH_COMPUTE)
};

struct rpc_msg_graph_compute_v2_rsp {
    uint64_t event_id;
    uint32_t device;
    uint32_t result;     // 0 = success, non-zero = error
};
```

For cached graphs, add `GRAPH_RECOMPUTE_V2`:
```cpp
struct rpc_msg_graph_recompute_v2_req {
    uint32_t device;
    uint64_t event_id;
};

struct rpc_msg_graph_recompute_v2_rsp {
    uint64_t event_id;
    uint32_t device;
    uint32_t result;
};
```

**Why this approach:** The original `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE` are fire-and-forget (no response). Adding a response version gives us free event signaling. The event fires when the response arrives, which TCP ordering guarantees is after the compute finishes.

#### Step 2: Client-side event struct

```cpp
// In ggml-rpc.cpp, add event struct

struct rpc_event_t {
    uint64_t id;
    std::shared_ptr<socket_t> sock;  // socket to read response from
    bool response_pending;           // true if we've sent the compute but haven't read the response
};

// Device-level event functions
static ggml_backend_event_t rpc_event_new(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    static std::atomic<uint64_t> next_id{1};
    auto * ev = new rpc_event_t();
    ev->id = next_id++;
    ev->sock = nullptr;
    ev->response_pending = false;
    return (ggml_backend_event_t)ev;
}

static void rpc_event_free(ggml_backend_dev_t device, ggml_backend_event_t event) {
    GGML_UNUSED(device);
    delete (rpc_event_t *)event;
}

static void rpc_event_synchronize(ggml_backend_dev_t device, ggml_backend_event_t event) {
    GGML_UNUSED(device);
    auto * ev = (rpc_event_t *)event;
    if (ev->response_pending) {
        // Read the GRAPH_COMPUTE_V2 response from the server
        rpc_msg_graph_compute_v2_rsp rsp;
        recv_data(ev->sock.get(), &rsp, sizeof(rsp));
        if (rsp.result != 0) {
            GGML_LOG_ERROR("RPC event %lu: graph compute failed with result %u\n", ev->id, rsp.result);
        }
        ev->response_pending = false;
    }
}
```

#### Step 3: Backend-level event_record and event_wait

```cpp
// Backend-level event_record (called by scheduler after graph_compute_async)
static void rpc_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    auto * ev = (rpc_event_t *)event;
    auto sock = get_socket(rpc_ctx->endpoint);
    
    ev->sock = sock;
    
    if (/* graph was recomputed (cached) */) {
        rpc_msg_graph_recompute_v2_req req = { rpc_ctx->device, ev->id };
        flush_set_tensor_batch(sock);
        // Send with response — the response IS the event completion signal
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE_V2,
            &req, sizeof(req), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    } else {
        // The full graph was sent via GRAPH_COMPUTE_V2 already
        // The response is already pending — just mark it
    }
    ev->response_pending = true;
}

// Backend-level event_wait (called by scheduler to make ANOTHER backend wait for event)
// For RPC events: the event MUST complete before any other backend can use the data.
// Since we use TCP ordering, the event_wait synchronizes the event (reads the response).
static void rpc_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    // Synchronize the event — reads the pending response
    auto * ev = (rpc_event_t *)event;
    if (ev->response_pending && ev->sock) {
        rpc_msg_graph_compute_v2_rsp rsp;
        recv_data(ev->sock.get(), &rsp, sizeof(rsp));
        ev->response_pending = false;
    }
}
```

#### Step 4: Modify graph_compute to use V2 when events are active

The scheduler calls `graph_compute_async` FIRST, then `event_record`. For the RPC backend:

- When events are NOT active: use original fire-and-forget `GRAPH_COMPUTE`/`GRAPH_RECOMPUTE`
- When events ARE active: use `GRAPH_COMPUTE_V2`/`GRAPH_RECOMPUTE_V2` that return a response

```cpp
// Store whether events are active in the RPC context
struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    graph_cache gc;
    bool events_active = false;  // NEW: set when scheduler creates events for this backend
};
```

The backend doesn't know if events are active. But `event_record` is only called when events exist (n_copies > 1). So:

- In `ggml_backend_rpc_graph_compute`: always use V1 (fire-and-forget) for now
- In `rpc_backend_event_record`: if this is called, it means events are active. The event_record sends the V2 command (or more precisely, the V2 command was already sent as part of graph_compute).

**Simplification:** Modify `graph_compute` to ALWAYS use V2 when events are available. Since events are only created when pipeline parallel is active, this is safe:

```cpp
static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    auto sock = get_socket(rpc_ctx->endpoint);
    flush_pending_get_tensor();
    flush_set_tensor_batch(sock);
    
    bool reuse = rpc_ctx->gc.is_cached(cgraph);
    if (reuse) {
        if (rpc_ctx->events_active) {
            rpc_msg_graph_recompute_v2_req req = { rpc_ctx->device };
            // Don't read response yet — event_record will own it
            // We need to store where to read the response
            flush_set_tensor_batch(sock);
            uint8_t cmd = RPC_CMD_GRAPH_RECOMPUTE_V2;
            send_data(sock.get(), &cmd, 1);
            uint64_t sz = sizeof(req);
            send_data(sock.get(), &sz, sizeof(sz));
            send_data(sock.get(), &req, sizeof(req));
            // Response is pending in the TCP buffer
        } else {
            rpc_msg_graph_recompute_req request;
            request.device = rpc_ctx->device;
            send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        }
    } else {
        rpc_ctx->gc.add(cgraph);
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        if (rpc_ctx->events_active) {
            // V2: prepend with event_id, send, response is pending
            // ... send GRAPH_COMPUTE_V2 ...
        } else {
            send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        }
    }
    return GGML_STATUS_SUCCESS;
}
```

#### Step 5: Server-side handlers

Add server-side handlers for `RPC_CMD_GRAPH_COMPUTE_V2` and `RPC_CMD_GRAPH_RECOMPUTE_V2`:

```cpp
case RPC_CMD_GRAPH_RECOMPUTE_V2: {
    rpc_msg_graph_recompute_v2_req request;
    if (!recv_msg(sockfd, &request, sizeof(request))) return;
    
    uint32_t device = request.device;
    if (device >= backends.size()) return;
    if (stored_graphs[device].graph == nullptr) return;
    
    ggml_cgraph * graph = stored_graphs[device].graph;
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    
    rpc_msg_graph_recompute_v2_rsp response;
    response.event_id = request.event_id;
    response.device = device;
    response.result = (status == GGML_STATUS_SUCCESS) ? 0 : 1;
    
    if (!send_msg(sockfd, &response, sizeof(response))) return;
    break;
}

case RPC_CMD_GRAPH_COMPUTE_V2: {
    // Similar to existing GRAPH_COMPUTE but includes event_id in request
    // and sends back a response after compute completes
    // ... parse request, compute, send response ...
    break;
}
```

#### Verification
- [ ] Server accepts `GRAPH_COMPUTE_V2` and sends response after compute
- [ ] Client reads response correctly in `event_wait`/`event_synchronize`
- [ ] TCP ordering guaranteed: V2 response is received AFTER compute completes
- [ ] No deadlock when both V1 and V2 commands are mixed on the same socket

---

### Phase B2: Register Event Functions on RPC Backend Interfaces

**Goal:** Wire up all event function pointers and set capability flags.

**Target file:** `ggml/src/ggml-rpc/ggml-rpc.cpp`

#### Step 1: Register device-level event functions

```cpp
// In ggml_backend_rpc_device_i (around line 2760):
static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ rpc_event_new,           // was NULL
    /* .event_free           = */ rpc_event_free,          // was NULL
    /* .event_synchronize    = */ rpc_event_synchronize,   // was NULL
};
```

#### Step 2: Register backend-level event functions

```cpp
// In ggml_backend_rpc_interface (around line 1569):
static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ rpc_backend_event_record,  // was NULL
    /* .event_wait              = */ rpc_backend_event_wait,    // was NULL
    /* .graph_optimize          = */ NULL,
};
```

#### Step 3: Update device capabilities

```cpp
// In ggml_backend_rpc_device_get_props (around line 2721):
props->caps = {
    /* .async                 = */ true,   // Changed from false — get_tensor_async exists
    /* .host_buffer           = */ false,
    /* .buffer_from_host_ptr  = */ false,
    /* .events                = */ true,   // Changed from false — event functions exist
};
```

**IMPORTANT:** With `caps.events = true`, the scheduler will create events (`ggml_backend_event_new`) when constructing the scheduler with `parallel=true`. But the scheduler also checks `caps.async` — which is now true.

#### Step 4: Set `events_active` on the backend context

The scheduler creates events when `n_copies > 1` AND the backend supports events. We need to detect this in the graph_compute path.

**Approach:** Check if the backend has events configured in the scheduler. Since the backend doesn't have direct access to the scheduler, use a different approach:

- Always use V2 (with response) when `caps.events = true`
- The response is read in `event_wait`/`event_synchronize`
- If no one calls `event_wait`/`event_synchronize`, the response sits in the TCP buffer (which blocks the next send)

**Better approach:** Track whether events are active via the `graph_compute` → `event_record` sequence:

```cpp
struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    graph_cache gc;
    
    // Event tracking — set by graph_compute, consumed by event_record
    std::shared_ptr<socket_t> compute_sock;   // socket used for last compute
    bool compute_response_pending = false;    // true if V2 was sent, response not yet read
};
```

In `graph_compute`:
```cpp
if (rpc_ctx->events_active /* always true when caps.events=true */) {
    // Send V2
    flush_set_tensor_batch(sock);
    uint8_t cmd = RPC_CMD_GRAPH_RECOMPUTE_V2;
    send_data(sock.get(), &cmd, 1);
    // ... send request ...
    rpc_ctx->compute_sock = sock;
    rpc_ctx->compute_response_pending = true;
}
```

In `event_record`:
```cpp
// Mark that the event should read the pending compute response
ev->sock = rpc_ctx->compute_sock;
ev->response_pending = rpc_ctx->compute_response_pending;
rpc_ctx->compute_response_pending = false;
```

**Alternative (simpler):** Don't track. Always send V2. The event_record always expects a response. The response is read in event_synchronize or event_wait.

```cpp
// graph_compute:
static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // ... existing logic ...
    if (reuse) {
        // Always send V2 — the response is pending
        uint8_t cmd = RPC_CMD_GRAPH_RECOMPUTE_V2;
        send_data(sock.get(), &cmd, 1);
        uint64_t req_size = sizeof(rpc_msg_graph_recompute_v2_req);
        send_data(sock.get(), &req_size, sizeof(req_size));
        rpc_msg_graph_recompute_v2_req req = { rpc_ctx->device };
        send_data(sock.get(), &req, sizeof(req));
        // Response is in the TCP receive buffer
        // It will be read by the first event_wait or event_synchronize call
    }
    // ...
}

// event_record:
static void rpc_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    auto * ev = (rpc_event_t *)event;
    auto sock = get_socket(rpc_ctx->endpoint);
    ev->sock = sock;
    ev->response_pending = true;  // The V2 response from graph_compute is pending
}

// event_synchronize / event_wait:
static void rpc_event_synchronize(ggml_backend_dev_t device, ggml_backend_event_t event) {
    auto * ev = (rpc_event_t *)event;
    if (ev->response_pending && ev->sock) {
        // Read the pending V2 response
        // Note: recv_data reads ENTIRE message, must read exact size
        // But send_rpc_cmd sends |size(8)|data|, and we sent via raw send_data...
        // We need to handle the framing correctly.
        
        // The V2 response was sent by the server using send_msg which sends:
        // uint64_t size + data
        // So we need to read the size first, then the data:
        uint64_t resp_size;
        recv_data(ev->sock.get(), &resp_size, sizeof(resp_size));
        rpc_msg_graph_recompute_v2_rsp rsp;
        recv_data(ev->sock.get(), &rsp, sizeof(rsp));
        ev->response_pending = false;
    }
}
```

**CRITICAL FRAMING CONSIDERATION:**

The existing `send_rpc_cmd` function does:
```
send_data(sock, &cmd, 1)
send_data(sock, &input_size, 8)
send_data(sock, input_data, input_size)
[if response_buf: recv_data(sock, &output_size, 8)]
[if response_buf: recv_data(sock, output_buf, output_size)]
```

For the V2 commands, we're sending the request via raw `send_data` (without `send_rpc_cmd`), and reading the response via raw `recv_data`. We must be consistent with the framing.

**Recommendation:** Use `send_rpc_cmd` for V2 as well, but pass a response buffer pointer. The response will be read in `event_wait`/`event_synchronize`.

But `send_rpc_cmd` is synchronous — it reads the response immediately. We need to DEFER the response read.

**Solution:** Create a new helper:

```cpp
// Send a command that expects a response later (deferred read)
static void send_rpc_cmd_deferred(const std::shared_ptr<socket_t> & sock, uint8_t cmd,
    const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    send_data(sock.get(), &cmd_byte, 1);
    uint64_t net_size = input_size;
    send_data(sock.get(), &net_size, sizeof(net_size));
    send_data(sock.get(), input, input_size);
    // Response is in the TCP buffer — read it later
}
```

And for reading the deferred response:

```cpp
// Read a deferred response (blocking)
static void recv_rpc_cmd_deferred(const std::shared_ptr<socket_t> & sock, void * output, size_t output_size) {
    uint64_t resp_size;
    recv_data(sock.get(), &resp_size, sizeof(resp_size));
    if (resp_size != output_size) {
        GGML_ABORT("RPC deferred response size mismatch: expected %zu, got %zu", output_size, (size_t)resp_size);
    }
    recv_data(sock.get(), output, output_size);
}
```

#### Verification
- [ ] `ggml_backend_event_new` returns a valid event (not NULL)
- [ ] `event_record` + `event_wait` sequence works correctly
- [ ] `caps.async = true` and `caps.events = true` are visible to the scheduler
- [ ] Scheduler creates events for RPC backend when pipeline parallel is enabled
- [ ] `ggml_backend_event_new` in the scheduler (line 1776) does not receive NULL

---

### Phase B3: Enable Pipeline Parallelism Detection

**Goal:** Ensure `pipeline_parallel = true` is set when RPC + local GPU are used.

**Target file:** `src/llama-context.cpp`

#### Step 1: No code changes needed

The detection logic at lines 350-377 of `llama-context.cpp` already checks:
```cpp
bool pipeline_parallel =
    model.n_devices() > 1 &&
    model.n_gpu_layers() > model.hparams.n_layer &&
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
    cparams.offload_kqv &&
    !model.has_tensor_overrides();

if (pipeline_parallel) {
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.events) {
            pipeline_parallel = false;
            break;
        }
    }
}
```

With `caps.async = true` and `caps.events = true` from Phase B2, this loop will pass for the RPC backend. `pipeline_parallel` remains `true`.

#### Step 2: Verify the log

Run with the patched code and verify:
```
pipeline parallelism enabled  ← should appear in the log
```

#### Step 3: Handle the `n_copies > 1` memory increase

The scheduler allocates `GGML_SCHED_MAX_COPIES` (default 4) copy buffers per tensor per backend when `parallel=true`. This increases memory usage. For the RPC backend, this means 4× the intermediate buffer memory on the RPC server.

Monitor VRAM usage to ensure no OOM.

#### Step 4: Test with `GGML_SCHED_DEBUG=1`

```bash
GGML_SCHED_DEBUG=1 ./build/bin/llama-cli --rpc SERVER:50052 -m model.gguf -p "hello" -n 256
```

Expected output should show:
- `n_splits > 1`
- `n_copies = 4`
- Clean event lifecycle (no assertion failures)

#### Verification
- [ ] Log shows `pipeline parallelism enabled`
- [ ] `n_copies` is set to 4 (GGML_SCHED_MAX_COPIES)
- [ ] RPC backend events are created successfully
- [ ] No OOM on RPC server

---

### Phase B4: Tuning and Edge Cases

**Goal:** Fix any edge cases in the event lifecycle and measure performance.

**Target files:** `ggml/src/ggml-rpc/ggml-rpc.cpp`, `ggml/src/ggml-backend.cpp`

#### Edge Case 1: Event creation fails

`ggml_backend_event_new` returns NULL if `device == NULL` or `device->iface.event_new == NULL`. Our `rpc_event_new` is non-NULL, so this should NOT return NULL. But verify:

```cpp
// In ggml_backend_sched_new, line 1776:
sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
// If this returns NULL, the scheduler will still work but without pipeline benefit
// The scheduler handles NULL events gracefully (falls back to synchronize)
```

#### Edge Case 2: Response framing mismatch

The V2 response from the server must use the EXACT same framing as what the client expects.

Server-side (`rpc_serve_client`):

```cpp
case RPC_CMD_GRAPH_RECOMPUTE_V2: {
    rpc_msg_graph_recompute_v2_req request;
    if (!recv_msg(sockfd, &request, sizeof(request))) return;
    // ... compute ...
    rpc_msg_graph_recompute_v2_rsp response = { request.event_id, request.device, 0 };
    if (!send_msg(sockfd, &response, sizeof(response))) return;  // send_msg handles framing
    break;
}
```

Client-side (`event_synchronize`):

The response was sent by `send_msg` which uses the framing: `| size (8) | data (size) |`. When reading:

```cpp
// In rpc_event_synchronize:
uint64_t resp_size;
recv_data(ev->sock.get(), &resp_size, sizeof(resp_size));  // read framing size
rpc_msg_graph_recompute_v2_rsp rsp;
recv_data(ev->sock.get(), &rsp, sizeof(rsp));              // read response data
```

This MUST match what `send_msg` produces. Look at `send_msg` definition to confirm framing format.

#### Edge Case 3: Mixing V1 and V2 commands on the same socket

The RPC backend uses V1 for `SET_TENSOR`, `GET_TENSOR`, `COPY_TENSOR`, and V2 for `GRAPH_COMPUTE`/`GRAPH_RECOMPUTE`. The TCP stream looks like:

```
| CMD_SET_TENSOR | ... | CMD_GRAPH_RECOMPUTE_V2 | req |  [response pending]
| CMD_GET_TENSOR | ... | 
```

After the `GRAPH_RECOMPUTE_V2` request, the next `recv_data` call (for `GET_TENSOR` response or `V2` response) must correctly identify what it's reading.

**This is a CRITICAL issue.** The TCP stream after a V2 request contains:
1. The V2 response (sent by the server after compute finishes)
2. Then responses to subsequent commands

If the client sends `GET_TENSOR` after `GRAPH_RECOMPUTE_V2`, the TCP stream is:
```
[client sends V2 request]
[client sends GET_TENSOR request]  ← get_tensor does send_rpc_cmd which reads response immediately
But the server hasn't processed V2 yet! The GET_TENSOR response isn't available.
```

The client's `send_rpc_cmd` for GET_TENSOR will try to read the response, but the server hasn't sent it yet (it's still computing). Worse, the first bytes in the receive buffer will be the V2 response (which will be misinterpreted as the GET_TENSOR response).

**SOLUTION:** When using V2 (events active), the `GET_TENSOR` response comes AFTER the V2 response. The client must drain the V2 response BEFORE processing GET_TENSOR.

This is naturally handled if event_wait/synchronize is called before any GET_TENSOR. The scheduler's flow ensures this:

```
Token N:
  Split 0 (RPC):
    event_wait → reads previous token's V2 response (if any)
    copy inputs (SET_TENSOR)
    graph_compute_async → sends V2 request (response pending)
    event_record → marks "V2 response is pending"
    
  Split 1 (GPU):
    copy inputs → calls GET_TENSOR to read RPC output
    BUT: GET_TENSOR calls flush_pending_get_tensor() which would
    try to read... wait. The GET_TENSOR is a new call, not a pipelined one.
```

Actually, the GET_TENSOR in the scheduler's copy loop goes through `ggml_backend_tensor_copy` → `ggml_backend_tensor_get` → `ggml_backend_rpc_buffer_get_tensor` → `send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, ..., data, size)`.

This `send_rpc_cmd` sends GET_TENSOR request and reads the response. The response arrives AFTER the server processes GET_TENSOR. But the V2 response is still pending in the TCP buffer!

The server processes commands in order:
1. V2 request → compute → send V2 response
2. GET_TENSOR request → send GET_TENSOR response

So the TCP receive buffer at the client contains:
```
| V2 response size | V2 response data | GET_TENSOR response size | GET_TENSOR response data |
```

When get_tensor's `send_rpc_cmd` reads:
```
recv_data(&resp_size, 8) → reads V2 response size (WRONG!)
```

This is a framing mismatch!

**FIX:** Before any GET_TENSOR, drain the V2 response:

```cpp
// In ggml_backend_rpc_buffer_get_tensor (sync version):
static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto sock = ctx->sock;
    flush_pending_get_tensor();
    flush_set_tensor_batch(sock);
    
    // NEW: drain any pending V2 event response
    // The event_ctx stores whether a V2 response is pending
    if (tls_event_response_pending) {
        uint64_t resp_size;
        recv_data(sock.get(), &resp_size, sizeof(resp_size));
        // We don't care about the content — just drain it
        std::vector<uint8_t> dummy(resp_size);
        recv_data(sock.get(), dummy.data(), resp_size);
        tls_event_response_pending = false;
    }
    
    // Proceed with normal GET_TENSOR
    rpc_msg_get_tensor_req request;
    // ...
}
```

**Better approach:** Instead of thread_local flags, manage this at the event level:

```cpp
// In rpc_backend_event_record:
// Store the expected response size for the pending event
tls_pending_event_resp_size = sizeof(rpc_msg_graph_recompute_v2_rsp);

// In any RPC function that reads from the socket:
// Before reading a response, check if there's a pending event response
static void drain_pending_event_response(const std::shared_ptr<socket_t> & sock) {
    if (tls_pending_event_resp_size > 0) {
        uint64_t resp_size;
        recv_data(sock.get(), &resp_size, sizeof(resp_size));
        std::vector<uint8_t> dummy(resp_size);
        recv_data(sock.get(), dummy.data(), resp_size);
        tls_pending_event_resp_size = 0;
    }
}
```

But this is fragile. A better design:

**Use a dedicated monitoring thread or non-blocking socket for V2 responses.**

**Simplest correct approach:** Abandon the deferred response approach. Instead:

1. `graph_compute_async` sends V1 command (fire-and-forget, no response)
2. `event_record` sends the EVENT notification as a SEPARATE command

```cpp
// Don't use V2 — use separate EVENT_RECORD command
RPC_CMD_EVENT_RECORD,

struct rpc_msg_event_record_req {
    uint64_t event_id;
    uint32_t device;
};

struct rpc_msg_event_record_rsp {
    uint64_t event_id;
    uint32_t result;
};
```

The server processes `EVENT_RECORD` after `GRAPH_RECOMPUTE` (TCP ordering). The response tells the client that the compute is done.

**This avoids the framing issue entirely** because:
1. `GRAPH_RECOMPUTE` is V1 (no response, as before)
2. `EVENT_RECORD` is a SEPARATE command with its own response
3. The event response is read by `event_synchronize`/`event_wait`, which are called by the scheduler

The only cost: 1 extra RTT per token (send EVENT_RECORD request, read response). But:
- The EVENT_RECORD request+response is very small (~20 bytes)
- The response can be deferred (sent with V1 compute, read later)
- Or it can be synchronous (sent after compute, read immediately)

**For maximum performance:** 
- Send EVENT_RECORD immediately (no extra RTT — it piggybacks on the same TCP stream)
- Read the EVENT_RECORD response in `event_wait` or `event_synchronize`
- If no one calls event_wait/synchronize, drain it before the next socket read

#### Recommended Implementation: Separate EVENT_RECORD Command

This is the cleanest approach:

```cpp
// In graph_compute (no changes needed — still V1 fire-and-forget):
// ... existing code ...

// In event_record:
static void rpc_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    auto * ev = (rpc_event_t *)event;
    auto sock = get_socket(rpc_ctx->endpoint);
    
    ev->sock = sock;
    
    // Send EVENT_RECORD command — server processes it after GRAPH_RECOMPUTE (TCP ordering)
    rpc_msg_event_record_req req = { ev->id, rpc_ctx->device };
    // Send WITHOUT reading response — response will be read by event_wait/synchronize
    uint8_t cmd = RPC_CMD_EVENT_RECORD;
    send_data(sock.get(), &cmd, 1);
    uint64_t req_size = sizeof(req);
    send_data(sock.get(), &req_size, sizeof(req_size));
    send_data(sock.get(), &req, sizeof(req));
    // Response is pending in the TCP buffer
    ev->response_pending = true;
}

// In event_synchronize / event_wait:
static void drain_event_response(rpc_event_t * ev) {
    if (ev->response_pending && ev->sock) {
        uint64_t resp_size;
        recv_data(ev->sock.get(), &resp_size, sizeof(resp_size));
        rpc_msg_event_record_rsp rsp;
        recv_data(ev->sock.get(), &rsp, sizeof(rsp));
        ev->response_pending = false;
    }
}
```

The server-side handler for EVENT_RECORD:

```cpp
case RPC_CMD_EVENT_RECORD: {
    rpc_msg_event_record_req request;
    if (!recv_msg(sockfd, &request, sizeof(request))) return;
    
    // The fact that we reach this handler means all prior commands
    // (including GRAPH_RECOMPUTE) have completed (TCP ordering)
    
    rpc_msg_event_record_rsp response = { request.event_id, 0 };
    if (!send_msg(sockfd, &response, sizeof(response))) return;
    break;
}
```

**Handling the response in other RPC functions:**

To prevent the pending EVENT_RECORD response from corrupting subsequent reads, ALL functions that read from the socket must drain it first:

```cpp
// Draining helper:
static void drain_any_pending_event_response(const std::shared_ptr<socket_t> & sock) {
    // Check thread-local list of pending events on this socket
    for (auto & ev : tls_pending_events) {
        if (ev->sock == sock && ev->response_pending) {
            drain_event_response(ev);
        }
    }
}

// Add drain calls in:
// - ggml_backend_rpc_buffer_get_tensor (before send_rpc_cmd for GET_TENSOR)
// - ggml_backend_rpc_buffer_cpy_tensor (before send_rpc_cmd for COPY_TENSOR)
// - flush_pending_get_tensor (before reading get_tensor responses)
```

#### Verification
- [ ] Event commands don't corrupt TCP stream framing
- [ ] V1 and V2/EVENT commands can be mixed on the same socket
- [ ] Pending event responses are drained before any non-event read
- [ ] Scheduler completes a full pipeline cycle without assertion failures

---

### Success Criteria for Path B

- Minimum **+20% generation throughput** improvement for batch inference (batch_size > 1)
- No regression for single-sequence inference
- `pipeline parallelism enabled` in log output
- `n_copies = 4` active
- No assertion failures or crashes
- Memory usage within 10% of baseline

---

### Fallback / Debug Mode

If pipeline parallelism causes issues:

1. The scheduler handles NULL events gracefully (falls back to `synchronize`)
2. Set `GGML_SCHED_DEBUG=1` to see detailed event lifecycle logging
3. Add `GGML_RPC_DEBUG=1` to trace RPC command sequences
4. As a last resort, set `caps.events = false` to disable pipeline parallelism and restore original behavior
