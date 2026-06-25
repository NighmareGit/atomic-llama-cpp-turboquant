# Path B Handover Script: Event-Based Pipeline Parallelism

## Overview

This document is a complete implementation handover for Path B. It references `RPC_OPTIMIZATION_REPORT.md` for the benchmark context, provides exact source locations in `build-a2/`, and specifies step-by-step changes for each phase. After each phase, run the matching test command and update `rpc-path-b-tracking.md` before proceeding.

---

## Table of Contents

1. [Source Map](#source-map)
2. [Phase B1: Add Event Command to RPC Protocol](#phase-b1-add-event-command-to-rpc-protocol)
3. [Phase B2: Register Event Functions on Backend Interfaces](#phase-b2-register-event-functions-on-backend-interfaces)
4. [Phase B3: Enable Pipeline Parallelism Detection](#phase-b3-enable-pipeline-parallelism-detection)
5. [Phase B4: Tuning and Edge Cases](#phase-b4-tuning-and-edge-cases)
6. [Testing Matrix (from RPC_OPTIMIZATION_REPORT.md)](#testing-matrix)
7. [Build Commands](#build-commands)
8. [Debugging Protocol](#debugging-protocol)

---

## Source Map

All changes in `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp` unless noted.

| Location | Lines | Description |
|----------|-------|-------------|
| `rpc_cmd` enum | 207-222 | Add `RPC_CMD_EVENT_RECORD` (new value 18) |
| struct defs (after existing batch structs) | ~291 | Add `rpc_msg_event_record_req` / `_rsp` |
| `send_rpc_cmd` (no response) | 1033-1045 | Reuse as-is for fire-and-forget |
| `send_rpc_cmd` (with response) | 1049-1064 | Reuse as-is for commands needing response |
| `ggml_backend_rpc_graph_compute` | 1546-1567 | **Add event-aware compute variant** |
| `ggml_backend_rpc_interface` | 1569-1586 | Wire `event_record`, `event_wait` |
| `ggml_backend_rpc_synchronize` | 1498-1502 | Add event response draining |
| `rpc_serve_client` switch | 2162-2602 | Add `RPC_CMD_EVENT_RECORD` handler |
| `ggml_backend_rpc_device_get_props` | 2721-2731 | Set `caps.async = true`, `caps.events = true` |
| `ggml_backend_rpc_device_i` | 2766-2782 | Wire `event_new`, `event_free`, `event_synchronize` |

Other files:
- `build-a2/ggml/include/ggml-rpc.h` (line 10): Bump `RPC_PROTO_MINOR_VERSION` to 2
- `build-a2/src/llama-context.cpp` (lines 348-377): No code change needed, re-check detection
- `build-a2/ggml/src/ggml-backend.cpp` (lines 1534-1714): No code change, verify event lifecycle

---

## Phase B1: Add Event Command to RPC Protocol

### B1.1 — Add enum value

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, line 207-222

Add `RPC_CMD_EVENT_RECORD` before `RPC_CMD_COUNT`:

```cpp
enum rpc_cmd {
    // ... (keep existing values 0-17 unchanged) ...
    RPC_CMD_SET_TENSOR_BATCH,
    RPC_CMD_EVENT_RECORD,    // NEW: event notification command (value 18)
    RPC_CMD_COUNT,           // was 18, now 19
};
```

### B1.2 — Define message structs

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, insert after the `set_tensor_batch_t` struct block (~line 291).

```cpp
// Path B: Event record command.
// Sent AFTER graph_compute to signal that compute has finished (TCP ordering).
// Server processes this after GRAPH_RECOMPUTE, sends response immediately.

struct rpc_msg_event_record_req {
    uint64_t event_id;
    uint32_t device;
};

struct rpc_msg_event_record_rsp {
    uint64_t event_id;
    uint32_t result;  // 0 = success
};
```

### B1.3 — Add deferred send/recv helpers

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, insert after `send_rpc_cmd` (with response, ~line 1064).

These helpers send a command WITHOUT reading the response, and read a deferred response later. This is the core mechanism that allows `event_record` to be non-blocking.

```cpp
// Path B: Send a command with deferred response.
// Unlike send_rpc_cmd (which reads response immediately), this version only sends.
// The response must be read later with recv_rpc_cmd_deferred().
// This allows pipelining: send multiple commands, read responses in order.
//
// Wire format (same as send_rpc_cmd):
//   Request:  | cmd(1) | request_size(8) | request_data(request_size) |
//   Response: | response_size(8) | response_data(response_size) |
// But the response is NOT read here — it stays in the TCP recv buffer.

static bool send_rpc_cmd_deferred(const std::shared_ptr<socket_t> & sock, enum rpc_cmd cmd,
                                   const void * input, size_t input_size) {
    uint8_t cmd_byte = (uint8_t)cmd;
    if (!send_data(sock.get(), &cmd_byte, sizeof(cmd_byte))) return false;
    if (!send_data(sock.get(), &input_size, sizeof(input_size))) return false;
    if (!send_data(sock.get(), input, input_size)) return false;
    return true;
}

// Read a response that was deferred by send_rpc_cmd_deferred.
// Blocks until the response arrives.
// The response MUST be next in the TCP stream (no intervening commands).
static bool recv_rpc_cmd_deferred(const std::shared_ptr<socket_t> & sock,
                                   void * output, size_t output_size) {
    uint64_t resp_size;
    if (!recv_data(sock.get(), &resp_size, sizeof(resp_size))) return false;
    if (resp_size != output_size) {
        GGML_LOG_ERROR("[%s] deferred response size mismatch: expected %zu, got %" PRIu64 "\n",
                       __func__, output_size, resp_size);
        return false;
    }
    if (!recv_data(sock.get(), output, output_size)) return false;
    return true;
}
```

### B1.4 — Client-side event struct

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, insert near the top of the file with the other structs (~after line 291, alongside the message structs).

```cpp
// Path B: Event synchronization primitives.
// RPC events use TCP ordering: event_record sends a command AFTER graph_compute.
// The server response proves that graph_compute finished (TCP ordering guarantee).

struct rpc_event_t {
    uint64_t id;
    std::shared_ptr<socket_t> sock;  // socket to read response from
    bool response_pending;           // true after event_record, cleared after synchronize
};
```

### B1.5 — Server-side handler

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, in `rpc_serve_client` switch statement, insert after the `RPC_CMD_SET_TENSOR_BATCH` case (after line 2583) and before `RPC_CMD_GET_DEVICE_MEMORY` (line 2584).

```cpp
            case RPC_CMD_EVENT_RECORD: {
                rpc_msg_event_record_req request;
                if (!recv_msg(sockfd, &request, sizeof(request))) {
                    return;
                }
                // Reaching this handler means all prior commands on this socket
                // (including GRAPH_RECOMPUTE) have completed — TCP ordering guarantee.
                rpc_msg_event_record_rsp response = {request.event_id, 0};
                if (!send_msg(sockfd, &response, sizeof(response))) {
                    return;
                }
                LOG_DBG("[%s] RPC_CMD_EVENT_RECORD: event_id=%lu, device=%u\n",
                        __func__, request.event_id, request.device);
                break;
            }
```

### B1.6 — Bump protocol version

**File:** `build-a2/ggml/include/ggml-rpc.h`, line 10

```cpp
#define RPC_PROTO_MINOR_VERSION    2   // was 1; bumped for RPC_CMD_EVENT_RECORD
```

### B1.7 — Implement event_record in graph_compute path

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, modify `ggml_backend_rpc_graph_compute` (lines 1546-1567).

The key change: after sending `GRAPH_RECOMPUTE`, immediately send `EVENT_RECORD` (deferred — don't read response yet). The response will be read in `event_wait`/`event_synchronize`.

**Strategy:** The RPC context tracks whether a graph compute just fired, so `event_record` can link the pending response to the event.

Add to `ggml_backend_rpc_context` struct (find it, around line 1588-1638 area):

```cpp
struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    graph_cache gc;
    // Path B: track pending compute response for event linkage
    std::shared_ptr<socket_t> last_compute_sock;
    bool last_compute_sent_event = false;  // true if EVENT_RECORD was sent after last compute
};
```

Modify `ggml_backend_rpc_graph_compute`:

```cpp
static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    auto sock = get_socket(rpc_ctx->endpoint);
    flush_pending_get_tensor();
    flush_set_tensor_batch(sock);
    bool reuse = rpc_ctx->gc.is_cached(cgraph);
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);

        // Path B: If event support is active, send EVENT_RECORD after GRAPH_RECOMPUTE.
        // The server processes EVENT_RECORD after GRAPH_RECOMPUTE (TCP ordering).
        // We do NOT read the response here — event_wait/synchronize will read it.
        rpc_msg_event_record_req ev_req;
        ev_req.event_id = 0;     // will be set by event_record
        ev_req.device = rpc_ctx->device;
        // Send with deferred response:
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, sizeof(ev_req));
        rpc_ctx->last_compute_sock = sock;
        rpc_ctx->last_compute_sent_event = true;
    } else {
        rpc_ctx->gc.add(cgraph);
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
        // Full compute doesn't send event (graph topology changed, cache miss)
        rpc_ctx->last_compute_sock = nullptr;
        rpc_ctx->last_compute_sent_event = false;
    }
    return GGML_STATUS_SUCCESS;
}
```

### B1.8 — Implement event_record backend function

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, add new function near the `ggml_backend_rpc_synchronize` function (after line 1502).

```cpp
// Path B: Backend-level event_record.
// Called by the scheduler after graph_compute_async returns.
// Links the pending EVENT_RECORD response to the event object.

static void rpc_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    auto * ev = (rpc_event_t *)event;

    if (rpc_ctx->last_compute_sent_event && rpc_ctx->last_compute_sock) {
        // The EVENT_RECORD was already sent in graph_compute.
        // Link the pending response to this event.
        ev->sock = rpc_ctx->last_compute_sock;
        ev->response_pending = true;
        rpc_ctx->last_compute_sent_event = false;  // consumed
    } else {
        // No pending event — this can happen on the first token or after cache miss.
        // Send an EVENT_RECORD now (though the compute is already done).
        auto sock = get_socket(rpc_ctx->endpoint);
        rpc_msg_event_record_req ev_req;
        ev_req.event_id = ev->id;
        ev_req.device = rpc_ctx->device;
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, sizeof(ev_req));
        ev->sock = sock;
        ev->response_pending = true;
    }
}

// Path B: Backend-level event_wait.
// Called by the scheduler to make another backend wait for this event.
// For RPC, we synchronize the event (read the pending response).
// This ensures the RPC compute has completed before another backend uses the output.

static void rpc_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    auto * ev = (rpc_event_t *)event;
    if (ev->response_pending && ev->sock) {
        rpc_msg_event_record_rsp rsp;
        if (!recv_rpc_cmd_deferred(ev->sock, &rsp, sizeof(rsp))) {
            GGML_LOG_ERROR("[%s] failed to read event response\n", __func__);
        }
        ev->response_pending = false;
    }
}
```

### B1.9 — Add event response draining to socket read functions

**Critical:** Any RPC function that reads from the socket must first drain pending event responses. Otherwise, the pending `EVENT_RECORD` response will be consumed by the wrong `recv_data` call, corrupting the TCP stream.

Add a drain helper:

```cpp
// Path B: Drain any pending event response from the given socket.
// Must be called before any recv_data that expects a non-event response.
// Uses thread-local pending event list (set by event_record).

static thread_local std::vector<rpc_event_t *> tls_active_events;

static void rpc_event_register_active(rpc_event_t * ev) {
    tls_active_events.push_back(ev);
}

static void rpc_event_drain_for_socket(const std::shared_ptr<socket_t> & sock) {
    for (auto * ev : tls_active_events) {
        if (ev->response_pending && ev->sock == sock) {
            rpc_msg_event_record_rsp rsp;
            if (!recv_rpc_cmd_deferred(sock, &rsp, sizeof(rsp))) {
                GGML_LOG_ERROR("[%s] drain: failed to read event response\n", __func__);
            }
            ev->response_pending = false;
        }
    }
}
```

**Simpler alternative (recommended):** Use a single thread-local flag per socket, since there's only one RPC backend socket per thread in practice.

```cpp
// Simpler drain: single pending event response per thread.
static thread_local struct {
    std::shared_ptr<socket_t> sock;
    bool pending;
} tls_pending_event;
```

Modify `rpc_backend_event_record` to use the simple version:

```cpp
static void rpc_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    auto * ev = (rpc_event_t *)event;

    if (rpc_ctx->last_compute_sent_event && rpc_ctx->last_compute_sock) {
        // The EVENT_RECORD was already sent in graph_compute.
        // The response is at the front of the TCP recv buffer.
        tls_pending_event.sock = rpc_ctx->last_compute_sock;
        tls_pending_event.pending = true;
        rpc_ctx->last_compute_sent_event = false;
        // Store event linkage for later read:
        ev->sock = rpc_ctx->last_compute_sock;
        ev->response_pending = true;
    } else {
        auto sock = get_socket(rpc_ctx->endpoint);
        rpc_msg_event_record_req ev_req = {ev->id, rpc_ctx->device};
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, sizeof(ev_req));
        tls_pending_event.sock = sock;
        tls_pending_event.pending = true;
        ev->sock = sock;
        ev->response_pending = true;
    }
}
```

Add drain calls in these functions (every function that reads from the socket):

1. **`ggml_backend_rpc_buffer_get_tensor`** (line ~1262) — add before `flush_pending_get_tensor()` call
2. **`ggml_backend_rpc_buffer_cpy_tensor`** (line ~1344) — add before `flush_pending_get_tensor()` call  
3. **`flush_pending_get_tensor`** (line ~1327) — add at start
4. **`ggml_backend_rpc_synchronize`** (line ~1498) — add at start
5. **`ggml_backend_rpc_graph_compute`** (line ~1546) — already flushing at start

The drain call:

```cpp
static void drain_pending_event_response() {
    if (tls_pending_event.pending) {
        rpc_msg_event_record_rsp rsp;
        if (!recv_rpc_cmd_deferred(tls_pending_event.sock, &rsp, sizeof(rsp))) {
            GGML_LOG_ERROR("[%s] drain: failed to read event response\n", __func__);
        }
        tls_pending_event.pending = false;
    }
}
```

**Modify each function:**

In `ggml_backend_rpc_buffer_get_tensor` (line 1262):
```cpp
static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto sock = ctx->sock;
    drain_pending_event_response();     // Path B: drain pending event response
    flush_pending_get_tensor();
    flush_set_tensor_batch(sock);
    // ... rest unchanged ...
}
```

In `ggml_backend_rpc_buffer_cpy_tensor` (line 1344):
```cpp
static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // ... existing checks ...
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto sock = ctx->sock;
    drain_pending_event_response();     // Path B: drain pending event response
    flush_pending_get_tensor();
    flush_set_tensor_batch(sock);
    // ... rest unchanged ...
}
```

In `flush_pending_get_tensor` (line 1327):
```cpp
static void flush_pending_get_tensor() {
    drain_pending_event_response();     // Path B: drain before reading get_tensor responses
    if (tls_pending_get_tensor.empty()) {
        return;
    }
    // ... rest unchanged ...
}
```

In `ggml_backend_rpc_graph_compute` (line 1546):
```cpp
static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto sock = get_socket(rpc_ctx->endpoint);
    drain_pending_event_response();     // Path B: drain before sending new commands
    flush_pending_get_tensor();
    flush_set_tensor_batch(sock);
    // ... rest unchanged ...
}
```

### B1.10 — Verification Test

Build and run a basic connectivity test:

```bash
# Terminal 1: start server
cd build-a2 && mkdir -p build-b1 && cd build-b1
cmake .. -DGGML_CUDA=ON && make -j$(nproc) rpc-server
./build/b1/bin/rpc-server -H 127.0.0.1 -p 50052

# Terminal 2: run client with debug
cd build-a2/build-b1
GGML_RPC_DEBUG=1 ./bin/llama-cli \
  --rpc 127.0.0.1:50052 \
  -m /path/to/Qwen3.5-9B-Q5_K_M.gguf \
  -p "Hello" -n 10 -ngl 99 -ctk q4_0 -ctv q4_0
```

Expected: server logs show `RPC_CMD_EVENT_RECORD` being processed. Client does NOT crash. Generation produces correct output (bit-exact with vanilla).

Update `rpc-path-b-tracking.md` after verification.

---

## Phase B2: Register Event Functions on Backend Interfaces

### B2.1 — Device-level event functions

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, add new functions before `ggml_backend_rpc_device_i` definition (before line 2766).

```cpp
// Path B: Device-level event functions.
// These manage the lifecycle of event objects.

static ggml_backend_event_t rpc_event_new(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    static std::atomic<uint64_t> next_id{1};
    auto * ev = new rpc_event_t();
    ev->id = next_id++;
    ev->sock = nullptr;
    ev->response_pending = false;
    LOG_DBG("[%s] allocated event %lu\n", __func__, ev->id);
    return (ggml_backend_event_t)ev;
}

static void rpc_event_free(ggml_backend_dev_t device, ggml_backend_event_t event) {
    GGML_UNUSED(device);
    if (event) {
        auto * ev = (rpc_event_t *)event;
        LOG_DBG("[%s] freeing event %lu\n", __func__, ev->id);
        delete ev;
    }
}

static void rpc_event_synchronize(ggml_backend_dev_t device, ggml_backend_event_t event) {
    GGML_UNUSED(device);
    auto * ev = (rpc_event_t *)event;
    if (ev->response_pending && ev->sock) {
        rpc_msg_event_record_rsp rsp;
        if (!recv_rpc_cmd_deferred(ev->sock, &rsp, sizeof(rsp))) {
            GGML_LOG_ERROR("[%s] event %lu: failed to read response\n", __func__, ev->id);
        }
        ev->response_pending = false;
        LOG_DBG("[%s] event %lu synchronized\n", __func__, ev->id);
    }
}
```

### B2.2 — Wire event functions into `ggml_backend_rpc_device_i`

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, lines 2779-2781.

```cpp
    /* .event_new            = */ rpc_event_new,             // was NULL
    /* .event_free           = */ rpc_event_free,            // was NULL
    /* .event_synchronize    = */ rpc_event_synchronize,     // was NULL
```

### B2.3 — Wire backend-level event functions into `ggml_backend_rpc_interface`

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, lines 1583-1584.

```cpp
    /* .event_record            = */ rpc_backend_event_record,  // was NULL
    /* .event_wait              = */ rpc_backend_event_wait,    // was NULL
```

### B2.4 — Update device capabilities

**File:** `build-a2/ggml/src/ggml-rpc/ggml-rpc.cpp`, lines 2726-2731.

```cpp
    props->caps = {
        /* .async                 = */ true,   // Changed from false — get_tensor_async exists
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,   // Changed from false — event functions exist
    };
```

### B2.5 — Verification Test

Same as B1.10, but additionally verify:
- `ggml_backend_event_new` returns non-NULL for RPC device
- Scheduler creates events for RPC backend (check with `GGML_SCHED_DEBUG=1`)
- Log shows events being created, recorded, and waited upon

```bash
GGML_SCHED_DEBUG=1 GGML_RPC_DEBUG=1 ./bin/llama-cli \
  --rpc 127.0.0.1:50052 \
  -m /path/to/Qwen3.5-9B-Q5_K_M.gguf \
  -p "Hello" -n 10 -ngl 99 -ctk q4_0 -ctv q4_0 2>&1 | grep -E "event|Event|split|copy"
```

Expected: scheduler output shows event_record and event_wait calls for the RPC backend.

---

## Phase B3: Enable Pipeline Parallelism Detection

### B3.1 — No code changes needed

The detection logic in `src/llama-context.cpp` (lines 348-377) checks `caps.async && caps.events` for all non-CPU devices. With `caps.async = true` and `caps.events = true` from B2.4, this passes.

**But verify:** the pipeline parallelism requires ALL of these conditions (line 350-355):
```cpp
pipeline_parallel =
    model.n_devices() > 1 &&
    model.n_gpu_layers() > model.hparams.n_layer &&  // all layers offloaded
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&  // layer-wise split
    cparams.offload_kqv &&                            // KV cache on GPU
    !model.has_tensor_overrides();                    // no manual overrides
```

Your test setup must satisfy ALL of these. The most common failure:
- `model.n_gpu_layers() > model.hparams.n_layer`: set `-ngl N` where N > total layers
- `model.split_mode() == LLAMA_SPLIT_MODE_LAYER`: default, OK
- `cparams.offload_kqv`: default is true, OK (unless `--no-kv-offload` is set)
- `model.n_devices() > 1`: requires at least 2 devices (RPC server + local GPU, or 2 RPC servers)

**For testing:** Use cross-GPU config (Config A from RPC_OPTIMIZATION_REPORT.md):
- AMD 7900 XTX as llama-server (client)
- NVIDIA 3060 Ti as rpc-server (worker)
- `-ngl 99 -ctk q4_0 -ctv q4_0`

### B3.2 — Verify the log

```bash
./bin/llama-cli --rpc 127.0.0.1:50052 \
  -m /path/to/Qwen3.5-9B-Q5_K_M.gguf \
  -p "Hello" -n 10 -ngl 99 -ctk q4_0 -ctv q4_0 2>&1 | grep "pipeline"
```

Expected output: `pipeline parallelism enabled`

### B3.3 — Verify n_copies

```bash
GGML_SCHED_DEBUG=1 ./bin/llama-cli --rpc 127.0.0.1:50052 \
  -m /path/to/Qwen3.5-9B-Q5_K_M.gguf \
  -p "Hello" -n 10 -ngl 99 -ctk q4_0 -ctv q4_0 2>&1 | grep "n_copies"
```

Expected: `n_copies = 4` (or `GGML_SCHED_MAX_COPIES`).

### B3.4 — Monitor VRAM

Pipeline parallelism allocates n_copies × buffer space. Monitor with:
```bash
nvidia-smi dmon -d 1 &
./bin/llama-cli --rpc 127.0.0.1:50052 \
  -m /path/to/Qwen3.5-9B-Q5_K_M.gguf \
  -p "Hello, how are you?" -n 256 -ngl 99 -ctk q4_0 -ctv q4_0
kill %1
```

Compare VRAM usage with baseline (Phase A2 without pipeline). Expect moderate increase.

---

## Phase B4: Tuning and Edge Cases

### B4.1 — Handle event_record without prior graph_compute

If `event_record` is called without a prior `graph_compute` (e.g., on the very first token), `last_compute_sent_event` will be false. The event_record should send an immediate `EVENT_RECORD`:

Already handled in `rpc_backend_event_record` — the `else` branch sends a fresh `EVENT_RECORD`.

### B4.2 — Test with single GPU (no pipeline)

Run without `--rpc` to verify no regression:

```bash
./bin/llama-cli -m /path/to/Qwen3.5-9B-Q5_K_M.gguf \
  -p "Hello" -n 10 -ngl 99 -ctk q4_0 -ctv q4_0
```

Compare tok/s with baseline Phase A2. Should be within ±2%.

### B4.3 — Test with pipeline parallelism disabled

Force disable by not meeting the pipeline conditions (e.g., `--no-kv-offload` or `-ngl` < total layers):

```bash
./bin/llama-cli --rpc 127.0.0.1:50052 \
  -m /path/to/Qwen3.5-9B-Q5_K_M.gguf \
  -p "Hello" -n 10 -ngl 20 --no-kv-offload
```

Should work without event-related crashes (falls back to synchronize).

### B4.4 — Cross-GPU performance benchmark

Run the full cross-GPU matrix from RPC_OPTIMIZATION_REPORT.md:

```bash
# Config A: AMD llama-server <=> NVIDIA rpc-server
# Server (NVIDIA):
docker run --rm --gpus all --network host \
  -v /mnt/models:/models \
  -w /app/build-b4/bin \
  --entrypoint ./rpc-server \
  llama-rpc-cuda-b4 -H 0.0.0.0 -p 50051

# Client (AMD):
docker run --rm --network host \
  --device=/dev/kfd --device=/dev/dri \
  -v /mnt/models:/models \
  -w /app/build-b4/bin \
  llama-rocm-b4 \
  --rpc 127.0.0.1:50051 \
  -m /models/Qwen_Qwen3.5-9B-Q5_K_M.gguf \
  -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 \
  --host 0.0.0.0 --port 8081

# Bench:
for i in 1 2 3 4 5; do
  curl -s http://127.0.0.1:8081/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"max_tokens":80}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('timings',{}); print(f'P={t.get(\"prompt_per_second\",0):.1f} G={t.get(\"predicted_per_second\",0):.1f}')"
done
```

---

## Testing Matrix

All test configurations from RPC_OPTIMIZATION_REPORT.md:

### Cross-GPU Tests (generation = primary metric)

| # | Config | Client GPU | Worker GPU | Model | Expected vs Baseline |
|---|--------|-----------|------------|-------|---------------------|
| 1 | A | AMD 7900 XTX (24GB) | NVIDIA 3060 Ti (8GB) | Qwen3.5-9B Q5_K_M | **Generation improvement expected** (pipeline overlap) |
| 2 | B | NVIDIA 3060 Ti (8GB) | AMD 7900 XTX (24GB) | Qwen3.5-9B Q5_K_M | Generation improvement expected |

### Single-GPU Regression Tests

| # | Config | Model | Expected |
|---|--------|-------|----------|
| 3 | Vanilla (no RPC) | Qwen3.5-9B Q5_K_M | ±2% of baseline |
| 4 | Vanilla (no RPC) | smollm3-3b Q4_K_M | ±2% of baseline |
| 5 | Vanilla (no RPC) | Qwen3.5-4B Q4_K_M | ±2% of baseline |

### Full GPU Offload Tests

| # | Config | Model | Expected |
|---|--------|-------|----------|
| 6 | Single 3060 Ti, full offload | smollm3-3b Q4_K_M | 143 t/s gen |
| 7 | Single 3060 Ti, full offload | Qwen3.5-4B Q4_K_M | 93 t/s gen |
| 8 | Single 3060 Ti, full offload | Qwen3.5-9B Q5_K_M | 54 t/s gen |

### Edge Case Tests

| # | Config | Expected |
|---|--------|----------|
| 9 | `--no-kv-offload` + RPC | No events created (caps.events check passes but pipeline_parallel condition `offload_kqv` fails) |
| 10 | `-ngl 20` (partial offload) + RPC | Pipeline not enabled, works with sync fallback |
| 11 | First token (cache miss) | Full GRAPH_COMPUTE sent, no event (B1.7 fallback) |
| 12 | 2 RPC workers + local GPU | Multiple events, verify no cross-socket timing issues |

---

## Build Commands

### First build (full, CUDA)
```bash
cd build-a2 && mkdir -p build-b && cd build-b
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) rpc-server
make -j$(nproc) llama-cli
```

### Quick rebuild (after .cpp changes)
```bash
cd build-a2/build-b
make -j$(nproc) rpc-server llama-cli
```

### ROCm build (for AMD client)
```bash
cd build-a2 && mkdir -p build-b-rocm && cd build-b-rocm
cmake .. -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) llama-cli
```

---

## Debugging Protocol

### Token-level tracing
```bash
GGML_RPC_DEBUG=1 GGML_SCHED_DEBUG=1 ./bin/llama-cli ... 2>&1 | tee debug.log
```

Keywords to grep:
- `RPC_CMD_EVENT_RECORD` — verify event commands are sent
- `event_record` / `event_wait` — verify scheduler calls event functions
- `pipeline` — verify pipeline is enabled
- `n_copies` — verify double-buffering
- `n_splits` — verify split count
- `ggml_backend_event_new` — verify RPC events are created

### TCP stream debugging
```bash
# Trace all socket I/O
strace -e trace=network -f -p $(pgrep llama-cli) 2>&1 | grep -E "send|recv"

# Monitor recv timing
strace -e trace=recvfrom -T -p $(pgrep llama-cli) 2>&1
```

### Common failure modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Hang after first token | Event response not drained before GET_TENSOR | Add `drain_pending_event_response()` to get_tensor |
| "response size mismatch" | Wrong response framing | Check `send_msg`/`recv_msg` vs `send_rpc_cmd_deferred`/`recv_rpc_cmd_deferred` consistency |
| Crash in `ggml_backend_event_record` (assert `event_record != NULL`) | Interface not wired | Check B2.3: `ggml_backend_rpc_interface` event_record/wait |
| Crash in `ggml_backend_event_new` (event_new NULL) | Device interface not wired | Check B2.2: `rpc_event_new` in device_i |
| Pipeline not enabled | `caps.async` or `caps.events` still false | Check B2.4 |
| No event commands sent | `last_compute_sent_event` always false | Check B1.7: graph_compute sends deferred EVENT_RECORD |

### Rollback

To disable Path B changes entirely:
```cpp
// In ggml_backend_rpc_device_get_props (line 2726-2731):
props->caps = {
    /* .async                 = */ false,  // revert to false
    /* .events                = */ false,  // revert to false
};
```

This restores original behavior. No other changes need reverting since the scheduler gracefully handles `NULL` events by falling back to `synchronize`.

---

## Appendix: Code Flow Summary

```
Scheduler calls:
  1. ggml_backend_graph_compute_async(backend=RPC, cgraph)
     → ggml_backend_rpc_graph_compute()
        → send GRAPH_RECOMPUTE (fire-and-forget)
        → send EVENT_RECORD (deferred, response pending)
        → rpc_ctx->last_compute_sock = sock
        → rpc_ctx->last_compute_sent_event = true

  2. ggml_backend_event_record(event, backend=RPC)
     → rpc_backend_event_record()
        → ev->sock = rpc_ctx->last_compute_sock
        → ev->response_pending = true
        → tls_pending_event = {sock, true}

Server (in order, TCP guarantees):
  3. RECV GRAPH_RECOMPUTE → compute on GPU → return (no response)
  4. RECV EVENT_RECORD → return response immediately

Client:
  5. Next split: copy input → get_tensor() → drain_pending_event_response()
     → RECV EVENT_RECORD response (reads it, discards)
  6. Next token: event_wait → rpc_backend_event_wait()
     → READ EVENT_RECORD response from TOKEN N's compute
     → confirms compute is done → proceeds with new SET_TENSOR
```
