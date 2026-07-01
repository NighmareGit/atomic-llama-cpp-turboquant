#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"

#include <array>
#include <cinttypes>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

static int rpc_trace_lvl() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_TRACE");
        v = e ? atoi(e) : 0;
    }
    return v;
}

static std::mutex rpc_trace_mutex;

static FILE * rpc_trace_file() {
    static FILE * trace_f = nullptr;
    static bool trace_f_init = false;
    if (!trace_f_init) {
        trace_f_init = true;
        const char * path = getenv("GGML_RPC_TRACE_FILE");
        if (path && path[0]) {
            trace_f = fopen(path, "a");
        }
    }
    return trace_f;
}

static void rpc_trace_emit_hotpath_fields(FILE * out) {
    const int32_t decode_id = ggml_pipeline_trace_get_decode_id();
    int32_t split_id = -1;
    int32_t backend_id = -1;
    ggml_hotpath_trace_get_sched_ctx(&split_id, &backend_id);
    if (decode_id >= 0) {
        fprintf(out, ",\"decode_id\":%d", decode_id);
    }
    if (split_id >= 0) {
        fprintf(out, ",\"split\":%d", split_id);
    }
    if (backend_id >= 0) {
        fprintf(out, ",\"backend\":%d", backend_id);
    }
}

static void rpc_trace_emit(const char * fn, const char * phase, int cmd, size_t bytes, bool blocking, int64_t elapsed_us) {
    if (!rpc_trace_lvl()) {
        return;
    }
    const auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(rpc_trace_mutex);
    FILE * out = rpc_trace_file();
    if (!out) {
        out = stderr;
    }
    fprintf(out,
        "{\"ts_us\":%lld,\"fn\":\"%s\",\"phase\":\"%s\",\"cmd\":%d,\"bytes\":%zu,\"blocking\":%s,\"elapsed_us\":%lld",
        (long long) ts_us, fn, phase, cmd, bytes, blocking ? "true" : "false", (long long) elapsed_us);
    rpc_trace_emit_hotpath_fields(out);
    fprintf(out, "}\n");
    fflush(out);
}

namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_SET_TENSOR_BATCH,
    RPC_CMD_EVENT_RECORD,    // Path B: event notification via TCP ordering (value 18)
    RPC_CMD_COPY_TENSOR_PEER,
    RPC_CMD_CHANNEL_BIND,    // B+11: pair response socket (value 20)
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

static void rpc_trace_emit_hello(const char * endpoint, int minor, bool peer_copy) {
    if (!rpc_trace_lvl()) {
        return;
    }
    const auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(rpc_trace_mutex);
    FILE * out = rpc_trace_file();
    if (!out) {
        out = stderr;
    }
    fprintf(out,
        "{\"ts_us\":%lld,\"fn\":\"negotiate_hello\",\"phase\":\"hello\",\"cmd\":%d,\"endpoint\":\"%s\",\"minor\":%d,\"peer_copy\":%s,\"elapsed_us\":0}\n",
        (long long) ts_us, RPC_CMD_HELLO, endpoint, minor, peer_copy ? "true" : "false");
    fflush(out);
}

static void rpc_trace_emit_copy_issue(int cmd, const char * src_ep, const char * dst_ep, bool peer_copy, bool defer) {
    if (!rpc_trace_lvl()) {
        return;
    }
    const auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(rpc_trace_mutex);
    FILE * out = rpc_trace_file();
    if (!out) {
        out = stderr;
    }
    fprintf(out,
        "{\"ts_us\":%lld,\"fn\":\"rpc_issue_copy_tensor\",\"phase\":\"copy_issue\",\"cmd\":%d,"
        "\"src_ep\":\"%s\",\"dst_ep\":\"%s\",\"peer_copy\":%s,\"defer\":%s,\"elapsed_us\":0",
        (long long) ts_us, cmd, src_ep ? src_ep : "", dst_ep ? dst_ep : "",
        peer_copy ? "true" : "false", defer ? "true" : "false");
    rpc_trace_emit_hotpath_fields(out);
    fprintf(out, "}\n");
    fflush(out);
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Path A1: batch SET_TENSOR
struct set_tensor_batch_t {
    std::vector<uint8_t> buf;
    uint32_t count = 0;
};

static thread_local set_tensor_batch_t tls_set_batch;
static thread_local socket_ptr tls_set_batch_sock;

#define RPC_SET_TENSOR_BATCH_MAX_SIZE (64 * 1024 * 1024)

static void set_tensor_batch_append(const rpc_tensor & tensor, uint64_t offset, const void * data, size_t size) {
    const size_t entry_size = sizeof(rpc_tensor) + sizeof(uint64_t) + sizeof(uint64_t) + size;
    const size_t old_size = tls_set_batch.buf.size();
    tls_set_batch.buf.resize(old_size + entry_size);
    uint8_t * p = tls_set_batch.buf.data() + old_size;
    memcpy(p, &tensor, sizeof(rpc_tensor)); p += sizeof(rpc_tensor);
    memcpy(p, &offset, sizeof(uint64_t));   p += sizeof(uint64_t);
    uint64_t net_size = size;
    memcpy(p, &net_size, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(p, data, size);
    tls_set_batch.count++;
}

static void flush_set_tensor_batch() {
    if (tls_set_batch.count == 0 || !tls_set_batch_sock) {
        return;
    }
    auto sock = tls_set_batch_sock;
    const size_t payload_size = sizeof(uint32_t) + tls_set_batch.buf.size();
    const size_t msg_size = 1 + sizeof(uint64_t) + payload_size;
    std::vector<uint8_t> msg(msg_size);
    uint8_t * p = msg.data();
    *p++ = (uint8_t) RPC_CMD_SET_TENSOR_BATCH;
    uint64_t net_payload = payload_size;
    memcpy(p, &net_payload, sizeof(net_payload)); p += sizeof(net_payload);
    uint32_t count = tls_set_batch.count;
    memcpy(p, &count, sizeof(count)); p += sizeof(count);
    memcpy(p, tls_set_batch.buf.data(), tls_set_batch.buf.size());
    sock->send_data(msg.data(), msg.size());
    tls_set_batch.buf.clear();
    tls_set_batch.count = 0;
    tls_set_batch_sock.reset();
}

// Path A2: pipelined GET_TENSOR (defer receive only)
struct rpc_pending_get_tensor {
    socket_ptr sock;
    void * data;
    size_t size;
};

static thread_local std::vector<rpc_pending_get_tensor> tls_pending_get_tensor;

// B+13: deferred GET_TENSOR -> local dst (completed at split graph_compute entry).
struct rpc_pending_download {
    socket_ptr sock;
    std::vector<uint8_t> staging;
    ggml_tensor * dst;
    ggml_backend_t dst_backend;
};

// Same-host multi-endpoint: queued at copy_issue, flushed sync at split barrier.
struct rpc_pending_relay {
    socket_ptr src_sock;
    socket_ptr dst_sock;
    const ggml_tensor * src;
    ggml_tensor * dst;
    std::vector<uint8_t> staging;
};

static thread_local std::vector<rpc_pending_relay> tls_pending_relays;

static thread_local std::vector<rpc_pending_download> tls_pending_downloads;

static uint64_t fnv_hash(const uint8_t * data, size_t len);

static void drain_pending_event_response(const socket_ptr & sock);
static void drain_pending_copy_response(const socket_ptr & sock);
static void flush_pending_get_tensor_for_socket(const socket_ptr & sock);
static void flush_pending_get_tensor();
static void flush_pending_hash_for_socket(const socket_ptr & sock);
static void flush_pending_hash_all();
static bool rpc_issue_relay_upload(const rpc_pending_relay & relay);
static void flush_pending_relays();
static bool rpc_pending_relay_matches_dst(const rpc_pending_relay & pr,
                                          const ggml_tensor * const * dst, size_t n_dst);
static void flush_pending_relays_for_dst(const ggml_tensor * const * dst, size_t n_dst);
static void flush_set_tensor_batch();
static void rpc_register_socket(const socket_ptr & sock);
static void rpc_drain_all_endpoints_pending();

int ggml_backend_rpc_server_count(void);
bool ggml_backend_rpc_event_defer_barrier(void);
bool ggml_backend_rpc_get_tensor_defer(void);
bool ggml_backend_rpc_hash_defer(void);
bool ggml_backend_rpc_dual_socket(void);

static socket_ptr rpc_response_sock(const socket_ptr & cmd);

static bool rpc_pipeline_plus_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_PIPELINE_PLUS");
        v = e ? (atoi(e) != 0) : 1;
    }
    return v != 0;
}

static int rpc_event_defer_min_servers() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_EVENT_DEFER_MIN_SERVERS");
        v = e ? atoi(e) : 1;
        if (v < 1) {
            v = 1;
        }
    }
    return v;
}

static int rpc_event_defer_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_EVENT_DEFER_BARRIER");
        v = e ? atoi(e) : (rpc_pipeline_plus_enabled() ? 1 : 0);
    }
    return v;
}

static int rpc_get_tensor_defer_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_GET_TENSOR_DEFER");
        v = e ? atoi(e) : (rpc_pipeline_plus_enabled() ? 1 : 0);
    }
    return v;
}

static int rpc_hash_defer_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_HASH_DEFER");
        // Default OFF until multi-socket defer ordering is proven on 5-GPU load.
        v = e ? atoi(e) : 0;
    }
    return v;
}

static bool rpc_event_defer_barrier() {
    return ggml_backend_rpc_event_defer_barrier();
}

static bool rpc_multi_socket_flush() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_MULTI_SOCKET_FLUSH");
        v = e ? atoi(e) : (rpc_pipeline_plus_enabled() ? 1 : 0);
    }
    return v != 0 && rpc_pipeline_plus_enabled();
}

static int rpc_dual_socket_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_DUAL_SOCKET");
        // Default OFF until all rpc-servers are proto 4.4 (bisect sets =1 explicitly).
        v = e ? atoi(e) : 0;
    }
    return v;
}

bool ggml_backend_rpc_dual_socket(void) {
    return rpc_dual_socket_env_enabled() != 0;
}

struct rpc_dual_pending {
    std::mutex              mutex;
    std::condition_variable cv;
    socket_ptr              rsp;
    bool                    paired = false;
};

static std::mutex g_dual_pending_mutex;
static std::unordered_map<uint32_t, std::shared_ptr<rpc_dual_pending>> g_dual_pending;

static uint32_t rpc_gen_session_id() {
    static std::atomic<uint32_t> seq{1};
    const uint32_t t = (uint32_t) std::chrono::steady_clock::now().time_since_epoch().count();
    return t ^ seq.fetch_add(1, std::memory_order_relaxed);
}

static socket_ptr rpc_response_sock(const socket_ptr & cmd) {
    return (cmd && cmd->rsp_channel) ? cmd->rsp_channel : cmd;
}

static std::mutex g_rpc_socket_registry_mutex;
static std::vector<std::weak_ptr<socket_t>> g_rpc_socket_registry;

static std::mutex g_rpc_reg_map_mutex;
static std::unordered_map<std::string, ggml_backend_reg_t> g_rpc_reg_map;
static uint32_t g_rpc_dev_id = 0;

static void rpc_register_socket(const socket_ptr & sock) {
    if (!sock) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_rpc_socket_registry_mutex);
    g_rpc_socket_registry.push_back(sock);
}

static thread_local struct {
    socket_ptr sock;
    bool pending;
} tls_pending_copy = {nullptr, false};

struct rpc_event_t;

// Path B: pending EVENT_RECORD response (one per thread)
static thread_local struct {
    socket_ptr sock;
    bool pending;
    rpc_event_t * ev;
} tls_pending_event = {nullptr, false, nullptr};

static void rpc_drain_all_endpoints_pending() {
    if (tls_pending_copy.pending && tls_pending_copy.sock) {
        drain_pending_copy_response(tls_pending_copy.sock);
    }
    if (!rpc_event_defer_barrier() && tls_pending_event.pending && tls_pending_event.sock) {
        drain_pending_event_response(tls_pending_event.sock);
    }
    flush_pending_get_tensor();
    flush_pending_hash_all();
    flush_set_tensor_batch();
    flush_pending_relays();

    if (!rpc_multi_socket_flush()) {
        return;
    }

    std::vector<socket_ptr> live;
    {
        std::lock_guard<std::mutex> lock(g_rpc_socket_registry_mutex);
        auto it = g_rpc_socket_registry.begin();
        while (it != g_rpc_socket_registry.end()) {
            if (auto sock = it->lock()) {
                live.push_back(sock);
                ++it;
            } else {
                it = g_rpc_socket_registry.erase(it);
            }
        }
    }
    for (const auto & sock : live) {
        flush_pending_get_tensor_for_socket(sock);
        flush_pending_hash_for_socket(sock);
        if (!rpc_event_defer_barrier() && tls_pending_event.pending && tls_pending_event.sock == sock) {
            drain_pending_event_response(sock);
        }
        if (tls_pending_copy.pending && tls_pending_copy.sock == sock) {
            drain_pending_copy_response(sock);
        }
    }
}

// B+4: skip redundant SET_TENSOR_HASH RTTs when server already confirmed hash
static thread_local std::unordered_map<uint64_t, bool> tls_hash_present;

// B+7f: pipelined SET_TENSOR_HASH (send deferred, recv at next set_tensor / barrier)
struct rpc_pending_hash {
    socket_ptr sock;
    uint64_t cache_key;
    rpc_tensor tensor;
    uint64_t offset;
    std::vector<uint8_t> staging;
};

static thread_local std::vector<rpc_pending_hash> tls_pending_hash;
static thread_local socket_ptr tls_hash_active_sock;

static uint64_t rpc_hash_cache_key(const socket_ptr & sock, uint64_t hash, uint64_t data_ptr, uint64_t offset) {
    const uintptr_t sk = (uintptr_t) sock.get();
    return hash ^ (data_ptr * 0x9e3779b97f4a7c15ULL) ^ (offset * 0xbf58476d1ce4e5b9ULL) ^ (sk * 0x94d049bb133111ebULL);
}

static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size);
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size);

static bool send_rpc_cmd_deferred(const socket_ptr & sock, enum rpc_cmd cmd,
                                   const void * input, size_t input_size) {
    flush_set_tensor_batch();
    uint8_t cmd_byte = (uint8_t) cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) return false;
    if (!sock->send_data(&input_size, sizeof(input_size))) return false;
    if (!sock->send_data(input, input_size)) return false;
    return true;
}

static bool recv_rpc_cmd_deferred(const socket_ptr & sock, void * output, size_t output_size) {
    const socket_ptr rsock = rpc_response_sock(sock);
    uint64_t resp_size;
    if (!rsock->recv_data(&resp_size, sizeof(resp_size))) return false;
    if (resp_size != output_size) {
        GGML_LOG_ERROR("[%s] deferred response size mismatch: expected %zu, got %" PRIu64 "\n",
                       __func__, output_size, resp_size);
        return false;
    }
    if (!rsock->recv_data(output, output_size)) return false;
    return true;
}

static constexpr size_t RPC_HELLO_REQ_V3_SIZE = RPC_CONN_CAPS_SIZE;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
    uint32_t session_id;
    uint8_t  dual_socket;
    uint8_t  reserved[3];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t flags; // bit0: dual_socket negotiated
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
    uint32_t session_id;
};

struct rpc_msg_hello_rsp_v3 {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_channel_bind_req {
    uint32_t session_id;
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_copy_tensor_peer_req {
    char src_endpoint[256];
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
};

// Path B: event record command (TCP ordering after GRAPH_RECOMPUTE)
struct rpc_msg_event_record_req {
    uint64_t event_id;
    uint32_t device;
};

struct rpc_msg_event_record_rsp {
    uint64_t event_id;
    uint32_t result;  // 0 = success
};

#pragma pack(pop)

// Path B: client-side event payload stored in ggml_backend_event::context
struct rpc_event_t {
    uint64_t id;
    socket_ptr sock;
    bool response_pending;
};

static void rpc_finish_event_response(rpc_event_t * ev) {
    if (!ev || !ev->response_pending || !ev->sock) {
        return;
    }
    if (tls_pending_event.pending && tls_pending_event.sock == ev->sock) {
        rpc_msg_event_record_rsp rsp;
        if (!recv_rpc_cmd_deferred(ev->sock, &rsp, sizeof(rsp))) {
            GGML_LOG_ERROR("[%s] failed to read event response\n", __func__);
        }
        tls_pending_event.pending = false;
        if (tls_pending_event.ev) {
            tls_pending_event.ev->response_pending = false;
            tls_pending_event.ev = nullptr;
        }
    }
    ev->response_pending = false;
}

static void drain_pending_event_response(const socket_ptr & sock) {
    if (tls_pending_event.pending && tls_pending_event.sock == sock) {
        const auto t0 = std::chrono::steady_clock::now();
        if (tls_pending_event.ev) {
            rpc_finish_event_response(tls_pending_event.ev);
        } else {
            rpc_msg_event_record_rsp rsp;
            if (!recv_rpc_cmd_deferred(sock, &rsp, sizeof(rsp))) {
                GGML_LOG_ERROR("[%s] failed to drain pending event response\n", __func__);
            }
            tls_pending_event.pending = false;
        }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "drain_event", RPC_CMD_EVENT_RECORD, sizeof(rpc_msg_event_record_rsp), true, us);
    }
}

static void drain_pending_copy_response(const socket_ptr & sock) {
    if (tls_pending_copy.pending && tls_pending_copy.sock == sock) {
        const auto t0 = std::chrono::steady_clock::now();
        rpc_msg_copy_tensor_rsp rsp;
        if (!recv_rpc_cmd_deferred(sock, &rsp, sizeof(rsp))) {
            GGML_LOG_ERROR("[%s] failed to drain pending copy response\n", __func__);
        }
        tls_pending_copy.pending = false;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "drain_copy", RPC_CMD_COPY_TENSOR, sizeof(rpc_msg_copy_tensor_rsp), true, us);
    }
}

static std::string rpc_endpoint_host(const std::string & endpoint) {
    const size_t pos = endpoint.find(':');
    return pos == std::string::npos ? endpoint : endpoint.substr(0, pos);
}

static bool rpc_same_host(const std::string & a, const std::string & b) {
    return rpc_endpoint_host(a) == rpc_endpoint_host(b);
}

static bool rpc_same_endpoint(const char * a, const char * b) {
    return a && b && strcmp(a, b) == 0;
}

static void flush_pending_get_tensor_for_socket(const socket_ptr & sock) {
    for (auto it = tls_pending_get_tensor.begin(); it != tls_pending_get_tensor.end(); ) {
        if (it->sock != sock) {
            ++it;
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const socket_ptr rsock = rpc_response_sock(it->sock);
        uint64_t out_size;
        if (!rsock->recv_data(&out_size, sizeof(out_size))) {
            GGML_LOG_ERROR("[%s] failed to read get_tensor response size\n", __func__);
            it = tls_pending_get_tensor.erase(it);
            continue;
        }
        if (out_size != it->size) {
            GGML_LOG_ERROR("[%s] get_tensor response size mismatch: expected %zu, got %zu\n",
                            __func__, it->size, (size_t) out_size);
        }
        if (!rsock->recv_data(it->data, it->size)) {
            GGML_LOG_ERROR("[%s] failed to read get_tensor response data\n", __func__);
        }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "flush_get", RPC_CMD_GET_TENSOR, it->size, true, us);
        it = tls_pending_get_tensor.erase(it);
    }
}

static void flush_pending_get_tensor() {
    if (!rpc_event_defer_barrier() && tls_pending_event.pending && tls_pending_event.sock) {
        drain_pending_event_response(tls_pending_event.sock);
    }
    while (!tls_pending_get_tensor.empty()) {
        flush_pending_get_tensor_for_socket(tls_pending_get_tensor.front().sock);
    }
}

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    uint64_t    last_graph_uid;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    // Path B: track pending compute for event linkage
    socket_ptr  last_compute_sock;
    bool        last_compute_sent_event = false;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
};

static const char * rpc_buft_endpoint(ggml_backend_buffer_t buf) {
    if (!buf) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (!buft || !buft->context) {
        return nullptr;
    }
    auto * ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
    return ctx->endpoint.c_str();
}

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool send_response(const socket_ptr & cmd, const void * msg, size_t msg_size) {
    return send_msg(rpc_response_sock(cmd), msg, msg_size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!tls_pending_hash.empty()) {
        flush_pending_hash_all();
    }
    flush_set_tensor_batch();
    // send header + data in one go to avoid TCP buffering issues on small packets
    std::vector<uint8_t> buf(1 + sizeof(uint64_t) + input_size);
    buf[0] = (uint8_t)cmd;
    memcpy(buf.data() + 1, &input_size, sizeof(input_size));
    if (input_size > 0 && input) {
        memcpy(buf.data() + 1 + sizeof(input_size), input, input_size);
    }
    if (!sock->send_data(buf.data(), buf.size())) {
        return false;
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    rpc_trace_emit(__func__, "send_only", cmd, input_size, false, us);
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!tls_pending_hash.empty()) {
        flush_pending_hash_all();
    }
    // Socket-scoped drain: protect TCP framing without stalling other RPC sockets.
    // B+9: defer EVENT recv until pipeline_barrier when GGML_RPC_EVENT_DEFER_BARRIER=1.
    if (cmd != RPC_CMD_HELLO && cmd != RPC_CMD_DEVICE_COUNT) {
        drain_pending_copy_response(sock);
        if (!rpc_event_defer_barrier()) {
            drain_pending_event_response(sock);
        }
        flush_pending_get_tensor_for_socket(sock);
        flush_pending_hash_for_socket(sock);
        flush_set_tensor_batch();
    }
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    const socket_ptr rsock = rpc_response_sock(sock);
    uint64_t out_size;
    if (!rsock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!rsock->recv_data(output, output_size)) {
        return false;
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    rpc_trace_emit(__func__, "send_recv", cmd, input_size + output_size, true, us);
    return true;
}

static void rpc_issue_set_tensor_payload(const socket_ptr & sock, const rpc_tensor & tensor,
                                         uint64_t offset, const void * data, size_t size) {
    if (sock->server_supports_batch) {
        if (tls_set_batch.count > 0 && tls_set_batch_sock && tls_set_batch_sock != sock) {
            flush_set_tensor_batch();
        }
        if (tls_set_batch.count == 0) {
            tls_set_batch_sock = sock;
        }
        set_tensor_batch_append(tensor, offset, data, size);
        if (tls_set_batch.buf.size() >= RPC_SET_TENSOR_BATCH_MAX_SIZE) {
            flush_set_tensor_batch();
        }
    } else {
        const size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
        std::vector<uint8_t> input(input_size, 0);
        memcpy(input.data(), &tensor, sizeof(rpc_tensor));
        memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
        memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
        bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
}

static void flush_pending_hash_for_socket(const socket_ptr & sock) {
    if (!sock) {
        return;
    }
    auto it = tls_pending_hash.begin();
    while (it != tls_pending_hash.end()) {
        if (it->sock != sock) {
            ++it;
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        rpc_msg_set_tensor_hash_rsp response;
        if (!recv_rpc_cmd_deferred(sock, &response, sizeof(response))) {
            GGML_LOG_ERROR("[%s] failed to read hash response\n", __func__);
            it = tls_pending_hash.erase(it);
            continue;
        }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "flush_hash", RPC_CMD_SET_TENSOR_HASH, sizeof(response), true, us);
        if (response.result) {
            tls_hash_present[it->cache_key] = true;
        } else {
            tls_hash_present[it->cache_key] = false;
            if (!it->staging.empty()) {
                rpc_issue_set_tensor_payload(sock, it->tensor, it->offset, it->staging.data(), it->staging.size());
            }
        }
        it = tls_pending_hash.erase(it);
    }
}

static void flush_pending_hash_all() {
    std::vector<socket_ptr> socks;
    for (const auto & ph : tls_pending_hash) {
        if (ph.sock && std::find(socks.begin(), socks.end(), ph.sock) == socks.end()) {
            socks.push_back(ph.sock);
        }
    }
    for (const auto & sock : socks) {
        flush_pending_hash_for_socket(sock);
    }
}

// RPC client-side implementation

static bool rpc_client_bind_response_channel(const socket_ptr & cmd_sock,
                                             const std::string & host, int port,
                                             uint32_t session_id) {
    auto rsp_sock = socket_t::connect(host.c_str(), port);
    if (!rsp_sock) {
        return false;
    }
    const uint8_t cmd = (uint8_t) RPC_CMD_CHANNEL_BIND;
    if (!rsp_sock->send_data(&cmd, sizeof(cmd))) {
        return false;
    }
    rpc_msg_channel_bind_req bind_req = { session_id };
    if (!send_msg(rsp_sock, &bind_req, sizeof(bind_req))) {
        return false;
    }
    cmd_sock->rsp_channel = rsp_sock;
    rpc_register_socket(rsp_sock);
    LOG_DBG("[%s] dual-socket response channel bound session=%u\n", __func__, session_id);
    return true;
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool rpc_exchange_hello(const socket_ptr & sock, const void * req, size_t req_size,
                               rpc_msg_hello_rsp & response, bool expect_v4_rsp) {
    const uint8_t cmd = (uint8_t) RPC_CMD_HELLO;
    const uint64_t input_size = req_size;
    if (!sock->send_data(&cmd, sizeof(cmd))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (req_size > 0 && !sock->send_data(req, req_size)) {
        return false;
    }
    if (expect_v4_rsp) {
        return recv_msg(sock, &response, sizeof(response));
    }
    rpc_msg_hello_rsp_v3 rsp3 = {};
    if (!recv_msg(sock, &rsp3, sizeof(rsp3))) {
        return false;
    }
    response.major = rsp3.major;
    response.minor = rsp3.minor;
    response.patch = rsp3.patch;
    response.flags = rsp3.padding;
    memcpy(response.conn_caps, rsp3.conn_caps, sizeof(response.conn_caps));
    response.session_id = 0;
    return true;
}

static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, const char * endpoint) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);
    request.session_id = rpc_gen_session_id();
    request.dual_socket = ggml_backend_rpc_dual_socket() ? 1 : 0;

    const bool use_v4_req = request.dual_socket != 0;
    const bool status = use_v4_req
        ? rpc_exchange_hello(sock, &request, sizeof(request), response, true)
        : rpc_exchange_hello(sock, request.conn_caps, RPC_HELLO_REQ_V3_SIZE, response, false);
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->server_supports_batch = (response.major == RPC_PROTO_MAJOR_VERSION && response.minor >= 1);
    sock->server_supports_peer_copy = (response.major == RPC_PROTO_MAJOR_VERSION && response.minor >= 3);
    sock->update_caps(response.conn_caps);

    if (request.dual_socket && response.minor >= 4 && (response.flags & 1) && response.session_id != 0) {
        std::string host;
        int port = 0;
        if (parse_endpoint(endpoint, host, port)) {
            if (!rpc_client_bind_response_channel(sock, host, port, response.session_id)) {
                GGML_LOG_WARN("RPC %s: dual-socket bind failed; falling back to single socket\n", endpoint);
                sock->rsp_channel.reset();
            }
        }
    }

    if (endpoint && endpoint[0]) {
        rpc_trace_emit_hello(endpoint, response.minor, sock->server_supports_peer_copy);
        GGML_LOG_INFO("RPC %s: proto %d.%d peer_copy=%s dual=%s\n", endpoint, response.major, response.minor,
                      sock->server_supports_peer_copy ? "yes" : "no",
                      sock->rsp_channel ? "yes" : "no");
    }
    // reset B+ pending state after fresh hello to avoid stale drain on subsequent cmds
    tls_pending_event = {nullptr, false, nullptr};
    tls_pending_copy = {nullptr, false};
    tls_pending_hash.clear();
    tls_hash_active_sock.reset();
    return true;
}

static socket_ptr rpc_ephemeral_connect(const std::string & endpoint) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return nullptr;
    }
    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (!sock) {
        return nullptr;
    }
    if (!negotiate_hello(sock, endpoint.c_str())) {
        return nullptr;
    }
    return sock;
}

static bool rpc_peer_get_tensor(const std::string & endpoint, const rpc_tensor & tensor,
                                size_t offset, size_t size, std::vector<uint8_t> & out) {
    auto sock = rpc_ephemeral_connect(endpoint);
    if (!sock) {
        return false;
    }
    rpc_msg_get_tensor_req request;
    request.tensor = tensor;
    request.offset = offset;
    request.size = size;
    out.resize(size);
    return send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), out.data(), size);
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!negotiate_hello(sock, endpoint.c_str())) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
    rpc_register_socket(sock);
    return sock;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto sock = ctx->sock;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (ggml_backend_rpc_hash_defer()) {
        if (tls_hash_active_sock && tls_hash_active_sock != sock) {
            flush_pending_hash_for_socket(tls_hash_active_sock);
        }
        flush_pending_hash_for_socket(sock);
        tls_hash_active_sock = sock;
    }
    if (size > HASH_THRESHOLD) {
        if (tls_set_batch.count > 0 && tls_set_batch_sock && tls_set_batch_sock != sock) {
            flush_set_tensor_batch();
        }
        flush_set_tensor_batch();
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        const uint64_t cache_key = rpc_hash_cache_key(sock, request.hash, request.tensor.data, request.offset);
        auto cache_it = tls_hash_present.find(cache_key);
        if (cache_it != tls_hash_present.end() && cache_it->second) {
            return;
        }
        if (ggml_backend_rpc_hash_defer()) {
            bool status = send_rpc_cmd_deferred(sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request));
            RPC_STATUS_ASSERT(status);
            rpc_pending_hash ph;
            ph.sock = sock;
            ph.cache_key = cache_key;
            ph.tensor = rpc_tensor;
            ph.offset = offset;
            ph.staging.resize(size);
            memcpy(ph.staging.data(), data, size);
            tls_pending_hash.push_back(std::move(ph));
            return;
        }
        rpc_msg_set_tensor_hash_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            tls_hash_present[cache_key] = true;
            return;
        }
        tls_hash_present[cache_key] = false;
    }
    rpc_issue_set_tensor_payload(sock, rpc_tensor, offset, data, size);
}

static void ggml_backend_rpc_buffer_get_tensor_async(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                     void * data, size_t offset, size_t size, bool batch_send) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto sock = ctx->sock;
    if (!batch_send && !rpc_event_defer_barrier()) {
        drain_pending_event_response(sock);
    }
    if (!batch_send) {
        flush_pending_get_tensor_for_socket(sock);
        flush_pending_hash_for_socket(sock);
        flush_set_tensor_batch();
    }
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    uint8_t cmd_byte = RPC_CMD_GET_TENSOR;
    sock->send_data(&cmd_byte, sizeof(cmd_byte));
    uint64_t input_size = sizeof(rpc_msg_get_tensor_req);
    sock->send_data(&input_size, sizeof(input_size));
    sock->send_data(&request, sizeof(request));
    tls_pending_get_tensor.push_back({sock, data, size});
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor,
                                              void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer == nullptr || !ggml_backend_buffer_is_rpc(buffer)) {
        ggml_backend_tensor_get(tensor, data, offset, size);
        return;
    }
    ggml_backend_rpc_buffer_get_tensor_async(buffer, tensor, data, offset, size, false);
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto sock = ctx->sock;
    if (!rpc_event_defer_barrier()) {
        drain_pending_event_response(sock);
    }
    flush_pending_get_tensor_for_socket(sock);
    flush_pending_hash_for_socket(sock);
    flush_set_tensor_batch();
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
}

// B+13: local/host/CUDA/HIP src -> RPC dst via fire-and-forget SET_TENSOR (no copy response wait).
static bool rpc_issue_upload_tensor(ggml_backend_t backend_src, const ggml_tensor * src, ggml_tensor * dst, bool defer_response) {
    if (src == nullptr || dst == nullptr || dst->buffer == nullptr) {
        return false;
    }
    if (!ggml_backend_buffer_is_rpc(dst->buffer)) {
        return false;
    }
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        return false;
    }

    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    if (src_buffer == nullptr) {
        return false;
    }

    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *) dst->buffer->context;
    auto sock = dst_ctx->sock;

    if (!defer_response) {
        drain_pending_copy_response(sock);
    }
    if (!rpc_event_defer_barrier()) {
        drain_pending_event_response(sock);
    }
    flush_pending_get_tensor_for_socket(sock);
    flush_set_tensor_batch();

    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }

    const void * data = src->data;
    thread_local std::vector<uint8_t> tls_upload_staging;

    if (!ggml_backend_buffer_is_host(src_buffer)) {
        if (!backend_src) {
            return false;
        }
        ggml_backend_synchronize(backend_src);
        tls_upload_staging.resize(size);
        ggml_backend_tensor_get(src, tls_upload_staging.data(), 0, size);
        data = tls_upload_staging.data();
    }

    rpc_tensor rpc_t = serialize_tensor(dst);
    const size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_t, sizeof(rpc_tensor));
    const uint64_t offset = 0;
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);

    const char * dst_ep = rpc_buft_endpoint(dst->buffer);
    rpc_trace_emit_copy_issue(RPC_CMD_SET_TENSOR, "", dst_ep ? dst_ep : "", false, defer_response);

    return send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
}

// B+13: RPC src -> local dst via deferred GET_TENSOR + async H2D at graph_compute entry.
static bool rpc_issue_download_tensor(ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst, bool defer_response) {
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        return false;
    }
    if (!ggml_backend_buffer_is_rpc(src->buffer) || ggml_backend_buffer_is_rpc(dst->buffer)) {
        return false;
    }
    if (!backend_dst) {
        return false;
    }

    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *) src->buffer->context;
    auto sock = src_ctx->sock;

    if (!defer_response) {
        drain_pending_copy_response(sock);
    }
    if (!defer_response && !rpc_event_defer_barrier()) {
        drain_pending_event_response(sock);
    }
    if (!defer_response) {
        flush_pending_get_tensor_for_socket(sock);
        flush_set_tensor_batch();
    }

    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }

    for (const auto & pending : tls_pending_downloads) {
        if (pending.dst == dst) {
            return true;
        }
    }

    rpc_pending_download pd;
    pd.sock = sock;
    pd.staging.resize(size);
    pd.dst = dst;
    pd.dst_backend = backend_dst;

    ggml_backend_rpc_buffer_get_tensor_async(src->buffer, src, pd.staging.data(), 0, size, defer_response);

    const char * src_ep = rpc_buft_endpoint(src->buffer);
    rpc_trace_emit_copy_issue(RPC_CMD_GET_TENSOR, src_ep ? src_ep : "", "", false, defer_response);

    tls_pending_downloads.push_back(std::move(pd));
    return true;
}

bool ggml_backend_rpc_try_download_tensor(ggml_backend_t dst_backend, const ggml_tensor * src, ggml_tensor * dst) {
    return rpc_issue_download_tensor(dst_backend, src, dst, true);
}

bool ggml_backend_rpc_download_pending_for_dst(const ggml_tensor * dst) {
    if (dst == nullptr) {
        return false;
    }
    for (const auto & pending : tls_pending_downloads) {
        if (pending.dst == dst) {
            return true;
        }
    }
    return false;
}

bool ggml_backend_rpc_try_upload_tensor(ggml_backend_t src_backend, const ggml_tensor * src, ggml_tensor * dst) {
    return rpc_issue_upload_tensor(src_backend, src, dst, true);
}

static bool rpc_pending_download_matches_dst(const rpc_pending_download & pd,
                                             const ggml_tensor * const * dst, size_t n_dst) {
    if (n_dst == 0) {
        return true;
    }
    for (size_t i = 0; i < n_dst; i++) {
        if (pd.dst == dst[i]) {
            return true;
        }
    }
    return false;
}

void ggml_backend_rpc_flush_pending_downloads_for_dst(const ggml_tensor * const * dst, size_t n_dst) {
    if (tls_pending_downloads.empty()) {
        return;
    }

    bool need_flush = n_dst == 0;
    if (!need_flush) {
        for (const auto & pd : tls_pending_downloads) {
            if (rpc_pending_download_matches_dst(pd, dst, n_dst)) {
                need_flush = true;
                break;
            }
        }
    }
    if (!need_flush) {
        return;
    }

    std::vector<socket_ptr> socks;
    for (const auto & pd : tls_pending_downloads) {
        if (std::find(socks.begin(), socks.end(), pd.sock) == socks.end()) {
            socks.push_back(pd.sock);
        }
    }
    for (const auto & sock : socks) {
        if (!rpc_event_defer_barrier() && tls_pending_event.pending && tls_pending_event.sock == sock) {
            drain_pending_event_response(sock);
        }
        flush_pending_get_tensor_for_socket(sock);
        flush_pending_hash_for_socket(sock);
    }

    const bool defer_h2d_sync = ggml_backend_rpc_get_tensor_defer();
    auto it = tls_pending_downloads.begin();
    while (it != tls_pending_downloads.end()) {
        if (rpc_pending_download_matches_dst(*it, dst, n_dst)) {
            if (defer_h2d_sync) {
                ggml_backend_tensor_set(it->dst, it->staging.data(), 0, it->staging.size());
            } else {
                ggml_backend_tensor_set_async(it->dst_backend, it->dst, it->staging.data(), 0, it->staging.size());
            }
            it = tls_pending_downloads.erase(it);
        } else {
            ++it;
        }
    }
}

bool ggml_backend_rpc_relay_pending_for_dst(const ggml_tensor * dst) {
    if (dst == nullptr) {
        return false;
    }
    for (const auto & relay : tls_pending_relays) {
        if (relay.dst == dst) {
            return true;
        }
    }
    return false;
}

static bool rpc_pending_relay_matches_dst(const rpc_pending_relay & pr,
                                          const ggml_tensor * const * dst, size_t n_dst) {
    if (n_dst == 0) {
        return true;
    }
    for (size_t i = 0; i < n_dst; i++) {
        if (pr.dst == dst[i]) {
            return true;
        }
    }
    return false;
}

static void flush_one_pending_relay(rpc_pending_relay & relay) {
    if (relay.src == nullptr || relay.dst == nullptr) {
        return;
    }
    const size_t size = ggml_nbytes(relay.src);
    if (size == 0) {
        return;
    }
    drain_pending_copy_response(relay.src_sock);
    drain_pending_copy_response(relay.dst_sock);
    if (!rpc_event_defer_barrier()) {
        drain_pending_event_response(relay.src_sock);
        drain_pending_event_response(relay.dst_sock);
    }
    flush_pending_get_tensor_for_socket(relay.src_sock);
    flush_pending_get_tensor_for_socket(relay.dst_sock);
    flush_set_tensor_batch();
    relay.staging.resize(size);
    ggml_backend_rpc_buffer_get_tensor(relay.src->buffer, relay.src, relay.staging.data(), 0, size);
    if (!rpc_issue_relay_upload(relay)) {
        GGML_LOG_ERROR("[%s] relay upload failed\n", __func__);
    }
}

static void flush_pending_relays_for_dst(const ggml_tensor * const * dst, size_t n_dst) {
    if (tls_pending_relays.empty()) {
        return;
    }
    auto it = tls_pending_relays.begin();
    while (it != tls_pending_relays.end()) {
        if (rpc_pending_relay_matches_dst(*it, dst, n_dst)) {
            flush_one_pending_relay(*it);
            it = tls_pending_relays.erase(it);
        } else {
            ++it;
        }
    }
}

void ggml_backend_rpc_flush_pending_relays_for_dst(const ggml_tensor * const * dst, size_t n_dst) {
    flush_pending_relays_for_dst(dst, n_dst);
}

void ggml_backend_rpc_flush_pending_downloads(void) {
    ggml_backend_rpc_flush_pending_downloads_for_dst(nullptr, 0);
}

static bool rpc_issue_relay_upload(const rpc_pending_relay & relay) {
    const size_t size = relay.staging.size();
    if (size == 0 || relay.dst == nullptr) {
        return true;
    }
    rpc_tensor rpc_t = serialize_tensor(relay.dst);
    const size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_t, sizeof(rpc_tensor));
    const uint64_t offset = 0;
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), relay.staging.data(), size);
    return send_rpc_cmd(relay.dst_sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
}

static void flush_pending_relays() {
    flush_pending_relays_for_dst(nullptr, 0);
}

// Same-host isolated RPC endpoints (e.g. triton :50054 + :50055 docker): client pulls
// from src worker and pushes to dst. COPY_TENSOR_PEER fails across separate processes.
static bool rpc_issue_relay_copy_tensor(const ggml_tensor * src, ggml_tensor * dst, bool defer_response) {
    ggml_backend_buffer_t src_buffer = src->buffer;
    ggml_backend_buffer_t dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *) src_buffer->context;
    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *) dst_buffer->context;
    auto src_sock = src_ctx->sock;
    auto dst_sock = dst_ctx->sock;

    if (!defer_response) {
        drain_pending_copy_response(dst_sock);
        drain_pending_copy_response(src_sock);
    }
    if (!rpc_event_defer_barrier()) {
        drain_pending_event_response(src_sock);
        drain_pending_event_response(dst_sock);
    }
    flush_pending_get_tensor_for_socket(src_sock);
    flush_pending_get_tensor_for_socket(dst_sock);
    flush_set_tensor_batch();

    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }

    const char * src_ep = rpc_buft_endpoint(src_buffer);
    const char * dst_ep = rpc_buft_endpoint(dst_buffer);
    if (!src_ep || !dst_ep) {
        return false;
    }

    if (defer_response) {
        rpc_pending_relay pr;
        pr.src_sock = src_sock;
        pr.dst_sock = dst_sock;
        pr.src = src;
        pr.dst = dst;
        rpc_trace_emit_copy_issue(RPC_CMD_GET_TENSOR, src_ep, dst_ep, false, true);
        tls_pending_relays.push_back(std::move(pr));
        return true;
    }

    thread_local std::vector<uint8_t> relay_staging;
    relay_staging.resize(size);
    ggml_backend_rpc_buffer_get_tensor(src_buffer, src, relay_staging.data(), 0, size);

    rpc_pending_relay pr;
    pr.src_sock = src_sock;
    pr.dst_sock = dst_sock;
    pr.src = src;
    pr.dst = dst;
    pr.staging = std::move(relay_staging);
    rpc_trace_emit_copy_issue(RPC_CMD_SET_TENSOR, src_ep, dst_ep, false, false);
    return rpc_issue_relay_upload(pr);
}

static bool rpc_issue_copy_tensor(const ggml_tensor * src, ggml_tensor * dst, bool defer_response) {
    if (!ggml_backend_buffer_is_rpc(src->buffer) || !ggml_backend_buffer_is_rpc(dst->buffer)) {
        return false;
    }

    ggml_backend_buffer_t src_buffer = src->buffer;
    ggml_backend_buffer_t dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *) src_buffer->context;
    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *) dst_buffer->context;
    auto sock = dst_ctx->sock;

    const char * src_ep = rpc_buft_endpoint(src_buffer);
    const char * dst_ep = rpc_buft_endpoint(dst_buffer);
    const bool cross_rpc = src_ctx->sock != dst_ctx->sock;
    const bool same_host = src_ep && dst_ep && rpc_same_host(src_ep, dst_ep);
    const bool same_endpoint = src_ep && dst_ep && rpc_same_endpoint(src_ep, dst_ep);
    const bool same_host_diff_endpoint = cross_rpc && same_host && !same_endpoint;

    // Peer copy only for same-endpoint workers (dual-socket single rpc-server).
    const bool peer_copy = cross_rpc
        && same_host
        && same_endpoint
        && sock->server_supports_peer_copy;

    if (cross_rpc && !peer_copy) {
        if (same_host_diff_endpoint) {
            return rpc_issue_relay_copy_tensor(src, dst, defer_response);
        }
        return false;
    }

    if (!defer_response) {
        drain_pending_copy_response(sock);
    }
    if (!rpc_event_defer_barrier()) {
        drain_pending_event_response(sock);
    }
    flush_pending_get_tensor_for_socket(sock);
    flush_set_tensor_batch();

    if (peer_copy) {
        rpc_trace_emit_copy_issue(RPC_CMD_COPY_TENSOR_PEER, src_ep, dst_ep, true, defer_response);
        rpc_msg_copy_tensor_peer_req request = {};
        snprintf(request.src_endpoint, sizeof(request.src_endpoint), "%s", src_ep);
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        if (defer_response) {
            if (!send_rpc_cmd(sock, RPC_CMD_COPY_TENSOR_PEER, &request, sizeof(request))) {
                return false;
            }
            tls_pending_copy.sock = sock;
            tls_pending_copy.pending = true;
            return true;
        }
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_COPY_TENSOR_PEER, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }

    rpc_trace_emit_copy_issue(RPC_CMD_COPY_TENSOR, src_ep, dst_ep, false, defer_response);
    rpc_msg_copy_tensor_req request;
    request.src = serialize_tensor(src);
    request.dst = serialize_tensor(dst);
    if (defer_response) {
        if (!send_rpc_cmd(sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request))) {
            return false;
        }
        tls_pending_copy.sock = sock;
        tls_pending_copy.pending = true;
        return true;
    }
    rpc_msg_copy_tensor_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.result;
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    return rpc_issue_copy_tensor(src, dst, false);
}

static bool ggml_backend_rpc_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                            const ggml_tensor * src, ggml_tensor * dst) {
    if (src && src->buffer && ggml_backend_buffer_is_rpc(src->buffer) &&
        dst && dst->buffer && !ggml_backend_buffer_is_rpc(dst->buffer)) {
        return rpc_issue_download_tensor(backend_dst, src, dst, true);
    }
    if (!ggml_backend_is_rpc(backend_dst)) {
        return false;
    }
    if (!ggml_backend_buffer_is_rpc(dst->buffer)) {
        return false;
    }
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        return rpc_issue_copy_tensor(src, dst, true);
    }
    return rpc_issue_upload_tensor(backend_src, src, dst, true);
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

// B+7: cache GET_ALLOC_SIZE responses (shape/op keyed; ignores pointers and strides)
struct rpc_alloc_shape_key {
    uint32_t type;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
};

static thread_local std::unordered_map<uint64_t, size_t> tls_alloc_size_cache;

static rpc_alloc_shape_key rpc_alloc_shape_key_from_tensor(const ggml_tensor * tensor) {
    rpc_alloc_shape_key key = {};
    if (!tensor) {
        return key;
    }
    key.type = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        key.ne[i] = tensor->ne[i];
    }
    key.op = tensor->op;
    memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
    key.flags = tensor->flags;
    return key;
}

static uint64_t rpc_hash_alloc_size_req(const std::string & endpoint,
                                        uint32_t device,
                                        const ggml_tensor * tensor) {
    uint64_t hash = fnv_hash(reinterpret_cast<const uint8_t *>(endpoint.data()), endpoint.size());
    hash ^= (uint64_t) device + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    const rpc_alloc_shape_key tkey = rpc_alloc_shape_key_from_tensor(tensor);
    hash ^= fnv_hash(reinterpret_cast<const uint8_t *>(&tkey), sizeof(tkey)) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const rpc_alloc_shape_key skey = rpc_alloc_shape_key_from_tensor(tensor->src[i]);
        hash ^= fnv_hash(reinterpret_cast<const uint8_t *>(&skey), sizeof(skey)) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return hash;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        const uint64_t cache_key = rpc_hash_alloc_size_req(buft_ctx->endpoint, buft_ctx->device, tensor);
        auto cached = tls_alloc_size_cache.find(cache_key);
        if (cached != tls_alloc_size_cache.end()) {
            return cached->second;
        }

        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        tls_alloc_size_cache.emplace(cache_key, response.alloc_size);
        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    rpc_drain_all_endpoints_pending();
}

void ggml_backend_rpc_drain_all_endpoints(void) {
    rpc_drain_all_endpoints_pending();
}

int ggml_backend_rpc_server_count(void) {
    std::lock_guard<std::mutex> lock(g_rpc_reg_map_mutex);
    return (int) g_rpc_reg_map.size();
}

bool ggml_backend_rpc_event_defer_barrier(void) {
    if (!rpc_pipeline_plus_enabled() || rpc_event_defer_env_enabled() == 0) {
        return false;
    }
    return ggml_backend_rpc_server_count() >= rpc_event_defer_min_servers();
}

bool ggml_backend_rpc_get_tensor_defer(void) {
    return rpc_pipeline_plus_enabled() && rpc_get_tensor_defer_env_enabled() != 0;
}

bool ggml_backend_rpc_hash_defer(void) {
    return rpc_pipeline_plus_enabled() && rpc_hash_defer_env_enabled() != 0;
}

static void rpc_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    auto * ev = (rpc_event_t *) event->context;

    if (rpc_ctx->last_compute_sent_event && rpc_ctx->last_compute_sock) {
        tls_pending_event.sock = rpc_ctx->last_compute_sock;
        tls_pending_event.pending = true;
        tls_pending_event.ev = ev;
        rpc_ctx->last_compute_sent_event = false;
        ev->sock = rpc_ctx->last_compute_sock;
        ev->response_pending = true;
    } else {
        auto sock = get_socket(rpc_ctx->endpoint);
        rpc_msg_event_record_req ev_req = {ev->id, rpc_ctx->device};
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, sizeof(ev_req));
        tls_pending_event.sock = sock;
        tls_pending_event.pending = true;
        tls_pending_event.ev = ev;
        ev->sock = sock;
        ev->response_pending = true;
    }
}

static void rpc_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    auto * ev = (rpc_event_t *) event->context;
    rpc_finish_event_response(ev);
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    auto sock = get_socket(rpc_ctx->endpoint);
    flush_pending_hash_all();
    flush_set_tensor_batch();

    const auto t0 = std::chrono::steady_clock::now();
    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);

        rpc_msg_event_record_req ev_req;
        ev_req.event_id = 0;
        ev_req.device = rpc_ctx->device;
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, sizeof(ev_req));
        tls_pending_event.sock = sock;
        tls_pending_event.pending = true;
        rpc_ctx->last_compute_sock = sock;
        rpc_ctx->last_compute_sent_event = true;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "graph_recompute", RPC_CMD_GRAPH_RECOMPUTE, sizeof(request), false, us);
    } else {
        rpc_dev_ctx->last_graph_uid = cgraph->uid;
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
        rpc_ctx->last_compute_sock = nullptr;
        rpc_ctx->last_compute_sent_event = false;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "graph_compute", RPC_CMD_GRAPH_COMPUTE, input.size(), false, us);
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_rpc_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ rpc_backend_event_record,
    /* .event_wait              = */ rpc_backend_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        compute_worker = std::thread([this]() { compute_worker_loop(); });
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool copy_tensor_peer(const rpc_msg_copy_tensor_peer_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    void enqueue_graph_compute(std::vector<uint8_t> input);
    void enqueue_graph_recompute(rpc_msg_graph_recompute_req request);
    void wait_compute_idle();

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    void compute_worker_loop();
    void submit_compute_job(std::function<void()> job);

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;

    std::mutex                    compute_mtx;
    std::condition_variable         compute_cv;
    std::deque<std::function<void()>> compute_queue;
    std::thread                     compute_worker;
    std::atomic<bool>               compute_shutdown{false};
    std::atomic<int>                compute_inflight{0};
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

bool rpc_server::copy_tensor_peer(const rpc_msg_copy_tensor_peer_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (dst == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing dst tensor\n", __func__);
        return false;
    }

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    if (src == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing src tensor\n", __func__);
        return false;
    }
    const size_t nbytes = ggml_nbytes(src);
    std::vector<uint8_t> data;
    if (!rpc_peer_get_tensor(request.src_endpoint, request.src, 0, nbytes, data)) {
        GGML_LOG_ERROR("[%s] peer get_tensor failed from %s\n", __func__, request.src_endpoint);
        return false;
    }

    ggml_backend_tensor_set(dst, data.data(), 0, nbytes);
    response.result = 1;
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (stored_graphs[device].buffer.size() < buf_size) {
        stored_graphs[device].buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    rpc_trace_emit("rpc_server::graph_compute", "server_compute", RPC_CMD_GRAPH_COMPUTE, input.size(), true, us);
    stored_graphs[device].graph = graph;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u\n", __func__, device);
    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    rpc_trace_emit("rpc_server::graph_recompute", "server_compute", RPC_CMD_GRAPH_RECOMPUTE, 0, true, us);
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

void rpc_server::submit_compute_job(std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lock(compute_mtx);
        compute_inflight++;
        compute_queue.push_back(std::move(job));
    }
    compute_cv.notify_one();
}

void rpc_server::compute_worker_loop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(compute_mtx);
            compute_cv.wait(lock, [this] {
                return compute_shutdown.load() || !compute_queue.empty();
            });
            if (compute_shutdown.load() && compute_queue.empty()) {
                break;
            }
            job = std::move(compute_queue.front());
            compute_queue.pop_front();
        }
        if (job) {
            job();
        }
        compute_inflight--;
        compute_cv.notify_all();
    }
}

void rpc_server::enqueue_graph_compute(std::vector<uint8_t> input) {
    submit_compute_job([this, input = std::move(input)]() mutable {
        if (!graph_compute(input)) {
            GGML_LOG_ERROR("[%s] async graph_compute failed\n", __func__);
        }
    });
}

void rpc_server::enqueue_graph_recompute(rpc_msg_graph_recompute_req request) {
    submit_compute_job([this, request]() {
        if (!graph_recompute(request)) {
            GGML_LOG_ERROR("[%s] async graph_recompute failed\n", __func__);
        }
    });
}

void rpc_server::wait_compute_idle() {
    std::unique_lock<std::mutex> lock(compute_mtx);
    compute_cv.wait(lock, [this] {
        return compute_queue.empty() && compute_inflight.load() == 0;
    });
}

rpc_server::~rpc_server() {
    {
        std::lock_guard<std::mutex> lock(compute_mtx);
        compute_shutdown = true;
    }
    compute_cv.notify_all();
    if (compute_worker.joinable()) {
        compute_worker.join();
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_channel_bind(socket_ptr sock);
static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock);

static void rpc_connection_thread(std::vector<ggml_backend_t> backends, std::string cache_dir, socket_ptr sock) {
    uint8_t cmd = 0;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd == RPC_CMD_CHANNEL_BIND) {
        rpc_serve_channel_bind(sock);
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO or CHANNEL_BIND, got %d\n", cmd);
        return;
    }
    rpc_serve_client(backends, cache_dir.c_str(), sock);
}

static void rpc_serve_channel_bind(socket_ptr sock) {
    rpc_msg_channel_bind_req req = {};
    if (!recv_msg(sock, &req, sizeof(req))) {
        return;
    }
    std::shared_ptr<rpc_dual_pending> pending;
    {
        std::lock_guard<std::mutex> lock(g_dual_pending_mutex);
        auto it = g_dual_pending.find(req.session_id);
        if (it != g_dual_pending.end()) {
            pending = it->second;
        }
    }
    if (!pending) {
        GGML_LOG_WARN("CHANNEL_BIND: unknown session %u\n", req.session_id);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pending->mutex);
        pending->rsp = sock;
        pending->paired = true;
    }
    pending->cv.notify_all();
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);

    // Read input_size and validate protocol version (HELLO cmd byte already consumed)
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    rpc_msg_hello_req req = {};
    uint8_t req_conn_caps[RPC_CONN_CAPS_SIZE] = {};
    uint32_t session_id = 0;
    uint8_t dual_want = 0;

    if (hello_input_size == RPC_HELLO_REQ_V3_SIZE) {
        if (!sock->recv_data(req_conn_caps, sizeof(req_conn_caps))) {
            return;
        }
        memcpy(req.conn_caps, req_conn_caps, sizeof(req_conn_caps));
    } else if (hello_input_size == sizeof(rpc_msg_hello_req)) {
        if (!sock->recv_data(&req, sizeof(req))) {
            return;
        }
        session_id = req.session_id;
        dual_want = req.dual_socket;
    } else {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu) — expected %zu or %zu\n",
                       (size_t) hello_input_size, RPC_HELLO_REQ_V3_SIZE, sizeof(rpc_msg_hello_req));
        return;
    }

    std::shared_ptr<rpc_dual_pending> pending;
    if (dual_want && RPC_PROTO_MINOR_VERSION >= 4 && session_id != 0) {
        pending = std::make_shared<rpc_dual_pending>();
        {
            std::lock_guard<std::mutex> lock(g_dual_pending_mutex);
            g_dual_pending[session_id] = pending;
        }
    }

    if (hello_input_size == RPC_HELLO_REQ_V3_SIZE) {
        rpc_msg_hello_rsp_v3 rsp3 = {};
        {
            rpc_msg_hello_rsp tmp = {};
            server.hello(tmp);
            rsp3.major = tmp.major;
            rsp3.minor = tmp.minor;
            rsp3.patch = tmp.patch;
        }
        sock->get_caps(rsp3.conn_caps);
        if (!send_msg(sock, &rsp3, sizeof(rsp3))) {
            return;
        }
    } else {
        rpc_msg_hello_rsp rsp = {};
        server.hello(rsp);
        if (pending) {
            rsp.flags |= 1;
            rsp.session_id = session_id;
        }
        sock->get_caps(rsp.conn_caps);
        if (!send_msg(sock, &rsp, sizeof(rsp))) {
            return;
        }
    }

    if (pending) {
        std::unique_lock<std::mutex> lock(pending->mutex);
        pending->cv.wait_for(lock, std::chrono::seconds(10), [&] { return pending->paired; });
        if (pending->paired) {
            sock->rsp_channel = pending->rsp;
            LOG_DBG("[%s] dual-socket response channel paired session=%u\n", __func__, session_id);
        } else {
            GGML_LOG_WARN("[%s] dual-socket bind timeout session=%u; single-socket fallback\n", __func__, session_id);
        }
        {
            std::lock_guard<std::mutex> glock(g_dual_pending_mutex);
            g_dual_pending.erase(session_id);
        }
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    uint8_t cmd = 0;
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_response(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_response(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_response(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_response(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR_PEER: {
                rpc_msg_copy_tensor_peer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor_peer(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                server.enqueue_graph_compute(std::move(input));
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                server.enqueue_graph_recompute(request);
                break;
            }
            case RPC_CMD_SET_TENSOR_BATCH: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (input.size() < sizeof(uint32_t)) {
                    GGML_LOG_ERROR("[%s] RPC_CMD_SET_TENSOR_BATCH: message too small (%zu)\n", __func__, input.size());
                    return;
                }

                uint32_t n_tensors;
                memcpy(&n_tensors, input.data(), sizeof(n_tensors));
                size_t pos = sizeof(n_tensors);

                for (uint32_t i = 0; i < n_tensors; i++) {
                    if (pos + sizeof(rpc_tensor) + sizeof(uint64_t) + sizeof(uint64_t) > input.size()) {
                        GGML_LOG_ERROR("[%s] RPC_CMD_SET_TENSOR_BATCH: truncated entry %u\n", __func__, i);
                        return;
                    }

                    const rpc_tensor * in_tensor = (const rpc_tensor *)(input.data() + pos);
                    pos += sizeof(rpc_tensor);

                    uint64_t offset;
                    memcpy(&offset, input.data() + pos, sizeof(offset));
                    pos += sizeof(offset);

                    uint64_t data_size;
                    memcpy(&data_size, input.data() + pos, sizeof(data_size));
                    pos += sizeof(data_size);

                    if (pos + data_size > input.size()) {
                        GGML_LOG_ERROR("[%s] RPC_CMD_SET_TENSOR_BATCH: entry %u data exceeds message\n", __func__, i);
                        return;
                    }

                    if (data_size > SIZE_MAX) {
                        GGML_LOG_ERROR("[%s] RPC_CMD_SET_TENSOR_BATCH: entry %u data_size overflow\n", __func__, i);
                        return;
                    }

                    const size_t single_size = sizeof(rpc_tensor) + sizeof(uint64_t) + (size_t)data_size;
                    std::vector<uint8_t> single_input(single_size);

                    memcpy(single_input.data(), in_tensor, sizeof(rpc_tensor));
                    memcpy(single_input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
                    memcpy(single_input.data() + sizeof(rpc_tensor) + sizeof(offset), input.data() + pos, (size_t)data_size);
                    pos += (size_t)data_size;

                    if (!server.set_tensor(single_input)) {
                        GGML_LOG_ERROR("[%s] RPC_CMD_SET_TENSOR_BATCH: set_tensor failed for entry %u\n", __func__, i);
                        return;
                    }
                }

                LOG_DBG("[%s] RPC_CMD_SET_TENSOR_BATCH: processed %u tensors\n", __func__, n_tensors);
                break;
            }
            case RPC_CMD_EVENT_RECORD: {
                rpc_msg_event_record_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                server.wait_compute_idle();
                rpc_msg_event_record_rsp response = {request.event_id, 0};
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                LOG_DBG("[%s] RPC_CMD_EVENT_RECORD: event_id=%lu, device=%u\n",
                        __func__, (unsigned long)request.event_id, request.device);
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        std::thread(rpc_connection_thread, backends, std::string(cache_dir ? cache_dir : ""), client_socket).detach();
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static ggml_backend_event_t rpc_event_new(ggml_backend_dev_t device) {
    static std::atomic<uint64_t> next_id{1};
    auto * ev = new rpc_event_t();
    ev->id = next_id++;
    ev->sock = nullptr;
    ev->response_pending = false;
    ggml_backend_event_t wrapper = new ggml_backend_event{device, ev};
    LOG_DBG("[%s] allocated event %lu\n", __func__, (unsigned long)ev->id);
    return wrapper;
}

static void rpc_event_free(ggml_backend_dev_t device, ggml_backend_event_t event) {
    if (event) {
        auto * ev = (rpc_event_t *)event->context;
        LOG_DBG("[%s] freeing event %lu\n", __func__, (unsigned long)ev->id);
        delete ev;
        delete event;
    }
    GGML_UNUSED(device);
}

static void rpc_event_synchronize(ggml_backend_dev_t device, ggml_backend_event_t event) {
    auto * ev = (rpc_event_t *)event->context;
    rpc_finish_event_response(ev);
    LOG_DBG("[%s] event %lu synchronized\n", __func__, (unsigned long)ev->id);
    GGML_UNUSED(device);
}

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
    /* .event_new            = */ rpc_event_new,
    /* .event_free           = */ rpc_event_free,
    /* .event_synchronize    = */ rpc_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    std::lock_guard<std::mutex> lock(g_rpc_reg_map_mutex);
    if (g_rpc_reg_map.find(endpoint) != g_rpc_reg_map.end()) {
        return g_rpc_reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(g_rpc_dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .last_graph_uid = */ 0,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        g_rpc_dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    g_rpc_reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
