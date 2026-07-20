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
    const char * e = getenv("GGML_RPC_TRACE");
    return e ? atoi(e) : 0;
}

static std::mutex rpc_trace_mutex;

static FILE * rpc_trace_file() {
    static FILE * trace_f = nullptr;
    static std::string last_path;
    const char * path = getenv("GGML_RPC_TRACE_FILE");
    if (!path || !path[0]) {
        return nullptr;
    }
    // re-open if path changed (e.g. between repeat runs with different trace dirs)
    if (trace_f && last_path != path) {
        fclose(trace_f);
        trace_f = nullptr;
    }
    if (!trace_f) {
        trace_f = fopen(path, "a");
        last_path = path;
    }
    return trace_f;
}

static void rpc_trace_emit_hotpath_fields(FILE * out) {
    const int32_t decode_id = ggml_pipeline_trace_get_decode_id();
    uint64_t trace_id = ggml_pipeline_trace_get_trace_id();
    int32_t split_id = -1;
    int32_t backend_id = -1;
    ggml_hotpath_trace_get_sched_ctx(&split_id, &backend_id);
    if (decode_id >= 0) {
        fprintf(out, ",\"decode_id\":%d", decode_id);
    }
    fprintf(out, ",\"trace_id\":%llu", (unsigned long long) trace_id);
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
    RPC_CMD_CHANNEL_BIND,         // B+11: pair response socket (value 20)
    RPC_CMD_GRAPH_COMPUTE_ALL,    // Path C: submit full graph for server-side multi-GPU compute (value 21)
    RPC_CMD_GRAPH_RECOMPUTE_ALL,  // Path C: re-execute cached graph (value 22)
    RPC_CMD_GRAPH_COMPUTE_STAGE,  // D6.9: per-stage split filtering for GPipe + profiler (value 23)
    RPC_CMD_COUNT,                 // updated from 22
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
void ggml_backend_rpc_flush_pending_downloads(void);

int ggml_backend_rpc_server_count(void);
bool ggml_backend_rpc_event_defer_barrier(void);
bool ggml_backend_rpc_get_tensor_defer(void);
bool ggml_backend_rpc_hash_defer(void);
bool ggml_backend_rpc_dual_socket(void);

static socket_ptr rpc_response_sock(const socket_ptr & cmd);

static bool rpc_pipeline_plus_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * legacy = getenv("GGML_PIPELINE_SCHED_LEGACY");
        if (legacy != nullptr && atoi(legacy) != 0) {
            v = 0;
        } else if (ggml_pipeline_multi_backend_seq_enabled()) {
            v = 0;
        } else {
            const char * e = getenv("GGML_PIPELINE_PLUS");
            v = e ? (atoi(e) != 0) : 1;
        }
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
        if (e && atoi(e) != 0) {
            GGML_LOG_WARN("GGML_RPC_DUAL_SOCKET is deprecated and will be removed in Phase R3. "
                          "It caused -9.1%% G regression on 4-GPU and a CUDA illegal-memory-access "
                          "crash on RTX 3060 Ti. The flag is blocked - dual socket will NOT be activated.\n");
        }
        // Always disabled - deprecated.
        v = 0;
    }
    return v;
}

static int rpc_multidevice_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_MULTIDEVICE");
        v = e ? atoi(e) : 0;
    }
    return v;
}

static bool rpc_server_telemetry_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_SERVER_TELEMETRY");
        v = e ? atoi(e) : 0;
    }
    return v != 0;
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
    if (tls_pending_event.pending && tls_pending_event.sock) {
        drain_pending_event_response(tls_pending_event.sock);
    }
    flush_pending_get_tensor();
    flush_pending_hash_all();
    flush_set_tensor_batch();
    flush_pending_relays();

    // B+16: flush pending downloads (staging -> tensor H2D copy) so that
    // deferred get_tensor data reaches destination tensors. Without this,
    // ggml_backend_synchronize on RPC backends drains get_tensor responses
    // but leaves staging data stranded, causing graph compute to read stale
    // tensor buffers and corrupt KV-cache attention outputs.
    ggml_backend_rpc_flush_pending_downloads();

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
        if (tls_pending_event.pending && tls_pending_event.sock == sock) {
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
static constexpr float RPC_WEIGHT_SPEED_RATIO = 1.75f;
static constexpr uint32_t GGML_RPC_MAX_DEVICES = 8;

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
// 20 bytes when trace_id supported (patch 3+ / RPC_CAP_TRACE_ID); legacy 12 bytes.
struct rpc_msg_event_record_req {
    uint64_t event_id;
    uint32_t device;
    uint64_t trace_id;
};

struct rpc_msg_event_record_rsp {
    uint64_t event_id;
    uint32_t result;  // 0 = success
    uint64_t trace_id;
};

// Path C: server-side multi-GPU scheduling
struct rpc_msg_graph_compute_all_req {
    uint32_t n_devices;
    uint32_t devices[GGML_RPC_MAX_DEVICES];
    uint32_t sync_mode;        // 0=fire-and-forget, 1=blocking with response
    uint32_t output_requested; // 0=no output, 1=include output in response
    uint32_t seq_id;           // D6.6: sequence ID for multi-seq pipeline dispatch
};

struct rpc_msg_graph_compute_all_rsp {
    uint32_t result;        // 0=success
    uint32_t output_device; // which GPU holds the output tensor
};

// D6.9: per-stage dispatch request (reuses ALL graph serialization, adds stage_id)
struct rpc_msg_graph_compute_stage_req {
    uint32_t n_devices;
    uint32_t devices[GGML_RPC_MAX_DEVICES];
    uint32_t stage_id;          // pipeline stage to filter for (maps to backend_id)
    uint32_t sync_mode;         // 0=fire-and-forget, 1=blocking with response
    uint32_t output_requested;  // 0=no output, 1=include output in response
};

// D4.10: response struct for single-device GRAPH_COMPUTE (telemetry carrier)
struct rpc_msg_graph_compute_rsp {
    uint32_t result;        // 0=success
};

struct rpc_msg_graph_recompute_all_req {
    uint32_t n_devices;
    uint32_t devices[GGML_RPC_MAX_DEVICES];
    uint64_t graph_hash;    // identifies cached graph
    uint32_t sync_mode;     // 0=fire-and-forget, 1=blocking with response
    uint32_t output_requested;
};

struct rpc_msg_graph_recompute_all_rsp {
    uint32_t result;
    uint32_t output_device;
};

// D4.10: telemetry constants
static constexpr uint32_t RPC_TELEMETRY_MAX_DEVICES   = 8;
static constexpr uint32_t RPC_TELEMETRY_MAX_PEER_PAIRS = 8;
static constexpr uint32_t RPC_TELEMETRY_MAX_SLOTS      = 256;

// D4.10: per-GPU metadata
struct rpc_telemetry_device_meta {
    char     name[64];
    uint64_t vram_mib;
    int32_t  backend_type; // ggml_backend_dev_type
    int32_t  pcie_gen;
    int32_t  pcie_width;
};

// D4.10: server telemetry frame appended to GRAPH_COMPUTE_ALL response
struct rpc_msg_server_telemetry {
    uint32_t n_devices;
    uint32_t n_peer_pairs;
    uint32_t n_slots;
    uint32_t reserved;
    uint64_t device_timings_us[RPC_TELEMETRY_MAX_DEVICES];
    int32_t  layer_assignments[RPC_TELEMETRY_MAX_DEVICES];
    uint64_t copy_times_us[RPC_TELEMETRY_MAX_PEER_PAIRS];
    rpc_telemetry_device_meta device_meta[RPC_TELEMETRY_MAX_DEVICES];
    uint64_t kv_read_times_us[RPC_TELEMETRY_MAX_SLOTS];
    uint64_t kv_write_times_us[RPC_TELEMETRY_MAX_SLOTS];
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
        rpc_msg_event_record_rsp rsp = {};
        size_t rsp_sz = ev->sock->server_supports_trace_id ? sizeof(rsp) : 12;
        if (!recv_rpc_cmd_deferred(ev->sock, &rsp, rsp_sz)) {
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
            rpc_msg_event_record_rsp rsp = {};
            size_t rsp_sz = sock->server_supports_trace_id ? sizeof(rsp) : 12;
            if (!recv_rpc_cmd_deferred(sock, &rsp, rsp_sz)) {
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
    if (tls_pending_event.pending && tls_pending_event.sock == sock) {
        drain_pending_event_response(sock);
    }
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
    // D4.5: multi-device dispatch state
    uint32_t    n_devices_on_endpoint = 0;
    bool        is_multi_device_capable = false;
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
    sock->server_supports_trace_id = (response.patch >= 3) || (response.conn_caps[0] & RPC_CAP_TRACE_ID);
    sock->server_supports_multi_device = (response.conn_caps[0] & RPC_CAP_MULTI_DEVICE) != 0;
    sock->server_supports_telemetry = (response.conn_caps[0] & RPC_CAP_SERVER_TELEMETRY) != 0;
    // D4.10 debug
    GGML_LOG_INFO("RPC %s: telemetry=%d conn_caps[0]=%d\n", endpoint,
                  sock->server_supports_telemetry ? 1 : 0,
                  response.conn_caps[0]);
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

// DO NOT USE: weak stub in libggml-base.so overrides this at dynamic link time.
// Use direct iface comparison in the same translation unit instead (see serialize_tensor).
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
    // Direct iface comparison bypasses weak symbol bug in ggml_backend_buffer_is_rpc()
    if (tensor->buffer && tensor->buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer) {
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
        // Always use known RPC buffer context (same reason as set_tensor)
        request.tensor.buffer = ctx->remote_ptr;
        request.tensor.data   = reinterpret_cast<uint64_t>(tensor->data);

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto sock = ctx->sock;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    // Always use known RPC buffer context - serialize_tensor may not detect
    // the buffer as RPC when tensor->buffer points to a multi-buffer wrapper.
    rpc_tensor.buffer = ctx->remote_ptr;
    rpc_tensor.data   = reinterpret_cast<uint64_t>(tensor->data);
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
    // Always use known RPC buffer context (same reason as set_tensor)
    request.tensor.buffer = ctx->remote_ptr;
    request.tensor.data   = reinterpret_cast<uint64_t>(tensor->data);
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
    // Always use known RPC buffer context (same reason as set_tensor)
    request.tensor.buffer = ctx->remote_ptr;
    request.tensor.data   = reinterpret_cast<uint64_t>(tensor->data);
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
        // Event was already sent deferred inside graph_compute.
        // If the response hasn't been drained yet, link the event struct
        // so event_wait/event_synchronize can drain it later.
        if (tls_pending_event.pending && tls_pending_event.sock == rpc_ctx->last_compute_sock) {
            tls_pending_event.ev = ev;
            ev->sock = rpc_ctx->last_compute_sock;
            ev->response_pending = true;
        }
        rpc_ctx->last_compute_sent_event = false;
    } else {
        auto sock = get_socket(rpc_ctx->endpoint);
        uint64_t tid = ggml_pipeline_trace_get_trace_id();
        rpc_msg_event_record_req ev_req = {ev->id, rpc_ctx->device, tid};
        size_t ev_sz = sock->server_supports_trace_id ? sizeof(ev_req) : 12;
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, ev_sz);
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

// D4.5: serialize graph for multi-device dispatch (GRAPH_COMPUTE_ALL)
// Format: | n_devices(4) | device_ids(n_devices*4) | n_nodes(4) | nodes(n_nodes*8) | n_tensors(4) | tensors(n_tensors*sizeof(rpc_tensor)) |
static void serialize_graph_for_all(
    const uint32_t * devices, uint32_t n_devices,
    ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    // Build tensor list first so we can allocate the exact size needed.
    // n_tensors may exceed n_nodes because add_tensor recursively visits
    // leaf weight tensors that are not node outputs.  Pre-allocating based
    // on n_nodes would cause a buffer overflow in that case.
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    tensors.reserve((size_t)n_nodes * 2);
    visited.reserve((size_t)n_nodes * 2);
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    uint32_t n_tensors = tensors.size();

    size_t total_size = sizeof(uint32_t) + n_devices * sizeof(uint32_t) +
        sizeof(uint32_t) + n_nodes * sizeof(uint64_t) +
        sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(total_size);

    uint8_t * dest = output.data();
    memcpy(dest, &n_devices, sizeof(n_devices));
    dest += sizeof(n_devices);
    memcpy(dest, devices, n_devices * sizeof(uint32_t));
    dest += n_devices * sizeof(uint32_t);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        uint64_t id = (uint64_t)(uintptr_t)cgraph->nodes[i];
        memcpy(dest, &id, sizeof(id));
        dest += sizeof(id);
    }
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    memcpy(dest, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

// D4.10: write a server_telemetry jsonl record (one JSON object per line)
static void rpc_write_server_telemetry_jsonl(const rpc_msg_server_telemetry & telem) {
    static std::mutex jsonl_mutex;
    std::lock_guard<std::mutex> lock(jsonl_mutex);

    const auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    uint64_t trace_id = ggml_pipeline_trace_get_trace_id();

    const char * tel_path = getenv("GGML_RPC_SERVER_TELEMETRY_FILE");
    if (!tel_path || !tel_path[0]) {
        tel_path = "server-telemetry.jsonl";
    }
    FILE * f = fopen(tel_path, "a");
    if (!f) {
        return;
    }
    fprintf(f, "{\"event\":\"server_telemetry\",\"ts_us\":%lld,\"trace_id\":%llu",
            (long long) ts_us, (unsigned long long) trace_id);

    fprintf(f, ",\"device_timings_us\":[");
    for (uint32_t i = 0; i < telem.n_devices; i++) {
        if (i) fprintf(f, ",");
        fprintf(f, "%llu", (unsigned long long) telem.device_timings_us[i]);
    }
    fprintf(f, "]");

    fprintf(f, ",\"layer_assignments\":[");
    for (uint32_t i = 0; i < telem.n_devices; i++) {
        if (i) fprintf(f, ",");
        fprintf(f, "%d", telem.layer_assignments[i]);
    }
    fprintf(f, "]");

    fprintf(f, ",\"copy_times_us\":[");
    for (uint32_t i = 0; i < telem.n_peer_pairs; i++) {
        if (i) fprintf(f, ",");
        fprintf(f, "%llu", (unsigned long long) telem.copy_times_us[i]);
    }
    fprintf(f, "]");

    fprintf(f, ",\"device_meta\":[");
    for (uint32_t i = 0; i < telem.n_devices; i++) {
        if (i) fprintf(f, ",");
        const rpc_telemetry_device_meta & meta = telem.device_meta[i];
        const char * backend_str = "unknown";
        switch (meta.backend_type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:  backend_str = "CPU"; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:  backend_str = "CUDA"; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: backend_str = "iGPU"; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: backend_str = "ACCEL"; break;
            default: break;
        }
        fprintf(f, "{\"name\":\"%.*s\",\"vram_mib\":%llu,\"backend\":\"%s\",\"pcie_gen\":%d,\"pcie_width\":%d}",
                (int) sizeof(meta.name), meta.name,
                (unsigned long long) meta.vram_mib, backend_str,
                meta.pcie_gen, meta.pcie_width);
    }
    fprintf(f, "]");

    fprintf(f, ",\"kv_read_times_us\":[");
    for (uint32_t i = 0; i < telem.n_slots; i++) {
        if (i) fprintf(f, ",");
        fprintf(f, "%llu", (unsigned long long) telem.kv_read_times_us[i]);
    }
    fprintf(f, "]");

    fprintf(f, ",\"kv_write_times_us\":[");
    for (uint32_t i = 0; i < telem.n_slots; i++) {
        if (i) fprintf(f, ",");
        fprintf(f, "%llu", (unsigned long long) telem.kv_write_times_us[i]);
    }
    fprintf(f, "]");

    fprintf(f, "}\n");
    fclose(f);
}

// issue 12: per-node timing for placement-grade heatmaps.
// Computes a graph one node at a time, recording per-node microseconds.
// Same ops + order as whole-graph compute; just slower (per-node sync).
struct rpc_node_timing {
    std::string name;
    uint64_t    us = 0;
};

static uint64_t compute_graph_per_node(
        ggml_backend_t backend, struct ggml_cgraph * graph,
        std::vector<rpc_node_timing> & out) {
    out.clear();
    if (!graph || graph->n_nodes == 0) return 0;
    out.reserve(graph->n_nodes);
    const auto t0 = std::chrono::steady_clock::now();
    for (int j = 0; j < graph->n_nodes; ++j) {
        struct ggml_cgraph gv = ggml_graph_view(graph, j, j + 1);
        const auto n0 = std::chrono::steady_clock::now();
        enum ggml_status ec = ggml_backend_graph_compute_async(backend, &gv);
        ggml_backend_synchronize(backend);
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - n0).count();
        if (ec != GGML_STATUS_SUCCESS) continue;
        rpc_node_timing nt;
        nt.name = graph->nodes[j]->name;
        nt.us = (uint64_t) us;
        out.push_back(nt);
    }
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

// issue 12: write a node_timings jsonl record to server-telemetry.jsonl.
static void rpc_write_node_timings_jsonl(const std::vector<rpc_node_timing> & nodes) {
    static std::mutex jsonl_mutex;
    std::lock_guard<std::mutex> lock(jsonl_mutex);

    const auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    uint64_t trace_id = ggml_pipeline_trace_get_trace_id();

    const char * tel_path = getenv("GGML_RPC_SERVER_TELEMETRY_FILE");
    if (!tel_path || !tel_path[0]) {
        tel_path = "server-telemetry.jsonl";
    }
    FILE * f = fopen(tel_path, "a");
    if (!f) return;

    fprintf(f, "{\"event\":\"node_timings\",\"ts_us\":%lld,\"trace_id\":%llu,\"entries\":[",
        (long long) ts_us, (unsigned long long) trace_id);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i) fprintf(f, ",");
        fprintf(f, "{\"name\":\"%s\",\"us\":%llu}",
            nodes[i].name.c_str(), (unsigned long long) nodes[i].us);
    }
    fprintf(f, "]}\n");
    fclose(f);
}

// D4.5: weighted weight placement for multi-device endpoints.
// Client assigns SET_TENSOR layers proportional to GPU speed ratio.
// For 3090 vs 3070 (1.75x ratio, 60 total layers):
//   fast_device gets 38 layers, slow_device gets 22 layers
//   send_set_tensor(dev[fast], layers[0..37], weights);
//   send_set_tensor(dev[slow], layers[38..59], weights);
// The server scheduler follows weight placement naturally via backend_from_buffer.
// Speed ratio constant: RPC_WEIGHT_SPEED_RATIO
// Split calculation: layers_fast = total_layers * speed_ratio / (speed_ratio + 1.0f)

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    auto sock = get_socket(rpc_ctx->endpoint);
    flush_pending_hash_all();
    flush_set_tensor_batch();

    // D6.9: per-stage dispatch via GRAPH_COMPUTE_STAGE
    // When the GPipe dispatch sets gpipe_active_stage, we send a stage-filtered
    // graph to the server so only this stage's backend computes its splits.
    // NOTE: we do NOT check server_supports_telemetry here. The stage-filtered
    // path avoids calling filter_null_src_nodes on the server, which would
    // incorrectly mark ALL nodes as GGML_OP_NONE when cross-backend tensors
    // (e.g. ROCm weights) have null data on the serialized graph. The server
    // handles telemetry being disabled gracefully (sends just the response
    // header without telemetry payload).
    int gpipe_stage = ggml_backend_sched_get_tls_gpipe_stage();
    // Fallback: if TLS is -1 but the scheduler has gpipe_active_stage set,
    // use the scheduler's value.  The scheduler field is set by set_gpipe_stage
    // (which sets both sched->gpipe_active_stage AND tls_gpipe_active_stage).
    // If only the scheduler field is set (via the deprecated set_gpipe_stage path),
    // detect it here.
    if (gpipe_stage < 0) {
        // Try to detect gpipe mode by checking if GGML_SCHED_GPIPE env var is set
        const char * gpipe_env = getenv("GGML_SCHED_GPIPE");
        if (gpipe_env && gpipe_env[0] == '1') {
            gpipe_stage = 0; // Assume stage 0 when in gpipe mode
            ggml_backend_sched_signal_gpipe_stage(0);
        }
    }
    if (gpipe_stage >= 0) {
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<uint8_t> input;
        // Use same serialization as GRAPH_COMPUTE_ALL (full graph)
        uint32_t n_devices = rpc_ctx->n_devices_on_endpoint > 0 ?
            rpc_ctx->n_devices_on_endpoint : 1;
        if (n_devices > GGML_RPC_MAX_DEVICES) {
            n_devices = GGML_RPC_MAX_DEVICES;
        }
        uint32_t devices[GGML_RPC_MAX_DEVICES];
        for (uint32_t i = 0; i < n_devices; i++) {
            devices[i] = i;
        }
        serialize_graph_for_all(devices, n_devices, cgraph, input);

        // Prefix stage_id before graph data (server extracts first 4 bytes)
        uint32_t stage_id = (uint32_t)gpipe_stage;
        std::vector<uint8_t> stage_input(sizeof(stage_id) + input.size());
        memcpy(stage_input.data(), &stage_id, sizeof(stage_id));
        memcpy(stage_input.data() + sizeof(stage_id), input.data(), input.size());

        rpc_msg_graph_compute_all_rsp rsp;
        size_t resp_size = sizeof(rsp) + sizeof(rpc_msg_server_telemetry);
        std::vector<uint8_t> resp_buf(resp_size);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE_STAGE,
                                   stage_input.data(), stage_input.size(),
                                   resp_buf.data(), resp_size);
        RPC_STATUS_ASSERT(status);
        memcpy(&rsp, resp_buf.data(), sizeof(rsp));
        if (rsp.result == 0) {
            rpc_msg_server_telemetry telem;
            memcpy(&telem, resp_buf.data() + sizeof(rsp), sizeof(telem));
            rpc_write_server_telemetry_jsonl(telem);
        }
        rpc_ctx->last_compute_sock = nullptr;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "graph_compute_stage",
                       RPC_CMD_GRAPH_COMPUTE_STAGE, input.size(), false, us);

        return GGML_STATUS_SUCCESS;
    }

    // D4.5: Path C - multi-device dispatch via GRAPH_COMPUTE_ALL
    // Fires when n_devices_on_endpoint > 1, i.e. a single RPC backend
    // has 2+ GPUs (e.g. rpc-server -d CUDA0,CUDA1). Telemetry collection
    // (D4.10) also works on the single-device GRAPH_COMPUTE path below --
    // the server collects telemetry for any GPU count. This ALL path adds
    // server-side multi-GPU scheduling on top of telemetry.
    if (rpc_multidevice_env_enabled() &&
        rpc_ctx->is_multi_device_capable &&
        rpc_ctx->n_devices_on_endpoint > 1) {
        const auto t0 = std::chrono::steady_clock::now();
        uint32_t n_devices = rpc_ctx->n_devices_on_endpoint;
        uint32_t devices[GGML_RPC_MAX_DEVICES];
        for (uint32_t i = 0; i < n_devices && i < GGML_RPC_MAX_DEVICES; i++) {
            devices[i] = i;
        }

        bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
        if (reuse) {
            // D4.5: fire-and-forget + EVENT_RECORD (matches existing GRAPH_COMPUTE pattern)
            rpc_msg_graph_recompute_all_req req = {};
            req.n_devices = n_devices;
            memcpy(req.devices, devices, n_devices * sizeof(uint32_t));
            req.graph_hash = cgraph->uid;
            req.sync_mode = 0; // fire-and-forget (use EVENT_RECORD for sync)
            req.output_requested = 0;
            bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE_ALL, &req, sizeof(req));
            RPC_STATUS_ASSERT(status);

            // Send EVENT_RECORD deferred: response drained later by event_wait
            // or at the next RPC operation on this socket. This avoids blocking
            // the scheduler's for-loop, allowing ROCm dispatch to overlap.
            uint64_t tid = ggml_pipeline_trace_get_trace_id();
            rpc_msg_event_record_req ev_req = {0, rpc_ctx->device, tid};
            size_t ev_sz = sock->server_supports_trace_id ? sizeof(ev_req) : 12;
            send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, ev_sz);
            rpc_ctx->last_compute_sent_event = true;
            rpc_ctx->last_compute_sock = sock;
            // Mark response pending now so drain functions (flush_pending_get_tensor, etc.)
            // see it before event_record links the rpc_event_t struct.
            tls_pending_event.sock = sock;
            tls_pending_event.pending = true;
            tls_pending_event.ev = nullptr;
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            rpc_trace_emit(__func__, "graph_recompute_all", RPC_CMD_GRAPH_RECOMPUTE_ALL, sizeof(req), false, us);
        } else {
            rpc_dev_ctx->last_graph_uid = cgraph->uid;
            std::vector<uint8_t> input;
            serialize_graph_for_all(devices, n_devices, cgraph, input);
            // D4.10: use response version; server sends telemetry when enabled
            rpc_msg_graph_compute_all_rsp rsp;
            size_t resp_size = sizeof(rsp);
            if (sock->server_supports_telemetry) {
                resp_size += sizeof(rpc_msg_server_telemetry);
            }
            std::vector<uint8_t> resp_buf(resp_size);
            bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE_ALL,
                                       input.data(), input.size(),
                                       resp_buf.data(), resp_size);
            RPC_STATUS_ASSERT(status);
            memcpy(&rsp, resp_buf.data(), sizeof(rsp));
            if (sock->server_supports_telemetry && rsp.result == 0) {
                rpc_msg_server_telemetry telem;
                memcpy(&telem, resp_buf.data() + sizeof(rsp), sizeof(telem));
                rpc_write_server_telemetry_jsonl(telem);
            }
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            rpc_trace_emit(__func__, "graph_compute_all", RPC_CMD_GRAPH_COMPUTE_ALL, input.size(), false, us);
        }
        return GGML_STATUS_SUCCESS;
    }

    const auto t0 = std::chrono::steady_clock::now();
    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);

        uint64_t tid = ggml_pipeline_trace_get_trace_id();
        rpc_msg_event_record_req ev_req = {0, rpc_ctx->device, tid};
        size_t ev_sz = sock->server_supports_trace_id ? sizeof(ev_req) : 12;
        // Defer the EVENT_RECORD response: the scheduler's event_record
        // already has a last_compute_sent_event fast-path, and the response
        // drains at event_wait/event_synchronize or at the next RPC op.
        // This makes graph_compute_async truly async (<50us TCP send only),
        // letting the scheduler dispatch ROCm splits while the RPC event
        // response is in-flight.
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, ev_sz);
        rpc_ctx->last_compute_sent_event = true;
        rpc_ctx->last_compute_sock = sock;
        // Mark response pending now so drain functions (flush_pending_get_tensor, etc.)
        // see it before event_record links the rpc_event_t struct.
        tls_pending_event.sock = sock;
        tls_pending_event.pending = true;
        tls_pending_event.ev = nullptr;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_emit(__func__, "graph_recompute", RPC_CMD_GRAPH_RECOMPUTE, sizeof(request), false, us);
    } else {
        rpc_dev_ctx->last_graph_uid = cgraph->uid;
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        // D4.10: use response version when server supports telemetry;
        // fall back to fire-and-forget for backward compatibility with old servers
        if (sock->server_supports_telemetry) {
            rpc_msg_graph_compute_rsp rsp = {};
            size_t resp_size = sizeof(rsp) + sizeof(rpc_msg_server_telemetry);
            std::vector<uint8_t> resp_buf(resp_size);
            bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE,
                                       input.data(), input.size(),
                                       resp_buf.data(), resp_size);
            RPC_STATUS_ASSERT(status);
            memcpy(&rsp, resp_buf.data(), sizeof(rsp));
            if (rsp.result == 0) {
                rpc_msg_server_telemetry telem;
                memcpy(&telem, resp_buf.data() + sizeof(rsp), sizeof(telem));
                rpc_write_server_telemetry_jsonl(telem);
            }
        } else {
            // Server always sends a 4-byte rpc_msg_graph_compute_rsp after GRAPH_COMPUTE.
            // Drain it to avoid corrupting subsequent deferred reads on the same socket
            // (e.g., EVENT_RECORD responses getting the GRAPH_COMPUTE response instead).
            rpc_msg_graph_compute_rsp rsp = {};
            bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE,
                                       input.data(), input.size(),
                                       &rsp, sizeof(rsp));
            RPC_STATUS_ASSERT(status);
        }
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

// D4.5: forward declaration
static uint32_t rpc_get_n_devices_on_endpoint(socket_ptr sock);

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint               = */ endpoint,
        /* .device                 = */ device,
        /* .name                   = */ dev_name,
        /* .last_compute_sock      = */ nullptr,
    };
    // D4.5: query endpoint device count for multi-device dispatch
    // NOTE: get_socket() calls negotiate_hello() synchronously before returning,
    // so server_supports_multi_device is guaranteed to be populated by this point.
    auto init_sock = get_socket(endpoint);
    ctx->n_devices_on_endpoint = rpc_get_n_devices_on_endpoint(init_sock);
    ctx->is_multi_device_capable = init_sock && init_sock->server_supports_multi_device;
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
        : backends(std::move(all_backends)), cache_dir(cache_dir),
          telemetry_enabled(rpc_server_telemetry_env_enabled()) {
        stored_graphs.resize(backends.size());
        if (telemetry_enabled) {
            // populate device_meta once at startup
            for (size_t i = 0; i < backends.size() && i < RPC_TELEMETRY_MAX_DEVICES; i++) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
                rpc_telemetry_device_meta & meta = startup_device_meta[i];
                memset(&meta, 0, sizeof(meta));
                if (!dev) {
                    continue;
                }
                struct ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                snprintf(meta.name, sizeof(meta.name), "%s", props.name);
                meta.vram_mib = props.memory_total / (1024 * 1024);
                meta.backend_type = (int32_t) ggml_backend_dev_type(dev);
                // pcie_gen/pcie_width not exposed via ggml api; leave 0
            }
        }
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
    bool graph_compute_all(const std::vector<uint8_t> & input);
    bool graph_recompute_all(const rpc_msg_graph_recompute_all_req & request);
    bool graph_compute_stage(const std::vector<uint8_t> & input, uint32_t stage_id);  // D6.9
    ggml_backend_sched_t create_multi_device_sched(
        const uint32_t * devices, uint32_t n_devices,
        const ggml_cgraph * graph);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    void enqueue_graph_compute(std::vector<uint8_t> input);
    void enqueue_graph_recompute(rpc_msg_graph_recompute_req request);
    void enqueue_graph_compute_all(std::vector<uint8_t> input);
    void enqueue_graph_recompute_all(rpc_msg_graph_recompute_all_req request);
    void enqueue_graph_compute_stage(std::vector<uint8_t> input, uint32_t stage_id);  // D6.9
    void wait_compute_idle();
    void collect_telemetry(const uint32_t * devices, uint32_t n_devices,
                           const int64_t * per_device_us);
    bool get_last_telemetry(rpc_msg_server_telemetry & out) const;

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
    // D4.5: dedicated storage for ALL-mode graph (separate from per-device stored_graphs)
    stored_graph all_graph;

    // D4.5: cached multi-device schedulers keyed by device set hash
    std::unordered_map<uint64_t, ggml_backend_sched_t> all_scheds;

    std::mutex                    compute_mtx;
    std::condition_variable         compute_cv;
    std::deque<std::function<void()>> compute_queue;
    std::thread                     compute_worker;
    std::atomic<bool>               compute_shutdown{false};
    std::atomic<int>                compute_inflight{0};

    // D4.10: telemetry state
    const bool                telemetry_enabled;
    rpc_telemetry_device_meta startup_device_meta[RPC_TELEMETRY_MAX_DEVICES];
    rpc_msg_server_telemetry  last_telemetry;
    mutable std::mutex        telemetry_mtx;
    std::atomic<uint64_t>     telemetry_decode_count{0};
    // issue 12: sampled per-node timing (node_timings events). Bounded overhead.
    std::atomic<uint64_t>     telemetry_node_sample_count{0};
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
        static std::atomic<int> warn_count{0};
        if (warn_count.fetch_add(1, std::memory_order_relaxed) < 5) {
            GGML_LOG_WARN("[%s] buffer %p not found in server buffer set (%zu buffers registered)\n",
                __func__, (void*)result->buffer, buffers.size());
        }
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        // NOTE: tensor->data is a client-side address that may not be valid in the server's address space
        // (e.g. ROCm client -> CUDA server). The actual offset is sent separately and validated
        // in set_tensor(). Skip cross-address-space bounds comparison here.
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

    // sanitize offset + size against buffer size
    // NOTE: in_tensor->data is a client-side address - do not use it for server-side bounds checking
    // (e.g. ROCm client -> CUDA server). Validate using offset and size only.
    {
        const size_t buf_size = ggml_backend_buffer_get_size(tensor->buffer);

        if (offset + size > buf_size) {
            GGML_LOG_ERROR("[%s] tensor data region (offset=%" PRIu64 ", size=%zu) exceeds buffer size %zu\n",
                           __func__, offset, size, buf_size);
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
        GGML_LOG_ERROR("[%s] invalid data ptr: id=%" PRIu64 " data=%p type=%d op=%d ne=[%lld %lld %lld %lld]\n",
                       __func__, id, result->data, tensor->type, tensor->op,
                       (long long)tensor->ne[0], (long long)tensor->ne[1],
                       (long long)tensor->ne[2], (long long)tensor->ne[3]);
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

// issue 12: sample every Nth decode for per-node timing (more expensive than
// device-level telemetry). Bounds overhead of per-node sync on the RPC server.
static constexpr int RPC_NODE_SAMPLE_INTERVAL = 16;

// Safety net: nodes whose LEAF src tensors have null data pointers cannot be computed.
// A leaf tensor is one that appears as an input (src) of some node but is NOT
// itself a node output. Leaves with null data are cross-backend tensors (ROCm/CPU)
// whose weights were never uploaded to this RPC server -- nodes depending on them
// must be skipped to avoid CUDA kernel crashes.
//
// Non-leaf src tensors (intermediate node outputs) naturally have null data before
// the CUDA backend allocates them during graph_compute -- those are NOT skipped,
// unless their producer was already filtered (propagation step).
//
// Graph nodes are assumed to be in topological order, so a single forward pass
// suffices: when we encounter a node whose intermediate src has null data, we
// check if its producer was filtered (op == GGML_OP_NONE).
static void filter_null_src_nodes(struct ggml_cgraph * graph) {
    if (!graph || graph->n_nodes == 0) {
        return;
    }
    // Build set of node output tensors (intermediates)
    std::unordered_set<struct ggml_tensor *> node_outputs;
    node_outputs.reserve((size_t) graph->n_nodes);
    for (int i = 0; i < (int)graph->n_nodes; i++) {
        if (graph->nodes[i]) {
            node_outputs.insert(graph->nodes[i]);
        }
    }
    int n_skipped = 0;
    for (int i = 0; i < (int)graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (!node) { continue; }
        bool should_filter = false;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] && node->src[j]->data == nullptr) {
                // Leaf (cross-backend weight, not produced by any graph node)
                if (node_outputs.find(node->src[j]) == node_outputs.end()) {
                    should_filter = true;
                    break;
                }
                // Intermediate produced by a node that was already filtered
                if (node->src[j]->op == GGML_OP_NONE) {
                    should_filter = true;
                    break;
                }
            }
        }
        if (should_filter) {
            node->op = GGML_OP_NONE;
            n_skipped++;
        }
    }
    if (n_skipped > 0) {
        GGML_LOG_DEBUG("[rpc-server] filtered %d nodes with null leaf src data\n", n_skipped);
    }
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    GGML_LOG_DEBUG("[rpc-server] graph_compute (single-device) called, input.size=%zu\n", input.size());
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
    // Filter out nodes with null src data (non-RPC buffers on client)
    filter_null_src_nodes(graph);

    // issue 12: on sampled decodes, compute per-node for placement-grade heatmaps.
    // Sampled to bound overhead; produces correct output (same ops + order).
    uint64_t us = 0;
    const uint64_t sample_id = telemetry_node_sample_count.fetch_add(1, std::memory_order_relaxed);
    const bool sample_nodes = telemetry_enabled && (sample_id % RPC_NODE_SAMPLE_INTERVAL) == 0;
    std::vector<rpc_node_timing> node_timings;

    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status;
    if (sample_nodes) {
        us = compute_graph_per_node(backends[device], graph, node_timings);
        status = GGML_STATUS_SUCCESS;
    } else {
        status = ggml_backend_graph_compute(backends[device], graph);
        us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    rpc_trace_emit("rpc_server::graph_compute", "server_compute", RPC_CMD_GRAPH_COMPUTE, input.size(), true, us);
    stored_graphs[device].graph = graph;
    // issue 12: emit per-node timings for this sampled decode.
    if (sample_nodes && !node_timings.empty()) {
        rpc_write_node_timings_jsonl(node_timings);
    }
    // D4.10: collect telemetry for single-device compute when enabled
    if (telemetry_enabled) {
        uint32_t dev = device;
        int64_t per_device_us = (int64_t) us;
        collect_telemetry(&dev, 1, &per_device_us);
    }
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
    // D7.8: collect telemetry for graph reuse (token generation) and write directly
    // (recompute is fire-and-forget, so telemetry can't piggyback on the response)
    if (telemetry_enabled) {
        uint32_t dev = device;
        int64_t per_device_us = (int64_t) us;
        collect_telemetry(&dev, 1, &per_device_us);
        {
            std::lock_guard<std::mutex> lock(telemetry_mtx);
            rpc_write_server_telemetry_jsonl(last_telemetry);
        }
    }
    return true;
}

static uint64_t hash_device_set(const uint32_t * devices, uint32_t n_devices) {
    uint64_t h = 0;
    for (uint32_t i = 0; i < n_devices; i++) {
        h ^= ((uint64_t)devices[i] << (i * 8));
    }
    return h;
}

ggml_backend_sched_t rpc_server::create_multi_device_sched(
    const uint32_t * devices, uint32_t n_devices,
    const ggml_cgraph * graph) {
    // D4.5: check cache first
    uint64_t key = hash_device_set(devices, n_devices);
    auto it = all_scheds.find(key);
    if (it != all_scheds.end()) {
        return it->second;
    }

    // Collect backends for each device index
    std::vector<ggml_backend_t> sched_backends;
    sched_backends.reserve(n_devices + 1);
    for (uint32_t i = 0; i < n_devices; i++) {
        if (devices[i] < backends.size()) {
            sched_backends.push_back(backends[devices[i]]);
        }
    }
    // Append CPU backend as fallback
    sched_backends.push_back(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));

    // Create scheduler with standard params
    // Default graph_size = 4096 if graph is null
    size_t graph_size = graph ? ggml_graph_overhead_custom(graph->n_nodes, false) : 4096;
    // No pipeline parallelism server-side (client handles it)
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        sched_backends.data(),
        nullptr, // default buft
        sched_backends.size(),
        graph_size,
        false, // parallel
        false  // op_offload
    );
    // D4.5: cache the scheduler
    all_scheds[key] = sched;
    return sched;
}

bool rpc_server::graph_compute_all(const std::vector<uint8_t> & input) {
    GGML_LOG_DEBUG("[rpc-server] graph_compute_all (multi-device) called, input.size=%zu\n", input.size());
    // Format: | n_devices(4) | device_ids(n_devices*4) | n_nodes(4) | nodes(n_nodes*8) | n_tensors(4) | tensors(n_devices*rpc_tensor) |
    if (input.size() < sizeof(uint32_t) * 3) {
        return false;
    }

    const uint8_t * src = input.data();
    uint32_t n_devices;
    memcpy(&n_devices, src, sizeof(n_devices));
    src += sizeof(n_devices);

    if (n_devices == 0 || n_devices > 8) {
        return false;
    }

    uint32_t devices[8];
    size_t devs_size = n_devices * sizeof(uint32_t);
    if (input.size() < sizeof(uint32_t) + devs_size + sizeof(uint32_t)) {
        return false;
    }
    memcpy(devices, src, devs_size);
    src += devs_size;

    // Parse graph data (reuse existing format minus the leading device field)
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < sizeof(uint32_t) + devs_size + sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes * sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < sizeof(uint32_t) + devs_size + sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;

    // Allocate context and deserialize
    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    std::vector<uint8_t> ctx_buf(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr{ ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;

    std::unordered_map<uint64_t, const rpc_tensor *> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        if (graph->nodes[i] == nullptr && id != 0) {
            return false;
        }
    }

    // Filter out nodes with null src data (non-RPC buffers on client)
    filter_null_src_nodes(graph);

    // Create multi-device scheduler
    ggml_backend_sched_t sched = create_multi_device_sched(devices, n_devices, graph);
    if (!sched) {
        return false;
    }

    // Compute
    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    rpc_trace_emit("rpc_server::graph_compute_all", "server_compute",
                   RPC_CMD_GRAPH_COMPUTE_ALL, input.size(), true, us);

    // D4.10: collect telemetry with per-device timing from scheduler when enabled
    if (telemetry_enabled) {
        int64_t per_device_us[RPC_TELEMETRY_MAX_DEVICES];
        for (uint32_t i = 0; i < n_devices && i < RPC_TELEMETRY_MAX_DEVICES; i++) {
            per_device_us[i] = ggml_backend_sched_get_backend_timing_us(sched, (int)i);
        }
        collect_telemetry(devices, n_devices, per_device_us);
    }

    // D4.5: store for recompute in dedicated ALL-mode storage
    if (all_graph.buffer.size() < buf_size) {
        all_graph.buffer.resize(buf_size);
    }
    std::copy(ctx_buf.begin(), ctx_buf.end(), all_graph.buffer.begin());
    all_graph.graph = graph;

    // NOTE: sched is cached in all_scheds, freed in ~rpc_server
    return true;
}

bool rpc_server::graph_recompute_all(const rpc_msg_graph_recompute_all_req & request) {
    if (all_graph.graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = all_graph.graph;

    // D4.5: reuse cached multi-device scheduler
    ggml_backend_sched_t sched = create_multi_device_sched(request.devices, request.n_devices, graph);
    if (!sched) {
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    rpc_trace_emit("rpc_server::graph_recompute_all", "server_compute",
                   RPC_CMD_GRAPH_RECOMPUTE_ALL, 0, true, us);

    // D7.8: collect telemetry for multi-device graph reuse and write directly
    if (telemetry_enabled) {
        int64_t per_device_us[RPC_TELEMETRY_MAX_DEVICES] = {0};
        per_device_us[0] = (int64_t) us;
        collect_telemetry(request.devices, request.n_devices, per_device_us);
        {
            std::lock_guard<std::mutex> lock(telemetry_mtx);
            rpc_write_server_telemetry_jsonl(last_telemetry);
        }
    }

    // NOTE: sched is cached in all_scheds, freed in ~rpc_server
    return true;
}

// D6.9: per-stage graph compute with split filtering.
// Identical to graph_compute_all but sets gpipe_active_stage on the scheduler
// so only splits matching the given stage's backend_id are computed.
bool rpc_server::graph_compute_stage(const std::vector<uint8_t> & input, uint32_t stage_id) {
    GGML_LOG_DEBUG("[rpc-server] graph_compute_stage called, stage_id=%u, input.size=%zu\n", stage_id, input.size());
    if (input.size() < sizeof(uint32_t) * 3) {
        return false;
    }

    const uint8_t * src = input.data();

    // Parse graph data: | n_devices(4) | devices(n_devices*4) | n_nodes(4) | nodes(n_nodes*8) | n_tensors(4) | tensors(n_tensors*sizeof(rpc_tensor)) |
    uint32_t n_devices;
    memcpy(&n_devices, src, sizeof(n_devices));
    src += sizeof(n_devices);

    if (n_devices == 0 || n_devices > 8) {
        return false;
    }

    uint32_t devices[8];
    size_t devs_size = n_devices * sizeof(uint32_t);
    if (input.size() < (size_t)(src - input.data()) + devs_size + sizeof(uint32_t)) {
        return false;
    }
    memcpy(devices, src, devs_size);
    src += devs_size;

    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < (size_t)(src - input.data()) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes * sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < (size_t)(src - input.data()) + n_tensors * sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;

    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    std::vector<uint8_t> ctx_buf(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr{ ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;

    std::unordered_map<uint64_t, const rpc_tensor *> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        if (graph->nodes[i] == nullptr && id != 0) {
            return false;
        }
    }

    // Filter out nodes with null src data (non-RPC buffers on client)
    filter_null_src_nodes(graph);

    ggml_backend_sched_t sched = create_multi_device_sched(devices, n_devices, graph);
    if (!sched) {
        return false;
    }

    // D6.9: filter splits to only this pipeline stage's backend
    ggml_backend_sched_set_gpipe_stage(sched, (int)stage_id);

    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    rpc_trace_emit("rpc_server::graph_compute_stage", "server_compute_stage",
                   RPC_CMD_GRAPH_COMPUTE_STAGE, input.size(), true, us);

    // Collect telemetry for this stage
    if (telemetry_enabled) {
        int64_t per_device_us[RPC_TELEMETRY_MAX_DEVICES];
        for (uint32_t i = 0; i < n_devices && i < RPC_TELEMETRY_MAX_DEVICES; i++) {
            per_device_us[i] = ggml_backend_sched_get_backend_timing_us(sched, (int)i);
        }
        collect_telemetry(devices, n_devices, per_device_us);
    }

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

void rpc_server::enqueue_graph_compute_all(std::vector<uint8_t> input) {
    submit_compute_job([this, input = std::move(input)]() {
        if (!graph_compute_all(input)) {
            GGML_LOG_ERROR("[%s] async graph_compute_all failed\n", __func__);
        }
    });
}

void rpc_server::enqueue_graph_recompute_all(rpc_msg_graph_recompute_all_req request) {
    submit_compute_job([this, request]() {
        if (!graph_recompute_all(request)) {
            GGML_LOG_ERROR("[%s] async graph_recompute_all failed\n", __func__);
        }
    });
}

// D6.9: enqueue per-stage compute with split filtering
void rpc_server::enqueue_graph_compute_stage(std::vector<uint8_t> input, uint32_t stage_id) {
    submit_compute_job([this, input = std::move(input), stage_id]() {
        if (!graph_compute_stage(input, stage_id)) {
            GGML_LOG_ERROR("[%s] async graph_compute_stage(%u) failed\n", __func__, stage_id);
        }
    });
}

void rpc_server::wait_compute_idle() {
    std::unique_lock<std::mutex> lock(compute_mtx);
    compute_cv.wait(lock, [this] {
        return compute_queue.empty() && compute_inflight.load() == 0;
    });
}

// D4.10: sample every Nth decode to keep overhead <1%
static constexpr int TELEMETRY_SAMPLE_INTERVAL = 1;

void rpc_server::collect_telemetry(const uint32_t * devices, uint32_t n_devices,
                                   const int64_t * per_device_us) {
    uint64_t decode_id = telemetry_decode_count.fetch_add(1, std::memory_order_relaxed);
    if (TELEMETRY_SAMPLE_INTERVAL > 1 && (decode_id % TELEMETRY_SAMPLE_INTERVAL) != 0) {
        return;
    }

    rpc_msg_server_telemetry t = {};
    t.n_devices = std::min<uint32_t>(n_devices, RPC_TELEMETRY_MAX_DEVICES);

    // device_timings: per-device compute time from scheduler or caller
    for (uint32_t i = 0; i < t.n_devices; i++) {
        t.device_timings_us[i] = per_device_us ? (uint64_t)per_device_us[i] : 0;
    }

    // layer_assignments: backend index per device (split boundary info not exposed)
    for (uint32_t i = 0; i < t.n_devices; i++) {
        t.layer_assignments[i] = (int32_t) i;
    }

    // copy_times: not tracked per peer pair yet; leave zero
    t.n_peer_pairs = 0;

    // device_meta: from startup snapshot
    for (uint32_t i = 0; i < t.n_devices; i++) {
        t.device_meta[i] = startup_device_meta[devices[i]];
    }

    // kv times: server has no direct kv cache context; leave zero as placeholder
    t.n_slots = 0;

    {
        std::lock_guard<std::mutex> lock(telemetry_mtx);
        last_telemetry = t;
    }
}

bool rpc_server::get_last_telemetry(rpc_msg_server_telemetry & out) const {
    std::lock_guard<std::mutex> lock(telemetry_mtx);
    if (!telemetry_enabled) {
        return false;
    }
    out = last_telemetry;
    return true;
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
    // D4.5: free cached multi-device schedulers
    for (auto & kv : all_scheds) {
        ggml_backend_sched_free(kv.second);
    }
    all_scheds.clear();
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
        // D4.5: only advertise multi-device capability if env var is enabled
        if (rpc_multidevice_env_enabled()) {
            rsp3.conn_caps[0] |= RPC_CAP_MULTI_DEVICE;
        }
        // D4.10: advertise telemetry capability when enabled (--telemetry flag / env var)
        if (rpc_server_telemetry_env_enabled()) {
            rsp3.conn_caps[0] |= RPC_CAP_SERVER_TELEMETRY;
        }
        sock->server_supports_trace_id = (rsp3.patch >= 3) || (req_conn_caps[0] & RPC_CAP_TRACE_ID);
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
        // D4.5: only advertise multi-device capability if env var is enabled
        if (rpc_multidevice_env_enabled()) {
            rsp.conn_caps[0] |= RPC_CAP_MULTI_DEVICE;
        }
        // D4.10: advertise telemetry capability
        if (rpc_server_telemetry_env_enabled()) {
            rsp.conn_caps[0] |= RPC_CAP_SERVER_TELEMETRY;
        }
        // trace_id support from client caps (for deciding 20B vs 12B recv on EVENT_RECORD)
        sock->server_supports_trace_id = (rsp.patch >= 3) || (req.conn_caps[0] & RPC_CAP_TRACE_ID);
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
                server.wait_compute_idle();
                // D4.10: send response (with telemetry appended if enabled)
                rpc_msg_graph_compute_rsp rsp = {};
                rsp.result = 0;
                rpc_msg_server_telemetry telem = {};
                if (server.get_last_telemetry(telem)) {
                    size_t resp_size = sizeof(rsp) + sizeof(telem);
                    std::vector<uint8_t> resp_buf(resp_size);
                    memcpy(resp_buf.data(), &rsp, sizeof(rsp));
                    memcpy(resp_buf.data() + sizeof(rsp), &telem, sizeof(telem));
                    send_response(sock, resp_buf.data(), resp_buf.size());
                } else {
                    send_response(sock, &rsp, sizeof(rsp));
                }
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
                rpc_msg_event_record_req request = {};
                size_t req_sz = sock->server_supports_trace_id ? sizeof(request) : 12;
                if (!recv_msg(sock, &request, req_sz)) {
                    return;
                }
                server.wait_compute_idle();
                rpc_msg_event_record_rsp response = {request.event_id, 0, request.trace_id};
                size_t rsp_sz = sock->server_supports_trace_id ? sizeof(response) : 12;
                if (!send_response(sock, &response, rsp_sz)) {
                    return;
                }
                LOG_DBG("[%s] RPC_CMD_EVENT_RECORD: event_id=%lu, device=%u, trace_id=%llu\n",
                        __func__, (unsigned long)request.event_id, request.device, (unsigned long long)request.trace_id);
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
            case RPC_CMD_GRAPH_COMPUTE_ALL: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                server.enqueue_graph_compute_all(std::move(input));
                server.wait_compute_idle();
                // D4.10: send response (with telemetry appended if enabled)
                rpc_msg_graph_compute_all_rsp rsp = {};
                rsp.result = 0;
                rpc_msg_server_telemetry telem = {};
                if (server.get_last_telemetry(telem)) {
                    size_t resp_size = sizeof(rsp) + sizeof(telem);
                    std::vector<uint8_t> resp_buf(resp_size);
                    memcpy(resp_buf.data(), &rsp, sizeof(rsp));
                    memcpy(resp_buf.data() + sizeof(rsp), &telem, sizeof(telem));
                    send_response(sock, resp_buf.data(), resp_buf.size());
                } else {
                    send_response(sock, &rsp, sizeof(rsp));
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE_ALL: {
                rpc_msg_graph_recompute_all_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                server.enqueue_graph_recompute_all(request);
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE_STAGE: {
                // D6.9: per-stage dispatch with split filtering
                // Format: | stage_id(4) | n_devices(4) | device_ids(...) | graph data |
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (input.size() < sizeof(uint32_t) * 2) {
                    return;
                }
                uint32_t stage_id;
                memcpy(&stage_id, input.data(), sizeof(stage_id));
                // Shift input to strip the stage_id prefix for graph_compute_stage
                std::vector<uint8_t> graph_input(input.begin() + sizeof(uint32_t), input.end());
                server.enqueue_graph_compute_stage(std::move(graph_input), stage_id);
                server.wait_compute_idle();
                rpc_msg_graph_compute_all_rsp rsp = {};
                rsp.result = 0;
                rpc_msg_server_telemetry telem = {};
                if (server.get_last_telemetry(telem)) {
                    size_t resp_size = sizeof(rsp) + sizeof(telem);
                    std::vector<uint8_t> resp_buf(resp_size);
                    memcpy(resp_buf.data(), &rsp, sizeof(rsp));
                    memcpy(resp_buf.data() + sizeof(rsp), &telem, sizeof(telem));
                    send_response(sock, resp_buf.data(), resp_buf.size());
                } else {
                    send_response(sock, &rsp, sizeof(rsp));
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
    if (std::strcmp(name, "ggml_backend_rpc_get_device_memory") == 0) {
        return (void *)ggml_backend_rpc_get_device_memory;
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

// D4.4: query endpoint device count for multi-device dispatch
static uint32_t rpc_get_n_devices_on_endpoint(socket_ptr sock) {
    if (!sock) {
        return 1;
    }
    rpc_msg_device_count_rsp rsp = {};
    bool ok = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &rsp, sizeof(rsp));
    return ok ? rsp.device_count : 1;
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
