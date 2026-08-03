#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "rpc-async-audit.h"

// For server-side CUDA split buffer type allocation (V0 row-split fix).
// ggml_backend_buft_is_cuda_split() is static in ggml-cuda.cu, so we
// detect CUDA split buffers by name suffix "_Split" (see rpc_buft_is_cuda_split).
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#  include "ggml-cuda.h"
#endif

#include <array>
#include <cinttypes>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <map>
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

// UDP listener for fire-and-forget graph submission (opt-in via GGML_RPC_UDP=1).
#ifndef _WIN32
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <unistd.h>
#endif

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

void rpc_trace_emit(const char * fn, const char * phase, int cmd, size_t bytes, bool blocking, int64_t elapsed_us) {
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

// Detect whether a buffer type is a native CUDA/HIP split buffer type.
// ggml_backend_buft_is_cuda_split() is static in ggml-cuda.cu and thus not
// linkable here. The CUDA split buffer type names are of the form
// "<BACKEND_NAME><dev>_Split" (e.g. "ROCm0_Split", "CUDA0_Split"), so we
// detect by suffix. This is only available when CUDA/HIP is compiled in.
static bool rpc_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    if (!buft || !buft->iface.get_name) {
        return false;
    }
    const char * name = buft->iface.get_name(buft);
    if (!name) {
        return false;
    }
    // Match the "_Split" suffix used by ggml_backend_cuda_split_buffer_type
    const char * suffix = "_Split";
    const size_t suffix_len = 6;
    size_t len = strlen(name);
    if (len < suffix_len) {
        return false;
    }
    return strcmp(name + len - suffix_len, suffix) == 0;
#else
    GGML_UNUSED(buft);
    return false;
#endif
}

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
    RPC_CMD_ALLOC_BUFFER_SPLIT,   // Allocate a buffer for a tensor row slice on a specific device (value 24)
    RPC_CMD_GET_TENSOR_BATCH,    // V1b: batch GET_TENSOR requests (value 25)
    RPC_CMD_COUNT,               // updated from 24
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
static void flush_get_tensor_batch();     // V1b: batch GET_TENSOR flush
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

// V1b: GET_TENSOR_BATCH opt-in. Default OFF (correctness-first — individual
// GET_TENSOR is the well-tested path). Set GGML_RPC_GET_TENSOR_BATCH=1 to
// enable. Even when enabled, batching only fires if the server advertises
// RPC_CAP_GET_TENSOR_BATCH (see server_supports_get_tensor_batch).
static int rpc_get_tensor_batch_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_GET_TENSOR_BATCH");
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

// UDP transport for fire-and-forget graph submission. On by default since
// Wayfinder Loop 5 (2026-07-24): +56% throughput vs TCP-only by eliminating
// TCP stream contention between GET_TENSOR and GRAPH_RECOMPUTE.
// Opt-out via GGML_RPC_UDP=0.
// EVENT_RECORD stays on TCP (needs reliable ordering for synchronization).
// Falls back to TCP if UDP is not initialized or send fails.
static bool rpc_udp_env_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_UDP");
        v = e ? atoi(e) : 0; // default OFF (reverted by BUG-011: UDP recompute broken by design)
    }
    return v != 0;
}

// rpc_udp_header and RPC_UDP_MAGIC / RPC_UDP_FLAG_ACK now live in transport.h
// (shared between client send/ACK-wait and the server listener).

// T2e: ACK timeout (ms). After sending a DATA frame we block up to this long
// waiting for the server's ACK (seq echo). On timeout the caller falls back to
// TCP GRAPH_RECOMPUTE. Tunable via GGML_RPC_UDP_ACK_TIMEOUT_MS. For a LAN the
// ACK RTT is <1 ms; the timeout is a safety net for genuine loss.
static int rpc_udp_ack_timeout_ms() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_UDP_ACK_TIMEOUT_MS");
        v = e ? atoi(e) : 100; // default 100 ms
    }
    return v;
}

// T2e: client-side fault injection (DUP only — DROP/REORDER live on the server
// side; see rpc_udp_listener). dup_pct (0-100) is the probability that the
// frame is sent twice. A duplicate DATA frame is idempotent on the server
// (recompute of the same graph_uid) and the client drains the extra ACK.
// Gated via GGML_RPC_UDP_FAULT_DUP_PCT so production runs are unaffected.
static int rpc_udp_fault_dup_pct() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_UDP_FAULT_DUP_PCT");
        v = e ? atoi(e) : 0; // default: no fault
    }
    return v;
}

// Send a graph recompute command over UDP with T2e reliability. Returns true
// only if the frame was sent AND the server's ACK arrived within the timeout
// (i.e. the frame was reliably delivered over UDP). Returns false if UDP is
// disabled, init/send failed, or the ACK timed out — the caller then falls
// back to the TCP GRAPH_RECOMPUTE path.
//
// udp_port is the remote UDP port (tcp_port + 1 by convention). The UDP socket
// is created lazily on first call per socket.
static bool rpc_udp_send_graph(const socket_ptr & sock,
                               const rpc_udp_header & hdr,
                               const uint32_t * devices, uint32_t n_devices,
                               int udp_port) {
    if (!rpc_udp_env_enabled()) {
        return false;
    }
    // Lazy-init the UDP socket on first use.
    if (!sock->udp_enabled()) {
        if (udp_port <= 0 || udp_port > 65535) {
            GGML_LOG_ERROR("[%s] invalid udp_port %d\n", __func__, udp_port);
            return false;
        }
        sock->init_udp(udp_port);
        if (!sock->udp_enabled()) {
            GGML_LOG_WARN("[%s] UDP init failed on port %d, will use TCP\n", __func__, udp_port);
            return false;
        }
        LOG_DBG("[%s] UDP enabled on port %d for graph submission\n", __func__, udp_port);
    }

    // Pack the datagram: header + optional device list.
    const size_t hdr_sz = sizeof(rpc_udp_header);
    const size_t dev_sz = static_cast<size_t>(n_devices) * sizeof(uint32_t);
    std::vector<uint8_t> pkt(hdr_sz + dev_sz);
    rpc_udp_header net_hdr = hdr;
    net_hdr.magic = RPC_UDP_MAGIC;
    net_hdr.flags = 0; // DATA frame (no ACK flag)
    const uint32_t seq = sock->udp_next_seq();
    net_hdr.seq = seq;
    memcpy(pkt.data(), &net_hdr, hdr_sz);
    if (dev_sz > 0 && devices) {
        memcpy(pkt.data() + hdr_sz, devices, dev_sz);
    }
    bool sent = sock->send_udp(pkt.data(), pkt.size());
    if (!sent) {
        return false;
    }

    // T2e fault injection: DUP — send the frame a second time. The server
    // enqueues both (idempotent recompute) and ACKs both; the client drains
    // the extra ACK as a stale frame on the next wait.
    if (rpc_udp_fault_dup_pct() > 0) {
        static thread_local uint32_t rng = 0x12345678u;
        // xorshift32 — deterministic, thread-local, no libc dependency.
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        int roll = static_cast<int>(rng % 100u);
        if (roll < rpc_udp_fault_dup_pct()) {
            GGML_LOG_DEBUG("[%s] FAULT dup: sending seq %u again\n", __func__, seq);
            sock->send_udp(pkt.data(), pkt.size());
        }
    }

    // T2e reliability: block for the server's ACK (seq echo). A lost frame
    // (or a server-side drop) means no ACK → timeout → caller falls back to
    // TCP GRAPH_RECOMPUTE. This makes the UDP path correct under loss.
    int timeout_ms = rpc_udp_ack_timeout_ms();
    bool acked = sock->recv_udp_ack(seq, timeout_ms);
    if (!acked) {
        GGML_LOG_WARN("[%s] UDP ACK timeout seq=%u (%u ms) — will fall back to TCP\n",
                      __func__, seq, timeout_ms);
    }
    return acked;
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

// Map from global device ID to endpoint for split buffer type lookups
static std::mutex g_rpc_dev_endpoint_mutex;
static std::unordered_map<uint32_t, std::string> g_rpc_dev_endpoint_map;

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

// B+15: per-socket FIFO queue for deferred EVENT_RECORD responses.
// Replaces thread-local tls_pending_event which races across sockets:
// the old single slot could only track one pending response per thread,
// so a second backend's graph_compute would clobber the first's slot,
// causing responses to interleave on the TCP response channel.
struct rpc_deferred_entry {
    enum rpc_cmd          cmd_type;
    std::vector<uint8_t>  rsp; // owned response buffer (read at drain time)
    struct rpc_event_t *  event = nullptr; // linked by event_record
};

struct rpc_socket_state {
    std::deque<rpc_deferred_entry> queue;
};

// Keyed by socket. get_socket() returns a cached stable socket_ptr per
// endpoint, so the key is stable for the lifetime of the connection.
static std::unordered_map<socket_ptr, rpc_socket_state> g_rpc_sockets;
static std::mutex g_rpc_sockets_mutex;

static void rpc_socket_queue_push(const socket_ptr & sock, rpc_deferred_entry entry);
static void rpc_socket_drain(const socket_ptr & sock);

static void rpc_drain_all_endpoints_pending() {
    if (tls_pending_copy.pending && tls_pending_copy.sock) {
        drain_pending_copy_response(tls_pending_copy.sock);
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

    std::vector<socket_ptr> live;

    // B+15: drain queued EVENT_RECORD responses for every socket that has
    // them. Done before multi_socket_flush so the response channel is clean.
    // Collect sockets under the lock, then drain outside it (rpc_socket_drain
    // takes the same lock internally).
    {
        std::lock_guard<std::mutex> lock(g_rpc_sockets_mutex);
        for (auto & [sock, state] : g_rpc_sockets) {
            live.push_back(sock);
        }
    }
    for (const auto & sock : live) {
        rpc_socket_drain(sock);
    }

    if (!rpc_multi_socket_flush()) {
        return;
    }

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
        if (tls_pending_copy.pending && tls_pending_copy.sock == sock) {
            drain_pending_copy_response(sock);
        }
        // B+15: drain any remaining queued EVENT_RECORD responses.
        rpc_socket_drain(sock);
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

// V1b: batch GET_TENSOR accumulator (Wayfinder Loop 5)
struct get_tensor_batch_entry {
    rpc_msg_get_tensor_req req;
    void * data;
    size_t size;
};
static thread_local std::vector<get_tensor_batch_entry> tls_get_batch;
static thread_local socket_ptr tls_get_batch_sock;

static void get_tensor_batch_append(const socket_ptr & sock,
                                     const rpc_msg_get_tensor_req & req,
                                     void * data, size_t size) {
    if (tls_get_batch_sock && tls_get_batch_sock != sock) {
        flush_get_tensor_batch();
    }
    tls_get_batch_sock = sock;
    tls_get_batch.push_back({req, data, size});
}

void flush_get_tensor_batch() {
    if (tls_get_batch.empty() || !tls_get_batch_sock) { return; }
    auto sock = tls_get_batch_sock;
    uint32_t count = (uint32_t)tls_get_batch.size();
    const size_t payload_size = sizeof(uint32_t) + count * sizeof(rpc_msg_get_tensor_req);
    const size_t msg_size = 1 + sizeof(uint64_t) + payload_size;
    std::vector<uint8_t> msg(msg_size);
    uint8_t * p = msg.data();
    *p++ = (uint8_t)RPC_CMD_GET_TENSOR_BATCH;
    uint64_t net_payload = payload_size;
    memcpy(p, &net_payload, sizeof(net_payload)); p += sizeof(net_payload);
    memcpy(p, &count, sizeof(count)); p += sizeof(count);
    for (const auto & entry : tls_get_batch) {
        memcpy(p, &entry.req, sizeof(entry.req)); p += sizeof(entry.req);
    }
    sock->send_data(msg.data(), msg.size());
    for (const auto & entry : tls_get_batch) {
        tls_pending_get_tensor.push_back({sock, entry.data, entry.size});
    }
    tls_get_batch.clear();
    tls_get_batch_sock.reset();
}

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

// F1 (Increment-1 T2a): graph_hash carries cgraph->uid from client to server so
// the server can verify the cached graph matches the one the client expects.
// Old servers (no RPC_CAP_RECOMPUTE_HASH) ignore this field; the client omits it
// when talking to them (4-byte backward-compatible request).
struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t graph_hash;
};

// F1 (Increment-1 T2a): server response to GRAPH_RECOMPUTE carrying hit/miss.
// result=0 → cache hit, server will recompute (client proceeds with EVENT_RECORD).
// result=1 → cache miss or uid mismatch, client MUST fall back to full GRAPH_COMPUTE.
struct rpc_msg_graph_recompute_rsp {
    uint32_t result;
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

// Split buffer allocation: allocate memory for a tensor row slice on a specific device
struct rpc_msg_alloc_buffer_split_req {
    uint32_t   device;
    rpc_tensor tensor;
    int64_t    row_low;
    int64_t    row_high;
};

struct rpc_msg_alloc_buffer_split_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
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
    // B+15: drain any queued responses for this socket. The drain reads
    // each response and signals its linked event.
    rpc_socket_drain(ev->sock);
    ev->response_pending = false;
}

static void drain_pending_event_response(const socket_ptr & sock) {
    // B+15: drain the per-socket queue instead of the thread-local slot.
    rpc_socket_drain(sock);
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

static void rpc_socket_queue_push(const socket_ptr & sock, rpc_deferred_entry entry) {
    std::lock_guard<std::mutex> lock(g_rpc_sockets_mutex);
    g_rpc_sockets[sock].queue.push_back(std::move(entry));
}

// Drain all queued responses for a socket. Swaps the queue under the lock,
// then does the blocking recv outside the lock so the global mutex is not
// held during I/O (which would serialize drains across all sockets).
static void rpc_socket_drain(const socket_ptr & sock) {
    std::deque<rpc_deferred_entry> to_drain;
    {
        std::lock_guard<std::mutex> lock(g_rpc_sockets_mutex);
        auto it = g_rpc_sockets.find(sock);
        if (it == g_rpc_sockets.end()) return;
        to_drain.swap(it->second.queue);
        if (it->second.queue.empty()) {
            g_rpc_sockets.erase(it);
        }
    }
    for (auto & entry : to_drain) {
        recv_rpc_cmd_deferred(sock, entry.rsp.data(), entry.rsp.size());
        if (entry.event) {
            entry.event->response_pending = false;
        }
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
    // B+15: drain queued EVENT_RECORD responses for this socket.
    rpc_socket_drain(sock);
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

// Global generation counter — incremented on every buffer alloc/free.
// Used to invalidate device memory caches (the memory is invariant during
// generation but changes during model load when buffers are allocated).
static std::atomic<uint64_t> g_memory_cache_gen{0};

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    uint32_t    global_index; // global rank across all backends (for split buffer)
    std::string name;
    std::string description;
    std::unordered_set<uint64_t> seen_graph_uids;
    // NW1 (Increment-1 T3c): bound the uid set so it cannot grow without limit
    // if the topology hash is imperfect (F9). ≤ SEEN_UIDS_MAX entries/device;
    // on overflow the whole set is cleared (graceful degradation — safe, just
    // falls back to GRAPH_COMPUTE more often until it re-populates).
    static constexpr size_t SEEN_UIDS_MAX = 32;
    static void seen_graph_uids_insert(std::unordered_set<uint64_t> & set, uint64_t uid) {
        if (set.size() >= SEEN_UIDS_MAX) {
            set.clear();
        }
        set.insert(uid);
    }
    // NW1 (T3c): hit/miss counters for the recompute reuse path, emitted to the
    // GGML_RPC_TRACE JSONL so AC2's "uid hit-rate" is measurable per device.
    uint64_t recompute_hits = 0;
    uint64_t recompute_misses = 0;
    // Cached device memory — invariant during generation, saves ~60 ms/token RPC
    uint64_t cached_gen  = 0;          // generation when cached
    size_t   cached_free  = 0;
    size_t   cached_total = 0;
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
    // B+15: true if the last graph_compute sent EVENT_RECORD deferred and the
    // response is queued, so event_record links the event instead of sending
    // a duplicate. Per-context (not thread-local), so it does not race.
    bool        last_compute_sent_event = false;
    // D4.5: multi-device dispatch state
    uint32_t    n_devices_on_endpoint = 0;
    bool        is_multi_device_capable = false;
    // UDP transport: TCP port of the remote RPC server, used to derive the
    // UDP port (tcp_port + 1). Populated at backend init from the endpoint.
    int         tcp_port = 0;
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

// ---- RPC split buffer helpers ----

#define GGML_RPC_MAX_DEVICES 16

// Compute row range for a specific device in a tensor split.
// tensor_split is an array of per-device non-normalized fractional weights
// (e.g., {60, 40} for a 60/40 split). n_dev is the number of non-zero entries.
// id is the device index within this split.
static void rpc_get_row_split(int64_t * row_low, int64_t * row_high,
                               const ggml_tensor * tensor,
                               const float * tensor_split, int n_dev, int id) {
    GGML_ASSERT(id >= 0 && id < n_dev);
    const int64_t nrows = (int64_t) tensor->ne[1];

    // Compute total split sum and cumulative fractions
    float split_sum = 0.0f;
    for (int i = 0; i < n_dev; i++) {
        split_sum += tensor_split[i];
    }
    GGML_ASSERT(split_sum > 0.0f);

    float cum_before = 0.0f;
    for (int i = 0; i < id; i++) {
        cum_before += tensor_split[i] / split_sum;
    }
    float cum_after = cum_before + tensor_split[id] / split_sum;

    *row_low  = (int64_t)(cum_before * (float) nrows);
    *row_high = (int64_t)(cum_after  * (float) nrows);

    // Clamp to valid range
    if (*row_low  < 0)     *row_low  = 0;
    if (*row_high > nrows) *row_high = nrows;
    if (*row_low  > *row_high) *row_low = *row_high;
}

// Compute byte size for nrows_split rows of a tensor.
// Matches ggml_nbytes_split from ggml-cuda.cu.
static size_t rpc_nbytes_split(const ggml_tensor * tensor, int64_t nrows_split) {
    if (nrows_split <= 0) return 0;
    const int64_t ne1 = (int64_t) tensor->ne[1];
    if (nrows_split > ne1) {
        // Clamp for safety: some tensors have ne[1]=0 or degenerate dims
        return (size_t) ne1 * tensor->nb[1];
    }
    return (size_t) nrows_split * tensor->nb[1];
}

// Count the number of non-zero entries in a tensor_split array (up to max_dev).
static int rpc_count_split_devices(const float * tensor_split, int max_dev) {
    int n_dev = 0;
    for (int i = 0; i < max_dev; i++) {
        if (tensor_split[i] > 0.0f) {
            n_dev = i + 1;
        }
    }
    return n_dev;
}

// ---- end RPC split buffer helpers ----

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
    flush_get_tensor_batch();  // V1b
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
        flush_get_tensor_batch();  // V1b
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
    // F1 (T2a): server supports graph_hash in GRAPH_RECOMPUTE + hit/miss response
    sock->server_supports_recompute_hash = (response.conn_caps[0] & RPC_CAP_RECOMPUTE_HASH) != 0;
    // V1b: server supports RPC_CMD_GET_TENSOR_BATCH (value 25)
    sock->server_supports_get_tensor_batch = (response.conn_caps[0] & RPC_CAP_GET_TENSOR_BATCH) != 0;
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
    // B+15: drain queued EVENT_RECORD responses after fresh hello to avoid
    // stale drain on subsequent cmds.
    rpc_socket_drain(sock);
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

// Split buffer context — data members needed early for serialize_tensor
struct ggml_backend_rpc_split_buffer_context {
    struct tensor_slice {
        uint64_t remote_ptr;
        uint64_t remote_size;
        int64_t  row_low;
        int64_t  row_high;
    };
    std::unordered_map<ggml_tensor*, tensor_slice> slices;
    std::shared_ptr<socket_t> sock;
    std::string endpoint;

    ~ggml_backend_rpc_split_buffer_context(); // defined later
};
static void ggml_backend_rpc_split_buffer_free_buffer(ggml_backend_buffer_t buffer);

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    g_memory_cache_gen.fetch_add(1, std::memory_order_release);
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
    } else if (tensor->buffer && tensor->buffer->iface.free_buffer == ggml_backend_rpc_split_buffer_free_buffer) {
        // Split buffer: look up the per-tensor slice remote_ptr and adjust
        // dimensions to the row slice so the server allocates only the needed
        // rows and computes with correct dimensions (no zeroed-row corruption).
        ggml_backend_rpc_split_buffer_context * buf_ctx = (ggml_backend_rpc_split_buffer_context *)tensor->buffer->context;
        auto it = buf_ctx->slices.find(const_cast<ggml_tensor*>(tensor));
        if (it != buf_ctx->slices.end()) {
            result.buffer = it->second.remote_ptr;
            int64_t nrows_split = it->second.row_high - it->second.row_low;
            result.ne[1] = nrows_split;
            result.nb[2] = result.ne[1] * result.nb[1];
            result.nb[3] = result.ne[2] * result.nb[2];
        } else {
            result.buffer = 0;
        }
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    // Re-apply split-buffer dimension adjustment after the unconditional overwrite
    // above (which clobbers the ne[1]/nb[2]/nb[3] values set in the split branch).
    if (tensor->buffer && tensor->buffer->iface.free_buffer == ggml_backend_rpc_split_buffer_free_buffer) {
        ggml_backend_rpc_split_buffer_context * buf_ctx = (ggml_backend_rpc_split_buffer_context *)tensor->buffer->context;
        auto it = buf_ctx->slices.find(const_cast<ggml_tensor*>(tensor));
        if (it != buf_ctx->slices.end()) {
            int64_t nrows_split = it->second.row_high - it->second.row_low;
            result.ne[1] = nrows_split;
            result.nb[2] = result.ne[1] * result.nb[1];
            result.nb[3] = result.ne[2] * result.nb[2];
        }
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

    // V1b: batching requires BOTH opt-in (GGML_RPC_GET_TENSOR_BATCH=1) AND server
    // capability negotiation (RPC_CAP_GET_TENSOR_BATCH). Without both, fall back
    // to the well-tested individual GET_TENSOR path. This guarantees an old
    // server (no batch handler) never receives RPC_CMD_GET_TENSOR_BATCH (25),
    // which would otherwise hit "Unknown command: 25" and drop the connection.
    const bool use_batch = batch_send && rpc_get_tensor_batch_env_enabled() != 0
                            && sock->server_supports_get_tensor_batch;

    if (!use_batch && !rpc_event_defer_barrier()) {
        drain_pending_event_response(sock);
    }
    if (!use_batch) {
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

    // V1b: when batching is enabled, accumulate instead of sending immediately.
    // This eliminates per-call TCP overhead and server dispatch cost.
    if (use_batch) {
        get_tensor_batch_append(sock, request, data, size);
        return;
    }

    // Non-batched path (unchanged): send immediately, defer receive
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
    flush_get_tensor_batch();  // V1b
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
    g_memory_cache_gen.fetch_add(1, std::memory_order_release);
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
    auto sock = get_socket(rpc_ctx->endpoint);

    if (rpc_ctx->last_compute_sent_event) {
        // B+15: EVENT_RECORD was already sent deferred inside graph_compute
        // and queued with event=nullptr. Link the event to the un-linked
        // queued entry (if still pending) so the drain signals it.
        std::lock_guard<std::mutex> lock(g_rpc_sockets_mutex);
        auto it = g_rpc_sockets.find(sock);
        if (it != g_rpc_sockets.end()) {
            for (auto & entry : it->second.queue) {
                if (entry.event == nullptr) {
                    entry.event = ev;
                    ev->sock = sock;
                    ev->response_pending = true;
                    break;
                }
            }
        }
        rpc_ctx->last_compute_sent_event = false;
    } else {
        // Non-reuse path: send EVENT_RECORD deferred and queue it.
        uint64_t tid = ggml_pipeline_trace_get_trace_id();
        rpc_msg_event_record_req ev_req = {ev->id, rpc_ctx->device, tid};
        size_t ev_sz = sock->server_supports_trace_id ? sizeof(ev_req) : 12;
        send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, ev_sz);
        rpc_deferred_entry de;
        de.cmd_type = RPC_CMD_EVENT_RECORD;
        de.rsp.resize(ev_sz);
        de.event = ev;
        rpc_socket_queue_push(sock, std::move(de));
        ev->sock = sock;
        ev->response_pending = true;
    }

    // B+15: drain all queued responses for this socket. Each entry's event
    // is signaled (response_pending = false) once its response is read.
    rpc_socket_drain(sock);
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

    // --- async vs blocking path audit -------------------------------------
    // This function has five distinct code paths. Each is classified below by
    // whether it BLOCKS waiting for a server response on the wire, or returns
    // immediately (async / fire-and-forget) with the response drained later.
    //
    //  PATH                        | TRIGGER                          | BLOCKING?  | SYNC
    //  -----------------------------|----------------------------------|------------|---------------------------
    //  GRAPH_COMPUTE_STAGE         | gpipe_stage >= 0 (D6.9 GPipe)    | BLOCKS     | response + telemetry payload
    //  GRAPH_RECOMPUTE_ALL (reuse) | n_devices>1 AND uid matches      | async      | deferred EVENT_RECORD (TCP)
    //  GRAPH_COMPUTE_ALL (first)   | n_devices>1 AND new uid          | BLOCKS     | response + telemetry payload
    //  GRAPH_RECOMPUTE (reuse)     | uid != 0 AND uid==last_graph_uid | async      | deferred EVENT_RECORD (TCP)
    //  GRAPH_COMPUTE (first)       | new uid (single-device)          | BLOCKS     | 4-byte response
    //
    //  BLOCKING paths use send_rpc_cmd(sock, cmd, input, input_size, output, output_size):
    //    the call does NOT return until the server finishes the graph compute
    //    and the full response is received over TCP. First-time graph submission
    //    also serializes the entire cgraph (expensive for large models).
    //
    //  ASYNC paths use send_rpc_cmd(sock, cmd, input, input_size) with no
    //    output buffer (fire-and-forget), followed by send_rpc_cmd_deferred()
    //    for EVENT_RECORD. The EVENT_RECORD response is NOT read here — it is
    //    queued via rpc_socket_queue_push() and drained later by rpc_socket_drain()
    //    at event_wait / event_synchronize / the next RPC op on this socket.
    //    This lets the scheduler dispatch ROCm splits while the EVENT_RECORD
    //    response is in-flight.
    //
    //  Conditions that select each path (in evaluation order):
    //    1. gpipe_stage >= 0             -> GRAPH_COMPUTE_STAGE (D6.9)
    //    2. multi_device && n_devices>1  -> GRAPH_COMPUTE_ALL / GRAPH_RECOMPUTE_ALL
    //    3. reuse (uid match)            -> GRAPH_RECOMPUTE (single) / GRAPH_RECOMPUTE_ALL (multi)
    //    4. else (first-time)            -> GRAPH_COMPUTE (single) / GRAPH_COMPUTE_ALL (multi)
    //
    //  rpc_ctx->is_multi_device_capable and rpc_ctx->n_devices_on_endpoint are
    //  populated at backend init from the server's DEVICE_COUNT response.
    //  reuse requires cgraph->uid != 0 (set by the scheduler for repeated
    //  token-generation graphs) AND last_graph_uid == uid.
    // -------------------------------------------------------------------------

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
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_graph_compute(RPC_PATH_GRAPH_COMPUTE_STAGE,
                                RPC_CMD_GRAPH_COMPUTE_STAGE, input.size(), us);

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

        bool reuse = cgraph->uid != 0 && rpc_dev_ctx->seen_graph_uids.count(cgraph->uid);
        bool recompute_ok = false; // F1 (T2a): server accepted the multi-device recompute (hit)
        if (reuse) {
            // D4.5: fire-and-forget + EVENT_RECORD (matches existing GRAPH_COMPUTE pattern)
            // UDP transport (opt-in): send GRAPH_RECOMPUTE_ALL over UDP for a
            // true fire-and-forget submission with no TCP round-trip. Falls
            // back to TCP if UDP is disabled or send fails.
            rpc_msg_graph_recompute_all_req req = {};
            req.n_devices = n_devices;
            memcpy(req.devices, devices, n_devices * sizeof(uint32_t));
            req.graph_hash = cgraph->uid;
            req.sync_mode = 0; // fire-and-forget (use EVENT_RECORD for sync)
            req.output_requested = 0;
            int udp_port = rpc_ctx->tcp_port > 0 ? rpc_ctx->tcp_port + 1 : 0;
            rpc_udp_header udp_hdr = {};
            udp_hdr.cmd = RPC_CMD_GRAPH_RECOMPUTE_ALL;
            udp_hdr.device = rpc_ctx->device;
            udp_hdr.graph_uid = cgraph->uid;
            bool udp_sent = rpc_udp_send_graph(sock, udp_hdr, req.devices, req.n_devices,
                                               udp_port);
            if (!udp_sent) {
                // F1 (T2a): read the synchronous hit/miss response; on miss fall
                // back to a full GRAPH_COMPUTE_ALL.
                if (sock->server_supports_recompute_hash) {
                    rpc_msg_graph_recompute_all_rsp rsp = {};
                    bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE_ALL,
                                               &req, sizeof(req), &rsp, sizeof(rsp));
                    if (!status) {
                        LOG_DBG("RPC-REUSE-ALL dev=%u recompute_all rpc failed → fallback\n", rpc_ctx->device);
                        reuse = false;
                    } else if (rsp.result != 0) {
                        LOG_DBG("RPC-REUSE-ALL dev=%u uid=%" PRIu64 " recompute MISS (result=%u) → fallback\n",
                                rpc_ctx->device, cgraph->uid, rsp.result);
                        reuse = false;
                    }
                } else {
                    bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE_ALL, &req, sizeof(req));
                    RPC_STATUS_ASSERT(status);
                }
            }

            if (reuse) {
                // Server accepted the recompute (hit) — synchronize via EVENT_RECORD.
                recompute_ok = true;
                rpc_dev_ctx->recompute_hits++;
                // Send EVENT_RECORD deferred: response drained later by event_wait
                // or at the next RPC operation on this socket. This avoids blocking
                // the scheduler's for-loop, allowing ROCm dispatch to overlap.
                // EVENT_RECORD stays on TCP even when GRAPH_RECOMPUTE used UDP —
                // it needs reliable ordering for synchronization.
                uint64_t tid = ggml_pipeline_trace_get_trace_id();
                rpc_msg_event_record_req ev_req = {0, rpc_ctx->device, tid};
                size_t ev_sz = sock->server_supports_trace_id ? sizeof(ev_req) : 12;
                send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, ev_sz);
                rpc_ctx->last_compute_sent_event = true;
                // B+15: queue the deferred EVENT_RECORD response instead of using
                // the thread-local single slot. event_record links the event.
                rpc_deferred_entry de;
                de.cmd_type = RPC_CMD_EVENT_RECORD;
                de.rsp.resize(ev_sz);
                de.event = nullptr;
                rpc_socket_queue_push(sock, std::move(de));
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
                rpc_trace_graph_compute(RPC_PATH_RECOMPUTE_ALL,
                                        RPC_CMD_GRAPH_RECOMPUTE_ALL, sizeof(req), us);
            }
        }
        if (!recompute_ok) {
            ggml_backend_rpc_device_context::seen_graph_uids_insert(rpc_dev_ctx->seen_graph_uids, cgraph->uid);
            rpc_dev_ctx->recompute_misses++;
            LOG_DBG("RPC-REUSE dev=%u uid=%" PRIu64 " MISS (set=%zu hits=%" PRIu64 " misses=%" PRIu64 ")\n",
                    rpc_ctx->device, cgraph->uid, rpc_dev_ctx->seen_graph_uids.size(),
                    rpc_dev_ctx->recompute_hits, rpc_dev_ctx->recompute_misses);
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
            rpc_trace_graph_compute(RPC_PATH_COMPUTE_ALL,
                                    RPC_CMD_GRAPH_COMPUTE_ALL, input.size(), us);
        }
        return GGML_STATUS_SUCCESS;
    }

    const auto t0 = std::chrono::steady_clock::now();
    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->seen_graph_uids.count(cgraph->uid);
    bool recompute_ok = false; // F1 (T2a): set true only if server accepted the recompute (hit)
    if (reuse) {
        // UDP transport (opt-in): send GRAPH_RECOMPUTE over UDP for a true
        // fire-and-forget submission with no TCP round-trip. Falls back to
        // TCP if UDP is disabled or send fails.
        int udp_port = rpc_ctx->tcp_port > 0 ? rpc_ctx->tcp_port + 1 : 0;
        rpc_udp_header udp_hdr = {};
        udp_hdr.cmd = RPC_CMD_GRAPH_RECOMPUTE;
        udp_hdr.device = rpc_ctx->device;
        udp_hdr.graph_uid = cgraph->uid;
        bool udp_sent = rpc_udp_send_graph(sock, udp_hdr, nullptr, 0, udp_port);
        if (!udp_sent) {
            // F1 (T2a): send the uid as graph_hash so the server can verify the
            // cached graph matches. Read the synchronous hit/miss response; on
            // miss, fall back to a full GRAPH_COMPUTE instead of proceeding.
            rpc_msg_graph_recompute_req request = {};
            request.device = rpc_ctx->device;
            size_t req_sz = sizeof(uint32_t);
            if (sock->server_supports_recompute_hash) {
                request.graph_hash = cgraph->uid;
                req_sz = sizeof(request);
            }
            if (sock->server_supports_recompute_hash) {
                rpc_msg_graph_recompute_rsp rsp = {};
                bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE,
                                           &request, req_sz, &rsp, sizeof(rsp));
                if (!status) {
                    // RPC failed — safest fallback is a full recompute.
                    LOG_DBG("RPC-REUSE dev=%u recompute rpc failed → fallback to GRAPH_COMPUTE\n", rpc_ctx->device);
                    reuse = false;
                } else if (rsp.result != 0) {
                    // F1: server reported MISS (uid mismatch or no cached graph).
                    // Fall back to a full GRAPH_COMPUTE for correctness.
                    LOG_DBG("RPC-REUSE dev=%u uid=%" PRIu64 " recompute MISS (result=%u) → fallback\n",
                            rpc_ctx->device, cgraph->uid, rsp.result);
                    reuse = false;
                }
            } else {
                // Old server (no recompute-hash cap): fire-and-forget as before.
                bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, req_sz);
                RPC_STATUS_ASSERT(status);
            }
        }

        if (reuse) {
            // Server accepted the recompute (hit) — synchronize via EVENT_RECORD.
            recompute_ok = true;
            rpc_dev_ctx->recompute_hits++;
            uint64_t tid = ggml_pipeline_trace_get_trace_id();
            rpc_msg_event_record_req ev_req = {0, rpc_ctx->device, tid};
            size_t ev_sz = sock->server_supports_trace_id ? sizeof(ev_req) : 12;
            // Defer the EVENT_RECORD response: the scheduler's event_record
            // already has a last_compute_sent_event fast-path, and the response
            // drains at event_wait/event_synchronize or at the next RPC op.
            // This makes graph_compute_async truly async (<50us TCP send only),
            // letting the scheduler dispatch ROCm splits while the RPC event
            // response is in-flight. EVENT_RECORD stays on TCP even when
            // GRAPH_RECOMPUTE used UDP — it needs reliable ordering for sync.
            send_rpc_cmd_deferred(sock, RPC_CMD_EVENT_RECORD, &ev_req, ev_sz);
            rpc_ctx->last_compute_sent_event = true;
            // B+15: queue the deferred EVENT_RECORD response instead of using
            // the thread-local single slot. event_record links the event.
            rpc_deferred_entry de;
            de.cmd_type = RPC_CMD_EVENT_RECORD;
            de.rsp.resize(ev_sz);
            de.event = nullptr;
            rpc_socket_queue_push(sock, std::move(de));
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            rpc_trace_graph_compute(RPC_PATH_RECOMPUTE,
                                    RPC_CMD_GRAPH_RECOMPUTE, sizeof(rpc_msg_graph_recompute_req), us);
        }
    }
    if (!recompute_ok) {
        ggml_backend_rpc_device_context::seen_graph_uids_insert(rpc_dev_ctx->seen_graph_uids, cgraph->uid);
        rpc_dev_ctx->recompute_misses++;
        LOG_DBG("RPC-REUSE dev=%u uid=%" PRIu64 " MISS (set=%zu hits=%" PRIu64 " misses=%" PRIu64 ")\n",
                rpc_ctx->device, cgraph->uid, rpc_dev_ctx->seen_graph_uids.size(),
                rpc_dev_ctx->recompute_hits, rpc_dev_ctx->recompute_misses);
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
        rpc_ctx->last_compute_sent_event = false;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        rpc_trace_graph_compute(RPC_PATH_COMPUTE,
                                RPC_CMD_GRAPH_COMPUTE, input.size(), us);
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
    };
    // UDP transport: parse the TCP port from the endpoint so graph_compute
    // can derive the UDP port (tcp_port + 1) for fire-and-forget submission.
    {
        std::string host;
        int port = 0;
        if (parse_endpoint(endpoint, host, port)) {
            ctx->tcp_port = port;
        }
    }
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

// Forward declaration of rpc_compute_engine and its nested split_buffer_meta type
class rpc_compute_engine;

// ============================================================================
// rpc_compute_engine: process-wide shared compute state (T2b extraction).
//
// Owns the GPU backends, the per-device stored_graphs cache, the multi-device
// scheduler cache, the single compute worker thread + queue, and all telemetry
// state. One instance is created in ggml_backend_rpc_start_server and shared
// (via shared_ptr) across all TCP connection handlers and the UDP listener.
//
// This resolves BUG-011a (UDP cache miss) and BUG-011b (event coupling): the
// UDP listener now hits the SAME stored_graphs + compute queue as TCP.
//
// NIT-1: stored_graph::uid is std::atomic<uint64_t> — read in recompute_allowed()
// from dispatch + UDP-listener threads, written by graph_compute() in the worker.
// NIT-2: try_enqueue_graph_recompute() makes check+enqueue atomic under compute_mtx.
// ============================================================================

// Split buffer metadata type — defined at global scope so both rpc_compute_engine
// and rpc_connection can reference it without nested-type issues.
struct rpc_split_buffer_meta {
    int64_t ne[GGML_MAX_DIMS];
    int64_t nrows_split;
    int64_t row_low;
    int64_t row_high;
};

class rpc_compute_engine {
public:
    rpc_compute_engine(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir),
          telemetry_enabled(rpc_server_telemetry_env_enabled()) {
        stored_graphs.resize(backends.size());
        if (telemetry_enabled) {
            for (size_t i = 0; i < backends.size() && i < RPC_TELEMETRY_MAX_DEVICES; i++) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
                rpc_telemetry_device_meta & meta = startup_device_meta[i];
                memset(&meta, 0, sizeof(meta));
                if (!dev) continue;
                struct ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                snprintf(meta.name, sizeof(meta.name), "%s", props.name);
                meta.vram_mib = props.memory_total / (1024 * 1024);
                meta.backend_type = (int32_t) ggml_backend_dev_type(dev);
            }
        }
        compute_worker = std::thread([this]() { compute_worker_loop(); });
    }
    ~rpc_compute_engine();

    // --- Graph computation (shared across all connections) ---
    // deserialize_tensor / create_node need the connection's buffer sets for
    // the split-buffer warning check; passed explicitly.
    // ram_buffers: VVRAM — set of buffers backed by host RAM (tier=RAM). The
    // staging pass copies these into VRAM just-in-time before compute.
    bool graph_compute(const std::vector<uint8_t> & input,
                       const std::unordered_set<ggml_backend_buffer_t> & buffers,
                       const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas,
                       const std::unordered_set<ggml_backend_buffer_t> & ram_buffers = {});
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool graph_compute_all(const std::vector<uint8_t> & input,
                           const std::unordered_set<ggml_backend_buffer_t> & buffers,
                           const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas);
    bool graph_recompute_all(const rpc_msg_graph_recompute_all_req & request);
    bool graph_compute_stage(const std::vector<uint8_t> & input, uint32_t stage_id,
                             const std::unordered_set<ggml_backend_buffer_t> & buffers,
                             const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas);
    bool recompute_allowed(const rpc_msg_graph_recompute_req & request) const;
    bool recompute_all_allowed(const rpc_msg_graph_recompute_all_req & request) const;

    // NIT-2: atomic check+enqueue under compute_mtx (fixes the multi-client race
    // where recompute_allowed() and enqueue_graph_recompute() could be split by
    // a concurrent graph_compute() uid-reset).
    bool try_enqueue_graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool try_enqueue_graph_recompute_all(const rpc_msg_graph_recompute_all_req & request);

    ggml_backend_sched_t create_multi_device_sched(
        const uint32_t * devices, uint32_t n_devices,
        const ggml_cgraph * graph);

    void enqueue_graph_compute(std::vector<uint8_t> input,
                               const std::unordered_set<ggml_backend_buffer_t> & buffers,
                               const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas,
                               const std::unordered_set<ggml_backend_buffer_t> & ram_buffers = {});
    void enqueue_graph_recompute(rpc_msg_graph_recompute_req request);
    void enqueue_graph_compute_all(std::vector<uint8_t> input,
                                   const std::unordered_set<ggml_backend_buffer_t> & buffers,
                                   const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas);
    void enqueue_graph_recompute_all(rpc_msg_graph_recompute_all_req request);
    void enqueue_graph_compute_stage(std::vector<uint8_t> input, uint32_t stage_id,
                                     const std::unordered_set<ggml_backend_buffer_t> & buffers,
                                     const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas);
    void wait_compute_idle();

    // T2c: drain all in-flight compute AND invalidate the cached-graph store
    // (stored_graphs, all_graph, all_scheds) atomically under compute_mtx.
    // Combining the wait + clear eliminates the race where a concurrent
    // graph_compute could write a slot while we clear it: holding compute_mtx
    // after the queue drains guarantees (a) no job is running (inflight==0)
    // and (b) no new job can be enqueued (submit_compute_job needs compute_mtx).
    // Cached graphs hold tensors whose buffer pointers may reference the
    // disconnecting connection's buffers; clearing them prevents a later
    // GRAPH_RECOMPUTE from adopting a stale graph and dereferencing freed GPU
    // memory. Sacrifices cross-client cache reuse (sanctioned by F6).
    void drain_and_invalidate();

    // T2d: lightweight cached-graph invalidation for new-client connection start.
    // Nulls stored_graphs[].graph/all_graph.graph and resets uids UNDER compute_mtx
    // so a concurrent graph_recompute() in the worker thread cannot adopt a stale
    // graph from the previous connection. Does NOT free all_scheds — those are owned
    // by drain_and_invalidate() and ~rpc_compute_engine(); freeing them here would
    // double-free. The old graphs themselves are pool-allocated (live inside
    // stored_graph::buffer), so nulling the pointer is sufficient — the next
    // graph_compute() reclaims the memory by reinitializing the pool at offset 0.
    void invalidate_cached_graphs();

    // Deserialization helpers (called by both engine and connection)
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor,
                                     const std::unordered_set<ggml_backend_buffer_t> & buffers,
                                     const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas);

    void collect_telemetry(const uint32_t * devices, uint32_t n_devices,
                           const int64_t * per_device_us);
    bool get_last_telemetry(rpc_msg_server_telemetry & out) const;

    const char * get_cache_dir() const { return cache_dir; }

    // Public accessors for rpc_connection (which delegates to engine)
    const std::vector<ggml_backend_t> & get_backends() const { return backends; }
    size_t get_backend_count() const { return backends.size(); }

    // NIT-1: uid is atomic — read in recompute_allowed() from dispatch + UDP
    // threads, written by graph_compute() in the worker thread.
    struct stored_graph {
        std::vector<uint8_t>      buffer;
        ggml_cgraph             * graph;
        std::atomic<uint64_t>     uid{0}; // F1 (T2a): uid bound to this cached graph (0 = unset)

        // VVRAM: staging buffers allocated during graph_compute to hold RAM-tier
        // weights copied into VRAM just-in-time. Freed in invalidate/destructor.
        std::vector<ggml_backend_buffer_t> vvram_staging_buffers;

        // VVRAM: snapshot of the RAM-tier buffer set taken when the layered
        // compute path (GGML_RPC_VVRAM_LRU=1) staged this graph. GRAPH_RECOMPUTE
        // (the decode loop) re-runs the layered path for these buffers — without
        // this set it would compute the stored graph directly against
        // RAM/dangling weight pointers.
        std::unordered_set<ggml_backend_buffer_t> vvram_ram_buffers;

        // std::atomic is not movable, so define explicit move constructor
        // for vector<stored_graph>::resize() to work.
        stored_graph() = default;
        stored_graph(stored_graph && other) noexcept
            : buffer(std::move(other.buffer)),
              graph(other.graph),
              uid(other.uid.load(std::memory_order_relaxed)),
              vvram_staging_buffers(std::move(other.vvram_staging_buffers)),
              vvram_ram_buffers(std::move(other.vvram_ram_buffers)) {}
        stored_graph & operator=(stored_graph && other) noexcept {
            buffer = std::move(other.buffer);
            graph = other.graph;
            uid.store(other.uid.load(std::memory_order_relaxed), std::memory_order_relaxed);
            vvram_staging_buffers = std::move(other.vvram_staging_buffers);
            vvram_ram_buffers = std::move(other.vvram_ram_buffers);
            return *this;
        }
        // Non-copyable (atomic member)
        stored_graph(const stored_graph &) = delete;
        stored_graph & operator=(const stored_graph &) = delete;
    };

    // Split buffer metadata type (shared with rpc_connection)
    // NOTE: this is a type alias for the global rpc_split_buffer_meta.
    using split_buffer_meta = rpc_split_buffer_meta;

private:
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map,
                              const std::unordered_set<ggml_backend_buffer_t> & buffers,
                              const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas);

    void compute_worker_loop();
    void submit_compute_job(std::function<void()> job);

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;

    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;
    stored_graph all_graph; // D4.5: dedicated ALL-mode storage
    std::unordered_map<uint64_t, ggml_backend_sched_t> all_scheds;

    std::mutex                    compute_mtx;
    std::condition_variable       compute_cv;
    std::deque<std::function<void()>> compute_queue;
    std::thread                   compute_worker;
    std::atomic<bool>             compute_shutdown{false};
    std::atomic<int>              compute_inflight{0};

    // D4.10: telemetry state
    const bool                telemetry_enabled;
    rpc_telemetry_device_meta startup_device_meta[RPC_TELEMETRY_MAX_DEVICES];
    rpc_msg_server_telemetry  last_telemetry;
    mutable std::mutex        telemetry_mtx;
    std::atomic<uint64_t>     telemetry_decode_count{0};
    std::atomic<uint64_t>     telemetry_node_sample_count{0};
};

// ============================================================================
// rpc_connection: per-connection server state (T2b extraction).
//
// Holds the per-connection buffer set + split metadata, and a shared_ptr to the
// process-wide rpc_compute_engine. Buffer/tensor management methods live here;
// graph computation is delegated to the engine.
// ============================================================================

// VVRAM LRU helpers (defined below): forward-declared for the alloc gate
// (rpc_connection::alloc_buffer reads the LRU staging budget for the
// BUG-015 headroom reservation).
static bool rpc_vvram_use_lru();
static size_t rpc_vvram_staging_headroom(size_t vram_total);

// [VVRAM-CLOSE] forward declarations for the discriminator's freed-buffer
// tracker (defined later, used by free_buffer/get_tensor earlier in the file).
static void vvram_close_record_free(ggml_backend_buffer_t buffer);
static bool vvram_close_was_freed(ggml_backend_buffer_t buffer);
// [VVRAM-CLOSE] diagnostics gate: GGML_VVRAM_CLOSE_DEBUG=1 enables the
// per-GET_TENSOR / per-free / per-cleanup prints. GET_TENSOR is the decode hot
// path — every print is an unbuffered stderr write() syscall plus raw inference
// data to stderr — so the default is OFF (code-review BLK-1/N7; follows the
// fork's GGML_RPC_DEBUG env-var convention).
static bool rpc_vvram_close_debug();

class rpc_connection {
public:
    rpc_connection(std::shared_ptr<rpc_compute_engine> engine, const char * cache_dir)
        : engine(std::move(engine)), cache_dir(cache_dir),
          telemetry_enabled(rpc_server_telemetry_env_enabled()) {
        // VVRAM: read RAM budget from env (default 32000 MiB). 0 disables VVRAM.
        const char * ram_budget_env = std::getenv("GGML_RPC_VVRAM_RAM_BUDGET_MB");
        if (ram_budget_env && ram_budget_env[0]) {
            vvram_ram_budget = (size_t)atoll(ram_budget_env) * 1024 * 1024;
            if (vvram_ram_budget > 0) {
                GGML_LOG_INFO("[VVRAM] enabled — RAM budget = %zu MiB\n", vvram_ram_budget / (1024*1024));
            }
        }
        if (telemetry_enabled) {
            const auto & backends = engine->get_backends();
            for (size_t i = 0; i < backends.size() && i < RPC_TELEMETRY_MAX_DEVICES; i++) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
                rpc_telemetry_device_meta & meta = startup_device_meta[i];
                memset(&meta, 0, sizeof(meta));
                if (!dev) continue;
                struct ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                snprintf(meta.name, sizeof(meta.name), "%s", props.name);
                meta.vram_mib = props.memory_total / (1024 * 1024);
                meta.backend_type = (int32_t) ggml_backend_dev_type(dev);
            }
        }
    }
    ~rpc_connection();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool alloc_buffer_split(const rpc_msg_alloc_buffer_split_req & request, rpc_msg_alloc_buffer_split_rsp & response);
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
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    // Accessors for the engine's graph methods
    const std::unordered_set<ggml_backend_buffer_t> & get_buffers() const { return buffers; }
    const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & get_split_metas() const { return split_buffer_metas; }
    std::shared_ptr<rpc_compute_engine> get_engine() const { return engine; }

    // Direct engine passthroughs (used by rpc_serve_client)
    bool recompute_allowed(const rpc_msg_graph_recompute_req & request) const { return engine->recompute_allowed(request); }
    bool recompute_all_allowed(const rpc_msg_graph_recompute_all_req & request) const { return engine->recompute_all_allowed(request); }
    bool try_enqueue_graph_recompute(const rpc_msg_graph_recompute_req & request) { return engine->try_enqueue_graph_recompute(request); }
    bool try_enqueue_graph_recompute_all(const rpc_msg_graph_recompute_all_req & request) { return engine->try_enqueue_graph_recompute_all(request); }
    void enqueue_graph_compute(std::vector<uint8_t> input) { engine->enqueue_graph_compute(std::move(input), buffers, split_buffer_metas, vvram_ram_buffers); }
    void enqueue_graph_recompute(rpc_msg_graph_recompute_req request) { engine->enqueue_graph_recompute(std::move(request)); }
    void enqueue_graph_compute_all(std::vector<uint8_t> input) { engine->enqueue_graph_compute_all(std::move(input), buffers, split_buffer_metas); }
    void enqueue_graph_recompute_all(rpc_msg_graph_recompute_all_req request) { engine->enqueue_graph_recompute_all(std::move(request)); }
    void enqueue_graph_compute_stage(std::vector<uint8_t> input, uint32_t stage_id) { engine->enqueue_graph_compute_stage(std::move(input), stage_id, buffers, split_buffer_metas); }
    void wait_compute_idle() { engine->wait_compute_idle(); }
    bool get_last_telemetry(rpc_msg_server_telemetry & out) const { return engine->get_last_telemetry(out); }

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);

    std::shared_ptr<rpc_compute_engine> engine;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    std::mutex buffers_mtx;

    std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> split_buffer_metas;

    // VVRAM: virtual VRAM state — RPC backend presents a flat virtual VRAM space
    // (GPU VRAM + host RAM budget). Buffers that don't fit in VRAM are placed in
    // pinned host RAM (tier=RAM) and staged into VRAM just-in-time during graph_compute.
    size_t vvram_ram_budget = 0;    // total RAM budget in bytes (env GGML_RPC_VVRAM_RAM_BUDGET_MB)
    size_t vvram_ram_used  = 0;     // bytes currently allocated to RAM-tier buffers
    size_t vvram_vram_used = 0;     // bytes currently allocated to VRAM-tier buffers
    size_t vvram_max_alloc = 0;     // largest single VRAM-tier alloc seen (compute-buffer exemption, BUG-015)
    size_t vvram_ram_last_alloc = 0; // size of the last RAM-tier alloc (context-phase detector, BUG-015e)
    std::unordered_set<ggml_backend_buffer_t> vvram_ram_buffers; // set of RAM-tier buffers

    // D4.10: per-connection telemetry state (for collect_telemetry calls from connection)
    const bool                telemetry_enabled;
    rpc_telemetry_device_meta startup_device_meta[RPC_TELEMETRY_MAX_DEVICES];
};

// OUT-OF-CLASS DESTRUCTOR DEFINITION below
using rpc_server = rpc_connection;

void rpc_connection::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_connection::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= engine->get_backends().size()) {
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
        buft = ggml_backend_get_default_buffer_type(engine->get_backends()[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_connection::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= engine->get_backends().size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(engine->get_backends()[dev_id]);

    // VVRAM tiering: decide VRAM vs RAM based on current free VRAM.
    // If the allocation fits in remaining VRAM, use the GPU buffer type.
    // Otherwise, fall back to pinned host RAM (CUDA/HIP host buffer type).
    bool use_ram = false;
    if (vvram_ram_budget > 0) {
        size_t vram_free = 0, vram_total = 0;
        ggml_backend_dev_t dev = ggml_backend_get_device(engine->get_backends()[dev_id]);
        ggml_backend_dev_memory(dev, &vram_free, &vram_total);
        // vram_free is the actual free VRAM (already accounts for all allocations).
        size_t ram_remaining = (vvram_ram_budget > vvram_ram_used) ? (vvram_ram_budget - vvram_ram_used) : 0;
        if (request.size <= vram_free) {
            // BUG-015 (I2-VVRAM-STAGING-HEADROOM): with the LRU layered compute
            // path on, reserve staging headroom on the GPU. The greedy gate used
            // to fill the device to ~98.5% with VRAM-tier weight buffers, leaving
            // the first RAM-tier layer's just-in-time staging zero physical room
            // (72B: cudaMalloc OOM at blk.50.attn_q.weight). Weight-sized allocs
            // that would dip below the headroom floor go to the RAM tier instead.
            // The floor must NOT catch the graph compute buffer / KV cache:
            //   - a new size record (larger than every VRAM alloc so far) — the
            //     shared buffer, big context buffers — stays resident;
            //   - context-phase allocs (KV cache, compute buffer) are far smaller
            //     than the recent RAM-tier layer buffers (< 1/4), so they stay
            //     resident too. Without this, the compute buffer landed in the RAM
            //     tier (BUG-015e: 72B, 55.4 MiB < 496 MiB max weight) and the
            //     layered path staged every activation -> decode stall.
            bool compute_like = request.size > vvram_max_alloc ||
                                (vvram_ram_last_alloc > 0 && request.size * 4 < vvram_ram_last_alloc);
            bool reserve_headroom = rpc_vvram_use_lru() &&
                                    !compute_like &&
                                    vram_free < request.size + rpc_vvram_staging_headroom(vram_total);
            if (reserve_headroom) {
                use_ram = (request.size <= ram_remaining);
            } else {
                use_ram = false;
            }
        } else if (request.size <= ram_remaining) {
            use_ram = true;
        } else {
            // Doesn't fit in either — let the VRAM alloc attempt fail gracefully
            // (it will return nullptr and the client will error out).
            use_ram = false;
        }
    }

    ggml_backend_buffer_t buffer = nullptr;
    if (use_ram) {
        // RAM tier: allocate pinned host memory. Prefer the GPU device's host
        // buffer type (pinned, faster H2D copy in staging). Fall back to the
        // plain CPU buffer type (non-pinned, still correct).
        ggml_backend_buffer_type_t host_buft = nullptr;
        ggml_backend_dev_t dev = ggml_backend_get_device(engine->get_backends()[dev_id]);
        if (dev) {
            host_buft = ggml_backend_dev_host_buffer_type(dev);
        }
        if (!host_buft) {
            host_buft = ggml_backend_cpu_buffer_type();
        }
        buffer = ggml_backend_buft_alloc_buffer(host_buft, request.size);
        if (buffer) {
            std::lock_guard<std::mutex> lock(buffers_mtx);
            vvram_ram_used += request.size;
            vvram_ram_last_alloc = request.size;
            vvram_ram_buffers.insert(buffer);
            GGML_LOG_INFO("[VVRAM] alloc_buffer: %zu MiB -> RAM tier (ram_used=%zu/%zu MiB)\n",
                request.size / (1024*1024), vvram_ram_used / (1024*1024), vvram_ram_budget / (1024*1024));
        }
    } else {
        // VRAM tier: normal GPU buffer.
        buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
        if (buffer) {
            std::lock_guard<std::mutex> lock(buffers_mtx);
            vvram_vram_used += request.size;
            if (request.size > vvram_max_alloc) {
                vvram_max_alloc = request.size;
            }
        }
    }

    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 " (%s)\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size,
            use_ram ? "RAM" : "VRAM");
        {
            std::lock_guard<std::mutex> lock(buffers_mtx);
            buffers.insert(buffer);
        }
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_connection::alloc_buffer_split(const rpc_msg_alloc_buffer_split_req & request, rpc_msg_alloc_buffer_split_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= engine->get_backends().size()) {
        return false;
    }

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
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    int64_t nrows_split = request.row_high - request.row_low;
    if (nrows_split <= 0) {
        response.remote_ptr = 0;
        response.remote_size = 0;
        return true;
    }

    // Allocate based on the tensor dimensions received from the client.
    // For split buffers, serialize_tensor adjusts ne[1] to nrows_split,
    // so ggml_nbytes(tensor) computes the exact slice size.
    size_t alloc_size = ggml_nbytes(tensor);

    // V0 row-split fix: use the native CUDA/HIP split buffer type so that
    // mul_mat on the server uses the correct row_low/row_high path for both
    // weights AND activations (via tensor->extra->data_device[id]).
    // The server has a single GPU (dev_id); build a tensor_split array where
    // that GPU owns 100% of the rows. The CUDA split buffer's init_tensor
    // allocates per-device memory lazily and populates tensor->extra.
    ggml_backend_buffer_t buffer = nullptr;
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    {
        float tensor_split[GGML_CUDA_MAX_DEVICES] = {};
        tensor_split[dev_id] = 1.0f;
        ggml_backend_buffer_type_t split_buft = ggml_backend_cuda_split_buffer_type((int)dev_id, tensor_split);
        if (split_buft) {
            // The CUDA split buffer type's alloc_buffer ignores the size
            // parameter (it just creates the context); init_tensor allocates
            // device memory lazily. Pass full tensor size for safety.
            buffer = ggml_backend_buft_alloc_buffer(split_buft, alloc_size);
        }
    }
#endif
    if (buffer == nullptr) {
        // Fallback: plain default buffer type (non-split path)
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(engine->get_backends()[dev_id]);
        buffer = ggml_backend_buft_alloc_buffer(buft, alloc_size);
    }

    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        // For CUDA split buffers, do NOT call buffer_clear — the split buffer
        // type's clear is a no-op and init_tensor handles zeroing padding.
        // Only zero for the fallback (non-split) path.
        if (!rpc_buft_is_cuda_split(buffer->buft)) {
            ggml_backend_buffer_clear(buffer, 0);
        }
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        {
            std::lock_guard<std::mutex> lock(buffers_mtx);
            buffers.insert(buffer);
            // Store split metadata for deserialize_tensor (fallback path only).
            // CUDA split buffers carry their own row_low/row_high context
            // in tensor->extra, so no metadata is needed for them.
            if (!rpc_buft_is_cuda_split(buffer->buft)) {
                rpc_split_buffer_meta meta;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    meta.ne[i] = tensor->ne[i];
                }
                meta.nrows_split = nrows_split;
                meta.row_low  = request.row_low;
                meta.row_high = request.row_high;
                split_buffer_metas[buffer] = meta;
            }
        }
    } else {
        LOG_DBG("[%s] device: %d, nrows_split: %" PRId64 ", alloc_size: %zu -> failed\n",
            __func__, dev_id, nrows_split, alloc_size);
    }
    return true;
}

bool rpc_connection::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= engine->get_backends().size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(engine->get_backends()[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_connection::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= engine->get_backends().size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(engine->get_backends()[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    // VVRAM: report virtual max = VRAM max + RAM budget. The client sees a flat
    // virtual VRAM space and allocates against it. Cap to avoid SIZE_MAX overflow
    // when the underlying buft reports SIZE_MAX (CUDA default).
    if (vvram_ram_budget > 0) {
        const size_t cap = (SIZE_MAX / 4) * 3; // avoid overflow
        if (max_size > cap) {
            max_size = cap;
        }
        max_size += vvram_ram_budget;
    }
    LOG_DBG("[%s] device: %d, max_size: %lu (vram=%zu + ram_budget=%zu)\n", __func__, dev_id, max_size,
            max_size - (vvram_ram_budget > 0 ? vvram_ram_budget : 0), vvram_ram_budget);
    response.max_size = max_size;
    return true;
}

bool rpc_connection::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(buffer) == buffers.end()) {
            GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
            return false;
        }
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_connection::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(buffer) == buffers.end()) {
            GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
            return false;
        }
    }
    // VVRAM: track tier usage before freeing.
    bool was_ram = false;
    size_t buf_size = buffer->size;
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        was_ram = (vvram_ram_buffers.find(buffer) != vvram_ram_buffers.end());
    }
    // N7: gated behind GGML_VVRAM_CLOSE_DEBUG=1 (default OFF).
    if (rpc_vvram_close_debug()) {
        fprintf(stderr, "[VVRAM-CLOSE-FREE] free_buffer base=%p size=%zu was_ram=%d\n",
                (void*)ggml_backend_buffer_get_base(buffer), buf_size, (int)was_ram);
    }
    vvram_close_record_free(buffer);
    ggml_backend_buffer_free(buffer);
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        buffers.erase(buffer);
        split_buffer_metas.erase(buffer); // clean up split metadata if present
        if (was_ram) {
            vvram_ram_buffers.erase(buffer);
            if (vvram_ram_used >= buf_size) {
                vvram_ram_used -= buf_size;
            } else {
                vvram_ram_used = 0;
            }
        } else {
            if (vvram_vram_used >= buf_size) {
                vvram_vram_used -= buf_size;
            } else {
                vvram_vram_used = 0;
            }
        }
    }
    return true;
}

bool rpc_connection::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(buffer) == buffers.end()) {
            GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
            return false;
        }
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_compute_engine::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor,
                                     const std::unordered_set<ggml_backend_buffer_t> & buffers,
                                     const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas) {
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
    bool is_split = false;
    bool is_cuda_split = false;
    split_buffer_meta split_meta;
    if (result->buffer) {
        // NOTE: buffers and split_metas are passed by const reference from the
        // connection. They are stable during graph compute (no concurrent mods).
        if (buffers.find(result->buffer) == buffers.end()) {
            static std::atomic<int> warn_count{0};
            if (warn_count.fetch_add(1, std::memory_order_relaxed) < 5) {
                GGML_LOG_WARN("[%s] buffer %p not found in server buffer set (%zu buffers registered)\n",
                    __func__, (void*)result->buffer, buffers.size());
            }
            result->buffer = nullptr;
        } else {
            // Check if this is a native CUDA split buffer (V0 row-split fix)
            is_cuda_split = rpc_buft_is_cuda_split(result->buffer->buft);
            // Check if this buffer has split metadata (fallback path)
            auto meta_it = split_metas.find(result->buffer);
            if (meta_it != split_metas.end()) {
                is_split = true;
                split_meta = meta_it->second;
            }
        }
    }

    if (is_cuda_split) {
        // V0 row-split fix: CUDA split buffer's get_base returns dummy 0x1000.
        // The real device pointers are in tensor->extra->data_device[id],
        // populated by init_tensor. Call it now (once per tensor) to allocate
        // device memory and set extra. Do NOT set result->data to the dummy.
        if (result->extra == nullptr && result->buffer->iface.init_tensor) {
            result->buffer->iface.init_tensor(result->buffer, result);
        }
        result->data = nullptr; // real pointers are in extra->data_device
    } else if (is_split) {
        // Split buffer: client-side serialize_tensor has already adjusted
        // ne[1] to nrows_split, so the serialized dimensions are correct.
        // Set data pointer to server-side buffer base — no ne[] changes needed.
        result->data = ggml_backend_buffer_get_base(result->buffer);
    } else {
        result->data = reinterpret_cast<void *>(tensor->data);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    ggml_set_name(result, tensor->name);
    return result;
}

// rpc_connection::deserialize_tensor: per-connection version that uses the
// connection's own buffers + split_buffer_metas (locked by buffers_mtas).
ggml_tensor * rpc_connection::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
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

    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    bool is_split = false;
    bool is_cuda_split = false;
    rpc_split_buffer_meta split_meta;
    if (result->buffer) {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(result->buffer) == buffers.end()) {
            static std::atomic<int> warn_count{0};
            if (warn_count.fetch_add(1, std::memory_order_relaxed) < 5) {
                GGML_LOG_WARN("[%s] buffer %p not found in server buffer set (%zu buffers registered)\n",
                    __func__, (void*)result->buffer, buffers.size());
            }
            result->buffer = nullptr;
        } else {
            is_cuda_split = rpc_buft_is_cuda_split(result->buffer->buft);
            auto meta_it = split_buffer_metas.find(result->buffer);
            if (meta_it != split_buffer_metas.end()) {
                is_split = true;
                split_meta = meta_it->second;
            }
        }
    }

    if (is_cuda_split) {
        if (result->extra == nullptr && result->buffer->iface.init_tensor) {
            result->buffer->iface.init_tensor(result->buffer, result);
        }
        result->data = nullptr;
    } else if (is_split) {
        result->data = ggml_backend_buffer_get_base(result->buffer);
    } else {
        result->data = reinterpret_cast<void *>(tensor->data);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_connection::set_tensor(const std::vector<uint8_t> & input) {
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

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);

    // Cache I/O outside the lock — does not touch tensor->buffer.
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }

    // Verify buffer and bounds under lock, then set.
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(tensor->buffer) == buffers.end()) {
            GGML_LOG_ERROR("[%s] buffer %p not found in server buffer set\n",
                __func__, (void*)tensor->buffer);
            return false;
        }
        // V0 row-split fix: CUDA split buffers store device pointers in
        // tensor->extra->data_device[id], not in tensor->data (which is null).
        // Use the buffer's own set_tensor interface, which handles the split.
        // The CUDA split buffer set_tensor asserts offset==0 and
        // size==ggml_nbytes(tensor), so skip the standard bounds check.
        if (rpc_buft_is_cuda_split(tensor->buffer->buft)) {
            tensor->buffer->iface.set_tensor(tensor->buffer, tensor, data, offset, size);
        } else if (vvram_ram_buffers.count(tensor->buffer)) {
            // VVRAM RAM-tier buffer: the client sends GPU-padded data but the
            // host buffer's tensor has unpadded dimensions. Write at the
            // tensor's actual position inside the buffer.
            // BUG-015d fix: the RPC 'offset' is the WITHIN-TENSOR offset
            // (0 for full-tensor writes from the loader), NOT the within-buffer
            // offset. The previous code wrote at base(buffer)+offset, i.e.
            // buffer base + 0 for every tensor — the last-written tensor
            // clobbered the buffer start and every other tensor's data stayed
            // zero (72B NaN decode at the first staged layer). tensor->data
            // already equals base + within-buffer offset.
            const size_t buf_size = ggml_backend_buffer_get_size(tensor->buffer);
            const size_t dst_off = (size_t)((char *) tensor->data - (char *) ggml_backend_buffer_get_base(tensor->buffer));
            if (dst_off + offset + size > buf_size) {
                GGML_LOG_ERROR("[%s] RAM-tier tensor data (off=%zu, size=%zu) exceeds buffer %zu\n",
                               __func__, dst_off, size, buf_size);
                return false;
            }
            memcpy((char *)tensor->data + offset, data, size);
        } else {
            const size_t buf_size = ggml_backend_buffer_get_size(tensor->buffer);
            if (offset + size > buf_size) {
                GGML_LOG_ERROR("[%s] tensor data region (offset=%" PRIu64 ", size=%zu) exceeds buffer size %zu\n",
                               __func__, offset, size, buf_size);
                return false;
            }
            ggml_backend_tensor_set(tensor, data, offset, size);
        }
    }
    return true;
}

bool rpc_connection::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
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

bool rpc_connection::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
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

    // V0 row-split fix: CUDA split buffers store device pointers in
    // tensor->extra->data_device[id]. Use the buffer's own set_tensor.
    if (rpc_buft_is_cuda_split(tensor->buffer->buft)) {
        tensor->buffer->iface.set_tensor(tensor->buffer, tensor, cached_file.data(), request.offset, size);
        response.result = 1;
        return true;
    }

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

bool rpc_connection::init_tensor(const rpc_msg_init_tensor_req & request) {
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
        // V0 row-split fix: CUDA split buffer's init_tensor populates
        // tensor->extra with device pointers (data_device[id]). This is
        // expected and correct for split buffers — do not reject.
        if (!rpc_buft_is_cuda_split(buffer->buft)) {
            GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
            return false;
        }
    }

    return true;
}

bool rpc_connection::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
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
    bool get_buf_host = tensor->buffer ? ggml_backend_buffer_is_host(tensor->buffer) : false;
    bool get_in_ram = tensor->buffer ? vvram_ram_buffers.count(tensor->buffer) : false;
    // BLK-1: gated behind GGML_VVRAM_CLOSE_DEBUG=1 (default OFF) — GET_TENSOR is
    // the decode hot path; no prints (or raw inference data) by default.
    if (rpc_vvram_close_debug()) {
        fprintf(stderr, "[VVRAM-GET-TENSOR] name=%s buf_host=%d in_ram_set=%d data=%p offset=%" PRIu64 " size=%" PRIu64 " buf_base=%p buf_size=%zu\n",
                tensor->name, (int)get_buf_host, (int)get_in_ram, tensor->data, request.offset, request.size,
                (void*)ggml_backend_buffer_get_base(tensor->buffer), ggml_backend_buffer_get_size(tensor->buffer));
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // [VVRAM-CLOSE] discriminator: was the output buffer freed between
    // VVRAM-DIAG (inside graph_compute) and this GET_TENSOR? If its base is in
    // the freed set -> H-lifetime CONFIRMED (memory freed/reused -> garbage).
    // If NOT freed, direct-deref distinguishes copy-path bug (memory valid here
    // but ggml_backend_tensor_get copy produces garbage) from H-lifetime.
    // BLK-1: the whole discriminator block is gated (hot path). BLK-2: the
    // direct-deref must only run on HOST buffers — tensor->data on a VRAM
    // (device) buffer is a device address; host-derefing it segfaults on the
    // CUDA servers. The gate AND the host guard are both required.
    if (rpc_vvram_close_debug()) {
        bool get_buf_freed = vvram_close_was_freed(tensor->buffer);
        bool get_buf_alive = false;
        {
            std::lock_guard<std::mutex> lock(buffers_mtx);
            get_buf_alive = (buffers.count(tensor->buffer) > 0);
        }
        fprintf(stderr, "[VVRAM-CLOSE] buffer_freed=%d buffer_alive=%d\n", (int)get_buf_freed, (int)get_buf_alive);
        if (get_buf_freed) {
            fprintf(stderr, "[VVRAM-CLOSE] RESULT=H-lifetime (output buffer freed before GET_TENSOR) -> skip deref\n");
        } else if (get_buf_alive && get_buf_host && tensor->data && request.size >= sizeof(float)) {
            const size_t doff = (size_t)((const char *) tensor->data - (const char *) ggml_backend_buffer_get_base(tensor->buffer));
            if (doff + (size_t) request.size <= ggml_backend_buffer_get_size(tensor->buffer)) {
                const float * gdp = (const float *)((const char *) tensor->data + (size_t) request.offset);
                int gn = (int)(request.size / sizeof(float));
                if (gn > 9) gn = 9;
                fprintf(stderr, "[VVRAM-CLOSE] direct-deref first-8=");
                for (int j = 0; j < gn; j++) fprintf(stderr, "%s%g", j ? "," : "", gdp[j]);
                fprintf(stderr, "\n");
            }
        }
    }

    // V0 row-split fix: CUDA split buffers store device pointers in
    // tensor->extra->data_device[id], not in tensor->data. Use the buffer's
    // own get_tensor interface, which handles the split. The CUDA split buffer
    // get_tensor asserts offset==0 and size==ggml_nbytes(tensor), so skip the
    // standard bounds check (which uses the dummy 0x1000 base pointer).
    if (rpc_buft_is_cuda_split(tensor->buffer->buft)) {
        response.resize(request.size, 0);
        tensor->buffer->iface.get_tensor(tensor->buffer, tensor, response.data(), request.offset, request.size);
        return true;
    }

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
    // [VVRAM-CLOSE] print the bytes the server actually copied into the response,
    // to localize server-side copy bug vs in-transit/client-side corruption.
    // BLK-1: gated (hot path).
    if (rpc_vvram_close_debug() && response.size() >= sizeof(float)) {
        const float * rp = (const float *) response.data();
        int rn = (int)(response.size() / sizeof(float));
        if (rn > 9) rn = 9;
        fprintf(stderr, "[VVRAM-CLOSE] response-bytes first-8=");
        for (int j = 0; j < rn; j++) fprintf(stderr, "%s%g", j ? "," : "", rp[j]);
        fprintf(stderr, "\n");
    }
    return true;
}

bool rpc_connection::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
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
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_connection::copy_tensor:\n"
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

bool rpc_connection::copy_tensor_peer(const rpc_msg_copy_tensor_peer_req & request, rpc_msg_copy_tensor_rsp & response) {
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

ggml_tensor * rpc_compute_engine::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map,
                                      const std::unordered_set<ggml_backend_buffer_t> & buffers,
                                      const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor, buffers, split_metas);
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
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map, buffers, split_metas);
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
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map, buffers, split_metas);
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
// Override with GGML_RPC_NODE_SAMPLE_INTERVAL env var (set to 1 for full profiling).
static int rpc_node_sample_interval() {
    static int cached = []() -> int {
        const char * e = getenv("GGML_RPC_NODE_SAMPLE_INTERVAL");
        return e ? atoi(e) : 16;
    }();
    return cached;
}

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

// VVRAM staging pass: copy RAM-tier weight tensors into VRAM before compute.
// For each tensor in the graph that references a RAM-tier buffer, allocate a
// small VRAM staging buffer, copy just that tensor's data, and redirect the
// tensor's data pointer. This per-tensor approach works even when the total
// RAM-tier data far exceeds VRAM (the model is staged layer-by-layer).
// Returns the list of staging buffers (owned by the caller, freed on invalidate).
static std::vector<ggml_backend_buffer_t> vvram_stage_graph_lru(
    struct ggml_cgraph * graph,
    const std::unordered_set<ggml_backend_buffer_t> & ram_buffers,
    ggml_backend_t backend);

// VVRAM LRU: enable the layered compute path (stage + compute + evict per layer).
// Cached — checked once. Used by both graph_compute and graph_recompute so the
// decode loop (GRAPH_RECOMPUTE) takes the same layered path as the first compute.
static bool rpc_vvram_use_lru() {
    static bool cached = []() {
        const char * e = std::getenv("GGML_RPC_VVRAM_LRU");
        return e && e[0] == '1';
    }();
    return cached;
}

// [VVRAM-CLOSE] diagnostic: record every buffer base pointer that gets freed,
// so the get_tensor discriminator can tell whether the output buffer was freed
// between VVRAM-DIAG (inside graph_compute) and GET_TENSOR. A set of freed
// base pointers (with a mutex) shared across cleanup()/free_buffer()/get_tensor.
namespace {
static std::mutex g_vvram_close_free_mtx;
static std::unordered_set<void *> g_vvram_close_freed_bases;
static bool g_vvram_close_recording = true; // set false once discriminator done
}
static void vvram_close_record_free(ggml_backend_buffer_t buffer) {
    if (!buffer || !g_vvram_close_recording) return;
    void * base = (void *) ggml_backend_buffer_get_base(buffer);
    std::lock_guard<std::mutex> lock(g_vvram_close_free_mtx);
    g_vvram_close_freed_bases.insert(base);
}
static bool vvram_close_was_freed(ggml_backend_buffer_t buffer) {
    if (!buffer) return false;
    void * base = (void *) ggml_backend_buffer_get_base(buffer);
    std::lock_guard<std::mutex> lock(g_vvram_close_free_mtx);
    return g_vvram_close_freed_bases.count(base) > 0;
}
// [VVRAM-CLOSE] diagnostics gate (defined here, forward-declared above so
// free_buffer/get_tensor — earlier in the file — can use it). Default OFF.
static bool rpc_vvram_close_debug() {
    static bool cached = []() {
        const char * e = std::getenv("GGML_VVRAM_CLOSE_DEBUG");
        return e && e[0] == '1';
    }();
    return cached;
}

// VVRAM LRU: staging budget as a percentage of total VRAM (env
// GGML_RPC_VVRAM_LRU_BUDGET_PCT, default 60). Cached. Shared by:
//   - vvram_compute_graph_layered — the logical LRU staging budget (max bytes
//     of staged weights the LRU may hold before evicting);
//   - rpc_connection::alloc_buffer — the staging headroom floor (BUG-015), so
//     VRAM-tier weight allocation stops short of the device and the layered
//     decode always has physical room for its per-layer staging buffers.
static int rpc_vvram_lru_budget_pct() {
    static int cached = []() {
        const char * e = std::getenv("GGML_RPC_VVRAM_LRU_BUDGET_PCT");
        int p = e ? atoi(e) : 0;
        return (p > 0 && p <= 100) ? p : 60;
    }();
    return cached;
}

// VVRAM LRU: bytes of VRAM reserved for the layered decode's per-layer staging.
// Derived from the LRU budget pct — reserve (100 - budget_pct)% of total VRAM,
// capped at 6%:
//   - the (100 - budget_pct) term ties the headroom to GGML_RPC_VVRAM_LRU_BUDGET_PCT:
//     the more VRAM the LRU may stage into, the less the alloc gate may consume
//     for resident weights;
//   - the 6% cap keeps models that (just) fit VRAM on the direct compute path —
//     e.g. the 35B on the 24.5 GiB 7900 leaves ~2 GiB free at the end of the
//     weight phase, so a larger floor would spill it into the RAM tier and break
//     byte-identical output. 6% (≈1.5 GiB) comfortably holds the first staged
//     layer (~0.5 GiB for the 72B) plus margin.
static size_t rpc_vvram_staging_headroom(size_t vram_total) {
    const int budget_pct = rpc_vvram_lru_budget_pct();
    size_t headroom_pct = (budget_pct < 100) ? (size_t)(100 - budget_pct) : 0;
    // [VVRAM-DIVERGE] the 6% cap keeps the 35B on the direct path (it fits in
    // VRAM with ~2 GiB spare). To force the layered >VRAM path for the
    // divergence test, raise the cap via GGML_VVRAM_DIVERGE_HEADROOM_CAP.
    int cap = 6;
    const char * cap_env = std::getenv("GGML_VVRAM_DIVERGE_HEADROOM_CAP");
    if (cap_env && cap_env[0]) {
        cap = atoi(cap_env);
        if (cap < 0) cap = 0;
        if (cap > 100) cap = 100;
    }
    if (headroom_pct > (size_t) cap) {
        headroom_pct = (size_t) cap;
    }
    return vram_total * headroom_pct / 100;
}

static std::vector<ggml_backend_buffer_t> vvram_stage_graph(
    struct ggml_cgraph * graph,
    const std::unordered_set<ggml_backend_buffer_t> & ram_buffers,
    ggml_backend_t backend) {

    std::vector<ggml_backend_buffer_t> staging;
    if (!graph || ram_buffers.empty()) {
        return staging;
    }
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(backend);

    // Collect unique (buffer, offset) pairs to avoid staging the same tensor twice.
    struct ram_ref_t {
        ggml_backend_buffer_t buf;
        size_t offset;
        size_t nbytes;
        bool operator==(const ram_ref_t & o) const {
            return buf == o.buf && offset == o.offset && nbytes == o.nbytes;
        }
    };
    struct ref_hash {
        size_t operator()(const ram_ref_t & r) const {
            return std::hash<ggml_backend_buffer_t>{}(r.buf) ^ (std::hash<size_t>{}(r.offset) << 1);
        }
    };
    std::unordered_set<ram_ref_t, ref_hash> seen;

    for (int i = 0; i < (int)graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (!node) continue;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || !src->buffer || !ram_buffers.count(src->buffer)) continue;
            size_t nbytes = ggml_nbytes(src);
            size_t offset = (size_t)src->data - (size_t)ggml_backend_buffer_get_base(src->buffer);
            ram_ref_t ref {src->buffer, offset, nbytes};
            if (seen.count(ref)) continue;
            seen.insert(ref);

            // Allocate a VRAM staging buffer for just this tensor.
            ggml_backend_buffer_t stage = ggml_backend_buft_alloc_buffer(gpu_buft, nbytes);
            if (!stage) {
                GGML_LOG_ERROR("[VVRAM] staging alloc failed (%zu MiB) for tensor %s\n",
                    nbytes / (1024*1024), src->name);
                for (auto s : staging) ggml_backend_buffer_free(s);
                staging.clear();
                return staging;
            }
            // Copy host RAM -> VRAM.
            struct ggml_tensor tmp{};
            tmp.type = GGML_TYPE_F32;
            tmp.ne[0] = (int64_t)((nbytes + sizeof(float) - 1) / sizeof(float));
            tmp.ne[1] = 1; tmp.ne[2] = 1; tmp.ne[3] = 1;
            tmp.nb[0] = sizeof(float);
            tmp.buffer = stage;
            tmp.data = ggml_backend_buffer_get_base(stage);
            ggml_backend_tensor_set(&tmp, (char *)ggml_backend_buffer_get_base(src->buffer) + offset, 0, nbytes);
            staging.push_back(stage);

            // Redirect this tensor (and any aliases) to the staging buffer.
            for (int ii = 0; ii < (int)graph->n_nodes; ii++) {
                struct ggml_tensor * n2 = graph->nodes[ii];
                if (!n2) continue;
                for (int jj = 0; jj < GGML_MAX_SRC; jj++) {
                    struct ggml_tensor * s2 = n2->src[jj];
                    if (s2 && s2->buffer == src->buffer) {
                        size_t o2 = (size_t)s2->data - (size_t)ggml_backend_buffer_get_base(src->buffer);
                        if (o2 == offset) {
                            s2->buffer = stage;
                            s2->data = ggml_backend_buffer_get_base(stage);
                        }
                    }
                }
            }
            GGML_LOG_DEBUG("[VVRAM] staged tensor %s (%zu MiB) RAM -> VRAM\n",
                src->name, nbytes / (1024*1024));
        }
    }
    // Synchronize to ensure all copies complete before compute.
    ggml_backend_synchronize(backend);
    return staging;
}

// VVRAM LRU eviction: extract layer number from a tensor name.
// Matches patterns like:
//   "blk.48.attn_k_norm.weight" -> 48 (weight tensor)
//   "norm-0", "attn_norm-0", "Qcur-0" -> 0 (intermediate tensor)
// Returns -1 if no layer number found.
static int vvram_extract_layer(const char * name) {
    if (!name) return -1;
    // First try "blk.N." pattern (weight tensors).
    const char * p = strstr(name, "blk.");
    if (p) {
        p += 4; // skip "blk."
        if (*p >= '0' && *p <= '9') return atoi(p);
    }
    // Then try "name-N" pattern (intermediate tensors).
    // Find the last dash followed by a number.
    const char * last_dash = strrchr(name, '-');
    if (last_dash && last_dash[1] >= '0' && last_dash[1] <= '9') {
        return atoi(last_dash + 1);
    }
    return -1;
}

// VVRAM LRU eviction: staging entry tracking for LRU eviction.
struct vvram_staging_entry {
    ggml_backend_buffer_t stage;   // VRAM staging buffer
    ggml_backend_buffer_t ram_buf;  // original RAM buffer
    size_t ram_offset;             // offset within RAM buffer
    size_t nbytes;                 // tensor size
    int layer;                     // layer number (-1 for shared)
    uint64_t last_use;             // monotonic counter for LRU
    // L0-T2 (defect A): EVERY tensor redirected to this stage is tracked here,
    // not just the first one. Alias views (e.g. "cache_k_l0" leaf and
    // "cache_k_l0 (view)" node output — same RAM buffer + offset) all get
    // redirected to the same stage, and eviction/cleanup must restore ALL of
    // them to their RAM pointers. Restoring only the first leaves the aliases
    // dangling into the freed stage (use-after-free -> reads return another
    // tensor's data). The first staged tensor is tensors.front().
    std::vector<ggml_tensor *> tensors;
};

// L0-T2 (defect A): restore EVERY tensor redirected to a stage back to its RAM
// pointer. Alias views (same RAM buffer + offset, different tensor object) are
// all tracked in the entry's tensors[]; eviction AND the layered-compute exit
// cleanup must restore all of them, or the aliases keep dangling pointers into
// the freed staging buffer (use-after-free -> reads return another tensor's
// data). Shared by vvram_stage_tensor_lru and vvram_compute_graph_layered.
static void vvram_entry_restore(vvram_staging_entry & e) {
    for (auto * t : e.tensors) {
        t->buffer = e.ram_buf;
        t->data = (char *) ggml_backend_buffer_get_base(e.ram_buf) + e.ram_offset;
    }
}

// VVRAM LRU eviction: stage a single tensor with LRU eviction.
// Maintains a VRAM budget; evicts least-recently-used staging buffers when full.
// Returns the staging buffer, or nullptr on failure.
static ggml_backend_buffer_t vvram_stage_tensor_lru(
    struct ggml_tensor * src,
    ggml_backend_t backend,
    std::vector<vvram_staging_entry> & lru_cache,
    size_t & lru_vram_used,
    size_t lru_vram_budget,
    uint64_t & lru_counter,
    int batch_layer) {

    if (!src || !src->buffer) return nullptr;
    size_t nbytes = ggml_nbytes(src);
    size_t offset = (size_t)src->data - (size_t)ggml_backend_buffer_get_base(src->buffer);
    int layer = vvram_extract_layer(src->name);

    // Check if already staged (dedup by buffer+offset+nbytes). NOTE: the caller
    // only invokes us for tensors whose buffer is still in the RAM-tier set, so
    // this fires for aliases of an already-staged tensor (same RAM buffer +
    // offset, different tensor object) — never for the same object twice.
    // L0-T2 (defect A): the dedup key must include nbytes. Two alias views of
    // the same base can have DIFFERENT extents (e.g. a subview); mapping the
    // larger one to a stage sized for the smaller lets reads/writes run past
    // the end of the staging buffer into a neighbour's stage. With the key
    // fixed, a hit only happens for byte-identical aliases, for which sharing
    // the stage is correct — and every redirected alias is added to the entry's
    // tensors[] so eviction/cleanup restores ALL of them (not just the first).
    for (auto & e : lru_cache) {
        if (e.ram_buf == src->buffer && e.ram_offset == offset && e.nbytes == nbytes) {
            e.last_use = ++lru_counter;
            GGML_LOG_DEBUG("[VVRAM-LRU] hit tensor %s (layer %d) — already staged\n",
                src->name, layer);
            // Redirect tensor to existing staging buffer.
            src->buffer = e.stage;
            src->data = ggml_backend_buffer_get_base(e.stage);
            // Track the alias so eviction/cleanup restores it too.
            bool known = false;
            for (auto * t : e.tensors) {
                if (t == src) { known = true; break; }
            }
            if (!known) {
                e.tensors.push_back(src);
            }
            return e.stage;
        }
    }

    // Evict LRU entries until we have enough VRAM. Two budgets are enforced:
    //   1. the logical LRU budget (lru_vram_budget, default 60% of total VRAM);
    //   2. the physical free VRAM (BUG-015): the alloc gate now reserves staging
    //      headroom, but that headroom (default ~1.5 GiB) is far smaller than the
    //      logical budget — without this guard the LRU would accumulate staged
    //      layers until the device OOMs again (the 72B crashed ~4 RAM-tier layers
    //      in: 4 × ~453 MiB > the reserved headroom).
    // A small margin keeps room for concurrent graph-compute scratch.
    //
    // CRITICAL: only tensors from STRICTLY OLDER BATCHES (entry.layer < the
    // batch currently being staged) may be evicted. Under steady-state memory
    // pressure the oldest entries eventually belong to the CURRENT batch —
    // evicting them mid-staging leaves this batch's subgraph computing against
    // restored RAM pointers (garbage logits -> repeated '?' tokens) and thrashes
    // the eviction loop (the 72B decode staged ~14x the layer count per pass).
    // Entries are tagged with batch_layer (NOT the tensor's extracted layer):
    // shared tensors (output_norm/output.weight, extracted layer -1) are staged
    // inside the last batch and must stay until that batch computes.
    const size_t evict_margin = 64 * 1024 * 1024;
    auto evict_one = [&]() {
        // Find the least-recently-used entry from an older batch.
        size_t lru_idx = (size_t) -1;
        uint64_t lru_time = UINT64_MAX;
        for (size_t i = 0; i < lru_cache.size(); i++) {
            // The batch guard (below) protects the CURRENT batch's staged
            // tensors from being evicted mid-staging: evicting them leaves the
            // batch's subgraph computing against restored RAM pointers.
            // L0-T2 (defect B): during the shared batch (layer -1) there are no
            // strictly-older entries, so the guard would make eviction
            // impossible and any stage-over-budget must fail. When the shared
            // batch's staging exceeds the (physical-free-aware) budget, fall
            // back to evicting the least-recently-used entry of ANY batch —
            // the shared batch's staged leaves are not written by its own
            // compute (KV writes happen in the per-layer batches), so a later
            // re-stage from RAM is correct, and the per-layer batches that do
            // write+read the same staged tensor still cannot evict their own
            // in-flight entries (guard applies for layer >= 0).
            if (batch_layer >= 0 && lru_cache[i].layer >= batch_layer) {
                continue;
            }
            if (lru_cache[i].last_use < lru_time) {
                lru_time = lru_cache[i].last_use;
                lru_idx = i;
            }
        }
        if (lru_idx == (size_t) -1) {
            return false;
        }
        auto & evict = lru_cache[lru_idx];
        GGML_LOG_DEBUG("[VVRAM-LRU] evict tensor %s (%zu MiB, layer %d, %zu redirects) to free VRAM\n",
            evict.tensors.front()->name, evict.nbytes / (1024*1024), evict.layer,
            evict.tensors.size());
        // Restore every redirected tensor's pointer to RAM.
        vvram_entry_restore(evict);
        lru_vram_used -= evict.nbytes;
        ggml_backend_buffer_free(evict.stage);
        lru_cache.erase(lru_cache.begin() + lru_idx);
        return true;
    };

    while (lru_vram_used + nbytes > lru_vram_budget) {
        if (!evict_one()) {
            break;
        }
    }
    ggml_backend_dev_t lru_dev = ggml_backend_get_device(backend);
    for (;;) {
        size_t vram_free = 0, vram_total = 0;
        if (lru_dev) {
            ggml_backend_dev_memory(lru_dev, &vram_free, &vram_total);
        }
        if (vram_free >= nbytes + evict_margin) {
            break;
        }
        if (!evict_one()) {
            break;
        }
    }

    // Allocate VRAM staging buffer.
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t stage = ggml_backend_buft_alloc_buffer(gpu_buft, nbytes);
    if (!stage) {
        GGML_LOG_ERROR("[VVRAM-LRU] alloc failed (%zu MiB) for tensor %s\n",
            nbytes / (1024*1024), src->name);
        return nullptr;
    }

    // Copy RAM -> VRAM.
    struct ggml_tensor tmp{};
    tmp.type = GGML_TYPE_F32;
    tmp.ne[0] = (int64_t)((nbytes + sizeof(float) - 1) / sizeof(float));
    tmp.ne[1] = 1; tmp.ne[2] = 1; tmp.ne[3] = 1;
    tmp.nb[0] = sizeof(float);
    tmp.buffer = stage;
    tmp.data = ggml_backend_buffer_get_base(stage);
    ggml_backend_tensor_set(&tmp, (char *)ggml_backend_buffer_get_base(src->buffer) + offset, 0, nbytes);

    // Redirect tensor to staging buffer.
    // BUG-FIX: capture ram_buf BEFORE redirecting src->buffer. The previous
    // code pushed {stage, src->buffer, ...} after the redirect, so entry.ram_buf
    // held the STAGING buffer — eviction then restored tensors to a just-freed
    // VRAM buffer (use-after-free) instead of their RAM buffer.
    ggml_backend_buffer_t ram_buf = src->buffer;
    src->buffer = stage;
    src->data = ggml_backend_buffer_get_base(stage);

    // Track in LRU cache. ram_buf is the original RAM buffer; eviction and the
    // layered-compute exit path restore tensors through this field. Entries are
    // tagged with batch_layer: the eviction guard only reclaims strictly-older
    // batches (see evict_one above).
    lru_cache.push_back({stage, ram_buf, offset, nbytes, batch_layer, ++lru_counter, {src}});
    lru_vram_used += nbytes;

    GGML_LOG_DEBUG("[VVRAM-LRU] staged tensor %s (%zu MiB, layer %d) RAM -> VRAM (vram_used=%zu/%zu MiB)\n",
        src->name, nbytes / (1024*1024), batch_layer,
        lru_vram_used / (1024*1024), lru_vram_budget / (1024*1024));
    return stage;
}

// VVRAM LRU eviction: build a valid per-layer subgraph for ggml_backend_graph_compute.
// The scaffold cgraph is pre-allocated with capacity for the full graph (nodes +
// hash table sized for all graph tensors). Each call:
//   - resets the scaffold (ggml_graph_clear + zeroed use_counts)
//   - fills nodes[] with the layer's node tensors (a subsequence of the graph's
//     topological order, so the subgraph is topologically valid)
//   - inserts the layer's nodes into the hash table and records their FULL-GRAPH
//     use counts, so fusion checks (ggml_node_get_use_count) are safe and correct
//     (a node consumed by a later layer must not be fused away by this pass)
// A memset'd ggml_cgraph is NOT sufficient: ggml_hash_find() does
// `ggml_hash(key) % hash_set->size` (ggml-impl.h:261) and a zeroed hash set
// (size == 0) crashes the server with SIGFPE; NULL keys/used also fault on the
// first bitset access. This mirrors ggml_new_graph_custom + ggml_visit_parents_graph.
static bool vvram_build_layer_subgraph(
    struct ggml_cgraph * subgraph,
    const struct ggml_cgraph * graph,
    const std::vector<int> & node_indices,
    const std::unordered_map<ggml_tensor *, int32_t> & full_use_counts) {

    ggml_graph_clear(subgraph);
    memset(subgraph->use_counts, 0, subgraph->visited_hash_set.size * sizeof(int32_t));

    const int n = (int) node_indices.size();
    if (n == 0 || n > subgraph->size) {
        return false;
    }

    for (int k = 0; k < n; k++) {
        ggml_tensor * node = graph->nodes[node_indices[k]];
        if (!node) {
            return false;
        }
        subgraph->nodes[k] = node;

        // Insert the node into the hash table. The table is sized for the full
        // graph (hash_size >= 2 * graph->n_nodes), so there is always room.
        const size_t pos = ggml_hash_insert(&subgraph->visited_hash_set, node);
        if (pos == GGML_HASHSET_ALREADY_EXISTS || pos == GGML_HASHSET_FULL) {
            GGML_LOG_ERROR("[VVRAM-LAYERED] hash table error (pos=%zu) building subgraph\n", pos);
            return false;
        }
        auto it = full_use_counts.find(node);
        subgraph->use_counts[pos] = (it != full_use_counts.end()) ? it->second : 0;
    }
    subgraph->n_nodes = n;
    subgraph->order = graph->order;
    return true;
}

// VVRAM LRU eviction: per-layer graph compute.
// Instead of staging ALL tensors at once (OOM for large models), this splits
// the graph into per-layer subgraphs and computes them sequentially. For each
// layer, it stages the layer's tensors (evicting previous layer's), computes
// the layer subgraph, then moves to the next.
//
// This is the key insight: a single decode pass processes layers sequentially.
// At any point, only ONE layer's weights need to be in VRAM (~500 MiB for 72B).
// This leaves plenty of room for activations and intermediate tensors.
static bool vvram_compute_graph_layered(
    struct ggml_cgraph * graph,
    const std::unordered_set<ggml_backend_buffer_t> & ram_buffers,
    ggml_backend_t backend) {

    if (!graph || ram_buffers.empty()) {
        // No RAM-tier buffers, compute normally.
        ggml_status status = ggml_backend_graph_compute(backend, graph);
        return status == GGML_STATUS_SUCCESS;
    }
    if (graph->n_nodes <= 0) {
        return true;
    }

    // Determine VRAM budget: 60% of total VRAM by default (leave room for
    // activations). Override with GGML_RPC_VVRAM_LRU_BUDGET_PCT.
    size_t vram_total = 0, vram_free = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) ggml_backend_dev_memory(dev, &vram_free, &vram_total);
    const int budget_pct = rpc_vvram_lru_budget_pct();
    size_t vram_budget = (vram_total * (size_t) budget_pct) / 100;
    // L0-T2 (defect B): the logical budget must never exceed the PHYSICALLY
    // available VRAM. At low headroom caps the weights fill VRAM and the
    // 60%-of-total budget (14.7 GiB on the 7900) far exceeds what cudaMalloc
    // can satisfy — the shared layer's 4 GiB KV staging OOMed against
    // weight-filled VRAM (CAP=10: cache_k_l15, CAP=20: cache_v_l19) even though
    // lru_vram_used was far below the logical budget. Capping the budget at the
    // free VRAM (measured at pass start) makes the eviction loop engage while
    // there is still something to reclaim instead of failing the first alloc.
    if (vram_free < vram_budget) {
        vram_budget = vram_free;
    }
    GGML_LOG_INFO("[VVRAM-LAYERED] VRAM budget = %zu MiB (total=%zu MiB free=%zu MiB, %d%%)\n",
        vram_budget / (1024*1024), vram_total / (1024*1024), vram_free / (1024*1024), budget_pct);

    // --- Layer classification ------------------------------------------------
    // Assign each node the MAXIMUM layer number across its source tensors.
    // (The previous implementation took the FIRST source with a layer number,
    // which systematically mis-assigned most layer-N compute to layer N-1, since
    // the first src of a layer-N node is usually the previous layer's residual
    // stream, e.g. "cur-N-1".)
    // Nodes with no layer-numbered source are shared (layer -1). These appear at
    // the graph START (embeddings / input) and graph END (output / final ops):
    //   - pre-nodes (idx <= last_nonneg): embeddings batch, computed first
    //   - post-nodes (idx > last_nonneg): output batch, folded into the LAST
    //     layer's batch so their inputs (final-layer intermediates) are ready.
    //
    // BUG-015c: the layer is taken from BOTH the src NAMES (blk.N.* weights,
    // norm-N / l_out-N activations) AND the ASSIGNED layers of src tensors that
    // are themselves graph nodes (dependency-chain propagation). Name-parsing
    // alone misplaces anonymous residual-stream tensors ("node_XXXX", RPC leaf
    // names): they carry no layer in their name, so chain links like
    //   ffn_inp-79 = add(node_2793, ...)   // node_2793 = add(node_2792, leaf)
    // classify to layer -1 and get computed in the FIRST batch, before the
    // mid-graph tensors they depend on exist -> garbage -> NaN logits.
    // The graph is topologically ordered, so every src that is a node output
    // has already been classified when we process node i.
    std::map<int, std::vector<int>> layer_nodes;
    int last_nonneg = -1;
    std::unordered_map<ggml_tensor *, int> assigned_layer;
    assigned_layer.reserve((size_t) graph->n_nodes * 2);
    for (int i = 0; i < (int)graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (!node) continue;
        int layer = -1; // default: shared
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                int l = vvram_extract_layer(node->src[j]->name);
                if (l > layer) { layer = l; }
                // Propagate the assigned layer of src tensors that are node
                // outputs (leaf tensors — weights, KV, inputs — have no layer).
                auto it = assigned_layer.find(node->src[j]);
                if (it != assigned_layer.end() && it->second > layer) {
                    layer = it->second;
                }
            }
        }
        if (layer >= 0 && i > last_nonneg) { last_nonneg = i; }
        layer_nodes[layer].push_back(i);
        assigned_layer[node] = layer;
    }

    const int max_layer = (int) layer_nodes.rbegin()->first;
    auto shared_it = layer_nodes.find(-1);
    if (shared_it != layer_nodes.end() && max_layer >= 0) {
        std::vector<int> pre_nodes, post_nodes;
        for (int idx : shared_it->second) {
            if (idx > last_nonneg) {
                post_nodes.push_back(idx);
            } else {
                pre_nodes.push_back(idx);
            }
        }
        if (!post_nodes.empty()) {
            auto & dst = layer_nodes[max_layer];
            dst.insert(dst.end(), post_nodes.begin(), post_nodes.end());
        }
        if (pre_nodes.empty()) {
            layer_nodes.erase(shared_it);
        } else {
            shared_it->second = std::move(pre_nodes);
        }
    }

    GGML_LOG_INFO("[VVRAM-LAYERED] graph has %zu layers (including shared)\n", layer_nodes.size());
    // Debug: show layer distribution.
    for (auto & [layer, node_indices] : layer_nodes) {
        GGML_LOG_INFO("[VVRAM-LAYERED]   layer %d: %zu nodes\n", layer, node_indices.size());
        if (layer == -1 || layer == 0 || layer == 79) {
            for (int idx : node_indices) {
                struct ggml_tensor * node = graph->nodes[idx];
                if (node && node->name[0]) {
                    GGML_LOG_INFO("[VVRAM-LAYERED]     node %d: %s (op=%d)\n", idx, node->name, (int) node->op);
                }
            }
        }
    }

    // Full-graph use counts for the subgraph hash tables (see vvram_build_layer_subgraph).
    std::unordered_map<ggml_tensor *, int32_t> full_use_counts;
    full_use_counts.reserve((size_t) graph->n_nodes * 2);
    for (int i = 0; i < (int)graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (!node) continue;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                full_use_counts[node->src[j]]++;
            }
        }
    }

    // Subgraph scaffold: a fully initialized ggml_cgraph (hash table sized for
    // the full graph) allocated from its own scratch context. Rebuilt per batch.
    const size_t scratch_size = ggml_graph_overhead_custom((size_t) graph->n_nodes, false);
    std::vector<uint8_t> scratch(scratch_size);
    struct ggml_init_params sparams = {
        /*.mem_size   =*/ scratch_size,
        /*.mem_buffer =*/ scratch.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr sctx { ggml_init(sparams) };
    GGML_ASSERT(sctx != nullptr);
    struct ggml_cgraph * subgraph = ggml_new_graph_custom(sctx.get(), (size_t) graph->n_nodes, false);

    // LRU cache state.
    std::vector<vvram_staging_entry> lru_cache;
    size_t lru_vram_used = 0;
    uint64_t lru_counter = 0;

    // Restore every staged tensor to its RAM buffer and free the staging VRAM.
    // The stored graph is reused by GRAPH_RECOMPUTE (the decode loop), which
    // re-runs this layered path — it must always start from RAM pointers, never
    // from dangling (freed) staging buffers.
    auto cleanup = [&]() {
        for (auto & e : lru_cache) {
            // L0-T2 (defect A): restore ALL tensors redirected to this stage,
            // not just the first — alias views would otherwise keep dangling
            // pointers into the freed staging buffer.
            vvram_entry_restore(e);
            // N7: gated behind GGML_VVRAM_CLOSE_DEBUG=1 (default OFF).
            if (rpc_vvram_close_debug()) {
                fprintf(stderr, "[VVRAM-CLOSE-FREE] cleanup-free stage base=%p size=%zu name=%s layer=%d redirects=%zu\n",
                        (void*)ggml_backend_buffer_get_base(e.stage), ggml_backend_buffer_get_size(e.stage),
                        e.tensors.front()->name, e.layer, e.tensors.size());
            }
            vvram_close_record_free(e.stage);
            ggml_backend_buffer_free(e.stage);
        }
        lru_cache.clear();
    };

    // Process layers in order: -1 (embeddings) first, then 0..max_layer.
    for (auto & [layer, node_indices] : layer_nodes) {
        if (node_indices.empty()) continue;
        GGML_LOG_DEBUG("[VVRAM-LAYERED] processing layer %d (%zu nodes)\n", layer, node_indices.size());

        // Guard: the whole batch must fit the VRAM budget. LRU eviction only
        // evicts the least-recently-used entry; if a single batch's weights
        // exceeded the budget, staging would evict the batch's own earlier
        // tensors and compute would read host pointers. (72B layers are
        // ~500 MiB << budget, so this is defensive.)
        size_t batch_weight_bytes = 0;
        std::unordered_set<ggml_tensor *> seen_srcs;
        for (int idx : node_indices) {
            struct ggml_tensor * node = graph->nodes[idx];
            if (!node) continue;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (!src || !src->buffer || !ram_buffers.count(src->buffer)) continue;
                if (!seen_srcs.insert(src).second) continue;
                batch_weight_bytes += ggml_nbytes(src);
            }
        }
        if (batch_weight_bytes > vram_budget && layer >= 0) {
            // L0-T2 (defect B): the guard now runs against the PHYSICAL-FREE
            // aware budget (min of pct×total and free VRAM). At low headroom
            // caps the budget is far smaller than any single batch's weights,
            // and for the shared batch (-1) the eviction-during--1 mechanism
            // exists precisely to stage+churn a batch larger than the budget
            // (KV cache, shared weights). The guard is kept only for per-layer
            // batches (>= 0), where the batch guard forbids evicting the
            // batch's own tensors — there a batch larger than the budget is
            // genuinely unstageable and failing fast beats an OOM loop.
            GGML_LOG_ERROR("[VVRAM-LAYERED] layer %d weights (%zu MiB) exceed VRAM budget (%zu MiB) — cannot stage safely\n",
                layer, batch_weight_bytes / (1024*1024), vram_budget / (1024*1024));
            cleanup();
            return false;
        }

        // Stage all RAM-tier tensors needed by this layer's nodes (LRU eviction
        // of previous batches when the budget is exceeded).
        for (int idx : node_indices) {
            struct ggml_tensor * node = graph->nodes[idx];
            if (!node) continue;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (!src || !src->buffer || !ram_buffers.count(src->buffer)) continue;

                ggml_backend_buffer_t stage = vvram_stage_tensor_lru(
                    src, backend, lru_cache, lru_vram_used, vram_budget, lru_counter, layer);
                if (!stage) {
                    GGML_LOG_ERROR("[VVRAM-LAYERED] staging failed for %s (layer %d)\n",
                        src->name, layer);
                    cleanup();
                    return false;
                }
            }
        }

        // Synchronize to ensure staging copies complete.
        ggml_backend_synchronize(backend);

        // Build a VALID subgraph (hash table + use_counts) and compute it.
        if (!vvram_build_layer_subgraph(subgraph, graph, node_indices, full_use_counts)) {
            GGML_LOG_ERROR("[VVRAM-LAYERED] failed to build subgraph for layer %d\n", layer);
            cleanup();
            return false;
        }
        ggml_status status = ggml_backend_graph_compute(backend, subgraph);
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("[VVRAM-LAYERED] compute failed for layer %d (%zu nodes)\n",
                layer, node_indices.size());
            cleanup();
            return false;
        }
        // BUG-015b (I2-VVRAM-STAGING-HEADROOM): the compute above is ASYNC — the
        // next batch's staging immediately evicts (frees) this batch's staged
        // weight buffers, and a freed buffer can be reused/overwritten while this
        // batch's kernels are still reading it (garbage activations -> NaN logits
        // -> repeated '?' output). Synchronize before the next batch's evictions.
        ggml_backend_synchronize(backend);
    }

    // TEMP DIAG: dump the final output tensor (logits) values after the layered pass.
    // N6: gated behind GGML_VVRAM_CLOSE_DEBUG=1 (diagnosis complete; no default prints).
    if (graph->n_nodes > 0 && rpc_vvram_close_debug()) {
        struct ggml_tensor * last = graph->nodes[graph->n_nodes - 1];
        if (last && last->data && last->type == GGML_TYPE_F32) {
            float * dp = (float *) last->data;
            bool last_in_ram = (last->buffer && ram_buffers.count(last->buffer)) ? true : false;
            bool last_buf_host = last->buffer ? ggml_backend_buffer_is_host(last->buffer) : false;
            fprintf(stderr, "[VVRAM-DIAG] last node=%s op=%d ne0=%" PRId64 " ne1=%" PRId64 " buf_host=%d in_ram_set=%d data=%p data[0..8]=",
                    last->name, (int) last->op, last->ne[0], last->ne[1], (int)last_buf_host, (int)last_in_ram, (void*)last->data);
            int n = last->ne[0] < 9 ? (int) last->ne[0] : 9;
            for (int j = 0; j < n; ++j) {
                fprintf(stderr, "%s%g", j ? "," : "", dp[j]);
            }
            fprintf(stderr, "\n");
        }
        // Also dump the l_out (final hidden state) of layer 79 if found.
        for (int i = graph->n_nodes - 1; i >= 0 && i > (int) graph->n_nodes - 20; --i) {
            struct ggml_tensor * t = graph->nodes[i];
            if (t && t->data && t->type == GGML_TYPE_F32 && strncmp(t->name, "l_out-79", 8) == 0) {
                float * dp = (float *) t->data;
                fprintf(stderr, "[VVRAM-DIAG] hidden l_out-79 ne0=%" PRId64 " ne1=%" PRId64 " data[0..4]=",
                        t->ne[0], t->ne[1]);
                int n = t->ne[0] < 5 ? (int) t->ne[0] : 5;
                for (int j = 0; j < n; ++j) {
                    fprintf(stderr, "%s%g", j ? "," : "", dp[j]);
                }
                fprintf(stderr, "\n");
                break;
            }
        }
    }
    GGML_LOG_INFO("[VVRAM-LAYERED] completed %zu layers (peak staged vram_used=%zu MiB)\n",
        layer_nodes.size(), lru_vram_used / (1024*1024));

    // Free all staging buffers and restore tensors to their RAM buffers.
    cleanup();

    return true;
}

// VVRAM LRU eviction: per-layer staging pass (legacy, kept for reference).
// This stages ALL tensors with LRU eviction but still requires all tensors
// during compute. Use vvram_compute_graph_layered instead for large models.
static std::vector<ggml_backend_buffer_t> vvram_stage_graph_lru(
    struct ggml_cgraph * graph,
    const std::unordered_set<ggml_backend_buffer_t> & ram_buffers,
    ggml_backend_t backend) {

    std::vector<ggml_backend_buffer_t> staging;
    if (!graph || ram_buffers.empty()) {
        return staging;
    }

    // Determine VRAM budget: 80% of total VRAM (leave room for activations).
    size_t vram_total = 0, vram_free = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) ggml_backend_dev_memory(dev, &vram_free, &vram_total);
    size_t vram_budget = (vram_total * 8) / 10; // 80% of total
    GGML_LOG_INFO("[VVRAM-LRU] VRAM budget = %zu MiB (total=%zu MiB)\n",
        vram_budget / (1024*1024), vram_total / (1024*1024));

    // Group nodes by layer. Nodes with no layer (shared tensors like embeddings)
    // go to layer -1 and are staged once at the start.
    // We classify each node by examining ALL its source tensors and picking
    // the maximum layer number. This correctly handles intermediate tensors
    // (like attn_out-0) that don't have a layer number but are produced by
    // layer-specific nodes.
    std::map<int, std::vector<int>> layer_nodes;
    for (int i = 0; i < (int)graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (!node) continue;
        int layer = -1; // default: shared
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                int l = vvram_extract_layer(node->src[j]->name);
                if (l > layer) { layer = l; }
            }
        }
        layer_nodes[layer].push_back(i);
    }

    GGML_LOG_INFO("[VVRAM-LRU] graph has %zu layers (including shared)\n", layer_nodes.size());

    // LRU cache state.
    std::vector<vvram_staging_entry> lru_cache;
    size_t lru_vram_used = 0;
    uint64_t lru_counter = 0;

    // Process layers in order.
    for (auto & [layer, node_indices] : layer_nodes) {
        GGML_LOG_DEBUG("[VVRAM-LRU] processing layer %d (%zu nodes)\n", layer, node_indices.size());

        // Stage all RAM-tier tensors needed by this layer's nodes.
        for (int idx : node_indices) {
            struct ggml_tensor * node = graph->nodes[idx];
            if (!node) continue;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (!src || !src->buffer || !ram_buffers.count(src->buffer)) continue;

                // Stage with LRU eviction.
                ggml_backend_buffer_t stage = vvram_stage_tensor_lru(
                    src, backend, lru_cache, lru_vram_used, vram_budget, lru_counter, layer);
                if (stage) {
                    // Track for later cleanup (dedup).
                    bool found = false;
                    for (auto s : staging) { if (s == stage) { found = true; break; } }
                    if (!found) staging.push_back(stage);
                }
            }
        }
    }

    // Synchronize to ensure all copies complete before compute.
    ggml_backend_synchronize(backend);
    GGML_LOG_INFO("[VVRAM-LRU] staged %zu tensors (vram_used=%zu MiB)\n",
        staging.size(), lru_vram_used / (1024*1024));
    return staging;
}

bool rpc_compute_engine::graph_compute(const std::vector<uint8_t> & input,
                       const std::unordered_set<ggml_backend_buffer_t> & buffers,
                       const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas,
                       const std::unordered_set<ggml_backend_buffer_t> & ram_buffers) {
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
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map, buffers, split_metas);

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

    // VVRAM: staging pass — copy RAM-tier weight tensors into VRAM before compute.
    // The staging buffers are stored alongside the cached graph so recompute can
    // reuse them without re-copying. They are freed on invalidate/destructor.
    stored_graphs[device].vvram_staging_buffers.clear();
    if (!ram_buffers.empty()) {
        // Use layered compute path if GGML_RPC_VVRAM_LRU=1 is set.
        if (rpc_vvram_use_lru()) {
            // Layered compute: stage + compute + evict per layer.
            // This is slower (per-node sync) but avoids OOM.
            // Snapshot the RAM-tier set so GRAPH_RECOMPUTE (the decode loop)
            // can route through the same layered path.
            stored_graphs[device].vvram_ram_buffers = ram_buffers;
            GGML_LOG_INFO("[VVRAM] using layered compute (LRU) for RAM-tier buffers\n");
            // Staging and compute happen in vvram_compute_graph_layered.
            // We skip the normal staging pass and compute below.
        } else {
            stored_graphs[device].vvram_ram_buffers.clear();
            stored_graphs[device].vvram_staging_buffers = vvram_stage_graph(graph, ram_buffers, backends[device]);
            if (!stored_graphs[device].vvram_staging_buffers.empty()) {
                GGML_LOG_INFO("[VVRAM] staged %zu RAM-tier buffers for compute\n",
                    stored_graphs[device].vvram_staging_buffers.size());
            }
        }
    } else {
        stored_graphs[device].vvram_ram_buffers.clear();
    }

    // issue 12: on sampled decodes, compute per-node for placement-grade heatmaps.
    // Sampled to bound overhead; produces correct output (same ops + order).
    uint64_t us = 0;
    const uint64_t sample_id = telemetry_node_sample_count.fetch_add(1, std::memory_order_relaxed);
    const int interval = rpc_node_sample_interval();
    const bool sample_nodes = telemetry_enabled && (sample_id % interval) == 0;
    std::vector<rpc_node_timing> node_timings;

    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status;
    if (rpc_vvram_use_lru() && !ram_buffers.empty()) {
        // Layered compute: stage + compute + evict per layer.
        // This is slower (per-node sync) but avoids OOM for large models.
        bool ok = vvram_compute_graph_layered(graph, ram_buffers, backends[device]);
        status = ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    } else if (sample_nodes) {
        us = compute_graph_per_node(backends[device], graph, node_timings);
        status = GGML_STATUS_SUCCESS;
    } else {
        status = ggml_backend_graph_compute(backends[device], graph);
        us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    rpc_trace_emit("rpc_compute_engine::graph_compute", "server_compute", RPC_CMD_GRAPH_COMPUTE, input.size(), true, us);
    // T2d: "free old graph on overwrite" — the previous stored_graphs[device].graph
    // (if any) is pool-allocated: its bytes live inside stored_graphs[device].buffer,
    // and ggml_init() on the next graph_compute() resets the pool offset to 0 so the
    // new graph reclaims that memory in place. There is no separate heap object to
    // free — calling free()/delete here would corrupt the pool. Dropping the pointer
    // (by overwriting it below) is the correct "free". This is single-source ownership:
    // the buffer pool owns the bytes, the pointer is a non-owning view. Verified no
    // double-free: drain_and_invalidate() also only nulls pointers (never frees).
    stored_graphs[device].graph = graph;
    // F1 (T2a): reset the bound uid for this slot. The uid is learned lazily on
    // the first GRAPH_RECOMPUTE after a fresh compute; resetting here guarantees a
    // new graph (post-eviction or post-recompute-miss fallback) re-binds cleanly.
    // NIT-1: uid is atomic — release so dispatch threads see the reset.
    stored_graphs[device].uid.store(0, std::memory_order_release);
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

bool rpc_compute_engine::recompute_allowed(const rpc_msg_graph_recompute_req & request) const {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    // NIT-1: uid is std::atomic<uint64_t> — read with acquire ordering so the
    // dispatch thread sees the latest value written by the compute worker.
    uint64_t slot_uid = stored_graphs[device].uid.load(std::memory_order_acquire);
    // F1 (T2a): uid gate. A zero hash means an old (pre-T2a) client that does
    // not send a hash — treat as unknown and allow (legacy behavior). Non-zero
    // hash must match the bound uid, or the slot must be unbound (lazy-bind on
    // the actual recompute). A mismatch means the cached graph is not the one the
    // client expects → signal MISS so the client falls back to GRAPH_COMPUTE.
    if (request.graph_hash != 0 && slot_uid != 0 && slot_uid != request.graph_hash) {
        return false;
    }
    return true;
}

bool rpc_compute_engine::recompute_all_allowed(const rpc_msg_graph_recompute_all_req & request) const {
    if (all_graph.graph == nullptr) {
        return false;
    }
    // NIT-1: uid is atomic — read with acquire ordering.
    uint64_t slot_uid = all_graph.uid.load(std::memory_order_acquire);
    // F1 (T2a): uid gate for the multi-device slot. Zero hash → old client, allow.
    if (request.graph_hash != 0 && slot_uid != 0 && slot_uid != request.graph_hash) {
        return false;
    }
    return true;
}

bool rpc_compute_engine::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    // F1 (T2a): uid verification. Lazy-bind on first recompute for this slot;
    // on mismatch the cached graph is NOT the one the client expects → return
    // false so the client falls back to a full GRAPH_COMPUTE. A zero hash means
    // the client is old (pre-T2a) and did not send a hash: skip verification.
    // NIT-1: uid is atomic — use acquire/release for proper synchronization
    // with the compute worker thread that resets uid in graph_compute().
    if (request.graph_hash != 0) {
        uint64_t slot_uid = stored_graphs[device].uid.load(std::memory_order_acquire);
        if (slot_uid == 0) {
            // First recompute since the last graph_compute for this slot: bind.
            stored_graphs[device].uid.store(request.graph_hash, std::memory_order_release);
        } else if (slot_uid != request.graph_hash) {
            LOG_DBG("[%s] device: %u uid mismatch: stored=%" PRIu64 " req=%" PRIu64 " (recompute miss)\n",
                    __func__, device, slot_uid, request.graph_hash);
            return false;
        }
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u uid=%" PRIu64 "\n", __func__, device, stored_graphs[device].uid.load(std::memory_order_relaxed));

    // E-3 probe (BUG-002a): on every recompute, log the cached graph's I32 leaf
    // tensors (tokens + s_copy, the snapshot-plane index vector) so we can see
    // which plane the replayed graph actually reads after an MTP rollback.
    // M1: server sees plane-0 indices while client computed rs_idx != 0 (stale
    // binding). M2: server sees the rollback indices but output is still wrong.
    // No behavior change. Keep only for diagnosis (may be reverted after E-1).
    {
        const ggml_cgraph * cg = stored_graphs[device].graph;
        fprintf(stderr, "[E3] recompute device=%u uid=%" PRIu64 " n_leafs=%d n_nodes=%d\n",
                device, stored_graphs[device].uid.load(std::memory_order_relaxed),
                cg ? cg->n_leafs : -1, cg ? cg->n_nodes : -1);
        if (cg) {
            for (int i = 0; i < cg->n_leafs && i < 24; ++i) {
                const ggml_tensor * t = cg->leafs[i];
                if (t == NULL) {
                    continue;
                }
                if (t->type == GGML_TYPE_I32 && t->ne[1] == 1) {
                    const int32_t * data = (const int32_t *) t->data;
                    fprintf(stderr, "[E3] leaf[%d] name=%s ne0=%" PRId64 " data=[", i, t->name, t->ne[0]);
                    const int N = t->ne[0] < 12 ? (int) t->ne[0] : 12;
                    for (int j = 0; j < N; ++j) {
                        fprintf(stderr, "%s%d", j ? "," : "", data ? data[j] : -999);
                    }
                    fprintf(stderr, "]\n");
                }
            }
        }
    }

    // issue 12: per-node timing on recompute path (same sampling as graph_compute).
    uint64_t us = 0;
    const uint64_t sample_id = telemetry_node_sample_count.fetch_add(1, std::memory_order_relaxed);
    const int interval = rpc_node_sample_interval();
    const bool sample_nodes = telemetry_enabled && (sample_id % interval) == 0;
    std::vector<rpc_node_timing> node_timings;

    const auto t0 = std::chrono::steady_clock::now();
    ggml_status status;
    if (rpc_vvram_use_lru() && !stored_graphs[device].vvram_ram_buffers.empty()) {
        // VVRAM LRU: the decode loop must take the same layered path as
        // graph_compute. The stored graph's weight tensors were restored to
        // their RAM buffers at the end of the previous layered pass, so we must
        // stage + compute + evict per layer again (computing directly against
        // RAM/dangling pointers would crash).
        bool ok = vvram_compute_graph_layered(graph, stored_graphs[device].vvram_ram_buffers, backends[device]);
        status = ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    } else if (sample_nodes) {
        us = compute_graph_per_node(backends[device], graph, node_timings);
        status = GGML_STATUS_SUCCESS;
    } else {
        status = ggml_backend_graph_compute(backends[device], graph);
        us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    rpc_trace_emit("rpc_compute_engine::graph_recompute", "server_compute", RPC_CMD_GRAPH_RECOMPUTE, 0, true, us);
    // issue 12: emit per-node timings for this sampled decode.
    if (sample_nodes && !node_timings.empty()) {
        rpc_write_node_timings_jsonl(node_timings);
    }
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

ggml_backend_sched_t rpc_compute_engine::create_multi_device_sched(
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

bool rpc_compute_engine::graph_compute_all(const std::vector<uint8_t> & input,
                           const std::unordered_set<ggml_backend_buffer_t> & buffers,
                           const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas) {
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
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map, buffers, split_metas);
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
    rpc_trace_emit("rpc_compute_engine::graph_compute_all", "server_compute",
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
    // T2d: "free old graph on overwrite" — same pool-allocation semantics as the
    // single-device path above. The previous all_graph.graph (if any) bytes live
    // in all_graph.buffer; resizing + std::copy below reclaims that memory in
    // place. No separate heap object to free — the buffer pool is single-source
    // ownership. Verified no double-free against drain_and_invalidate().
    if (all_graph.buffer.size() < buf_size) {
        all_graph.buffer.resize(buf_size);
    }
    std::copy(ctx_buf.begin(), ctx_buf.end(), all_graph.buffer.begin());
    all_graph.graph = graph;
    // F1 (T2a): reset bound uid; re-learned lazily on first GRAPH_RECOMPUTE_ALL.
    // NIT-1: uid is atomic — release so dispatch threads see the reset.
    all_graph.uid.store(0, std::memory_order_release);

    // NOTE: sched is cached in all_scheds, freed in ~rpc_server
    return true;
}

bool rpc_compute_engine::graph_recompute_all(const rpc_msg_graph_recompute_all_req & request) {
    if (all_graph.graph == nullptr) {
        return false;
    }
    // F1 (T2a): uid verification for the multi-device path. Lazy-bind on first
    // recompute after a fresh compute; mismatch → client falls back to GRAPH_COMPUTE_ALL.
    // NIT-1: uid is atomic — use acquire/release for proper synchronization.
    if (request.graph_hash != 0) {
        uint64_t slot_uid = all_graph.uid.load(std::memory_order_acquire);
        if (slot_uid == 0) {
            all_graph.uid.store(request.graph_hash, std::memory_order_release);
        } else if (slot_uid != request.graph_hash) {
            LOG_DBG("[%s] uid mismatch: stored=%" PRIu64 " req=%" PRIu64 " (recompute_all miss)\n",
                    __func__, slot_uid, request.graph_hash);
            return false;
        }
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
    rpc_trace_emit("rpc_compute_engine::graph_recompute_all", "server_compute",
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
bool rpc_compute_engine::graph_compute_stage(const std::vector<uint8_t> & input, uint32_t stage_id,
                             const std::unordered_set<ggml_backend_buffer_t> & buffers,
                             const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas) {
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
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map, buffers, split_metas);
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
    rpc_trace_emit("rpc_compute_engine::graph_compute_stage", "server_compute_stage",
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

bool rpc_connection::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= engine->get_backends().size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(engine->get_backends()[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    // VVRAM: report virtual free/total = real VRAM + remaining RAM budget.
    // The scheduler uses this to decide how much it can allocate; the RAM budget
    // is "free" address space that the backend can tier into.
    size_t ram_free = 0;
    if (vvram_ram_budget > 0) {
        ram_free = (vvram_ram_budget > vvram_ram_used) ? (vvram_ram_budget - vvram_ram_used) : 0;
        response.free_mem = free + ram_free;
        response.total_mem = total + vvram_ram_budget;
    } else {
        response.free_mem = free;
        response.total_mem = total;
    }
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 " (vram_free=%zu + ram_free=%zu), total_mem: %" PRIu64 "\n",
        __func__, dev_id, response.free_mem, free, ram_free, response.total_mem);
    return true;
}

void rpc_compute_engine::submit_compute_job(std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lock(compute_mtx);
        compute_inflight++;
        compute_queue.push_back(std::move(job));
    }
    compute_cv.notify_one();
}

void rpc_compute_engine::compute_worker_loop() {
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

void rpc_compute_engine::enqueue_graph_compute(std::vector<uint8_t> input,
                               const std::unordered_set<ggml_backend_buffer_t> & buffers,
                               const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas,
                               const std::unordered_set<ggml_backend_buffer_t> & ram_buffers) {
    // Capture buffers + split_metas + ram_buffers by value: the connection that
    // owns them may be destroyed before the compute job runs (shared engine).
    submit_compute_job([this, input = std::move(input),
                        buffers = std::unordered_set<ggml_backend_buffer_t>(buffers),
                        split_metas = std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta>(split_metas),
                        ram_buffers = std::unordered_set<ggml_backend_buffer_t>(ram_buffers)]() mutable {
        if (!graph_compute(input, buffers, split_metas, ram_buffers)) {
            GGML_LOG_ERROR("[%s] async graph_compute failed\n", __func__);
        }
    });
}

void rpc_compute_engine::enqueue_graph_recompute(rpc_msg_graph_recompute_req request) {
    submit_compute_job([this, request]() {
        if (!graph_recompute(request)) {
            GGML_LOG_ERROR("[%s] async graph_recompute failed\n", __func__);
        }
    });
}

void rpc_compute_engine::enqueue_graph_compute_all(std::vector<uint8_t> input,
                                   const std::unordered_set<ggml_backend_buffer_t> & buffers,
                                   const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas) {
    // Capture buffers + split_metas by value: the connection that owns them may
    // be destroyed before the compute job runs (shared engine across connections).
    submit_compute_job([this, input = std::move(input),
                        buffers = std::unordered_set<ggml_backend_buffer_t>(buffers),
                        split_metas = std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta>(split_metas)]() {
        if (!graph_compute_all(input, buffers, split_metas)) {
            GGML_LOG_ERROR("[%s] async graph_compute_all failed\n", __func__);
        }
    });
}

void rpc_compute_engine::enqueue_graph_recompute_all(rpc_msg_graph_recompute_all_req request) {
    submit_compute_job([this, request]() {
        if (!graph_recompute_all(request)) {
            GGML_LOG_ERROR("[%s] async graph_recompute_all failed\n", __func__);
        }
    });
}

// NIT-2: atomic check+enqueue for GRAPH_RECOMPUTE. Locks compute_mtx, verifies
// the cached graph slot, and enqueues the recompute job — all under the same
// lock. This prevents the multi-client race where recompute_allowed() and
// enqueue_graph_recompute() could be split by a concurrent graph_compute()
// uid-reset. Returns true if the recompute was enqueued (hit), false if the
// caller should fall back to GRAPH_COMPUTE (miss).
bool rpc_compute_engine::try_enqueue_graph_recompute(const rpc_msg_graph_recompute_req & request) {
    std::lock_guard<std::mutex> lock(compute_mtx);
    if (!recompute_allowed(request)) {
        return false;
    }
    // Directly enqueue the job (we already hold the lock, so we can't call
    // submit_compute_job which also tries to lock). Inline the enqueue.
    compute_inflight++;
    compute_queue.push_back([this, request]() {
        if (!graph_recompute(request)) {
            GGML_LOG_ERROR("[%s] async graph_recompute failed\n", __func__);
        }
    });
    compute_cv.notify_one();
    return true;
}

// NIT-2: atomic check+enqueue for GRAPH_RECOMPUTE_ALL (multi-device path).
bool rpc_compute_engine::try_enqueue_graph_recompute_all(const rpc_msg_graph_recompute_all_req & request) {
    std::lock_guard<std::mutex> lock(compute_mtx);
    if (!recompute_all_allowed(request)) {
        return false;
    }
    compute_inflight++;
    compute_queue.push_back([this, request]() {
        if (!graph_recompute_all(request)) {
            GGML_LOG_ERROR("[%s] async graph_recompute_all failed\n", __func__);
        }
    });
    compute_cv.notify_one();
    return true;
}

// D6.9: enqueue per-stage compute with split filtering
void rpc_compute_engine::enqueue_graph_compute_stage(std::vector<uint8_t> input, uint32_t stage_id,
                                     const std::unordered_set<ggml_backend_buffer_t> & buffers,
                                     const std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta> & split_metas) {
    // Capture buffers + split_metas by value: the connection that owns them may
    // be destroyed before the compute job runs (shared engine across connections).
    submit_compute_job([this, input = std::move(input), stage_id,
                        buffers = std::unordered_set<ggml_backend_buffer_t>(buffers),
                        split_metas = std::unordered_map<ggml_backend_buffer_t, rpc_split_buffer_meta>(split_metas)]() {
        if (!graph_compute_stage(input, stage_id, buffers, split_metas)) {
            GGML_LOG_ERROR("[%s] async graph_compute_stage(%u) failed\n", __func__, stage_id);
        }
    });
}

void rpc_compute_engine::wait_compute_idle() {
    std::unique_lock<std::mutex> lock(compute_mtx);
    compute_cv.wait(lock, [this] {
        return compute_queue.empty() && compute_inflight.load() == 0;
    });
}

void rpc_compute_engine::drain_and_invalidate() {
    std::unique_lock<std::mutex> lock(compute_mtx);
    // Wait until the worker has drained. While we hold compute_mtx here, no
    // other thread can call submit_compute_job (it needs compute_mtx), so once
    // the queue is empty and inflight==0, the system stays idle until we
    // release the lock. This makes the subsequent slot clear race-free.
    compute_cv.wait(lock, [this] {
        return compute_queue.empty() && compute_inflight.load() == 0;
    });
    for (auto & sg : stored_graphs) {
        // Graph nodes live in sg.buffer (mem pool owned by the vector); we do
        // not free them individually — just drop the pointer so recompute_allowed
        // sees a miss and the next graph_compute reuses the pool.
        sg.graph = nullptr;
        sg.uid.store(0, std::memory_order_release);
        // VVRAM: free staging buffers (VRAM copies of RAM-tier weights).
        for (auto stage : sg.vvram_staging_buffers) {
            ggml_backend_buffer_free(stage);
        }
        sg.vvram_staging_buffers.clear();
        // VVRAM: drop the RAM-buffer snapshot (stale — connection buffers may die).
        sg.vvram_ram_buffers.clear();
    }
    for (auto stage : all_graph.vvram_staging_buffers) {
        ggml_backend_buffer_free(stage);
    }
    all_graph.vvram_staging_buffers.clear();
    all_graph.graph = nullptr;
    all_graph.uid.store(0, std::memory_order_release);
    for (auto & kv : all_scheds) {
        ggml_backend_sched_free(kv.second);
    }
    all_scheds.clear();
    LOG_DBG("[%s] drained compute + invalidated cached graphs + schedulers\n", __func__);
}

void rpc_compute_engine::invalidate_cached_graphs() {
    // T2d: lightweight invalidation for new-client connection start. We only
    // hold compute_mtx briefly to null the cached-graph pointers + reset uids —
    // we do NOT wait for the worker to drain (unlike drain_and_invalidate())
    // because a new connection has no in-flight jobs of its own yet, and
    // blocking on other connections' jobs would stall the handshake. The race
    // window is closed by compute_mtx: graph_recompute() reads stored_graphs
    // under compute_mtx (via try_enqueue_graph_recompute), so a concurrent
    // recompute either sees the old pointer (and completes against the old
    // graph, which is still valid — its buffers are alive until that
    // connection's destructor runs drain_and_invalidate) or sees nullptr (and
    // falls back to GRAPH_COMPUTE). Either way is correct. all_scheds are NOT
    // freed here — they remain owned by drain_and_invalidate() and the
    // destructor; freeing them would double-free.
    std::lock_guard<std::mutex> lock(compute_mtx);
    for (auto & sg : stored_graphs) {
        sg.graph = nullptr;
        sg.uid.store(0, std::memory_order_release);
        // VVRAM: free staging buffers on invalidation.
        for (auto stage : sg.vvram_staging_buffers) {
            ggml_backend_buffer_free(stage);
        }
        sg.vvram_staging_buffers.clear();
        // VVRAM: drop the RAM-buffer snapshot.
        sg.vvram_ram_buffers.clear();
    }
    for (auto stage : all_graph.vvram_staging_buffers) {
        ggml_backend_buffer_free(stage);
    }
    all_graph.vvram_staging_buffers.clear();
    all_graph.graph = nullptr;
    all_graph.uid.store(0, std::memory_order_release);
    LOG_DBG("[%s] invalidated cached graphs for new client\n", __func__);
}

// D4.10: sample every Nth decode to keep overhead <1%
static constexpr int TELEMETRY_SAMPLE_INTERVAL = 1;

void rpc_compute_engine::collect_telemetry(const uint32_t * devices, uint32_t n_devices,
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

bool rpc_compute_engine::get_last_telemetry(rpc_msg_server_telemetry & out) const {
    std::lock_guard<std::mutex> lock(telemetry_mtx);
    if (!telemetry_enabled) {
        return false;
    }
    out = last_telemetry;
    return true;
}

rpc_compute_engine::~rpc_compute_engine() {
    {
        std::lock_guard<std::mutex> lock(compute_mtx);
        compute_shutdown = true;
    }
    compute_cv.notify_all();
    if (compute_worker.joinable()) {
        compute_worker.join();
    }
    // D4.5: free cached multi-device schedulers
    for (auto & kv : all_scheds) {
        ggml_backend_sched_free(kv.second);
    }
    all_scheds.clear();
}

rpc_connection::~rpc_connection() {
    // T2c: ordered teardown. Buffers owned by this connection may still be
    // referenced by (a) in-flight compute jobs and (b) cached graphs in the
    // shared engine. Freeing them while either lives = use-after-free of GPU
    // memory. Ordering:
    //   1. drain_and_invalidate() — under compute_mtx: drain jobs that reference
    //      our buffers, then drop cached graphs/scheds BEFORE freeing buffers so
    //      a concurrent recompute can never adopt a graph whose buffers we are
    //      about to free (F6 mitigation). Holding compute_mtx across the drain
    //      + clear makes the slot clear race-free (worker can't dequeue and
    //      submit_compute_job can't enqueue while we hold the lock).
    //   2. free buffers + drop split metadata.
    //   3. bump g_memory_cache_gen so any cached device-memory values (V1a)
    //      observe the freed memory.
    // Engine validity: we hold a shared_ptr, so the engine cannot be destroyed
    // before this destructor completes.
    if (engine) {
        engine->drain_and_invalidate();
    }
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        for (auto buffer : buffers) {
            ggml_backend_buffer_free(buffer);
        }
        buffers.clear();
        split_buffer_metas.clear();
    }
    g_memory_cache_gen.fetch_add(1, std::memory_order_release);
}

static void rpc_serve_channel_bind(socket_ptr sock);
static void rpc_serve_client(std::shared_ptr<rpc_compute_engine> engine, socket_ptr sock);

static void rpc_connection_thread(std::shared_ptr<rpc_compute_engine> engine, socket_ptr sock) {
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
    rpc_serve_client(engine, sock);
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

static void rpc_serve_client(std::shared_ptr<rpc_compute_engine> engine, socket_ptr sock) {
    rpc_connection connection(engine, engine->get_cache_dir());

    // T2d: invalidate the previous client's cached graphs for this shared
    // engine BEFORE the new client's first GRAPH_COMPUTE/RECOMPUTE. Without
    // this, a new client's GRAPH_RECOMPUTE could match the stale uid bound to
    // the previous client's cached graph and replay the wrong graph (the
    // uid-reset in graph_compute only covers same-client reuse). Keying by
    // connection-start + compute_mtx makes this race-free against the worker.
    engine->invalidate_cached_graphs();

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
            connection.hello(tmp);
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
        // V1b: advertise GET_TENSOR_BATCH support (server handler is always present in this build)
        rsp3.conn_caps[0] |= RPC_CAP_GET_TENSOR_BATCH;
        sock->server_supports_trace_id = (rsp3.patch >= 3) || (req_conn_caps[0] & RPC_CAP_TRACE_ID);
        // F1 (T2a): does the client understand graph_hash in GRAPH_RECOMPUTE?
        sock->server_supports_recompute_hash = (req_conn_caps[0] & RPC_CAP_RECOMPUTE_HASH) != 0;
        if (!send_msg(sock, &rsp3, sizeof(rsp3))) {
            return;
        }
    } else {
        rpc_msg_hello_rsp rsp = {};
        connection.hello(rsp);
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
        // V1b: advertise GET_TENSOR_BATCH support (server handler is always present in this build)
        rsp.conn_caps[0] |= RPC_CAP_GET_TENSOR_BATCH;
        // trace_id support from client caps (for deciding 20B vs 12B recv on EVENT_RECORD)
        sock->server_supports_trace_id = (rsp.patch >= 3) || (req.conn_caps[0] & RPC_CAP_TRACE_ID);
        // F1 (T2a): does the client understand graph_hash in GRAPH_RECOMPUTE?
        sock->server_supports_recompute_hash = (req_conn_caps[0] & RPC_CAP_RECOMPUTE_HASH) != 0;
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
                response.device_count = connection.get_engine()->get_backend_count();
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
                if (!connection.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_response(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER_SPLIT: {
                rpc_msg_alloc_buffer_split_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_split_rsp response;
                if (!connection.alloc_buffer_split(request, response)) {
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
                if (!connection.get_alloc_size(request, response)) {
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
                if (!connection.get_alignment(request, response)) {
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
                if (!connection.get_max_size(request, response)) {
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
                if (!connection.buffer_get_base(request, response)) {
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
                if (!connection.free_buffer(request)) {
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
                if (!connection.buffer_clear(request)) {
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
                if (!connection.set_tensor(input)) {
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
                if (!connection.set_tensor_hash(request, response)) {
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
                if (!connection.init_tensor(request)) {
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
                if (!connection.get_tensor(request, response)) {
                    return;
                }
                if (!send_response(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_BATCH: {
                // V1b: batched GET_TENSOR (Wayfinder Loop 5)
                // Payload: count (4 bytes) + N × rpc_msg_get_tensor_req
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (input.size() < sizeof(uint32_t)) {
                    GGML_LOG_ERROR("[%s] RPC_CMD_GET_TENSOR_BATCH: message too small (%zu)\n", __func__, input.size());
                    return;
                }

                uint32_t count;
                memcpy(&count, input.data(), sizeof(count));
                size_t pos = sizeof(count);

                for (uint32_t i = 0; i < count; i++) {
                    if (pos + sizeof(rpc_msg_get_tensor_req) > input.size()) {
                        GGML_LOG_ERROR("[%s] RPC_CMD_GET_TENSOR_BATCH: truncated entry %u\n", __func__, i);
                        return;
                    }

                    rpc_msg_get_tensor_req request;
                    memcpy(&request, input.data() + pos, sizeof(request));
                    pos += sizeof(request);

                    std::vector<uint8_t> response;
                    if (!connection.get_tensor(request, response)) {
                        return;
                    }
                    // Send individual response on response socket
                    if (!send_response(sock, response.data(), response.size())) {
                        return;
                    }
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!connection.copy_tensor(request, response)) {
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
                if (!connection.copy_tensor_peer(request, response)) {
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
                connection.enqueue_graph_compute(std::move(input));
                connection.wait_compute_idle();
                // D4.10: send response (with telemetry appended if enabled)
                rpc_msg_graph_compute_rsp rsp = {};
                rsp.result = 0;
                rpc_msg_server_telemetry telem = {};
                if (connection.get_last_telemetry(telem)) {
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
                // F1 (T2a): variable-size request. New clients (advertising
                // RPC_CAP_RECOMPUTE_HASH) send 12 bytes (device + graph_hash); old
                // clients send 4 bytes (device only). Read based on negotiated cap.
                rpc_msg_graph_recompute_req request = {};
                size_t req_sz = sock->server_supports_recompute_hash ? sizeof(request) : sizeof(uint32_t);
                if (!recv_msg(sock, &request, req_sz)) {
                    return;
                }
                // Synchronous hit/miss decision BEFORE enqueueing the async compute.
                // The client needs to know whether to proceed (EVENT_RECORD) or fall
                // back to a full GRAPH_COMPUTE.
                rpc_msg_graph_recompute_rsp rsp = {};
                // NIT-2: atomic check+enqueue (fixes multi-client race where
                // recompute_allowed() and enqueue_graph_recompute() could be
                // split by a concurrent graph_compute() uid-reset).
                if (connection.try_enqueue_graph_recompute(request)) {
                    rsp.result = 0; // hit → server will recompute
                } else {
                    rsp.result = 1; // miss → client must fall back to GRAPH_COMPUTE
                }
                if (sock->server_supports_recompute_hash) {
                    if (!send_response(sock, &rsp, sizeof(rsp))) {
                        return;
                    }
                }
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

                    if (!connection.set_tensor(single_input)) {
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
                connection.wait_compute_idle();
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
                if (!connection.get_device_memory(request, response)) {
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
                connection.enqueue_graph_compute_all(std::move(input));
                connection.wait_compute_idle();
                // D4.10: send response (with telemetry appended if enabled)
                rpc_msg_graph_compute_all_rsp rsp = {};
                rsp.result = 0;
                rpc_msg_server_telemetry telem = {};
                if (connection.get_last_telemetry(telem)) {
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
                // F1 (T2a): synchronous uid check before enqueueing the async
                // multi-device recompute. On miss, signal the client to fall back.
                // The existing response struct carries result + output_device; reuse
                // result=0 (hit) / result=1 (miss). Only respond when the client
                // advertises recompute-hash support (it always sends graph_hash then).
                rpc_msg_graph_recompute_all_rsp rsp = {};
                // NIT-2: atomic check+enqueue for multi-device path.
                if (connection.try_enqueue_graph_recompute_all(request)) {
                    rsp.result = 0; // hit
                } else {
                    rsp.result = 1; // miss → client falls back to GRAPH_COMPUTE_ALL
                }
                if (sock->server_supports_recompute_hash) {
                    if (!send_response(sock, &rsp, sizeof(rsp))) {
                        return;
                    }
                }
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
                connection.enqueue_graph_compute_stage(std::move(graph_input), stage_id);
                connection.wait_compute_idle();
                rpc_msg_graph_compute_all_rsp rsp = {};
                rsp.result = 0;
                rpc_msg_server_telemetry telem = {};
                if (connection.get_last_telemetry(telem)) {
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

// T2e: server-side fault injection env helpers. All default to 0 (off) so
// production runs are unaffected. drop_pct/reorder_pct are probabilities 0-100.
static int rpc_udp_fault_server_drop_pct() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_UDP_FAULT_DROP_PCT");
        v = e ? atoi(e) : 0;
    }
    return v;
}
static int rpc_udp_fault_server_reorder_pct() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_RPC_UDP_FAULT_REORDER_PCT");
        v = e ? atoi(e) : 0;
    }
    return v;
}

// UDP listener for fire-and-forget graph submission. Runs a dedicated thread
// that recvfrom()s datagrams on udp_port, parses the rpc_udp_header, and
// enqueues GRAPH_RECOMPUTE / GRAPH_RECOMPUTE_ALL on the compute worker —
// exactly like the TCP dispatch path does for these commands.
//
// T2e reliability: after a successful enqueue the listener sends an ACK
// datagram back to the sender (the send channel). The ACK is an
// rpc_udp_header with RPC_UDP_FLAG_ACK set and seq echoing the DATA frame's
// seq. The client matches the ACK to its outstanding frame; on ACK timeout it
// falls back to TCP GRAPH_RECOMPUTE. This closes BUG-011's "receive-only"
// gap: the listener can now both receive and send.
//
// Loss tolerance: if the magic or cmd is invalid, the frame is dropped
// silently. Sequence gaps (detected via seq number) are logged at most once
// per second. Out-of-order and duplicate frames are accepted — the server's
// graph_recompute is idempotent and the next token's frame supersedes.
static void rpc_udp_listener(rpc_compute_engine & engine, int udp_port) {
    if (udp_port <= 0 || udp_port > 65535) {
        return;
    }
    auto sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        GGML_LOG_ERROR("[udp-listener] socket(SOCK_DGRAM) failed\n");
        return;
    }
    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(reuse));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(udp_port));
    if (bind(sockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        GGML_LOG_ERROR("[udp-listener] bind(port %d) failed\n", udp_port);
        close(sockfd);
        return;
    }
    GGML_LOG_INFO("[udp-listener] listening on UDP port %d\n", udp_port);

    // Track last-seen seq per remote so we can log gaps (rate-limited) and
    // dedup exact duplicates (T2e: a duplicate UDP frame must be a no-op).
    struct peer_state {
        uint32_t last_seq = UINT32_MAX;     // highest seq seen (gap detection)
        uint32_t last_ack_seq = UINT32_MAX; // seq of last frame we ACKed (dedup key)
        int64_t  last_gap_log_us = 0;
    };
    std::unordered_map<uint64_t, peer_state> peers; // key = (addr << 16) | port

    // T2e: reorder simulation. The RPC protocol is stop-and-wait, so the server
    // always receives frames in seq order. To exercise delayed-frame tolerance
    // without deadlock, the server ACKs every frame IMMEDIATELY (unblocking the
    // client) but, with probability reorder_pct, holds the frame's *enqueue* for
    // one recvfrom cycle — it is enqueued when the NEXT frame arrives. This keeps
    // the enqueue cadence at exactly one frame per recvfrom (identical to the
    // non-reordered baseline), so the compute-worker interleaving is unchanged and
    // the pre-existing MTP recurrent-state bug (BUG-002, ops.cpp:4916) is not
    // triggered. It proves the server tolerates per-frame enqueue delay without
    // stall or corruption.
    struct held_frame {
        std::vector<uint8_t> bytes;
        sockaddr_in from = {};
        socklen_t from_len = 0;
        uint32_t seq = 0;
    };
    held_frame delayed;      // the frame held back by the current reorder
    bool delayed_valid = false;

    const int drop_pct  = rpc_udp_fault_server_drop_pct();
    const int reorder_pct = rpc_udp_fault_server_reorder_pct();
    // xorshift32 PRNG state (per-listener thread).
    uint32_t rng = 0x9E3779B9u;
    auto rng_pct = [&]() -> int {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return static_cast<int>(rng % 100u);
    };
    auto maybe_drop = [&]() -> bool {
        if (drop_pct <= 0) return false;
        return rng_pct() < drop_pct;
    };

    std::vector<uint8_t> buf(1024); // max UDP datagram we expect

    // T2e: enqueue one already-parsed DATA frame on the compute worker.
    // Gap logging, dedup, and ACK are handled in the main loop BEFORE this is
    // called, so this only does the actual (idempotent) recompute enqueue.
    // Returns true if a job was enqueued. Does NOT send an ACK.
    auto do_enqueue = [&](const uint8_t * data, size_t got,
                          const rpc_udp_header & hdr) -> bool {
        const size_t hdr_sz = sizeof(rpc_udp_header);
        if (got < hdr_sz) return false;
        if (hdr.cmd == RPC_CMD_GRAPH_RECOMPUTE) {
            if (got != hdr_sz) {
                return false;
            }
            // F1 (T2a): carry the uid from the UDP header into the request so the
            // server can verify the cached graph matches.
            rpc_msg_graph_recompute_req req = {};
            req.device = hdr.device;
            req.graph_hash = hdr.graph_uid;
            // NIT-2: atomic check+enqueue (fixes multi-client race).
            return engine.try_enqueue_graph_recompute(req);
        }
        if (hdr.cmd == RPC_CMD_GRAPH_RECOMPUTE_ALL) {
            // Parse the device list that follows the header.
            const uint32_t n_devices = (got > hdr_sz) ?
                static_cast<uint32_t>((got - hdr_sz) / sizeof(uint32_t)) : 0;
            rpc_msg_graph_recompute_all_req req = {};
            req.graph_hash = hdr.graph_uid;
            req.sync_mode = 0; // fire-and-forget
            req.output_requested = 0;
            for (uint32_t i = 0; i < n_devices && i < GGML_RPC_MAX_DEVICES; i++) {
                uint32_t dev;
                memcpy(&dev, data + hdr_sz + i * sizeof(uint32_t), sizeof(dev));
                req.devices[i] = dev;
                req.n_devices++;
            }
            // NIT-2: atomic check+enqueue for multi-device path.
            return engine.try_enqueue_graph_recompute_all(req);
        }
        // Unknown cmd: drop silently.
        return false;
    };

    // T2e: send ACK for a successfully-handled frame. Echo the DATA seq so the
    // client can match the ACK to its outstanding frame. This is the send
    // channel that BUG-011 flagged as missing. Called immediately on receipt
    // (before any reorder delay) so the stop-and-wait client is unblocked.
    auto send_ack = [&](const rpc_udp_header & hdr,
                        const sockaddr_in & from, socklen_t from_len) {
        rpc_udp_header ack_hdr = {};
        ack_hdr.magic     = RPC_UDP_MAGIC;
        ack_hdr.seq       = hdr.seq;          // echo DATA seq
        ack_hdr.cmd       = hdr.cmd;
        ack_hdr.flags     = RPC_UDP_FLAG_ACK; // marks this as an ACK
        ack_hdr.device    = hdr.device;
        ack_hdr.graph_uid = hdr.graph_uid;
        ssize_t sent = sendto(sockfd, reinterpret_cast<const char *>(&ack_hdr),
                              sizeof(ack_hdr), 0,
                              reinterpret_cast<const sockaddr *>(&from), from_len);
        if (sent < 0) {
            GGML_LOG_WARN("[udp-listener] ACK sendto failed for seq %u\n", hdr.seq);
        } else {
            // T2e: record the ACKed seq so a later exact duplicate is a no-op.
            // Find the peer_state for this sender.
            uint64_t peer_key = (static_cast<uint64_t>(from.sin_addr.s_addr) << 16)
                              | static_cast<uint64_t>(ntohs(from.sin_port));
            peers[peer_key].last_ack_seq = hdr.seq;
            LOG_DBG("[udp-listener] ACK seq=%u cmd=%u dev=%u\n", hdr.seq, hdr.cmd, hdr.device);
        }
    };

    while (true) {
        sockaddr_in from = {};
        socklen_t from_len = sizeof(from);
        ssize_t n = recvfrom(sockfd, reinterpret_cast<char *>(buf.data()), buf.size(), 0,
                             reinterpret_cast<sockaddr *>(&from), &from_len);
        if (n < 0) {
            continue;
        }
        size_t got = static_cast<size_t>(n);
        const size_t hdr_sz = sizeof(rpc_udp_header);
        if (got < hdr_sz) continue;
        rpc_udp_header hdr;
        memcpy(&hdr, buf.data(), hdr_sz);
        if (hdr.magic != RPC_UDP_MAGIC) continue;
        if (hdr.flags & RPC_UDP_FLAG_ACK) continue; // ignore stray ACKs

        // T2e fault injection: DROP. With probability drop_pct the frame is
        // discarded before any processing — simulating datagram loss. The client
        // will not receive an ACK and will time out → fall back to TCP.
        if (maybe_drop()) {
            GGML_LOG_DEBUG("[udp-listener] FAULT drop: dropped %zu-byte frame from %s:%u\n",
                           got, inet_ntoa(from.sin_addr), ntohs(from.sin_port));
            continue;
        }

        // Log sequence gaps (rate-limited to 1 Hz per peer).
        uint64_t peer_key = (static_cast<uint64_t>(from.sin_addr.s_addr) << 16)
                          | static_cast<uint64_t>(ntohs(from.sin_port));
        auto & ps = peers[peer_key];
        if (ps.last_seq != UINT32_MAX && hdr.seq != ps.last_seq + 1) {
            const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now_us - ps.last_gap_log_us > 1000000) {
                GGML_LOG_WARN("[udp-listener] seq gap: expected %u, got %u (frame dropped)\n",
                              ps.last_seq + 1, hdr.seq);
                ps.last_gap_log_us = now_us;
            }
        }
        ps.last_seq = hdr.seq;

        // T2e dedup: an exact duplicate (same seq as the last frame we ACKed) is
        // a no-op on the compute side — the recompute for this seq was already
        // enqueued and superseded. ACK it (the client waits on this seq) but skip
        // re-enqueue. This prevents a double-enqueue of the same graph.
        if (ps.last_ack_seq != UINT32_MAX && hdr.seq == ps.last_ack_seq) {
            LOG_DBG("[udp-listener] dup seq=%u (no-op, already ACKed)\n", hdr.seq);
            send_ack(hdr, from, from_len);
            continue;
        }

        // T2e: ACK immediately so the stop-and-wait client is unblocked and the
        // protocol keeps flowing regardless of any reorder delay below. We ACK
        // even on a uid miss (the client proceeds and may TCP-fallback).
        send_ack(hdr, from, from_len);

        // T2e fault injection: REORDER. With probability reorder_pct, hold the
        // current frame's *enqueue* for one recvfrom cycle instead of dispatching
        // it immediately. Because we already ACKed, the client is unblocked (no
        // deadlock). On the NEXT recvfrom, the held frame is flushed first, then
        // the new frame is processed — a one-frame delay that exercises delayed-
        // frame tolerance while keeping the enqueue cadence at exactly one frame
        // per recvfrom (identical to baseline interleaving, so the pre-existing
        // MTP recurrent-state bug BUG-002 at ops.cpp:4916 is not triggered).
        bool reorder_hold = (reorder_pct > 0 && rng_pct() < reorder_pct);
        if (delayed_valid) {
            // Flush the previously-held frame first (it arrived one cycle ago).
            rpc_udp_header dh;
            memcpy(&dh, delayed.bytes.data(), hdr_sz);
            do_enqueue(delayed.bytes.data(), delayed.bytes.size(), dh);
            LOG_DBG("[udp-listener] FAULT reorder: flushed delayed seq=%u\n", dh.seq);
            delayed_valid = false;
        }
        if (reorder_hold) {
            // Hold the current frame; it will be flushed on the next recvfrom.
            delayed.bytes.assign(buf.data(), buf.data() + got);
            delayed.from = from;
            delayed.from_len = from_len;
            delayed.seq = hdr.seq;
            delayed_valid = true;
            LOG_DBG("[udp-listener] FAULT reorder: holding seq=%u for one cycle\n", hdr.seq);
            continue;
        }

        // Normal path: dispatch to compute worker immediately.
        do_enqueue(buf.data(), got, hdr);
    }
    // Note: a held frame is lost on teardown — acceptable (server shutting down).
    close(sockfd);
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
    // UDP transport (prototype): a connectionless UDP listener that receives
    // fire-and-forget graph recompute frames. T2b: shares the SAME rpc_compute_engine
    // as the TCP connections, so UDP recompute hits the same stored_graphs + compute
    // queue. This resolves BUG-011a (UDP cache miss) and BUG-011b (event coupling:
    // wait_compute_idle() now sees all compute, not just the UDP listener's).
    // The shared_ptr keeps the engine alive even if the accept loop returns early
    // (the listener + connection threads capture the shared_ptr).
    auto engine = std::make_shared<rpc_compute_engine>(backends, cache_dir ? cache_dir : "");
    int udp_port = (port > 0 && port < 65535) ? port + 1 : 0;
    std::thread([engine, udp_port]() { rpc_udp_listener(*engine, udp_port); }).detach();
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        std::thread(rpc_connection_thread, engine, client_socket).detach();
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    // engine destructor joins the compute worker thread (via shared_ptr last ref)
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

    // Return cached values if the generation hasn't changed.
    // Device memory is invariant during generation (no buffers allocated/freed),
    // and this function is called ~30×/token, accounting for 71% of blocking RPC time.
    uint64_t gen = g_memory_cache_gen.load(std::memory_order_acquire);
    if (ctx->cached_gen == gen && gen > 0) {
        *free  = ctx->cached_free;
        *total = ctx->cached_total;
        return;
    }

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);

    // Populate cache
    ctx->cached_free  = *free;
    ctx->cached_total = *total;
    ctx->cached_gen   = gen;
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

// Forward declaration for split buffer type (defined later)
static const char * ggml_backend_rpc_split_buffer_type_get_name(ggml_backend_buffer_type_t buft);

// Split buffer type context — needed by supports_buft for scheduler routing
struct ggml_backend_rpc_split_buffer_type_context {
    int         device;       // RPC-local device index
    int         split_id;     // global split rank for this RPC device
    int         n_dev;        // total number of devices in the split
    float       tensor_split[GGML_RPC_MAX_DEVICES];
    std::string name;
    std::string endpoint;
};

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft) {
        return false;
    }
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;

    if (buft->iface.get_name == ggml_backend_rpc_buffer_type_name) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
    }

    if (buft->iface.get_name == ggml_backend_rpc_split_buffer_type_get_name) {
        ggml_backend_rpc_split_buffer_type_context * buft_ctx = (ggml_backend_rpc_split_buffer_type_context *)buft->context;
        return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
    }

    return false;
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

// ---- RPC split buffer type (client-side) ----

ggml_backend_rpc_split_buffer_context::~ggml_backend_rpc_split_buffer_context() {
    // Free each per-tensor allocation on the remote server
    g_memory_cache_gen.fetch_add(1, std::memory_order_release);
    if (!sock) return;
    for (const auto & kv : slices) {
        if (kv.second.remote_ptr != 0) {
            rpc_msg_free_buffer_req req = {kv.second.remote_ptr};
            send_rpc_cmd(sock, RPC_CMD_FREE_BUFFER, &req, sizeof(req), nullptr, 0);
        }
    }
}

static void ggml_backend_rpc_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    g_memory_cache_gen.fetch_add(1, std::memory_order_release);
    ggml_backend_rpc_split_buffer_context * ctx = (ggml_backend_rpc_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_rpc_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // Return a dummy address (like CUDA split buffer does)
    return (void *)0x2000;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_rpc_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    g_memory_cache_gen.fetch_add(1, std::memory_order_release);
    ggml_backend_rpc_split_buffer_context * buf_ctx = (ggml_backend_rpc_split_buffer_context *)buffer->context;
    ggml_backend_rpc_split_buffer_type_context * buft_ctx = (ggml_backend_rpc_split_buffer_type_context *)buffer->buft->context;

    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    int64_t row_low, row_high;
    rpc_get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, buft_ctx->n_dev, buft_ctx->split_id);

    int64_t nrows_split = row_high - row_low;
    if (nrows_split == 0) {
        return GGML_STATUS_SUCCESS;
    }

    // Allocate remote buffer for this tensor slice
    auto sock = get_socket(buft_ctx->endpoint);
    if (!sock) {
        GGML_LOG_ERROR("[%s] failed to connect to %s\n", __func__, buft_ctx->endpoint.c_str());
        return GGML_STATUS_ABORTED;
    }
    // Store socket shared_ptr to keep connection alive across alloc/set_tensor phases.
    // The socket is the same for all tensors in this buffer, so only set on first call.
    if (!buf_ctx->sock) {
        buf_ctx->sock = sock;
    }

    // Store slice metadata early so serialize_tensor can adjust ne[1] for the
    // ALLOC_BUFFER_SPLIT request (server allocates based on serialized dimensions).
    {
        ggml_backend_rpc_split_buffer_context::tensor_slice early_slice;
        early_slice.remote_ptr = 0;
        early_slice.remote_size = 0;
        early_slice.row_low  = row_low;
        early_slice.row_high = row_high;
        buf_ctx->slices[tensor] = early_slice;
    }

    rpc_msg_alloc_buffer_split_req request;
    request.device = buft_ctx->device;
    request.tensor = serialize_tensor(tensor);
    request.tensor.buffer = 0; // No buffer context yet on server for this slice
    request.tensor.data   = 0;
    request.row_low  = row_low;
    request.row_high = row_high;

    rpc_msg_alloc_buffer_split_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER_SPLIT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.remote_ptr != 0) {
        // Update the early slice with server's remote_ptr
        buf_ctx->slices[tensor].remote_ptr = response.remote_ptr;
        buf_ctx->slices[tensor].remote_size = response.remote_size;
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // Split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_rpc_split_buffer_context * buf_ctx = (ggml_backend_rpc_split_buffer_context *)buffer->context;
    ggml_backend_rpc_split_buffer_type_context * buft_ctx = (ggml_backend_rpc_split_buffer_type_context *)buffer->buft->context;

    auto it = buf_ctx->slices.find(tensor);
    if (it == buf_ctx->slices.end()) {
        return; // no data for this device
    }

    const auto & slice = it->second;
    socket_ptr sock = buf_ctx->sock ? buf_ctx->sock : get_socket(buft_ctx->endpoint);
    GGML_ASSERT(sock != nullptr);

    // Compute offset within the full tensor data for this device's rows.
    // The server buffer is full-size, so data goes at row_low * nb[1].
    size_t slice_size = rpc_nbytes_split(tensor, slice.row_high - slice.row_low);
    size_t slice_offset = (size_t) slice.row_low * tensor->nb[1];
    const uint8_t * slice_data = (const uint8_t *)data + slice_offset;

    // Serialize tensor with the split buffer's remote_ptr
    rpc_tensor rpc_t = serialize_tensor(tensor);
    rpc_t.buffer = slice.remote_ptr;
    rpc_t.data   = reinterpret_cast<uint64_t>(tensor->data);

    rpc_issue_set_tensor_payload(sock, rpc_t, 0, slice_data, slice_size);
}

static void ggml_backend_rpc_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // Split tensors must always be retrieved in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_rpc_split_buffer_context * buf_ctx = (ggml_backend_rpc_split_buffer_context *)buffer->context;
    ggml_backend_rpc_split_buffer_type_context * buft_ctx = (ggml_backend_rpc_split_buffer_type_context *)buffer->buft->context;

    auto it = buf_ctx->slices.find(const_cast<ggml_tensor*>(tensor));
    if (it == buf_ctx->slices.end()) {
        return;
    }

    const auto & slice = it->second;
    socket_ptr sock = buf_ctx->sock ? buf_ctx->sock : get_socket(buft_ctx->endpoint);
    GGML_ASSERT(sock != nullptr);

    size_t slice_size = rpc_nbytes_split(tensor, slice.row_high - slice.row_low);
    size_t slice_offset = (size_t) slice.row_low * tensor->nb[1];
    uint8_t * slice_data = (uint8_t *)data + slice_offset;

    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.tensor.buffer = slice.remote_ptr;
    request.tensor.data   = reinterpret_cast<uint64_t>(tensor->data);
    request.offset = 0;
    request.size = slice_size;

    bool status = send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), slice_data, slice_size);
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // No-op for split buffers (consistent with CUDA split buffer)
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_rpc_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_rpc_split_buffer_clear,
    /* .reset           = */ NULL,
};

// RPC split buffer type

static const char * ggml_backend_rpc_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_split_buffer_type_context * ctx = (ggml_backend_rpc_split_buffer_type_context *)buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // We don't know the exact split after rounding, so allocate the device buffers
    // for each tensor separately in init_tensor. However, the size still represents
    // the maximum cumulative size as returned by get_alloc_size.
    ggml_backend_rpc_split_buffer_context * ctx = new ggml_backend_rpc_split_buffer_context();
    ctx->endpoint = ((ggml_backend_rpc_split_buffer_type_context *)buft->context)->endpoint;
    return ggml_backend_buffer_init(buft, ggml_backend_rpc_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_rpc_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_rpc_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_rpc_split_buffer_type_context * ctx = (ggml_backend_rpc_split_buffer_type_context *)buft->context;
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    int64_t row_low, row_high;
    rpc_get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, ctx->n_dev, ctx->split_id);

    int64_t nrows_split = row_high - row_low;
    if (nrows_split == 0) {
        return 0;
    }

    return rpc_nbytes_split(tensor, nrows_split);
}

static bool ggml_backend_rpc_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_rpc_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_rpc_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_rpc_split_buffer_type_is_host,
};

// Factory function for the RPC split buffer type.
// tensor_split is an array of per-device fractional weights (e.g., {60, 40} for 60/40).
// device is the RPC-local device index on the remote server.
// The factory computes the global split position by counting non-zero entries
// in tensor_split and assuming RPC devices occupy the last positions.
static ggml_backend_buffer_type_t ggml_backend_rpc_split_buffer_type(
    int device, const float * tensor_split) {

    int _n_dev_ts = rpc_count_split_devices(tensor_split, GGML_RPC_MAX_DEVICES);
    fprintf(stderr, "[rpc-split] factory called: device=%d, n_dev=%d\n",
            device, _n_dev_ts);
    fflush(stderr);
    for (int i = 0; i < GGML_RPC_MAX_DEVICES; i++) {
        fprintf(stderr, "[rpc-split]   tensor_split[%d] = %.1f\n", i, tensor_split[i]);
    }
    fflush(stderr);

    int n_dev = rpc_count_split_devices(tensor_split, GGML_RPC_MAX_DEVICES);

    if (n_dev == 0) {
        GGML_LOG_ERROR("[%s] tensor_split has no non-zero entries; cannot create split buffer type\n", __func__);
        return nullptr;
    }

    // Count total RPC devices registered so far to determine the base offset.
    // RPC devices occupy the last n_rpc positions in the tensor_split array.
    // The first RPC device is at split_id = n_dev - n_rpc, the last at n_dev - 1.
    int n_rpc_total = 0;
    {
        std::lock_guard<std::mutex> lock(g_rpc_reg_map_mutex);
        for (const auto & pair : g_rpc_reg_map) {
            ggml_backend_rpc_reg_context * rctx = (ggml_backend_rpc_reg_context *)pair.second->context;
            n_rpc_total += (int) rctx->devices.size();
        }
    }

    // This device's global split position
    int split_id = n_dev - n_rpc_total + device;
    if (split_id < 0 || split_id >= n_dev) {
        GGML_LOG_ERROR("[%s] invalid split_id=%d for device=%d, n_dev=%d, n_rpc=%d\n",
                       __func__, split_id, device, n_dev, n_rpc_total);
        // Fallback: use device as split_id
        split_id = device;
    }

    ggml_backend_rpc_split_buffer_type_context * ctx = new ggml_backend_rpc_split_buffer_type_context;
    ctx->device     = device;
    ctx->split_id   = split_id;
    ctx->n_dev      = n_dev;
    ctx->name       = "RPC" + std::to_string(device) + "_Split";

    // Copy tensor_split
    for (int i = 0; i < GGML_RPC_MAX_DEVICES && i < n_dev; i++) {
        ctx->tensor_split[i] = tensor_split ? tensor_split[i] : 0.0f;
    }

    // Find endpoint from the global device-to-endpoint map
    {
        std::lock_guard<std::mutex> lock(g_rpc_reg_map_mutex);
        for (const auto & pair : g_rpc_reg_map) {
            ggml_backend_rpc_reg_context * rctx = (ggml_backend_rpc_reg_context *)pair.second->context;
            for (auto * dev : rctx->devices) {
                ggml_backend_rpc_device_context * dctx = (ggml_backend_rpc_device_context *)dev->context;
                if (dctx->device == (uint32_t) device) {
                    ctx->endpoint = dctx->endpoint;
                    break;
                }
            }
            if (!ctx->endpoint.empty()) break;
        }
    }

    if (ctx->endpoint.empty()) {
        GGML_LOG_WARN("[%s] no endpoint found for device %d\n", __func__, device);
        delete ctx;
        return nullptr;
    }

    struct ggml_backend_buffer_type buft;
    buft.iface   = ggml_backend_rpc_split_buffer_type_interface;
    buft.device  = nullptr; // Will be resolved by the caller
    buft.context = ctx;

    // Allocate the buffer type on the heap (caller manages lifetime)
    ggml_backend_buffer_type_t result = new ggml_backend_buffer_type(buft);
    return result;
}

// ---- end RPC split buffer type ----

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    fprintf(stderr, "[rpc-get_proc] called with name='%s'\n", name);
    fflush(stderr);
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        fprintf(stderr, "[rpc-get_proc] returning rpc_add_server\n");
        fflush(stderr);
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        fprintf(stderr, "[rpc-get_proc] returning rpc_start_server\n");
        fflush(stderr);
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_get_device_memory") == 0) {
        fprintf(stderr, "[rpc-get_proc] returning get_device_memory\n");
        fflush(stderr);
        return (void *)ggml_backend_rpc_get_device_memory;
    }
    if (std::strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        fprintf(stderr, "[rpc-get_proc] split_buffer_type requested — returning factory\n");
        fflush(stderr);
        return (void *)ggml_backend_rpc_split_buffer_type;
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
    // Create the registration context first
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";

    // Create per-endpoint registration BEFORE devices so devices can reference it
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };

    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(g_rpc_dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint     = */    endpoint,
            /* .device       = */    ind,
            /* .global_index = */    g_rpc_dev_id,
            /* .name         = */    dev_name,
            /* .description  = */    dev_desc,
            // seen_graph_uids default-constructs empty
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ reg,  // FIX: use per-endpoint registration, not global (which has context=NULL)
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        {
            std::lock_guard<std::mutex> ep_lock(g_rpc_dev_endpoint_mutex);
            g_rpc_dev_endpoint_map[g_rpc_dev_id] = std::string(endpoint);
        }
        g_rpc_dev_id++;
    }
    g_rpc_reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
