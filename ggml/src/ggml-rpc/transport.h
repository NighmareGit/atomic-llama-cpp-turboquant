#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

// T2e: UDP datagram header for graph recompute, shared between the client
// (transport.cpp send + ACK wait) and the server (ggml-rpc.cpp listener).
// Fixed 22-byte header. Followed by 0 or more uint32_t device indices for the
// GRAPH_RECOMPUTE_ALL variant.
//
// Two frame kinds share this header:
//   DATA (client→server): flags=0, seq=monotonic per socket, cmd=GRAPH_RECOMPUTE[*],
//                         device + graph_uid identify the cached graph.
//   ACK  (server→client): flags=RPC_UDP_FLAG_ACK, seq=echo of the DATA frame's seq.
//                         The client matches seq to confirm the specific frame arrived.
struct rpc_udp_header {
    uint32_t magic;      // RPC_UDP_MAGIC
    uint32_t seq;        // DATA: monotonic per socket; ACK: echo of DATA seq
    uint8_t  cmd;        // RPC_CMD_GRAPH_RECOMPUTE or GRAPH_RECOMPUTE_ALL
    uint8_t  flags;      // RPC_UDP_FLAG_ACK set on ACK frames
    uint32_t device;     // device index
    uint64_t graph_uid;  // identifies cached graph on server
};

static constexpr uint32_t RPC_UDP_MAGIC = 0x474D4C01;
// T2e: ACK flag in rpc_udp_header::flags. Server sets this on the ACK frame
// that echoes a received DATA frame's seq back to the client.
static constexpr uint8_t  RPC_UDP_FLAG_ACK = 1u << 0;

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;
static constexpr uint8_t  RPC_CAP_TRACE_ID        = 1u << 0; // for trace_id in EVENT_RECORD (proto patch 3+)
static constexpr uint8_t  RPC_CAP_MULTI_DEVICE    = 1u << 1; // Path C: server-side multi-GPU scheduling
static constexpr uint8_t  RPC_CAP_SERVER_TELEMETRY = 1u << 2; // D4.10: server appends telemetry to GRAPH_COMPUTE_ALL response
static constexpr uint8_t  RPC_CAP_RECOMPUTE_HASH  = 1u << 3; // F1 (T2a): graph_hash in GRAPH_RECOMPUTE req + hit/miss rsp
static constexpr uint8_t  RPC_CAP_GET_TENSOR_BATCH = 1u << 4; // V1b: server understands RPC_CMD_GET_TENSOR_BATCH (value 25)

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    // UDP transport for fire-and-forget graph submission (opt-in via GGML_RPC_UDP=1).
    // init_udp() creates a UDP socket aimed at the remote's UDP port (tcp_port+1
    // by default). send_udp() fires a single datagram. For T2e reliability,
    // recv_udp_ack() waits for the server's ACK (echoing the DATA frame's seq)
    // with a timeout; the caller falls back to TCP on timeout.
    bool init_udp(int udp_port);
    bool send_udp(const void * data, size_t size) const;
    // T2e: blocking wait for an ACK frame whose seq == expected_seq.
    // Returns true if the matching ACK arrived within timeout_ms, false on
    // timeout or socket error. Loops past stale/duplicate ACKs (wrong seq).
    bool recv_udp_ack(uint32_t expected_seq, int timeout_ms);
    bool udp_enabled() const;
    uint32_t udp_next_seq();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    // Set after HELLO: server supports RPC_CMD_SET_TENSOR_BATCH when minor >= 1
    bool server_supports_batch = false;
    // Set after HELLO: server supports RPC_CMD_COPY_TENSOR_PEER when minor >= 3
    bool server_supports_peer_copy = false;
    // B+11: paired response socket (cmd/response split); nullptr = single-socket
    socket_ptr rsp_channel;
    // Set after HELLO (patch 3+ or RPC_CAP_TRACE_ID): trace_id carried in EVENT_RECORD (20B wire)
    bool server_supports_trace_id = false;
    // Path C: server supports GRAPH_COMPUTE_ALL (multi-device scheduling)
    bool server_supports_multi_device = false;
    // D4.10: server appends rpc_msg_server_telemetry to GRAPH_COMPUTE_ALL response
    bool server_supports_telemetry = false;
    // F1 (T2a): server understands graph_hash in GRAPH_RECOMPUTE and sends hit/miss response
    bool server_supports_recompute_hash = false;
    // V1b: server understands RPC_CMD_GET_TENSOR_BATCH (value 25)
    bool server_supports_get_tensor_batch = false;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
