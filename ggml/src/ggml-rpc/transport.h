#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;
static constexpr uint8_t  RPC_CAP_TRACE_ID     = 1u << 0; // for trace_id in EVENT_RECORD (proto patch 3+)
static constexpr uint8_t  RPC_CAP_MULTI_DEVICE = 1u << 1; // Path C: server-side multi-GPU scheduling

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

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

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
