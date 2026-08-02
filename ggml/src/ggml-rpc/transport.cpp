#include "transport.h"
#include "ggml-impl.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <cstdlib>
#include <mutex>
#include <optional>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

#ifdef GGML_RPC_RDMA
static constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
static constexpr int    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
static constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd  = nullptr;
    struct ibv_cq * scq = nullptr;   // send completions
    struct ibv_cq * rcq = nullptr;   // recv completions
    struct ibv_qp * qp  = nullptr;

    void          * tx_buf = nullptr;
    struct ibv_mr * tx_mr  = nullptr;

    void          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    struct ibv_mr * rx_mr  = nullptr;
    int             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(int i) const {
        return static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    bool post_rx(int i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        return ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        if (tx_mr) ibv_dereg_mr(tx_mr);
        if (rx_mr) ibv_dereg_mr(rx_mr);
        free(tx_buf);
        free(rx_buf);
        if (qp)  ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
struct rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    int      gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;
};

struct rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA

struct socket_t::impl {
    impl(sockfd_t fd) : udp_fd(-1), udp_port(0), use_rdma(false), fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    // UDP transport for fire-and-forget graph submission (opt-in).
    bool init_udp(int udp_port);
    bool send_udp(const void * data, size_t size) const;
    bool recv_udp_ack(uint32_t expected_seq, int timeout_ms);
    bool udp_enabled() const { return udp_fd >= 0; }
    uint32_t udp_next_seq();    // next monotonic seq for this socket (thread-safe)
    mutable std::mutex udp_mu;  // guards lazy init of udp_fd and udp_seq
    sockfd_t         udp_fd;    // client UDP socket, -1 = not initialized
    int              udp_port;  // remote UDP port (tcp_port + 1)
    uint32_t         udp_seq = 0; // monotonic UDP sequence number

#ifdef GGML_RPC_RDMA
    bool tcp_peer_closed();
    std::optional<rdma_gid_t> rdma_build_target_gid();
    bool rdma_probe();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid);
    bool rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc);
    bool rdma_send(const void * data, size_t size);
    bool rdma_recv(void * data, size_t size);

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#endif // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
    if (fd != INVALID_SOCKET) closesocket(this->fd);
    if (udp_fd != INVALID_SOCKET) closesocket(udp_fd);
#else
    if (fd >= 0) close(this->fd);
    if (udp_fd >= 0) close(udp_fd);
#endif
}

#ifdef GGML_RPC_RDMA

bool socket_t::impl::tcp_peer_closed() {
    if (fd < 0) return false;
#ifndef _WIN32
    struct pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    int r = poll(&pfd, 1, 0);
    return r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#else
    return false;
#endif
}

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        return std::nullopt;
    }
    rdma_gid_t target = {};
    if (addr.ss_family == AF_INET) {
        const auto * a = reinterpret_cast<const sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        return target;
    }
    if (addr.ss_family == AF_INET6) {
        const auto * a = reinterpret_cast<const sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        return target;
    }
    return std::nullopt;
}

bool socket_t::impl::rdma_probe() {
    const char * dev_env = std::getenv("GGML_RDMA_DEV");
    const char * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    if (!target_gid) {
        return false;
    }

    const uint8_t ib_port = 1;
    int num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    if (!devs || num_devs == 0) return false;

    ibv_context * ibctx = nullptr;
    const char * matched_dev = nullptr;
    int gid_idx = gid_env ? atoi(gid_env) : -1;
    int gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    for (int d = 0; d < num_devs; d++) {
        const char * dn = ibv_get_device_name(devs[d]);
        if (dev_env && strcmp(dev_env, dn) != 0) continue;

        ibv_context * ctx = ibv_open_device(devs[d]);
        if (!ctx) continue;

        ibv_port_attr pa;
        if (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); continue; }

        int found_gid = gid_idx;
        int found_version = IBV_GID_TYPE_IB;
        if (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            int v2_idx = -1;
            int v1_idx = -1;
            for (int i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                if (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) continue;
                if (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) continue;
                if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } else if (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            if (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } else if (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } else {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            if (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        if (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            break;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    if (!ibctx) return false;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    if (!rdma->pd) return false;

    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, nullptr, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, nullptr, 0);
    if (!rdma->scq || !rdma->rcq) return false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    if (!rdma->qp) return false;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    if (!rdma->tx_buf || !rdma->rx_buf) return false;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!rdma->tx_mr || !rdma->rx_mr) return false;

    ibv_gid local_gid;
    if (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) return false;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    const char * ver_str = "";
    if (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } else if (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    GGML_LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    return true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid) {
    // RESET -> INIT
    {
        struct ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            return false;
        }
    }

    for (int i = 0; i < RDMA_RX_DEPTH; i++) {
        if (!rdma->post_rx(i)) return false;
    }

    // INIT -> RTR
    {
        struct ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            return false;
        }
    }

    // RTR -> RTS
    {
        struct ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            return false;
        }
    }

    GGML_LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    return true;
}

bool socket_t::impl::rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc) {
    for (uint64_t s = 0; ; s++) {
        int n = ibv_poll_cq(cq, 1, wc);
        if (n > 0) {
            if (wc->status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            return wc->status == IBV_WC_SUCCESS;
        }
        if (n < 0) return false;
        if ((s & 0xFFFFF) == 0 && s > 0) {
            if (tcp_peer_closed()) {
                return false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } else {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;
        struct ibv_wc wc;
        if (!rdma_poll(c->scq, &wc)) return false;

        src += chunk;
        rem -= chunk;
    }
    return true;
}

bool socket_t::impl::rdma_recv(void * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        struct ibv_wc wc;
        if (!rdma_poll(c->rcq, &wc)) return false;

        int slot = (int)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        if (!c->post_rx(slot)) return false;

        dst += got;
        rem -= got;
    }
    return true;
}

#endif // GGML_RPC_RDMA

bool socket_t::impl::send_data(const void * data, size_t size) {
#ifdef GGML_RPC_RDMA
    if (use_rdma) {
        return rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            GGML_LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

bool socket_t::impl::recv_data(void * data, size_t size) {
#ifdef GGML_RPC_RDMA
    if (use_rdma) {
        return rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (char *)data + bytes_recv, size_to_recv, 0);
        if (n < 0) {
            GGML_LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            return false;
        }
        if (n == 0) {
            LOG_DBG("recv returned 0 (peer closed?)\n");
            return false;
        }
        bytes_recv += (size_t)n;
    }
    return true;
}

// UDP transport for fire-and-forget graph submission.
//
// This is opt-in (GGML_RPC_UDP=1) and additive: TCP is untouched. The UDP
// socket is created lazily on first init_udp() and aimed at the remote's UDP
// port (tcp_port + 1 by default). send_udp() fires a single datagram with no
// ACK and no retransmit — loss is tolerated because the next token's graph
// call replaces a lost frame. The caller falls back to TCP if UDP is not
// enabled or send_udp() fails.
static bool is_valid_fd(sockfd_t sockfd); // defined later in this file
bool socket_t::impl::init_udp(int udp_port) {
    if (udp_enabled()) {
        return true;
    }
    std::lock_guard<std::mutex> lock(udp_mu);
    if (udp_enabled()) {
        return true;
    }
    this->udp_port = udp_port;

    // Resolve the remote host from the existing TCP connection's peer address.
    sockaddr_storage peer_addr = {};
    socklen_t peer_len = sizeof(peer_addr);
    if (getpeername(fd, reinterpret_cast<sockaddr *>(&peer_addr), &peer_len) != 0) {
        GGML_LOG_ERROR("[%s] getpeername failed for UDP target\n", __func__);
        return false;
    }

    sockfd_t sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (!is_valid_fd(sockfd)) {
        GGML_LOG_ERROR("[%s] socket(SOCK_DGRAM) failed\n", __func__);
        return false;
    }

    // Build the target address: same host as TCP, but the UDP port.
    sockaddr_in target_addr = {};
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(static_cast<uint16_t>(udp_port));
    if (peer_addr.ss_family == AF_INET) {
        target_addr.sin_addr = reinterpret_cast<const sockaddr_in *>(&peer_addr)->sin_addr;
    } else if (peer_addr.ss_family == AF_INET6) {
        // Extract IPv4 from IPv4-mapped IPv6 if present; otherwise fail.
        const auto * a6 = reinterpret_cast<const sockaddr_in6 *>(&peer_addr);
        if (IN6_IS_ADDR_V4MAPPED(&a6->sin6_addr)) {
            memcpy(&target_addr.sin_addr, &a6->sin6_addr.s6_addr[12], 4);
        } else {
            GGML_LOG_ERROR("[%s] IPv6 peer not supported for UDP (no v4-mapped addr)\n", __func__);
#ifdef _WIN32
            closesocket(sockfd);
#else
            close(sockfd);
#endif
            return false;
        }
    } else {
        GGML_LOG_ERROR("[%s] unexpected peer address family %d\n", __func__, peer_addr.ss_family);
#ifdef _WIN32
        closesocket(sockfd);
#else
        close(sockfd);
#endif
        return false;
    }

    // Store the target as a connected UDP socket so send_udp() can use send().
    // connect() on a UDP socket does not wire anything — it just sets the
    // default destination, which is exactly what we want.
    if (::connect(sockfd, reinterpret_cast<const sockaddr *>(&target_addr),
                  sizeof(target_addr)) != 0) {
        GGML_LOG_ERROR("[%s] connect(udp) failed for port %d\n", __func__, udp_port);
#ifdef _WIN32
        closesocket(sockfd);
#else
        close(sockfd);
#endif
        return false;
    }

    udp_fd = sockfd;
    LOG_DBG("[%s] UDP socket %d -> port %d (tcp peer)\n", __func__, udp_fd, udp_port);
    return true;
}

bool socket_t::impl::send_udp(const void * data, size_t size) const {
    if (!udp_enabled()) {
        return false;
    }
    // send() on a connected UDP socket; single datagram, no retransmit.
    ssize_t n = send(udp_fd, static_cast<const char *>(data), size, 0);
    if (n < 0 || static_cast<size_t>(n) != size) {
        GGML_LOG_ERROR("[%s] send_udp failed (sent=%zd, size=%zu)\n", __func__, n, size);
        return false;
    }
    return true;
}

uint32_t socket_t::impl::udp_next_seq() {
    std::lock_guard<std::mutex> lock(udp_mu);
    return udp_seq++;
}

// T2e: blocking wait for an ACK frame echoing expected_seq.
//
// The UDP socket is connected (see init_udp), so recv() filters to the remote
// UDP peer. We set SO_RCVTIMEO to timeout_ms and loop on recv(): each datagram
// is parsed as an rpc_udp_header; we accept the first frame that carries the
// ACK flag AND seq == expected_seq. Stale ACKs (seq != expected_seq) are
// drained and ignored — they belong to prior tokens whose DATA frame was
// already superseded. Returns false on timeout or socket error.
//
// This is called immediately after send_udp() in the client's graph-compute
// path, so the wait is bounded (typically <1 ms on a LAN; the timeout is a
// safety net for genuine loss). On timeout the caller falls back to TCP.
bool socket_t::impl::recv_udp_ack(uint32_t expected_seq, int timeout_ms) {
    if (!udp_enabled()) {
        return false;
    }
    // Set receive timeout for the duration of this wait.
    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    if (setsockopt(udp_fd, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char *>(&tv), sizeof(tv)) != 0) {
        GGML_LOG_ERROR("[%s] setsockopt(SO_RCVTIMEO) failed\n", __func__);
        return false;
    }
    // Loop until we see the expected ACK or the timeout fires.
    std::vector<uint8_t> buf(256);
    while (true) {
        ssize_t n = recv(udp_fd, reinterpret_cast<char *>(buf.data()), buf.size(), 0);
        if (n < 0) {
            // EAGAIN/EWOULDBLOCK = timeout fired. Genuine loss → caller falls back.
            break;
        }
        if (static_cast<size_t>(n) < sizeof(rpc_udp_header)) {
            continue; // too small to be our ACK
        }
        auto * hdr = reinterpret_cast<const rpc_udp_header *>(buf.data());
        if (hdr->magic != RPC_UDP_MAGIC) {
            continue; // not our protocol
        }
        if (!(hdr->flags & RPC_UDP_FLAG_ACK)) {
            continue; // not an ACK (shouldn't happen, but be defensive)
        }
        if (hdr->seq == expected_seq) {
            // Matched ACK — restore blocking mode (no timeout) for any future use.
            struct timeval zero = {};
            setsockopt(udp_fd, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char *>(&zero), sizeof(zero));
            return true;
        }
        // else: stale ACK (seq != expected_seq) — drain and keep waiting.
    }
    // Timeout (or error) — restore blocking mode before returning.
    struct timeval zero = {};
    setsockopt(udp_fd, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char *>(&zero), sizeof(zero));
    return false;
}

void socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
    local_caps[0] |= RPC_CAP_TRACE_ID; // advertise trace_id support (EVENT_RECORD 20B)
    local_caps[0] |= RPC_CAP_RECOMPUTE_HASH; // F1 (T2a): graph_hash in GRAPH_RECOMPUTE
    local_caps[0] |= RPC_CAP_GET_TENSOR_BATCH; // V1b: RPC_CMD_GET_TENSOR_BATCH (value 25)
    local_caps[0] |= RPC_CAP_RECOMPUTE_REBIND; // E-2 (BUG-002a): input-rebind descriptors in GRAPH_RECOMPUTE
#ifdef GGML_RPC_RDMA
    rdma_local = {};
    if (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, sizeof(rc));
    } else {
        rdma.reset();
    }
#endif // GGML_RPC_RDMA
}

void socket_t::impl::update_caps(const uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    if (!rdma) {
        return;
    }
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, sizeof(rc));
    if (rc.qpn == 0) {
        rdma.reset();
        return;
    }
    if (rdma_activate(rc.qpn, rc.psn, rc.gid)) {
        use_rdma = true;
    } else {
        GGML_LOG_ERROR("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#else
    (void)remote_caps;
#endif // GGML_RPC_RDMA
}


/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

void socket_t::get_caps(uint8_t * local_caps) {
    return pimpl->get_caps(local_caps);
}

void socket_t::update_caps(const uint8_t * remote_caps) {
    return pimpl->update_caps(remote_caps);
}

bool socket_t::init_udp(int udp_port) {
    return pimpl->init_udp(udp_port);
}

bool socket_t::send_udp(const void * data, size_t size) const {
    return pimpl->send_udp(data, size);
}

bool socket_t::recv_udp_ack(uint32_t expected_seq, int timeout_ms) {
    return pimpl->recv_udp_ack(expected_seq, timeout_ms);
}

bool socket_t::udp_enabled() const {
    return pimpl->udp_enabled();
}

uint32_t socket_t::udp_next_seq() {
    return pimpl->udp_next_seq();
}

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    return sockfd != INVALID_SOCKET;
#else
    return sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, NULL, NULL);
    if (!is_valid_fd(client_socket_fd)) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        GGML_LOG_ERROR("Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        GGML_LOG_ERROR("Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        GGML_LOG_ERROR("Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (::connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
static std::mutex g_rpc_transport_mu;
static bool g_rpc_transport_wsa_started = false;
#endif

bool rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (g_rpc_transport_wsa_started) {
        return true;
    }
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        return false;
    }
    g_rpc_transport_wsa_started = true;
    return true;
#else
    return true;
#endif
}

void rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (!g_rpc_transport_wsa_started) {
        return;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = false;
#endif
}
