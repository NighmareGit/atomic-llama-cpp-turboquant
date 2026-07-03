#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    4   // + RPC_CMD_CHANNEL_BIND dual-socket (4)
#define RPC_PROTO_PATCH_VERSION    3   // + trace_id in EVENT_RECORD (20B wire, RPC_CAP_TRACE_ID)

#ifdef  __cplusplus
// 98 = upstream 97 + the fork's GGML_OP_TURBO_WHT. Bumped patch version because
// adding an op shifts the GGML_OP enum used in the RPC wire protocol.
static_assert(GGML_OP_COUNT == 98, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// B+7a'/B+9: drain deferred EVENT/GET/COPY across all RPC sockets (pipeline_barrier hook).
GGML_BACKEND_API void ggml_backend_rpc_drain_all_endpoints(void);

// B+13: complete deferred RPC->local downloads before local split graph_compute.
GGML_BACKEND_API void ggml_backend_rpc_flush_pending_downloads(void);

// B+13: recv+H2D only downloads whose dst matches; n_dst==0 flushes all pending.
GGML_BACKEND_API void ggml_backend_rpc_flush_pending_downloads_for_dst(const struct ggml_tensor * const * dst, size_t n_dst);

// B+13: buffer/input_backend may disagree on RPC; route by buffer type in sched gather.
GGML_BACKEND_API bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool ggml_backend_rpc_try_download_tensor(ggml_backend_t dst_backend, const struct ggml_tensor * src, struct ggml_tensor * dst);
GGML_BACKEND_API bool ggml_backend_rpc_try_upload_tensor(ggml_backend_t src_backend, const struct ggml_tensor * src, struct ggml_tensor * dst);
GGML_BACKEND_API bool ggml_backend_rpc_download_pending_for_dst(const struct ggml_tensor * dst);

// Same-host multi-endpoint relay: queued at copy_issue, flushed after producer ready.
GGML_BACKEND_API bool ggml_backend_rpc_relay_pending_for_dst(const struct ggml_tensor * dst);
GGML_BACKEND_API void ggml_backend_rpc_flush_pending_relays_for_dst(const struct ggml_tensor * const * dst, size_t n_dst);

// B+9: unique RPC server endpoints registered (host:port strings).
GGML_BACKEND_API int ggml_backend_rpc_server_count(void);

// B+9: defer EVENT recv to pipeline_barrier (default on for 2-GPU when pipeline plus is on).
GGML_BACKEND_API bool ggml_backend_rpc_event_defer_barrier(void);

// B+12: defer GET_TENSOR recv to sched graph_compute boundary (default on when pipeline plus is on).
GGML_BACKEND_API bool ggml_backend_rpc_get_tensor_defer(void);

// B+7f: pipeline SET_TENSOR_HASH send/recv during load (default on when pipeline plus is on).
GGML_BACKEND_API bool ggml_backend_rpc_hash_defer(void);

// B+11: cmd/response dual-socket RPC (default OFF; proto minor >= 4). See docs/rpc-multi-backend-pipeline-plus/RPC-PROTOCOL.md
GGML_BACKEND_API bool ggml_backend_rpc_dual_socket(void);

#ifdef  __cplusplus
}
#endif
