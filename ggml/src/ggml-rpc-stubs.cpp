// Weak symbol stubs for RPC functions called from ggml-base.
//
// ggml-backend.cpp (part of libggml-base.so) calls several RPC-specific
// functions that are defined in libggml-rpc.so.  When shared libraries are
// used these symbols are unresolved at link time, causing errors for any
// target that links ggml-base (or ggml) without also pulling in ggml-rpc.
//
// These stubs provide default implementations that return safe "no RPC"
// values (false, nullptr, no-op).  They are marked __attribute__((weak))
// so that the strong definitions in ggml-rpc override them when the RPC
// backend is actually loaded (either statically linked or via dlopen).
//
// Only the RPC functions that are actually referenced from ggml-base are
// listed here.

#include "ggml-rpc.h"

#define WEAK __attribute__((weak))

extern "C" {

WEAK bool ggml_backend_rpc_event_defer_barrier(void) {
    return false;
}

WEAK bool ggml_backend_rpc_get_tensor_defer(void) {
    return false;
}

WEAK bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    (void)buffer;
    return false;
}

WEAK bool ggml_backend_is_rpc(ggml_backend_t backend) {
    (void)backend;
    return false;
}

WEAK bool ggml_backend_rpc_download_pending_for_dst(const struct ggml_tensor * dst) {
    (void)dst;
    return false;
}

WEAK bool ggml_backend_rpc_relay_pending_for_dst(const struct ggml_tensor * dst) {
    (void)dst;
    return false;
}

WEAK bool ggml_backend_rpc_try_download_tensor(ggml_backend_t dst_backend, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    (void)dst_backend;
    (void)src;
    (void)dst;
    return false;
}

WEAK bool ggml_backend_rpc_try_upload_tensor(ggml_backend_t src_backend, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    (void)src_backend;
    (void)src;
    (void)dst;
    return false;
}

WEAK void ggml_backend_rpc_flush_pending_relays_for_dst(const struct ggml_tensor * const * dst, size_t n_dst) {
    (void)dst;
    (void)n_dst;
}

WEAK void ggml_backend_rpc_flush_pending_downloads_for_dst(const struct ggml_tensor * const * dst, size_t n_dst) {
    (void)dst;
    (void)n_dst;
}

WEAK void ggml_backend_rpc_flush_pending_downloads(void) {
}

} // extern "C"
