#pragma once

// Async-path audit markers for ggml_backend_rpc_graph_compute.
// Each marker maps to one of the five code paths in graph_compute.
// Use as a compile-time tag and as a runtime trace key (see rpc_path_kind).

// Async-safe: returns immediately after TCP send, response drained later
#define RPC_ASYNC_GRAPH_RECOMPUTE  1
// Blocking: waits for server response (first-time graph compute)
#define RPC_BLOCKING_GRAPH_COMPUTE 1
// D6.9 GPipe stage dispatch: blocking, full response + telemetry
#define RPC_BLOCKING_GRAPH_COMPUTE_STAGE 1
// Path C multi-device reuse: async fire-and-forget via GRAPH_RECOMPUTE_ALL
#define RPC_ASYNC_GRAPH_RECOMPUTE_ALL 1
// Path C multi-device first-time: blocking GRAPH_COMPUTE_ALL with telemetry
#define RPC_BLOCKING_GRAPH_COMPUTE_ALL 1

// Path classification for the five graph_compute code paths.
enum rpc_path_kind {
    RPC_PATH_UNKNOWN = 0,
    // D6.9: per-stage GPipe dispatch (gpipe_stage >= 0)
    RPC_PATH_GRAPH_COMPUTE_STAGE,
    // Path C multi-device reuse (n_devices > 1, uid match)
    RPC_PATH_RECOMPUTE_ALL,
    // Path C multi-device first-time (n_devices > 1, new uid)
    RPC_PATH_COMPUTE_ALL,
    // Single-device reuse (uid != 0, uid == last_graph_uid)
    RPC_PATH_RECOMPUTE,
    // Single-device first-time (new uid)
    RPC_PATH_COMPUTE,
};

// Human-readable path name for trace output.
inline const char * rpc_path_name(rpc_path_kind p) {
    switch (p) {
        case RPC_PATH_GRAPH_COMPUTE_STAGE: return "stage";
        case RPC_PATH_RECOMPUTE_ALL:       return "recompute_all";
        case RPC_PATH_COMPUTE_ALL:         return "compute_all";
        case RPC_PATH_RECOMPUTE:           return "recompute";
        case RPC_PATH_COMPUTE:             return "compute";
        default:                           return "unknown";
    }
}

// Whether a path blocks waiting for a server response on the wire.
inline bool rpc_path_blocking(rpc_path_kind p) {
    switch (p) {
        case RPC_PATH_GRAPH_COMPUTE_STAGE:
        case RPC_PATH_COMPUTE_ALL:
        case RPC_PATH_COMPUTE:
            return true;
        case RPC_PATH_RECOMPUTE_ALL:
        case RPC_PATH_RECOMPUTE:
        default:
            return false;
    }
}

// rpc_trace_emit is defined in ggml-rpc.cpp (made non-static for audit use).
// Forward-declared here so the audit header is self-contained.
void rpc_trace_emit(const char * fn, const char * phase, int cmd, size_t bytes, bool blocking, int64_t elapsed_us);

// Emit a classified graph_compute trace event. Tags the trace line with the
// path kind and the correct blocking flag so the trace file can distinguish
// async vs blocking calls at runtime without reading code.
inline void rpc_trace_graph_compute(rpc_path_kind path, int cmd, size_t bytes, int64_t elapsed_us) {
    rpc_trace_emit("ggml_backend_rpc_graph_compute",
                   rpc_path_name(path), cmd, bytes,
                   rpc_path_blocking(path), elapsed_us);
}
