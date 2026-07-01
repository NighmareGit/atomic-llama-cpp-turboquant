#!/usr/bin/env bash
# Topology-aware performance tuning for Path-B-Plus RPC clients.
#
# usage:
#   source scripts/b6-gate-performance-env.sh
#   b6_apply_performance_tuning "$BENCH_RPC_ENDPOINT"
#
# overrides (force; skip auto rules):
#   B6_PERF_AUTO=0
#   GGML_RPC_DUAL_SOCKET=0|1
#   GGML_RPC_HASH_DEFER=0|1

b6_count_rpc_endpoints() {
    local rpc="${1:-}"
    if [[ -z "$rpc" ]]; then
        echo 0
        return
    fi
    local n=1
    local rest="$rpc"
    while [[ "$rest" == *","* ]]; do
        rest="${rest#*,}"
        n=$((n + 1))
    done
    echo "$n"
}

# Rules from bisect evidence (2026-07-01). See rpc-patch/docs/b6-gate/PERFORMANCE-RULES.md
b6_recommend_dual_socket() {
    local rpc_count="$1"
    case "$rpc_count" in
        4) echo 1 ;;  # 5-GPU Linux (4 RPC + local): +18% n128, +10% n2048
        3) echo 0 ;;  # 4-GPU (3 RPC + local): -9% G when ON (until 3gpu retest updates)
        2) echo 0 ;;  # 3-GPU (2 RPC + local): pending retest; default OFF
        1) echo 0 ;;  # 2-GPU
        *) echo 0 ;;
    esac
}

b6_recommend_hash_defer() {
    local rpc_count="$1"
    if [[ "$rpc_count" -ge 4 ]]; then
        echo 0
    else
        echo 0
    fi
}

b6_apply_performance_tuning() {
    local rpc="${1:-${BENCH_RPC_ENDPOINT:-}}"
    local rpc_count
    rpc_count="$(b6_count_rpc_endpoints "$rpc")"

    if [[ "${B6_PERF_AUTO:-1}" != "1" ]]; then
        return 0
    fi

    if [[ -z "${GGML_RPC_DUAL_SOCKET:-}" ]]; then
        export GGML_RPC_DUAL_SOCKET
        GGML_RPC_DUAL_SOCKET="$(b6_recommend_dual_socket "$rpc_count")"
    fi

    if [[ -z "${GGML_RPC_HASH_DEFER:-}" ]]; then
        export GGML_RPC_HASH_DEFER
        GGML_RPC_HASH_DEFER="$(b6_recommend_hash_defer "$rpc_count")"
    fi
}

b6_print_performance_tuning() {
    local rpc="${1:-${BENCH_RPC_ENDPOINT:-}}"
    local rpc_count
    rpc_count="$(b6_count_rpc_endpoints "$rpc")"
    echo "b6_rpc_endpoints=${rpc_count}"
    echo "GGML_RPC_DUAL_SOCKET=${GGML_RPC_DUAL_SOCKET:-$(b6_recommend_dual_socket "$rpc_count")}"
    echo "GGML_RPC_HASH_DEFER=${GGML_RPC_HASH_DEFER:-$(b6_recommend_hash_defer "$rpc_count")}"
    echo "B6_PERF_AUTO=${B6_PERF_AUTO:-1}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    rpc="${1:-${BENCH_RPC_ENDPOINT:-}}"
    b6_apply_performance_tuning "$rpc"
    b6_print_performance_tuning "$rpc"
fi