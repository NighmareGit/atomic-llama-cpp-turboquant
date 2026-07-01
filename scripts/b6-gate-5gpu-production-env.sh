#!/usr/bin/env bash
# Production env for 5-GPU Linux cluster (b6-5gpu-g).
#
# usage:
#   source scripts/b6-gate-5gpu-production-env.sh
#   bash scripts/b6-gate-run-remote.sh b6-5gpu-g-prod
#
# env overrides:
#   B6_5GPU_DUAL_SOCKET=0   disable B+11 dual-socket (default 1 on prod label)
#   B6_5GPU_HASH_DEFER=1    enable GGML_RPC_HASH_DEFER (default 0)

b6_5gpu_base_env() {
    export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054,192.168.8.23:50055}"
    export BENCH_TS="${BENCH_TS:-20,10,30,10,30}"
    export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
}

b6_5gpu_production_env() {
    b6_5gpu_base_env
    if [[ "${B6_5GPU_DUAL_SOCKET:-1}" == "1" ]]; then
        export GGML_RPC_DUAL_SOCKET=1
    else
        export GGML_RPC_DUAL_SOCKET=0
    fi
    if [[ "${B6_5GPU_HASH_DEFER:-0}" == "1" ]]; then
        export GGML_RPC_HASH_DEFER=1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    b6_5gpu_production_env
    echo "BENCH_RPC_ENDPOINT=${BENCH_RPC_ENDPOINT}"
    echo "BENCH_TS=${BENCH_TS}"
    echo "GGML_PIPELINE_PLUS=${GGML_PIPELINE_PLUS}"
    echo "GGML_RPC_DUAL_SOCKET=${GGML_RPC_DUAL_SOCKET:-0}"
    echo "GGML_RPC_HASH_DEFER=${GGML_RPC_HASH_DEFER:-0}"
fi