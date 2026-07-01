#!/usr/bin/env bash
# 6-GPU Linux cluster: 5-GPU prod + jupiter 5070 Ti RPC (:50053).
#
# Devices (tensor-split order):
#   0 remus 5060      192.168.8.176:50051
#   1 romulus 3060    127.0.0.1:50051
#   2 triton 3090     192.168.8.23:50054
#   3 triton 3070     192.168.8.23:50055
#   4 jupiter 5070    192.168.8.21:50053
#   5 romulus 7900    local ROCm client
#
# usage:
#   source scripts/b6-gate-6gpu-env.sh
#   b6_6gpu_production_env

b6_6gpu_base_env() {
    local jupiter_ip="${B6_JUPITER_HOST:-192.168.8.21}"
    export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054,192.168.8.23:50055,${jupiter_ip}:50053}"
    export BENCH_TS="${BENCH_TS:-16,10,30,10,16,22}"
    export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
}

b6_6gpu_production_env() {
    b6_6gpu_base_env
    if [[ -n "${GGML_RPC_DUAL_SOCKET:-}" ]]; then
        :
    elif [[ "${B6_6GPU_DUAL_SOCKET:-0}" == "1" ]]; then
        export GGML_RPC_DUAL_SOCKET=1
    else
        export GGML_RPC_DUAL_SOCKET=0
    fi
    if [[ "${B6_6GPU_HASH_DEFER:-0}" == "1" ]]; then
        export GGML_RPC_HASH_DEFER=1
    fi
}

# Default KV for frontier loads: CKV turbo3 + CKD q8_0 (pathb config-f pattern).
b6_6gpu_frontier_kv_env() {
    export BENCH_CTK="${BENCH_CTK:-q8_0}"
    export BENCH_CTV="${BENCH_CTV:-turbo3}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    b6_6gpu_production_env
    b6_6gpu_frontier_kv_env
    echo "BENCH_RPC_ENDPOINT=${BENCH_RPC_ENDPOINT}"
    echo "BENCH_TS=${BENCH_TS}"
    echo "GGML_PIPELINE_PLUS=${GGML_PIPELINE_PLUS}"
    echo "GGML_RPC_DUAL_SOCKET=${GGML_RPC_DUAL_SOCKET:-0}"
    echo "BENCH_CTK=${BENCH_CTK:-} BENCH_CTV=${BENCH_CTV:-}"
fi