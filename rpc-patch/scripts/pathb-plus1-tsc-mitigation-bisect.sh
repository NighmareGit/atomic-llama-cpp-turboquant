#!/usr/bin/env bash
# Phase 1f: Plus=1 TSC mitigation bisect (strategy D).
# One flag OFF per arm; GGML_PIPELINE_PLUS=1 always.
#
# usage: pathb-plus1-tsc-mitigation-bisect.sh [arm...]
#   default arms: canonical no-partial no-defer no-async-copy no-get-defer
#
# docs: docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-semantic-collapse.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .scratch/cluster-access.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .scratch/cluster-access.env
    set +a
fi

BENCH="${SCRIPT_DIR}/pathb-romulus-2gpu-bench.sh"
LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-bisect}"
mkdir -p "$LOG_DIR"

run_arm() {
    local arm="$1"
    local label="trace-g-2gpu-tsc-bisect-${arm}"

    unset GGML_PIPELINE_BARRIER_PARTIAL
    unset GGML_RPC_EVENT_DEFER_BARRIER
    unset GGML_SCHED_MOE_ASYNC_COPY
    unset GGML_RPC_GET_TENSOR_DEFER
    unset GGML_RPC_MULTI_SOCKET_FLUSH

    case "$arm" in
        canonical) ;;
        no-partial)     export GGML_PIPELINE_BARRIER_PARTIAL=0 ;;
        no-defer)       export GGML_RPC_EVENT_DEFER_BARRIER=0 ;;
        no-async-copy)  export GGML_SCHED_MOE_ASYNC_COPY=0 ;;
        no-get-defer)   export GGML_RPC_GET_TENSOR_DEFER=0 ;;
        all-off)
            export GGML_PIPELINE_BARRIER_PARTIAL=0
            export GGML_RPC_EVENT_DEFER_BARRIER=0
            export GGML_RPC_MULTI_SOCKET_FLUSH=0
            export GGML_SCHED_MOE_ASYNC_COPY=0
            export GGML_RPC_GET_TENSOR_DEFER=0
            ;;
        *)
            echo "error: unknown arm '$arm'" >&2
            exit 1
            ;;
    esac

    echo "=== arm=${arm} label=${label} ==="
    echo "  GGML_PIPELINE_PLUS=1"
    echo "  GGML_PIPELINE_BARRIER_PARTIAL=${GGML_PIPELINE_BARRIER_PARTIAL:-<default>}"
    echo "  GGML_RPC_EVENT_DEFER_BARRIER=${GGML_RPC_EVENT_DEFER_BARRIER:-<default>}"
    echo "  GGML_SCHED_MOE_ASYNC_COPY=${GGML_SCHED_MOE_ASYNC_COPY:-<default>}"
    echo "  GGML_RPC_GET_TENSOR_DEFER=${GGML_RPC_GET_TENSOR_DEFER:-<default>}"

    GGML_PIPELINE_PLUS=1 \
        BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_TRACE=0 \
        "$BENCH" "$label" 2>&1 | tee "${LOG_DIR}/${arm}.log"

    echo "BENCH_DONE arm=${arm} log=${LOG_DIR}/${arm}.log"
}

ARMS=()
if [[ $# -gt 0 ]]; then
    ARMS=("$@")
else
    ARMS=(canonical no-partial no-defer no-async-copy no-get-defer)
fi

for arm in "${ARMS[@]}"; do
    run_arm "$arm"
done

echo "=== bisect complete; logs in ${LOG_DIR} ==="