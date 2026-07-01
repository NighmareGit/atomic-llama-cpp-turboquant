#!/usr/bin/env bash
# B+6 gate OFF-bisect runner (ADR-0001, DESIGN-b6-bisect-protocol.md).
#
# usage: bash scripts/b6-gate-bisect-run.sh <bisect> [extra profiler args...]
#
# bisect (2-GPU triton):
#   no-partial        B+8 OFF  (GGML_PIPELINE_BARRIER_PARTIAL=0)
#   no-async-copy     B+10 OFF (GGML_SCHED_MOE_ASYNC_COPY=0)
#   no-defer          B+9 OFF  (GGML_RPC_EVENT_DEFER_BARRIER=0)
#   no-get-defer      B+12 OFF (GGML_RPC_GET_TENSOR_DEFER=0)
#   no-dual-socket    B+11 OFF (GGML_RPC_DUAL_SOCKET=0)
#   canonical-romulus romulus-native canonical re-bench (all mitigations ON)
#
# bisect (4-GPU, set B6_GATE_PRESET=b6-4gpu-g or b6-4gpu-g-triton):
#   no-flush          B+7a' OFF (GGML_RPC_MULTI_SOCKET_FLUSH=0)
#   no-dual-socket    B+11 OFF (GGML_RPC_DUAL_SOCKET=0)
#
# env:
#   B6_GATE_CLIENT=romulus|remus-docker  (default: romulus)
#   B6_GATE_PRESET=b6-2gpu-f-triton|b6-4gpu-g  (default: b6-2gpu-f-triton)
#   PROFILER_OUT_DIR                     (optional override)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BISECT="${1:?bisect required}"
shift || true

CLIENT="${B6_GATE_CLIENT:-romulus}"
PRESET="${B6_GATE_PRESET:-b6-2gpu-f-triton}"

export PROFILER_SKIP_VALIDATE=1
export BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-384}"
export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
export BENCH_TRACE=1

case "$BISECT" in
    no-partial)
        export GGML_PIPELINE_BARRIER_PARTIAL=0
        SUFFIX="no-partial"
        ;;
    no-async-copy)
        export GGML_SCHED_MOE_ASYNC_COPY=0
        SUFFIX="no-async-copy"
        ;;
    no-defer)
        export GGML_RPC_EVENT_DEFER_BARRIER=0
        SUFFIX="no-defer"
        ;;
    no-get-defer)
        export GGML_RPC_GET_TENSOR_DEFER=0
        SUFFIX="no-get-defer"
        ;;
    no-dual-socket)
        export GGML_RPC_DUAL_SOCKET=0
        SUFFIX="no-dual-socket"
        ;;
    no-flush)
        export GGML_RPC_MULTI_SOCKET_FLUSH=0
        SUFFIX="no-flush"
        ;;
    canonical-romulus)
        SUFFIX="romulus-native"
        ;;
    -h|--help)
        sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "error: unknown bisect: ${BISECT}" >&2
        exit 1
        ;;
esac

case "$PRESET" in
    b6-2gpu-f-triton)
        OUT_NAME="b6-2gpu-f-triton-n384-${SUFFIX}"
        OUT_PATTERN='^b6-2gpu-f-triton-n384-'
        ;;
    b6-4gpu-g)
        if [[ "$BISECT" == "canonical-romulus" ]]; then
            OUT_NAME="b6-4gpu-g-n384-romulus-native"
        else
            OUT_NAME="b6-4gpu-g-n384-romulus-native-${SUFFIX}"
        fi
        OUT_PATTERN='^b6-4gpu-g-n384-romulus-native'
        ;;
    b6-4gpu-g-triton)
        if [[ "$BISECT" == "canonical-romulus" ]]; then
            OUT_NAME="b6-4gpu-g-triton-n384-romulus-native"
        else
            OUT_NAME="b6-4gpu-g-triton-n384-romulus-native-${SUFFIX}"
        fi
        OUT_PATTERN='^b6-4gpu-g-triton-n384-romulus-native'
        ;;
    *)
        echo "error: unknown B6_GATE_PRESET: ${PRESET}" >&2
        exit 1
        ;;
esac

GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "$GIT_SHA" == "unknown" ]]; then
    echo "warning: GIT_SHA unknown; bisect audit trail degraded" >&2
fi

case "$CLIENT" in
    romulus)
        export B6_CLIENT_KIND=rocm-native
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${OUT_NAME}}"
        ;;
    remus-docker)
        export B6_CLIENT_KIND=cuda-docker
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-/src/benches/path-b-plus/${OUT_NAME}}"
        ;;
    *)
        echo "error: B6_GATE_CLIENT must be romulus or remus-docker (got ${CLIENT})" >&2
        exit 1
        ;;
esac

BASE="$(basename "$PROFILER_OUT_DIR")"
if [[ ! "$BASE" =~ $OUT_PATTERN ]]; then
    echo "error: PROFILER_OUT_DIR basename must match ${OUT_PATTERN}* (got ${BASE})" >&2
    exit 1
fi

echo "=== b6 bisect: ${BISECT} preset=${PRESET} client=${CLIENT} out=${PROFILER_OUT_DIR} ==="

EXTRA=(--skip-rpc-validate "$@")

case "$CLIENT" in
    romulus)
        exec bash "${ROOT}/scripts/b6-gate-profiler-romulus.sh" "$PRESET" "${EXTRA[@]}"
        ;;
    remus-docker)
        exec bash "${ROOT}/scripts/b6-gate-remus-docker.sh" "$PRESET" "${EXTRA[@]}"
        ;;
esac