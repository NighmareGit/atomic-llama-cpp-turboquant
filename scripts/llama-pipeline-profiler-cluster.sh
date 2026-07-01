#!/usr/bin/env bash
# Cluster wrapper for llama-pipeline-profiler (romulus 4-GPU primary preset).
#
# Runs the native profiler on romulus via SSH, or locally when PROFILER_LOCAL=1.
#
# usage:
#   ./scripts/llama-pipeline-profiler-cluster.sh [label] [extra profiler args...]
#
# env:
#   PROFILER_MODE     throughput|trace|profile|ab-plus (default: trace)
#   PROFILER_LOCAL=1  run on this host (skip SSH)
#   BENCH_MODEL, BENCH_RPC_ENDPOINT, BENCH_TS, BENCH_GEN_TOKENS, BENCH_N_PROMPT, BENCH_PROMPT_FILE
#   BENCH_TRACE=1     alias for --trace when mode is ab-plus or throughput
#   PATHB_ROMULUS_SSH, PATHB_ROMULUS_SSH_PASS

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:-profiler-4gpu-primary}"
shift || true

SSH_HOST="${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}"
SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"
WIN_IP="${PATHB_WIN_RPC_IP:-192.168.8.21}"
MODE="${PROFILER_MODE:-trace}"
GEN="${BENCH_GEN_TOKENS:-256}"
N_PROMPT="${BENCH_N_PROMPT:-0}"
PROMPT_FILE="${BENCH_PROMPT_FILE:-}"
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,${WIN_IP}:50053}"
TS="${BENCH_TS:-36,24,24,16}"
OUT_BASE="${PROFILER_OUT_BASE:-${ROOT}/benches/path-b-plus/profiler-cluster}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${PROFILER_OUT_DIR:-${OUT_BASE}/${STAMP}-${LABEL}}"
resolve_profiler_bin() {
    local cand="${1}"
    if [[ -x "$cand" || -f "$cand" ]]; then
        echo "$cand"
        return
    fi
    for alt in \
        "${cand}.exe" \
        "${ROOT}/build/bin/Release/llama-pipeline-profiler" \
        "${ROOT}/build/bin/Release/llama-pipeline-profiler.exe"; do
        if [[ -x "$alt" || -f "$alt" ]]; then
            echo "$alt"
            return
        fi
    done
    echo "$cand"
}

PROFILER_BIN="$(resolve_profiler_bin "${PROFILER_BIN:-${ROOT}/build/bin/llama-pipeline-profiler}")"
TRACE_FLAG="${BENCH_TRACE:-1}"

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

if [[ "${LABEL}" == "--help" || "${LABEL}" == "-h" ]]; then
    usage
fi

b6_mitigation_default() {
    local v="${!1-}"
    if [[ -n "$v" ]]; then
        echo "$v"
    elif [[ "${GGML_PIPELINE_PLUS:-0}" == "1" ]]; then
        echo "1"
    else
        echo "0"
    fi
}

b6_resolve_client_kind() {
    if [[ -n "${B6_CLIENT_KIND:-}" ]]; then
        echo "$B6_CLIENT_KIND"
        return
    fi
    if [[ "$OUT_DIR" == /src/* ]]; then
        echo "cuda-docker"
    else
        echo "rocm-native"
    fi
}

b6_append_env_audit() {
    local kind
    kind="$(b6_resolve_client_kind)"
    local git_sha
    git_sha="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    {
        echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "OUT_DIR=${OUT_DIR}"
        echo "PROFILER_MODE=${MODE}"
        echo "client_kind=${kind}"
        echo "GIT_SHA=${git_sha}"
        echo "RPC=${ENDPOINT}"
        echo "TS=${TS}"
        echo "N_GEN=${GEN}"
        echo "GGML_PIPELINE_PLUS=${GGML_PIPELINE_PLUS:-1}"
        echo "GGML_PIPELINE_BARRIER_PARTIAL=$(b6_mitigation_default GGML_PIPELINE_BARRIER_PARTIAL)"
        echo "GGML_PIPELINE_BARRIER_PARTIAL_STRICT=$(b6_mitigation_default GGML_PIPELINE_BARRIER_PARTIAL_STRICT)"
        echo "GGML_RPC_EVENT_DEFER_BARRIER=$(b6_mitigation_default GGML_RPC_EVENT_DEFER_BARRIER)"
        echo "GGML_SCHED_MOE_ASYNC_COPY=$(b6_mitigation_default GGML_SCHED_MOE_ASYNC_COPY)"
        echo "GGML_RPC_MULTI_SOCKET_FLUSH=$(b6_mitigation_default GGML_RPC_MULTI_SOCKET_FLUSH)"
        echo "GGML_RPC_GET_TENSOR_DEFER=$(b6_mitigation_default GGML_RPC_GET_TENSOR_DEFER)"
        echo "GGML_RPC_DUAL_SOCKET=$(b6_mitigation_default GGML_RPC_DUAL_SOCKET)"
    } >>"${OUT_DIR}/env.txt"
}

build_profiler_args() {
    local args=(
        -m "$MODEL"
        -rpc "$ENDPOINT"
        -ts "$TS"
        -n "$GEN"
        --mode "$MODE"
        --out-dir "$OUT_DIR"
    )
    if [[ -n "$PROMPT_FILE" ]]; then
        args+=(-f "$PROMPT_FILE")
    elif [[ "$N_PROMPT" != "0" ]]; then
        args+=(-p "$N_PROMPT")
    fi
    case "$MODE" in
        trace|profile|spike-check) ;;
        ab-plus|throughput)
            if [[ "$TRACE_FLAG" == "1" ]]; then
                args+=(--trace)
            fi
            ;;
    esac
    if [[ "$MODE" == "profile" ]]; then
        args+=(--with-gpu-telemetry)
    fi
    args+=("$@")
    if [[ "${PROFILER_SKIP_VALIDATE:-0}" == "1" ]]; then
        local has_skip=0 a
        for a in "${args[@]}"; do
            [[ "$a" == "--skip-rpc-validate" ]] && has_skip=1
        done
        if [[ "$has_skip" -eq 0 ]]; then
            args+=(--skip-rpc-validate)
        fi
    fi
    printf '%s\n' "${args[@]}"
}

validate_rpc() {
    echo "=== rpc preflight (R5) ==="
    echo "endpoint=${ENDPOINT} ts=${TS}"
    "$PROFILER_BIN" --validate-rpc -rpc "$ENDPOINT" -ts "$TS"
}

run_local() {
    if [[ ! -f "$PROFILER_BIN" ]] && [[ ! -x "$PROFILER_BIN" ]]; then
        echo "error: profiler not found: ${PROFILER_BIN}" >&2
        echo "hint: cmake --build build --target llama-pipeline-profiler" >&2
        exit 1
    fi
    if [[ "${PROFILER_SKIP_VALIDATE:-0}" != "1" ]]; then
        validate_rpc || exit 1
    fi
    mkdir -p "$OUT_DIR"
    mapfile -t PROF_ARGS < <(build_profiler_args "$@")
    echo "=== local profiler: ${LABEL} ==="
    echo "out=${OUT_DIR} mode=${MODE}"
    cd "$ROOT"
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH
    fi
    "$PROFILER_BIN" "${PROF_ARGS[@]}"
    echo "LABEL=${LABEL}" >>"${OUT_DIR}/env.txt"
    b6_append_env_audit
    echo "PROFILER_CLUSTER_DONE label=${LABEL} out=${OUT_DIR}"
}

run_remote() {
    REMOTE_ROOT="${PATHB_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
    REMOTE_OUT="${PATHB_ROMULUS_OUT:-${REMOTE_ROOT}/benches/path-b-plus/profiler-cluster/${STAMP}-${LABEL}}"
    REMOTE_PROF="${PATHB_ROMULUS_PROFILER:-${REMOTE_ROOT}/build/bin/llama-pipeline-profiler}"

    REMOTE_ENV=(
        "BENCH_MODEL=${MODEL}"
        "BENCH_RPC_ENDPOINT=${ENDPOINT}"
        "BENCH_TS=${TS}"
        "BENCH_GEN_TOKENS=${GEN}"
        "BENCH_N_PROMPT=${N_PROMPT}"
        "BENCH_PROMPT_FILE=${PROMPT_FILE}"
        "PROFILER_MODE=${MODE}"
        "PROFILER_OUT_DIR=${REMOTE_OUT}"
        "PROFILER_BIN=${REMOTE_PROF}"
        "BENCH_TRACE=${TRACE_FLAG}"
        "PROFILER_LOCAL=1"
    )

    REMOTE_CMD="cd ${REMOTE_ROOT} && ${REMOTE_ENV[*]} bash scripts/llama-pipeline-profiler-cluster.sh ${LABEL}"

    SSH_BASE=(ssh -o StrictHostKeyChecking=no)
    if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
        SSH_BASE=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no)
    fi

    echo "=== romulus profiler: ${LABEL} ==="
    echo "remote: ${SSH_HOST}"
    echo "out=${REMOTE_OUT} mode=${MODE} gen=${GEN}"
    "${SSH_BASE[@]}" "$SSH_HOST" "$REMOTE_CMD"
    echo "remote out: ${REMOTE_OUT}"
}

if [[ "${PROFILER_LOCAL:-0}" == "1" ]]; then
    run_local "$@"
else
    run_remote "$@"
fi