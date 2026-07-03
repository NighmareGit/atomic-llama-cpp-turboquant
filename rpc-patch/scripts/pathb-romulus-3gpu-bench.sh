#!/usr/bin/env bash
# Run 3-GPU bench on romulus (7900 ROCm client + 2 RPC workers).
#
# usage: pathb-romulus-3gpu-bench.sh [preset] [label]
#   preset: primary-5060-3060 | primary-5060-5070 | experimental-5060-6600
#   label default: trace-g-3gpu-<preset>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESET="${1:-primary-5060-3060}"
shift || true
LABEL="${1:-trace-g-3gpu-${PRESET}}"
shift || true

SSH_HOST="${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}"
SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"

case "$PRESET" in
    primary-5060-3060)
        ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051}"
        TS="${BENCH_TS:-50,28,22}"
        ;;
    primary-5060-5070)
        ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,192.168.8.21:50053}"
        TS="${BENCH_TS:-50,28,22}"
        ;;
    experimental-5060-6600)
        ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,192.168.8.176:50052}"
        TS="${BENCH_TS:-58,28,14}"
        ;;
    *)
        echo "error: unknown preset '$PRESET'" >&2
        exit 1
        ;;
esac

GEN="${BENCH_GEN_TOKENS:-128}"
TIMEOUT="${BENCH_LOAD_TIMEOUT:-1200}"
MODEL="${BENCH_MODEL:-}"
CTK="${BENCH_CTK:-}"
CTV="${BENCH_CTV:-}"
NCMOE="${BENCH_NCMOE:-}"
CTX="${BENCH_CTX:-}"
TRACE="${BENCH_TRACE:-0}"
RUNS="${BENCH_RUNS:-}"
EXTRA="${BENCH_EXTRA:-}"
PROMPTS="${BENCH_PROMPTS_FILE:-}"
CURL_TIMEOUT="${BENCH_CURL_TIMEOUT:-}"
SAVE_FULL="${BENCH_SAVE_FULL:-}"
MULTITURN="${BENCH_MULTITURN_KVFILL:-}"
PIPE="${GGML_PIPELINE_PLUS:-1}"
P0_FULL="${GGML_PIPELINE_P0_FULL_SYNC:-}"
SCHED_LEGACY="${GGML_PIPELINE_SCHED_LEGACY:-}"
MULTI_BACKEND_SEQ="${GGML_PIPELINE_MULTI_BACKEND_SEQ:-}"

REMOTE_ENV=(
    "BENCH_GEN_TOKENS=${GEN}"
    "BENCH_LOAD_TIMEOUT=${TIMEOUT}"
    "BENCH_RPC_ENDPOINT=${ENDPOINT}"
    "BENCH_TS=${TS}"
    "BENCH_TRACE=${TRACE}"
    "GGML_PIPELINE_PLUS=${PIPE}"
)
[[ -n "$MODEL" ]] && REMOTE_ENV+=("BENCH_MODEL=${MODEL}")
[[ -n "$CTK" ]] && REMOTE_ENV+=("BENCH_CTK=${CTK}")
[[ -n "$CTV" ]] && REMOTE_ENV+=("BENCH_CTV=${CTV}")
[[ -n "$RUNS" ]] && REMOTE_ENV+=("BENCH_RUNS=${RUNS}")
[[ -n "$CTX" ]] && REMOTE_ENV+=("BENCH_CTX=${CTX}")
[[ -n "$EXTRA" ]] && REMOTE_ENV+=("BENCH_EXTRA=${EXTRA}")
[[ -n "$PROMPTS" ]] && REMOTE_ENV+=("BENCH_PROMPTS_FILE=${PROMPTS}")
[[ -n "$CURL_TIMEOUT" ]] && REMOTE_ENV+=("BENCH_CURL_TIMEOUT=${CURL_TIMEOUT}")
[[ -n "$SAVE_FULL" ]] && REMOTE_ENV+=("BENCH_SAVE_FULL=${SAVE_FULL}")
[[ -n "$MULTITURN" ]] && REMOTE_ENV+=("BENCH_MULTITURN_KVFILL=${MULTITURN}")
[[ -n "$P0_FULL" ]] && REMOTE_ENV+=("GGML_PIPELINE_P0_FULL_SYNC=${P0_FULL}")
[[ -n "$SCHED_LEGACY" ]] && REMOTE_ENV+=("GGML_PIPELINE_SCHED_LEGACY=${SCHED_LEGACY}")
[[ -n "$MULTI_BACKEND_SEQ" ]] && REMOTE_ENV+=("GGML_PIPELINE_MULTI_BACKEND_SEQ=${MULTI_BACKEND_SEQ}")
if [[ -n "$NCMOE" ]]; then
    REMOTE_ENV+=("BENCH_NCMOE=${NCMOE}")
else
    REMOTE_ENV+=("BENCH_NCMOE=")
fi
REMOTE_ENV+=("$@")

REMOTE_PREFIX=""
for _e in "${REMOTE_ENV[@]}"; do
    REMOTE_PREFIX+="$(printf '%q ' "$_e")"
done
REMOTE_CMD="cd /home/hunter/atomic-llama-cpp-turboquant && ${REMOTE_PREFIX}bash rpc-patch/scripts/pathb-romulus-3gpu-bench-host.sh $(printf '%q' "$LABEL")"

SSH_BASE=(ssh -o StrictHostKeyChecking=no)
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_BASE=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no)
fi

echo "=== romulus 3-GPU bench: ${LABEL} (preset=${PRESET}) ==="
echo "remote: ${SSH_HOST}"
echo "endpoint=${ENDPOINT} ts=${TS} gen=${GEN} ctx=${CTX:-4096} pipeline_plus=${PIPE} multi_backend_seq=${MULTI_BACKEND_SEQ:-0}"
"${SSH_BASE[@]}" "$SSH_HOST" "$REMOTE_CMD"