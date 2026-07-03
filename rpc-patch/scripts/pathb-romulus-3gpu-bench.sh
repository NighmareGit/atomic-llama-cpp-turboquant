#!/usr/bin/env bash
# Run 3-GPU bench on romulus (7900 ROCm client + 2 RPC workers).
#
# usage: pathb-romulus-3gpu-bench.sh [preset] [label]
#   preset: primary-5060-3060 | primary-5060-5070 | experimental-5060-6600
#   label default: trace-g-3gpu-<preset>
#
# env overrides: BENCH_RPC_ENDPOINT, BENCH_TS, BENCH_CTK, BENCH_CTV, BENCH_TRACE, etc.

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
        echo "error: unknown preset '$PRESET' (primary-5060-3060 | primary-5060-5070 | experimental-5060-6600)" >&2
        exit 1
        ;;
esac

GEN="${BENCH_GEN_TOKENS:-128}"
TIMEOUT="${BENCH_LOAD_TIMEOUT:-1200}"
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
CTK="${BENCH_CTK:-q8_0}"
CTV="${BENCH_CTV:-q8_0}"
NCMOE="${BENCH_NCMOE:-}"
TRACE="${BENCH_TRACE:-0}"
PIPE="${GGML_PIPELINE_PLUS:-1}"

REMOTE_ENV=(
    "BENCH_GEN_TOKENS=${GEN}"
    "BENCH_LOAD_TIMEOUT=${TIMEOUT}"
    "BENCH_RPC_ENDPOINT=${ENDPOINT}"
    "BENCH_MODEL=${MODEL}"
    "BENCH_CTK=${CTK}"
    "BENCH_CTV=${CTV}"
    "BENCH_TS=${TS}"
    "BENCH_TRACE=${TRACE}"
    "GGML_PIPELINE_PLUS=${PIPE}"
)
if [[ -n "$NCMOE" ]]; then
    REMOTE_ENV+=("BENCH_NCMOE=${NCMOE}")
else
    REMOTE_ENV+=("BENCH_NCMOE=")
fi
REMOTE_ENV+=("$@")

REMOTE_CMD="cd /home/hunter/atomic-llama-cpp-turboquant && ${REMOTE_ENV[*]} bash /home/hunter/bench-5gpu.sh ${LABEL}"

SSH_BASE=(ssh -o StrictHostKeyChecking=no)
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_BASE=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no)
fi

echo "=== romulus 3-GPU bench: ${LABEL} (preset=${PRESET}) ==="
echo "remote: ${SSH_HOST}"
echo "endpoint=${ENDPOINT} ts=${TS} gen=${GEN} load_timeout=${TIMEOUT} trace=${TRACE} pipeline_plus=${PIPE}"
"${SSH_BASE[@]}" "$SSH_HOST" "$REMOTE_CMD"