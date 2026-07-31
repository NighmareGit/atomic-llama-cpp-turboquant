#!/usr/bin/env bash
# Run 4-GPU bench on romulus (7900 ROCm client + 3 CUDA RPC workers).
#
# Default topology (no RX6600):
#   client 7900 XTX + workers 3060 Ti (:50051 local), 5060 Ti (remus :50051), 5070 Ti (win :50053)
#
# usage: pathb-romulus-4gpu-bench.sh [label] [extra env assignments...]
#   label default: trace-g-4gpu-primary
#
# env:
#   BENCH_4GPU_PRESET   primary (default) | legacy-6600
#   PATHB_ROMULUS_SSH_PASS, BENCH_GEN_TOKENS, BENCH_LOAD_TIMEOUT, BENCH_TRACE, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="${1:-trace-g-4gpu-primary}"
shift || true

SSH_HOST="${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}"
SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"

GEN="${BENCH_GEN_TOKENS:-128}"
TIMEOUT="${BENCH_LOAD_TIMEOUT:-4200}"
# primary = 7900 + 3060 + 5060 + 5070; legacy-6600 = old 6600 topology (experimental)
PRESET="${BENCH_4GPU_PRESET:-primary}"
WIN_IP="${PATHB_WIN_RPC_IP:-192.168.8.21}"
if [[ -z "${BENCH_RPC_ENDPOINT:-}" ]]; then
    case "$PRESET" in
        legacy-6600) ENDPOINT="192.168.8.176:50051,192.168.8.176:50052,127.0.0.1:50051" ;;
        *)           ENDPOINT="192.168.8.176:50051,127.0.0.1:50051,${WIN_IP}:50053" ;;
    esac
else
    ENDPOINT="${BENCH_RPC_ENDPOINT}"
fi
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
CTK="${BENCH_CTK:-q4_0}"
CTV="${BENCH_CTV:-q4_0}"
NCMOE="${BENCH_NCMOE:-}"
# ts order: ROCm0 client, RPC0, RPC1, RPC2 (endpoint order above)
if [[ -z "${BENCH_TS:-}" ]]; then
    case "$PRESET" in
        legacy-6600) TS="28,12,28,32" ;;
        *)           TS="36,24,24,16" ;;  # 7900, 5060, 5070, 3060 by VRAM share
    esac
else
    TS="${BENCH_TS}"
fi
TRACE="${BENCH_TRACE:-1}"
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

echo "=== romulus 4-GPU bench: ${LABEL} ==="
echo "remote: ${SSH_HOST}"
echo "endpoint=${ENDPOINT} ts=${TS} gen=${GEN} load_timeout=${TIMEOUT} trace=${TRACE} pipeline_plus=${PIPE}"
"${SSH_BASE[@]}" "$SSH_HOST" "$REMOTE_CMD"