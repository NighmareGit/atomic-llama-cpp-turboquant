#!/usr/bin/env bash
# Run 2-GPU bench on romulus (7900 ROCm client + single remus 5060 RPC).
#
# usage: pathb-romulus-2gpu-bench.sh [label] [extra env assignments...]
#   label default: trace-g-2gpu-primary
#
# env overrides: BENCH_RPC_ENDPOINT, BENCH_TS, BENCH_CTK, BENCH_CTV, BENCH_TRACE, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="${1:-trace-g-2gpu-primary}"
shift || true

SSH_HOST="${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}"
SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"

ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
TS="${BENCH_TS:-50,50}"
GEN="${BENCH_GEN_TOKENS:-128}"
TIMEOUT="${BENCH_LOAD_TIMEOUT:-1200}"
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
CTK="${BENCH_CTK:-q8_0}"
CTV="${BENCH_CTV:-q8_0}"
NCMOE="${BENCH_NCMOE:-}"
TRACE="${BENCH_TRACE:-0}"
PIPE="${GGML_PIPELINE_PLUS:-1}"
P0_FULL="${GGML_PIPELINE_P0_FULL_SYNC:-}"
P1_FULL="${GGML_PIPELINE_P1_FULL_SYNC:-}"
SCHED_LEGACY="${GGML_PIPELINE_SCHED_LEGACY:-}"

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
if [[ -n "$P0_FULL" ]]; then
    REMOTE_ENV+=("GGML_PIPELINE_P0_FULL_SYNC=${P0_FULL}")
fi
if [[ -n "$P1_FULL" ]]; then
    REMOTE_ENV+=("GGML_PIPELINE_P1_FULL_SYNC=${P1_FULL}")
fi
if [[ -n "$SCHED_LEGACY" ]]; then
    REMOTE_ENV+=("GGML_PIPELINE_SCHED_LEGACY=${SCHED_LEGACY}")
fi
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

echo "=== romulus 2-GPU bench: ${LABEL} ==="
echo "remote: ${SSH_HOST}"
echo "endpoint=${ENDPOINT} ts=${TS} gen=${GEN} load_timeout=${TIMEOUT} trace=${TRACE} pipeline_plus=${PIPE} p0_full=${P0_FULL:-0} p1_full=${P1_FULL:-0} sched_legacy=${SCHED_LEGACY:-0}"
"${SSH_BASE[@]}" "$SSH_HOST" "$REMOTE_CMD"