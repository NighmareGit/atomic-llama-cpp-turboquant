#!/bin/bash
# 5-GPU MTP test (b6-5gpu-g-prod preset) with VRAM preflight + lateral config.
# Run on Romulus. Models in /mnt/models/
#
# Usage:
#   ./scripts/test-5gpu-mtp.sh [MODEL_BASENAME]
#
# 1. Run VRAM preflight first (it will suggest -ts for MTP + 5-GPU)
# 2. Use the output TS in this script or set TS= env var
# 3. Lateral tools (preflight, probes) help avoid OOM and bad splits for MTP draft

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRESET="b6-5gpu-g-prod"
MODEL_BASENAME="${1:-gemma-4-26B-A4B-APEX-I-Compact.gguf}"
MODEL=/mnt/models/gemma-4-26B-A4B-APEX-I-Compact.gguf
DRAFT=/mnt/models/gemma-4-12B-it-assistant-Q8_0.gguf

N_PREDICT="${N_PREDICT:-384}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# Source production env for 5-GPU (blessed config: dual off, wavefront off, etc.)
if [ -f "scripts/b6-gate-5gpu-production-env.sh" ]; then
  # shellcheck source=scripts/b6-gate-5gpu-production-env.sh
  source scripts/b6-gate-5gpu-production-env.sh
  b6_5gpu_production_env || true
fi

echo "=== 5-GPU MTP + VRAM preflight (lateral config) ==="
PF_OUTPUT=$(python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \
  --preset "${PRESET}" \
  --gguf "${MODEL}" \
  --ts-mode equal \
  --phase load \
  --mtp 2>&1 || true)
echo "$PF_OUTPUT"

# Capture BENCH_TS
TS=$(echo "$PF_OUTPUT" | grep -o 'BENCH_TS=[^ ]*' | cut -d= -f2 | tr -d '"' || true)
if [ -z "$TS" ]; then
  TS=$(echo "$PF_OUTPUT" | grep -o '#.*ts=.*' | head -1 | sed 's/.*ts=//;s/ .*//' || true)
fi
TS="${TS:-30,14,16,40,20}"  # fallback; override from preflight
export TS

echo "Using TS=${TS}"
echo "Model: ${MODEL}"

export GGML_PIPELINE_PLUS=1
export LLAMA_PIPELINE_DEPTH2=1

# Traces
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE=1
export GGML_PIPELINE_TRACE=1
export GGML_SCHED_TRACE_FILE=/tmp/5gpu-mtp-sched.jsonl
export GGML_RPC_TRACE_FILE=/tmp/5gpu-mtp-rpc.jsonl
export GGML_PIPELINE_TRACE_FILE=/tmp/5gpu-mtp-pipeline.jsonl

# 5-GPU RPC list (example; use your actual workers)
RPC_LIST="${RPC_LIST:-192.168.8.23:50054,192.168.8.176:50051,...}"  # extend

SERVER="${LLAMA_SERVER:-/home/hunter/projects/atomic-llama-cpp-turboquant/build-a2/build-check/bin/llama-server}"

echo "=== Starting 5-GPU MTP ==="
$SERVER \
  -m "${MODEL}" \
  --n-gpu-layers 99 \
  -c 4096 \
  --split-mode layer \
  -ts "${TS}" \
  --rpc "${RPC_LIST}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict "${N_PREDICT}" \
  --model-draft "${DRAFT}" --spec-type mtp --draft-block-size 3 --draft-max 8 \
  2>&1 | tee /tmp/5gpu-mtp-server.log &

SERVER_PID=$!
sleep 20

curl -s -X POST "http://127.0.0.1:${PORT}/completion" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"The quick brown fox...\",\"n_predict\":${N_PREDICT}}" \
  | tee /tmp/5gpu-mtp-result.json

kill $SERVER_PID 2>/dev/null || true

./rpc-patch/scripts/pathb-hotpath-summary.sh /tmp/ | tee /tmp/5gpu-mtp-summary.txt

echo "=== 5-GPU MTP test complete ==="
echo "Always run preflight + lateral probes before changing models or GPU count."