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
MODEL="/mnt/models/${MODEL_BASENAME}"
DRAFT="${DRAFT_GGUF:-/mnt/models/gemma-assistant-mtp.gguf}"

N_PREDICT="${N_PREDICT:-384}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "=== 5-GPU MTP + VRAM preflight (lateral config) ==="
python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \
  --preset "${PRESET}" \
  --gguf "${MODEL}" \
  --ts-mode equal \
  --phase load \
  --mtp || true

# Example from plan/history for 5-GPU; override after preflight
TS="${TS:-30,14,16,40,20}"   # adjust based on preflight output for your MTP model

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

SERVER="${LLAMA_SERVER:-./build/bin/llama-server}"

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
  --draft 1 \
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