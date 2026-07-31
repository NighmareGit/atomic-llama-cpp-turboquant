#!/bin/bash
# 4-GPU MTP test (triton preset) with VRAM preflight + lateral config.
# Run on Romulus.
# Models in /mnt/models/
#
# Prereqs:
#   - Latest commit pulled (after push)
#   - Built with trace_id + stubs (or use Docker image)
#   - Run vram preflight first to get recommended -ts
#
# Usage:
#   ./scripts/test-4gpu-mtp.sh [MODEL_BASENAME]
#   e.g. ./scripts/test-4gpu-mtp.sh gemma-4-26B-A4B-APEX-I-Compact.gguf

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRESET="b6-4gpu-g-triton"
MODEL_BASENAME="${1:-gemma-4-26B-A4B-APEX-I-Compact.gguf}"
MODEL=/mnt/models/gemma-4-26B-A4B-APEX-I-Compact.gguf

# MTP draft (assistant) - adjust or use quantize script if needed
DRAFT=/mnt/models/gemma-4-12B-it-assistant-Q8_0.gguf

N_PREDICT="${N_PREDICT:-384}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# First, run VRAM preflight (lateral config tool) to get -ts recommendation
echo "=== Running VRAM preflight for ${PRESET} (MTP-aware) ==="
PF_OUTPUT=$(python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \
  --preset "${PRESET}" \
  --gguf "${MODEL}" \
  --ts-mode equal \
  --phase load \
  --mtp 2>&1 || true)
echo "$PF_OUTPUT"

# Capture BENCH_TS (preferred) or fallback to comment
TS=14,39,23,24
if [ -z "$TS" ]; then
  TS=14,39,23,24
fi
TS=14,39,23,24
export TS  # for any sub-calls

echo "Using TS=14,39,23,24
echo "Model: ${MODEL}"
echo "Draft: ${DRAFT}"

echo "Using TS=14,39,23,24
echo "Model: ${MODEL}"
echo "Draft: ${DRAFT}"

# Optional: run lateral config matrix or ts probe for this preset
# bash rpc-patch/scripts/pathb-config-c-matrix.sh ... or rpc-ts-fit-probe.sh

export GGML_PIPELINE_PLUS=1
export LLAMA_PIPELINE_DEPTH2=1   # enable Layer A stubs / MTP depth-2 surface

# Traces (from item 1) - optional for verification
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE=1
export GGML_PIPELINE_TRACE=1
export GGML_SCHED_TRACE_FILE=/tmp/4gpu-mtp-sched.jsonl
export GGML_RPC_TRACE_FILE=/tmp/4gpu-mtp-rpc.jsonl
export GGML_PIPELINE_TRACE_FILE=/tmp/4gpu-mtp-pipeline.jsonl

# Example RPC for 4-GPU triton on romulus (adjust IPs/ports)
RPC_LIST=192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054

SERVER="${LLAMA_SERVER:-./build/bin/llama-server}"

echo "=== Starting 4-GPU MTP server ==="
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
  --draft 1 \   # or MTP specific flags from your quant/run scripts
  2>&1 | tee /tmp/4gpu-mtp-server.log &

SERVER_PID=$!
sleep 15

echo "=== Sending test prompt (MTP) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/completion" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"The quick brown fox jumps over the lazy dog. \",\"n_predict\":${N_PREDICT}}" \
  | tee /tmp/4gpu-mtp-result.json

kill $SERVER_PID 2>/dev/null || true

echo "=== Running hotpath summary (with trace_id from item 1) ==="
./rpc-patch/scripts/pathb-hotpath-summary.sh /tmp/ | tee /tmp/4gpu-mtp-summary.txt

echo "=== Done. Check /tmp/4gpu-mtp-* for results ==="
echo "Re-run preflight + adjust TS for your exact MTP model VRAM usage."