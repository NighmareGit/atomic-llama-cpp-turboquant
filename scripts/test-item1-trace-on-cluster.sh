#!/bin/bash
# Deploy and run live test for Item 1 (unified trace_id) on the cluster
# Run this on Romulus (192.168.8.108)
#
# Prerequisites:
# - Code synced to Path-B-Event-Support-Pipeline-Plus with item 1 changes
# - Built with cmake (GGML_RPC etc enabled)
# - Models in /mnt/models or appropriate path
# - Triton workers up (192.168.8.23 etc)
#
# Usage on cluster:
#   chmod +x scripts/test-item1-trace-on-cluster.sh
#   ./scripts/test-item1-trace-on-cluster.sh

set -euo pipefail

echo "=== Item 1 Live Test: Unified trace_id on 4-GPU (romulus-native + triton) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Config - adjust as needed for your canonical 4-GPU romulus-native + triton setup
# From project history: romulus 7900XTX client + triton 3090 worker(s)
MODEL=${MODEL:-/mnt/models/llama-3.1-70b-instruct.Q4_K_M.gguf}  # or your model
N_PREDICT=${N_PREDICT:-384}
TS=${TS:-25,12,25,38}
RPC_ENDPOINTS=${RPC_ENDPOINTS:-"192.168.8.23:50054"}  # primary triton 3090 ; extend for full 4-GPU
PORT=${PORT:-8080}
PROMPT=${PROMPT:-"The quick brown fox jumps over the lazy dog. "}

# Trace files
TELEMETRY_DIR=/tmp/item1-telemetry-$(date +%Y%m%d-%H%M%S)
mkdir -p "$TELEMETRY_DIR"

echo "Model: $MODEL"
echo "n_predict: $N_PREDICT"
echo "TS: $TS"
echo "RPC: $RPC_ENDPOINTS"
echo "Telemetry: $TELEMETRY_DIR"

# 1. Baseline (no traces) - verify no regression
echo ""
echo "=== 1. BASELINE RUN (traces OFF) ==="
export GGML_PIPELINE_PLUS=1
unset GGML_SCHED_TRACE GGML_RPC_TRACE GGML_PIPELINE_TRACE

./build/bin/llama-server \
  -m "$MODEL" \
  --n-gpu-layers 99 \
  -c 4096 \
  --split-mode layer \
  -ts "$TS" \
  --rpc "$RPC_ENDPOINTS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --n-predict "$N_PREDICT" \
  2>&1 | tee "$TELEMETRY_DIR/baseline.log" &
SERVER_PID=$!

sleep 20  # allow server start

curl -s -X POST "http://127.0.0.1:$PORT/completion" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT}" \
  | tee "$TELEMETRY_DIR/baseline-result.json"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=== Baseline G check ==="
grep -iE 't/s|tokens per second|eval time|G =' "$TELEMETRY_DIR/baseline.log" | tail -5 || true

# 2. Trace run for item 1
echo ""
echo "=== 2. TRACE RUN (item 1) ==="
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE=1
export GGML_PIPELINE_TRACE=1
export GGML_SCHED_TRACE_FILE="$TELEMETRY_DIR/sched-trace.jsonl"
export GGML_RPC_TRACE_FILE="$TELEMETRY_DIR/rpc-trace.jsonl"
export GGML_PIPELINE_TRACE_FILE="$TELEMETRY_DIR/pipeline-trace.jsonl"

./build/bin/llama-server \
  -m "$MODEL" \
  --n-gpu-layers 99 \
  -c 4096 \
  --split-mode layer \
  -ts "$TS" \
  --rpc "$RPC_ENDPOINTS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --n-predict "$N_PREDICT" \
  2>&1 | tee "$TELEMETRY_DIR/trace.log" &
SERVER_PID=$!

sleep 20

curl -s -X POST "http://127.0.0.1:$PORT/completion" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N_PREDICT}" \
  | tee "$TELEMETRY_DIR/trace-result.json"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# 3. Verification
echo ""
echo "=== 3. VERIFICATION (pathb-hotpath-summary.sh) ==="
./rpc-patch/scripts/pathb-hotpath-summary.sh "$TELEMETRY_DIR/" | tee "$TELEMETRY_DIR/summary.txt"

echo ""
echo "=== ITEM 1 VERIFICATION RESULTS ==="
cat "$TELEMETRY_DIR/summary.txt"

# Check criteria
JOIN_RATE=$(grep -o 'trace_id_join_rate=[0-9.]*' "$TELEMETRY_DIR/summary.txt" | cut -d= -f2 || echo "0")
G_VALUE=$(grep -oE 'G = [0-9.]+|eval time.*[0-9.]+ t/s' "$TELEMETRY_DIR/trace.log" | head -1 || echo "")

echo ""
echo "=== FINAL CHECK ==="
if (( $(echo "$JOIN_RATE > 0.95" | bc -l) )); then
  echo "PASS: trace_id_join_rate=$JOIN_RATE (>0.95)"
else
  echo "FAIL: trace_id_join_rate=$JOIN_RATE (<=0.95)"
fi

echo "G check: $G_VALUE (target >=66 t/s on 4-GPU baseline)"
echo "Telemetry saved to: $TELEMETRY_DIR"
echo "Copy this dir back for analysis if needed."

echo "Test complete. Report the summary.txt and G numbers."