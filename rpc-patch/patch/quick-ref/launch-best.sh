#!/usr/bin/env bash
# Quick-launch the best placement plan (1.20 tok/s, Exp W)
# GPU topology: ROCm(7900XTX) + 3060Ti(Docker:50051) + 3070/3090(triton:50052 single-endpoint)
#
# Usage:
#   ./launch-best.sh [label]     # label defaults to "quick-$(date +%H%M)"
#
# Prerequisites:
#   - rpc-server running on 127.0.0.1:50051 (3060Ti, Docker)
#   - rpc-server running on 192.168.8.23:50052 with CUDA0(3070) + CUDA1(3090) (triton)
#   - Prototype binary at ./build-rocm/bin/llama-server (or set LLAMA_SERVER_BIN)

set -euo pipefail

MODEL="${LLAMA_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf}"
LLAMA_BIN="${LLAMA_SERVER_BIN:-./build-rocm/bin/llama-server}"
QUICKREF_DIR="$(dirname "$(readlink -f "$0")")"
PLAN="${QUICKREF_DIR}/best-plan.json"
LABEL="${1:-quick-$(date +%H%M)}"
RUNDIR="${QUICKREF_DIR}/runs/${LABEL}"
mkdir -p "$RUNDIR"

# --- Telemetry ---
export GGML_SCHED_TRACE=1
export GGML_SCHED_PER_NODE_TIMING=1
export GGML_RPC_SERVER_TELEMETRY=1
export GGML_SCHED_TRACE_FILE="${RUNDIR}/sched-trace.jsonl"

echo "=== Best Plan Quick-Launch ==="
echo "Plan:   ${PLAN}"
echo "Rundir: ${RUNDIR}"
echo "Model:  ${MODEL}"
echo ""

timeout --signal=SIGINT 360 \
  "$LLAMA_BIN" \
    -m "$MODEL" \
    --rpc 127.0.0.1:50051,192.168.8.23:50052 \
    -c 512 -b 512 -ub 512 -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --port 8086 --no-warmup -np 1 \
    --placement-plan "$PLAN" \
    --pipeline-plus \
    --pplus-rpc-defer-barrier --pplus-rpc-get-defer --pplus-rpc-flush \
    2>&1 | tee "${RUNDIR}/llama-server-stderr.log" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# --- Wait for model load ---
echo -n "Waiting for model load..."
for i in $(seq 1 120); do
  resp=$(curl -s --max-time 2 http://127.0.0.1:8086/health 2>/dev/null || true)
  if echo "$resp" | grep -q '"ok"'; then
    echo " LOADED"
    break
  fi
  echo -n "."
  sleep 2
done
echo ""

# --- Run benchmark ---
BENCH_PROMPT='Explain gravity in simple terms.'
echo "=== Benchmark: 3 requests, 32 tokens each ==="
for j in 1 2 3; do
  echo "--- Request $j ---"
  curl -s --max-time 300 http://127.0.0.1:8086/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${BENCH_PROMPT}\",\"max_tokens\":32,\"temperature\":0.3,\"stream\":false}" \
    | tee "${RUNDIR}/request-${j}.json"
  echo ""
done

echo ""
echo "=== Done ==="
echo "Results: ${RUNDIR}/"
echo "To stop server: kill $SERVER_PID"
