#!/bin/bash
# Phase 1 feedback loop: Check for get_alloc_size(nil) deadlock
# Usage: ./scripts/deadlock-check.sh <binary-path> [model-path] [timeout-sec]
#
# Returns:
#   0 (GREEN)  - Model loaded successfully (set_tensor found, no deadlock loop)
#   1 (RED)    - Deadlock detected (get_alloc_size(nil) loop with no set_tensor progress)
#   2          - Other error (binary not found, docker down, etc.)

set -euo pipefail

BINARY="${1:-./build-rocm/bin/llama-server}"
MODEL="${2:-/mnt/models/arcee-ai_Trinity-Mini-Q5_K_M.gguf}"
TIMEOUT="${3:-60}"  # seconds to wait before checking

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=9989

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    exit 2
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 2
fi

echo "=== DEADLOCK CHECK ==="
echo "Binary: $BINARY ($(du -h "$BINARY" | cut -f1))"
echo "Model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo "Timeout: ${TIMEOUT}s"
echo ""

# 1. Restart both dockers to clear stale state
echo "[1/5] Restarting dockers..."
docker restart pathd-rpc-romulus > /dev/null 2>&1 || { echo "WARN: pathd-rpc-romulus restart failed"; }
ssh -o ConnectTimeout=5 "user@192.168.8.22" "docker restart pathd-rpc-remus" > /dev/null 2>&1 || { echo "WARN: remus restart failed"; }
sleep 3

# Verify dockers are up
docker ps --filter name=pathd-rpc-romulus --format "{{.Names}} {{.Status}}" 2>/dev/null || { echo "ERROR: pathd-rpc-romulus not found"; exit 2; }
ssh -o ConnectTimeout=5 "user@192.168.8.22" "docker ps --filter name=pathd-rpc-remus --format '{{.Names}} {{.Status}}'" 2>/dev/null || { echo "ERROR: pathd-rpc-remus not found on remus"; exit 2; }

# 2. Start llama-server in background
echo "[2/5] Starting llama-server (timeout=${TIMEOUT}s)..."
# Use a simple config: no flash-attention, small context, no MTP spec
timeout $((TIMEOUT + 15)) "$BINARY" \
    --model "$MODEL" \
    --rpc "192.168.8.22:50052,127.0.0.1:50051" \
    --n-gpu-layers 99 \
    -ts "36,15,49" \
    -c 512 \
    -np 1 \
    --no-warmup \
    --fit off \
    -lv 1 \
    --port "$PORT" \
    --host 127.0.0.1 \
    > /tmp/deadlock-check-server.log 2>&1 &
SERVER_PID=$!

# 3. Wait for the timeout duration
echo "[3/5] Waiting ${TIMEOUT}s for model load..."
sleep "$TIMEOUT"

# 4. Check result
echo "[4/5] Analyzing..."

# Kill the server
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# Get docker logs from both servers
DOCKER_3060=$(docker logs pathd-rpc-romulus 2>&1)
DOCKER_REMUS=$(ssh -o ConnectTimeout=5 "user@192.168.8.22" "docker logs pathd-rpc-remus" 2>&1 || echo "(ssh failed)")

# Count key patterns
NIL_COUNT_3060=$(echo "$DOCKER_3060" | grep -c "get_alloc_size.*nil.*nil" 2>/dev/null || echo 0)
SET_TENSOR_COUNT_3060=$(echo "$DOCKER_3060" | grep -c "set_tensor" 2>/dev/null || echo 0)
GRAPH_COMPUTE_COUNT_3060=$(echo "$DOCKER_3060" | grep -c "graph_compute" 2>/dev/null || echo 0)

NIL_COUNT_REMUS=$(echo "$DOCKER_REMUS" | grep -c "get_alloc_size.*nil.*nil" 2>/dev/null || echo 0)
SET_TENSOR_COUNT_REMUS=$(echo "$DOCKER_REMUS" | grep -c "set_tensor" 2>/dev/null || echo 0)
GRAPH_COMPUTE_COUNT_REMUS=$(echo "$DOCKER_REMUS" | grep -c "graph_compute" 2>/dev/null || echo 0)

echo ""
echo "=== Results ==="
echo "3060 Ti:  nil=$NIL_COUNT_3060  set_tensor=$SET_TENSOR_COUNT_3060  graph_compute=$GRAPH_COMPUTE_COUNT_3060"
echo "Remus:    nil=$NIL_COUNT_REMUS  set_tensor=$SET_TENSOR_COUNT_REMUS  graph_compute=$GRAPH_COMPUTE_COUNT_REMUS"
echo ""

# Determine pass/fail:
# GREEN: At least one device received set_tensor or graph_compute (loading made progress)
# RED: Both devices stuck in get_alloc_size(nil) with no set_tensor at all
if [ "$SET_TENSOR_COUNT_3060" -gt 0 ] || [ "$SET_TENSOR_COUNT_REMUS" -gt 0 ] || \
   [ "$GRAPH_COMPUTE_COUNT_3060" -gt 0 ] || [ "$GRAPH_COMPUTE_COUNT_REMUS" -gt 0 ]; then
    echo "RESULT: GREEN - Model loading made progress (set_tensor/graph_compute found)"
    echo "Deadlock not detected."
    exit 0
elif [ "$NIL_COUNT_3060" -gt 5 ] && [ "$NIL_COUNT_REMUS" -gt 5 ]; then
    echo "RESULT: RED - Both devices stuck in get_alloc_size(nil) loop"
    echo "Deadlock detected!"
    echo ""
    echo "Last 5 lines from 3060 docker:"
    echo "$DOCKER_3060" | tail -5
    echo ""
    echo "Last 5 lines from remus docker:"
    echo "$DOCKER_REMUS" | tail -5
    exit 1
elif [ "$NIL_COUNT_3060" -gt 5 ]; then
    echo "RESULT: RED - 3060 Ti stuck in get_alloc_size(nil) loop"
    echo "Deadlock detected (asymmetric)"
    echo ""
    echo "Last 5 lines from 3060 docker:"
    echo "$DOCKER_3060" | tail -5
    exit 1
else
    echo "RESULT: AMBIGUOUS - Can't determine from patterns"
    echo "Server log:"
    head -20 /tmp/deadlock-check-server.log 2>/dev/null
    exit 2
fi
