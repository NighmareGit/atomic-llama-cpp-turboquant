#!/bin/bash
# Test: RPC endpoints in args but dockers are DOWN
set -euo pipefail
exec > /tmp/test-rpc-down.log 2>&1
set -x
ROOT="/home/hunter/scratch/prototype-repo"
SERVER="${ROOT}/build-rocm/bin/llama-server"
MODEL="/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf"
PORT=9996
TAG="debug-rpc-down"

# Start server with --rpc but dockers DOWN (they were removed before starting this)
timeout 60 $SERVER -m $MODEL -c 512 -np 1 -ngl 99 --no-warmup --port $PORT --host 127.0.0.1 -lv 1 --rpc 192.168.8.22:50052,127.0.0.1:50051 -ts 36,15,49 > /tmp/${TAG}-server.log 2>&1 &
SPID=$!
echo "Server PID: $SPID"

for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then break; fi
  sleep 1
done

curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" -H "Content-Type: application/json" -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Count from 1 to 5."}],"temperature":0.0,"max_tokens":20,"stream":false}' > /tmp/${TAG}-result.json 2>&1

kill -9 $SPID 2>/dev/null || true
