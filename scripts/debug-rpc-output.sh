#!/bin/bash
# Debug loop: check if RPC causes wrong output
# Usage: ./scripts/debug-rpc-output.sh [norpc|rpc|rpc0]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${ROOT}/build-rocm/bin/llama-server"
MODEL="/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf"
PORT=9997
MODE="${1:-rpc}"
TAG="debug-${MODE}"

# Kill leftover
for pid in $(ps aux | grep "[l]lama-server.*${PORT}" | awk '{print $2}'); do kill $pid 2>/dev/null; done
sleep 1

SERVER_ARGS="-m $MODEL -c 512 -np 1 -ngl 99 --no-warmup --port $PORT --host 127.0.0.1 -lv 1"
if [ "$MODE" = "rpc" ]; then
    SERVER_ARGS="$SERVER_ARGS --rpc 192.168.8.22:50052,127.0.0.1:50051 -ts 36,15,49"
elif [ "$MODE" = "rpc0" ]; then
    SERVER_ARGS="$SERVER_ARGS --rpc 192.168.8.22:50052,127.0.0.1:50051 -ts 0,0,100"
fi

# Start server in background, capture PID
timeout 60 $SERVER $SERVER_ARGS > /tmp/${TAG}-server.log 2>&1 &
SPID=$!

# Wait for HTTP
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then break; fi
    sleep 1
done

# Send request
curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Count from 1 to 5."}],"temperature":0.0,"max_tokens":20,"stream":false}' \
    > /tmp/${TAG}-result.json 2>&1

# Kill safely - use separate process group
kill $SPID 2>/dev/null || true
(sleep 5; kill -9 $SPID 2>/dev/null) &
WAIT_PID=$!
wait $SPID 2>/dev/null || true
kill $WAIT_PID 2>/dev/null || true

# Analyze
exec python3 << PYEOF
import json, os
mode = "${MODE}"
tag = "debug-" + mode
with open("/tmp/" + tag + "-result.json") as f:
    d = json.load(f)
txt = d['choices'][0]['message'].get('content','')
reasoning = d['choices'][0]['message'].get('reasoning_content','')
finish = d['choices'][0]['finish_reason']
tokens = d['usage']['completion_tokens']
print('MODE=' + mode)
print('finish=' + str(finish) + ' tokens=' + str(tokens))
print('content=' + repr(txt))
print('reasoning=' + repr(reasoning[:120]))
if tokens >= 10:
    print('VERDICT: PASS')
else:
    print('VERDICT: FAIL (too few tokens)')
PYEOF
