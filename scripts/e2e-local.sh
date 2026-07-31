#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build-rocm/bin/llama-server}"
PORT=9988
HOST="127.0.0.1"
MAIN="${MAIN:-/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf}"
RESULT_DIR="/tmp/e2e-local-$$"
mkdir -p "$RESULT_DIR"

echo "=== E2E LOCAL TEST ==="
echo "Server: $SERVER"
echo "Model:  $MAIN"
echo "RPC:    127.0.0.1:50051"

# Kill leftover
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true; sleep 1

docker restart pathd-rpc-romulus > /dev/null 2>&1; sleep 2

timeout 300 "$SERVER" \
    -m "$MAIN" \
    -md "$MAIN" --spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-n-min 0 \
    --rpc "127.0.0.1:50051" \
    -ngl 99 -ngld 99 \
    -ts "0,100" \
    -c 4096 -np 1 -fa 1 \
    --host "$HOST" --port "$PORT" \
    --no-warmup --cont-batching --log-disable \
    > "$RESULT_DIR/server.log" 2>&1 &
SERVER_PID=$!
sleep 2

for i in $(seq 1 60); do
    if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        echo "  HTTP ready after ${i}s"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: Server failed to start"
        cat "$RESULT_DIR/server.log" | tail -20
        exit 1
    fi
    sleep 2
done

for i in $(seq 1 120); do
    MODEL_JSON=$(curl -s "http://${HOST}:${PORT}/v1/models" 2>/dev/null)
    MODEL_COUNT=$(echo "$MODEL_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('data',[])))" 2>/dev/null || echo "0")
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "  Model ready after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "  ERROR: Model failed to load"
        cat "$RESULT_DIR/server.log" | tail -30
        exit 1
    fi
    sleep 2
done

echo "[Prompt 1] IOU calculation..."
curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Calculate the IoU of two bounding boxes: box1: [10, 10, 50, 50] and box2: [30, 30, 70, 70]. Show your reasoning."}],"temperature":0.0,"max_tokens":512,"stream":false}' \
    > "$RESULT_DIR/p1.json" 2>&1

echo "[Prompt 2] Thermodynamics..."
curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"What are the three laws of thermodynamics? Explain each briefly."}],"temperature":0.0,"max_tokens":256,"stream":false}' \
    > "$RESULT_DIR/p2.json" 2>&1

echo "[Prompt 3] Sky color..."
curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Write exactly one paragraph about why the sky is blue."}],"temperature":0.7,"max_tokens":150,"stream":false}' \
    > "$RESULT_DIR/p3.json" 2>&1

echo ""
echo "=== RESULTS ==="
PASS=0; FAIL=0
for p in 1 2 3; do
    RESP="$RESULT_DIR/p${p}.json"
    TEXT=$(python3 -c "
import json,sys
try:
    d=json.load(open('$RESP'))
    print(d['choices'][0]['message']['content'][:300])
except Exception as e:
    print('PARSE_ERROR:', e)" 2>/dev/null) || TEXT="PARSE_ERROR"
    
    TOKENS=$(python3 -c "
import json,sys
d=json.load(open('$RESP'))
u=d.get('usage',{})
print(f'prompt={u.get(\"prompt_tokens\",\"?\")}  completion={u.get(\"completion_tokens\",\"?\")}  total={u.get(\"total_tokens\",\"?\")}')" 2>/dev/null) || TOKENS="unknown"
    
    echo "  Prompt $p: Tokens: $TOKENS"
    
    GARBLE_CHECK=$(python3 -c "
import re
txt='''$TEXT'''
if len(txt) < 10: print('TOO_SHORT')
elif 'PARSE_ERROR' in txt: print('PARSE_ERROR')
else:
    words=txt.split()
    fourgrams=[' '.join(words[i:i+4]) for i in range(len(words)-3)]
    from collections import Counter
    c=Counter(fourgrams)
    r=sum(1 for v in c.values() if v>1)
    if r>3: print(f'LOOPING: {r} repeated 4-grams')
    elif len(txt)<50: print('TOO_SHORT')
    else: print('OK')" 2>/dev/null)
    
    if echo "$GARBLE_CHECK" | grep -q "OK"; then
        echo "    Quality: OK (no garble/looping)"; PASS=$((PASS+1))
    elif echo "$GARBLE_CHECK" | grep -q "TOO_SHORT"; then
        echo "    Quality: TOO_SHORT"; FAIL=$((FAIL+1))
    elif echo "$GARBLE_CHECK" | grep -q "LOOPING"; then
        echo "    Quality: LOOPING"; FAIL=$((FAIL+1))
    else
        echo "    Quality: $GARBLE_CHECK"; FAIL=$((FAIL+1))
    fi
    echo "    Preview: $(echo "$TEXT" | head -c 200)"
    echo ""
done

kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true
echo "=== OVERALL ==="
echo "Passed: $PASS / $((PASS+FAIL))"
echo "Failed: $FAIL / $((PASS+FAIL))"
exit $FAIL
