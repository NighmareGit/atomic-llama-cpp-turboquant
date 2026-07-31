#!/bin/bash
# End-to-end MTP model test: performance + garble + logic + looping
# Tests both 9B and 35B MTP models with speculative decoding (nextn)
# Usage: ./scripts/e2e-mtp-test.sh [model] [model-draft]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build-rocm/bin/llama-server}"
RPC_3060="127.0.0.1:50051"
RPC_REMUS="192.168.8.22:50052"
PORT=9988
HOST="127.0.0.1"

# Default: 9B MTP model
MAIN="${MAIN:-/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf}"
DRAFT="${DRAFT:-$MAIN}"  # Same GGUF, built-in MTP heads
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B-MTP}"

if [ $# -ge 1 ]; then MAIN="$1"; fi
if [ $# -ge 2 ]; then DRAFT="$2"; fi

# Verify files
for f in "$SERVER" "$MAIN"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

echo "=== E2E MTP TEST ==="
echo "Server: $SERVER ($(du -h "$SERVER" | cut -f1))"
echo "Model:  $MAIN ($(du -h "$MAIN" | cut -f1))"
echo "Draft:  $DRAFT ($(du -h "$DRAFT" | cut -f1))"
echo "RPC:    $RPC_3060, $RPC_REMUS"
echo ""

RESULT_DIR="/tmp/e2e-mtp-$$"
mkdir -p "$RESULT_DIR"

# Kill any leftover server
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 1

echo "[1/5] Starting llama-server..."
# Use known-good config from Phase 2/3 tuning
timeout 300 "$SERVER" \
    -m "$MAIN" \
    -md "$DRAFT" \
    --spec-type draft-mtp \
    --spec-draft-n-max 16 \
    --spec-draft-n-min 0 \
    --rpc "$RPC_REMUS,$RPC_3060" \
    -ngl 99 -ngld 99 \
    -ts "36,15,49" \
    -c 4096 \
    -np 1 \
    -fa 1 \
    --host "$HOST" \
    --port "$PORT" \
    --no-warmup \
    --cont-batching \
    --log-disable \
    > "$RESULT_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo "  PID: $SERVER_PID, waiting for model load..."
sleep 2

# Wait for server HTTP endpoint
for i in $(seq 1 60); do
    if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        echo "  HTTP ready after ${i}s"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: Server failed to start within 60s"
        cat "$RESULT_DIR/server.log" | tail -20
        exit 1
    fi
    sleep 2
done

# Wait for model to finish loading (check /v1/models returns actual model data)
echo "  Waiting for model to load..."
for i in $(seq 1 120); do
    MODEL_JSON=$(curl -s "http://${HOST}:${PORT}/v1/models" 2>/dev/null)
    MODEL_COUNT=$(echo "$MODEL_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('data',[])))" 2>/dev/null || echo "0")
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "  Model ready after ${i}s (${MODEL_COUNT} model(s))"
        echo "$MODEL_JSON" > "$RESULT_DIR/models.json"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "  ERROR: Model failed to load within 240s"
        echo "$MODEL_JSON" > "$RESULT_DIR/models.json"
        cat "$RESULT_DIR/server.log" | tail -30
        exit 1
    fi
    sleep 2
done
echo ""
echo ""

echo "[3/5] PROMPT 1: Logic/reasoning (IOU calculation)..."
curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Calculate the IoU (Intersection over Union) of two bounding boxes: box1: [10, 10, 50, 50] and box2: [30, 30, 70, 70]. Show your step-by-step reasoning."}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": false
    }' > "$RESULT_DIR/prompt1.json" 2>&1 &
PID1=$!

echo "[4/5] PROMPT 2: Facts/knowledge (no looping)..."
curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "What are the three laws of thermodynamics? Explain each briefly."}
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": false
    }' > "$RESULT_DIR/prompt2.json" 2>&1 &
PID2=$!

echo "[5/5] PROMPT 3: Follow-up test (short generation, check no garbled repetition)..."
curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Write exactly one paragraph about why the sky is blue."}
        ],
        "temperature": 0.7,
        "max_tokens": 150,
        "stream": false
    }' > "$RESULT_DIR/prompt3.json" 2>&1 &
PID3=$!

# Wait for all prompts
wait $PID1 $PID2 $PID3 2>/dev/null || true

echo ""
echo "=== RESULTS ==="

# Analyze results
PASS=0
FAIL=0

for p in 1 2 3; do
    RESPONSE_FILE="$RESULT_DIR/prompt${p}.json"
    if [ ! -f "$RESPONSE_FILE" ]; then
        echo "  PROMPT $p: NO RESPONSE"
        FAIL=$((FAIL+1))
        continue
    fi
    
    # Extract text from response
    TEXT=$(python3 -c "
import json,sys
try:
    d=json.load(open('$RESPONSE_FILE'))
    txt=d['choices'][0]['message']['content']
    print(txt[:500])
except Exception as e:
    print('PARSE_ERROR:', e)
" 2>/dev/null) || TEXT="PARSE_ERROR"
    
    # Check for errors
    ERROR_MSG=$(python3 -c "
import json,sys
try:
    d=json.load(open('$RESPONSE_FILE'))
    print('ok')
except Exception as e:
    print('json_error:', e)
" 2>/dev/null)
    
    if echo "$ERROR_MSG" | grep -q "json_error"; then
        echo "  PROMPT $p: JSON PARSE ERROR"
        head -5 "$RESPONSE_FILE"
        FAIL=$((FAIL+1))
        continue
    fi
    
    # Check token usage
    TOKENS=$(python3 -c "
import json,sys
d=json.load(open('$RESPONSE_FILE'))
u=d.get('usage',{})
print(f'prompt={u.get(\"prompt_tokens\",\"?\")}  completion={u.get(\"completion_tokens\",\"?\")}  total={u.get(\"total_tokens\",\"?\")}')
" 2>/dev/null) || TOKENS="unknown"
    
    echo "  PROMPT $p: Tokens: $TOKENS"
    
    # Check for garbled output (repeated characters, n-gram looping)
    GARBLE_CHECK=$(python3 -c "
import re
txt='''$TEXT'''
# Check for truncation/empty
if len(txt) < 10:
    print('TOO_SHORT')
elif 'PARSE_ERROR' in txt:
    print('PARSE_ERROR')
else:
    # Check for n-gram repetition (sign of looping)
    words = txt.split()
    bigrams = {' '.join(words[i:i+2]) for i in range(len(words)-1)}
    # If more than 30% of bigrams are unique, probably not looping
    # But check for 4-gram repetition
    fourgrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
    from collections import Counter
    fourgram_counts = Counter(fourgrams)
    repeated = sum(1 for c in fourgram_counts.values() if c > 1)
    if repeated > 3:
        print(f'LOOPING: {repeated} repeated 4-grams')
    elif len(txt) < 50:
        print('TOO_SHORT')
    else:
        print('OK')
" 2>/dev/null)
    
    if echo "$GARBLE_CHECK" | grep -q "OK"; then
        echo "    Quality: OK (no garble/looping)"
        PASS=$((PASS+1))
    elif echo "$GARBLE_CHECK" | grep -q "TOO_SHORT"; then
        echo "    Quality: TOO_SHORT - possible generation issue"
        FAIL=$((FAIL+1))
    elif echo "$GARBLE_CHECK" | grep -q "LOOPING"; then
        echo "    Quality: LOOPING DETECTED"
        FAIL=$((FAIL+1))
    else
        echo "    Quality: $GARBLE_CHECK"
        FAIL=$((FAIL+1))
    fi
    
    # Show first 200 chars of response
    echo "    Preview: $(echo "$TEXT" | head -c 200)"
    echo ""
done

# Kill server
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo ""
echo "=== OVERALL ==="
echo "Passed: $PASS / $((PASS+FAIL))"
echo "Failed: $FAIL / $((PASS+FAIL))"

# Get timing metrics from server log
echo ""
echo "[Performance]"
grep -oP 'prompt tokens: \d+.*tokens per second' "$RESULT_DIR/server.log" 2>/dev/null | tail -5 || echo "  (no perf metrics in log)"
grep -oP 'predicted \d+ tokens in \S+ seconds, \S+ tokens/second' "$RESULT_DIR/server.log" 2>/dev/null | tail -5 || echo "  (no decode perf in log)"

exit $FAIL
