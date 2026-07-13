#!/bin/bash
# Multi-GPU KV-cache fix verification: 1-GPU and 2-GPU test matrix
# Usage: bash test-kv-fix.sh [model_path]
set -euo pipefail

MODEL="${1:-/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf}"
PORT_BASE=8095
TOKENS=128
BINARY="build-rocm-docker/bin/llama-server"

PASS=0
FAIL=0
RESULTS=""

# Prompt list: short factual, longer reasoning, code
PROMPTS=(
  "What is the capital of France?"
  "Explain the difference between TCP and UDP in detail."
  "Write a Python function that computes the Fibonacci sequence."
)
SEEDS=(42 123 999)

run_test() {
  local label="$1" port="$2" env_vars="$3"
  shift 3

  echo ""
  echo "========== $label (port $port) =========="
  echo "env: $env_vars"

  # Start server
  env $env_vars $BINARY \
    -m "$MODEL" --host 127.0.0.1 --port "$port" \
    --n-gpu-layers 99 --ctx-size 2048 \
    > /tmp/llama-server-${port}.log 2>&1 &
  local pid=$!

  # Wait for ready
  for i in $(seq 1 60); do
    if curl -s --max-time 2 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; then
      break
    fi
    if ! kill -0 $pid 2>/dev/null; then
      echo "SERVER DIED! Check /tmp/llama-server-${port}.log"
      tail -20 /tmp/llama-server-${port}.log
      return 1
    fi
    sleep 1
  done

  # Run prompts
  for prompt in "${PROMPTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local resp
      resp=$(curl -s --max-time 120 "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$TOKENS,\"temperature\":0,\"seed\":$seed,\"stream\":false}" 2>/dev/null)

      if [ -z "$resp" ]; then
        echo "  FAIL: empty response (prompt=$prompt seed=$seed)"
        FAIL=$((FAIL + 1))
        RESULTS="${RESULTS}${label}|${prompt}|seed=${seed}|FAIL:empty\n"
        continue
      fi

      # Parse and check
      local verdict
      verdict=$(python3 -c "
import sys, json, re
try:
    d = json.loads(sys.stdin.read())
    if 'error' in d:
        print('FAIL:server_error:' + str(d['error'])[:100])
        sys.exit(0)
    c = d['choices'][0]['message']['content']
    t = d['usage']['completion_tokens']
    text = c.replace('\n',' ')

    # Garble detection: substring repetition, low content, word repetition
    garbled = False
    reasons = []

    # Heuristic 1: substring repetition (len>=3, repeated >=4 times)
    for l in range(3, min(9, len(text)//4 + 1)):
        for i in range(len(text) - l*4 + 1):
            chunk = text[i:i+l]
            j = i + l
            count = 1
            while j + l <= len(text) and text[j:j+l] == chunk:
                count += 1
                j += l
            if count >= 4 and not chunk.isspace() and chunk.strip() and re.search(r'[a-zA-Z\u4e00-\u9fff]', chunk):
                garbled = True
                reasons.append('repetition')
                break
        if garbled:
            break

    # Heuristic 2: very low non-whitespace content
    nw = sum(1 for c in c if not c.isspace())
    if len(c) > 0 and nw / max(len(c), 1) < 0.02 and t >= 16:
        garbled = True
        reasons.append('low-content')

    # Heuristic 3: no words but many tokens
    words = re.findall(r'[a-zA-Z]{2,}', c.lower())
    if t >= 16 and len(words) == 0:
        garbled = True
        reasons.append('no-words')

    # Heuristic 4: word repetition (very few unique words vs many total)
    if t >= 32 and len(set(words)) <= 2 and len(words) >= 8:
        garbled = True
        reasons.append('word-rep')

    verdict = 'GARBLED' if garbled else 'CLEAN'
    preview = c[:120].replace('\n','\\n')
    print(f'{verdict} tokens={t} preview=[{preview}...]' + (f' reasons={reasons}' if reasons else ''))
except Exception as e:
    print(f'FAIL:parse_error:{e}')
" <<< "$resp" 2>/dev/null) || verdict="FAIL:python_error"

      local status="PASS"
      if echo "$verdict" | grep -q "^CLEAN"; then
        PASS=$((PASS + 1))
        echo "  PASS: prompt=$(echo $prompt | cut -c1-30)... seed=$seed $verdict"
      else
        FAIL=$((FAIL + 1))
        status="FAIL"
        echo "  $status: prompt=$(echo $prompt | cut -c1-30)... seed=$seed $verdict"
      fi
      RESULTS="${RESULTS}${label}|$(echo $prompt | cut -c1-40)|seed=${seed}|$status|$verdict\n"
    done
  done

  # Kill server
  kill $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  sleep 1
}

# ===== TEST 1: 1-GPU ROCm only, PIPELINE_PLUS=1 =====
run_test "1-GPU-ROCm-PPLUS" 8095 "GGML_PIPELINE_PLUS=1"

# ===== TEST 2: 2-GPU ROCm+RPC, PIPELINE_PLUS=1 =====
run_test "2-GPU-ROCm-RPC-PPLUS" 8096 "GGML_PIPELINE_PLUS=1 GGML_RPC_ENDPOINTS=127.0.0.1:50051"

# ===== TEST 3: 1-GPU ROCm only, NO pipeline-plus (baseline) =====
run_test "1-GPU-ROCm-NOPLUS" 8097 ""

# ===== TEST 4: 2-GPU ROCm+RPC, NO pipeline-plus (baseline) =====
run_test "2-GPU-ROCm-RPC-NOPLUS" 8098 "GGML_RPC_ENDPOINTS=127.0.0.1:50051"

# ===== TEST 5: 3-GPU ROCm+2xRPC, PIPELINE_PLUS=1 =====
# remus 5060Ti at 192.168.8.22:50052, local 3060Ti at 127.0.0.1:50051
run_test "3-GPU-ROCm-RPCx2-PPLUS" 8099 "GGML_PIPELINE_PLUS=1 GGML_RPC_ENDPOINTS=127.0.0.1:50051,192.168.8.22:50052"

# ===== TEST 6: 3-GPU ROCm+2xRPC, NO pipeline-plus (baseline) =====
run_test "3-GPU-ROCm-RPCx2-NOPLUS" 8100 "GGML_RPC_ENDPOINTS=127.0.0.1:50051,192.168.8.22:50052"

# ===== Summary =====
echo ""
echo "============================================"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "============================================"
echo -e "$RESULTS" | column -t -s'|'

if [ $FAIL -gt 0 ]; then
  exit 1
fi
echo "ALL PASSED"
