#!/bin/bash
# Large-model multi-GPU test: 27B, 35B+MoE+MTP, 70B
# Usage: bash test-large-models.sh
set -euo pipefail

PROFILER="build-rocm-docker/bin/llama-gpipe-profiler"
TEST_SCRIPT="rpc-patch/patch/test-kv-fix.sh"
OUTDIR="profiler-out"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_FILE="${OUTDIR}/${TIMESTAMP}-summary.txt"

# Cluster VRAM budget (90% usable)
# 7900XTX: 24560 MiB -> 22104 usable
# 3060Ti (docker): ~8000 MiB -> 7200 usable
# 5060Ti (remus): 15850 MiB -> 14265 usable
# Total: ~43569 MiB usable

RPC_ENDPOINTS="127.0.0.1:50051,192.168.8.22:50052"

mkdir -p "$OUTDIR"
echo "=== Large Model Multi-GPU Test ===" > "$RESULTS_FILE"
echo "Timestamp: $TIMESTAMP" >> "$RESULTS_FILE"
echo "RPC: $RPC_ENDPOINTS" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

run_model_test() {
  local label="$1" model="$2" ts="$3" ctx="$4" extra_args="$5"
  local out="${OUTDIR}/${TIMESTAMP}-${label}"

  echo "========== $label =========="
  echo "model: $model"
  echo "ts: $ts  ctx: $ctx"
  echo "extra: $extra_args"
  echo ""

  # Build TS args
  local ts_args=""
  [ -n "$ts" ] && ts_args="-sm layer -ts $ts"

  # Step 1: Probe VRAM fit with profiler (load only, no inference)
  echo "--- VRAM probe ---"
  local probe_out probe_rc
  probe_out=$(env GGML_RPC_ENDPOINTS="$RPC_ENDPOINTS" GGML_PIPELINE_PLUS=1 \
    $PROFILER -m "$model" \
    --rpc "$RPC_ENDPOINTS" \
    --tasks pp -p 1 -n 1 -r 1 --no-warmup \
    -ngl 99 --ctx-size "$ctx" \
    $ts_args \
    $extra_args 2>&1) && probe_rc=0 || probe_rc=$?

  # Extract per-device model buffer sizes (3-GPU load)
  echo "$probe_out" | grep "model buffer size" | grep "RPC0" | sort -u || true
  echo "  Model distribution:"

  if [ "$probe_rc" -ne 0 ]; then
    echo "  WARN: profiler exit=$probe_rc (may be benign)"
    # Check if we at least got RPC distribution
    if ! echo "$probe_out" | grep -q "RPC.*model buffer size"; then
      echo "  FAIL: model load failed" | tee -a "$RESULTS_FILE"
      echo "$probe_out" | grep -E "error|OOM|failed" | tail -5
      echo "$label: LOAD_FAILED" >> "$RESULTS_FILE"
      return 1
    fi
  fi

  # Extract per-device model buffer sizes (3-GPU load, not 1-GPU warmup)
  echo "$probe_out" | grep "model buffer size" | grep "RPC0" | sort -u
  echo "  Model distribution:"

  echo "  LOAD OK" | tee -a "$RESULTS_FILE"

  # Step 2: Profiler run (PP + TG)
  echo ""
  echo "--- Profiler (PP=64, TG=256, 3 reps) ---"
  env GGML_RPC_ENDPOINTS="$RPC_ENDPOINTS" GGML_PIPELINE_PLUS=1 \
    $PROFILER -m "$model" \
    --rpc "$RPC_ENDPOINTS" \
    --tasks pp,tg -p 64 -n 256 -r 3 \
    --warmup --trace --server-telemetry \
    --gpipe-stages 2 \
    -ngl 99 --ctx-size "$ctx" \
    $ts_args \
    $extra_args \
    --out-dir "${out}" \
    2>&1 | tee "${out}-profiler.log" | grep -E "wall_ms|tps|heatmap|done|error|FAIL" | head -20

  local tps_line
  tps_line=$(grep "wall_ms.*tps=" "${out}-profiler.log" | tail -1)
  if [ -n "$tps_line" ]; then
    echo "  TPS: $tps_line" | tee -a "$RESULTS_FILE"
  fi

  # Step 3: Quick correctness check via server
  echo ""
  echo "--- Correctness check ---"
  local port=8190
  env GGML_RPC_ENDPOINTS="$RPC_ENDPOINTS" GGML_PIPELINE_PLUS=1 \
    build-rocm-docker/bin/llama-server \
    -m "$model" --host 127.0.0.1 --port "$port" \
    -ngl 99 --ctx-size "$ctx" \
    $ts_args \
    $extra_args \
    > /tmp/llama-server-large-${label}.log 2>&1 &
  local srv_pid=$!

  # Wait for ready
  local ready=0
  for i in $(seq 1 120); do
    if curl -s --max-time 3 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; then
      ready=1; break
    fi
    if ! kill -0 $srv_pid 2>/dev/null; then
      echo "  SERVER DIED" | tee -a "$RESULTS_FILE"
      tail -30 /tmp/llama-server-large-${label}.log
      echo "$label: SERVER_DIED" >> "$RESULTS_FILE"
      return 1
    fi
    sleep 2
  done

  if [ "$ready" -ne 1 ]; then
    echo "  TIMEOUT waiting for server" | tee -a "$RESULTS_FILE"
    kill $srv_pid 2>/dev/null || true
    echo "$label: LOAD_TIMEOUT" >> "$RESULTS_FILE"
    return 1
  fi

  # Quick generation test
  local resp
  resp=$(curl -s --max-time 120 "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32,\"temperature\":0,\"stream\":false}" 2>/dev/null)

  kill $srv_pid 2>/dev/null || true
  wait $srv_pid 2>/dev/null || true

  if [ -n "$resp" ] && echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:80])" 2>/dev/null; then
    echo "  CORRECTNESS OK" | tee -a "$RESULTS_FILE"
  else
    echo "  CORRECTNESS FAIL" | tee -a "$RESULTS_FILE"
    echo "$resp" | head -5
  fi

  echo "$label: COMPLETE" >> "$RESULTS_FILE"
  echo ""
}

# ===== Model 1: Qwen3.5 27B Q5_K_M (18.6GB dense, 64 layers) =====
# Auto-split: let GGML distribute by VRAM proportion
# Expected: 7900XTX ~10GB, 3060Ti ~3GB, 5060Ti ~6GB
run_model_test "qwen27b-dense-3gpu" \
  "/mnt/models/Qwen3.5-27B-Q5_K_M.gguf" \
  "" "4096" ""

# ===== Model 2: Qwen3.5 35B MoE A3B Q4_K_M (19.7GB MoE, 40 layers) =====
# Auto-split: VRAM-proportional with MoE expert awareness
run_model_test "qwen35b-moe-3gpu" \
  "/mnt/models/Qwen3.5-35B-A3B.i1-Q4_K_M.gguf" \
  "" "4096" ""

# ===== Model 3: Qwen3.5 35B MoE + MTP =====
# Same model, enable MTP speculative decoding
# MTP adds ~1 extra draft head per layer — small VRAM overhead
run_model_test "qwen35b-moe-mtp-3gpu" \
  "/mnt/models/Qwen3.5-35B-A3B.i1-Q4_K_M.gguf" \
  "" "4096" "--spec-type draft-mtp --spec-draft-n-max 2"

# ===== Model 4: 70B Llama-3 Q4_K_M (40GB dense) — if it fits =====
# Auto-split across 3 GPUs: 40GB model + KV + compute ~44GB total
# Total usable: 43.5GB — very tight, may need ctx=1024
run_model_test "llama70b-3gpu" \
  "/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf" \
  "" "2048" ""

echo ""
echo "=== Results ==="
cat "$RESULTS_FILE"
