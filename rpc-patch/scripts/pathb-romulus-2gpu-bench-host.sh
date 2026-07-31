#!/usr/bin/env bash
# Run ON romulus: 2-GPU pathb bench (7900 ROCm + remus RPC).
set -euo pipefail

LABEL="${1:?label required}"
GEN="${BENCH_GEN_TOKENS:-128}"
ROOT="/home/hunter/atomic-llama-cpp-turboquant"

cd "$ROOT"
docker rm -f bench-rpc bench-llama 2>/dev/null || true

export LLAMA_TURBOQUANT_ROOT="$ROOT"
export BENCH_MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
export BENCH_RPC_MODE=multi
export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
export BENCH_TS="${BENCH_TS:-50,50}"
export BENCH_CTX="${BENCH_CTX:-4096}"
export BENCH_CTK="${BENCH_CTK:-q8_0}"
export BENCH_CTV="${BENCH_CTV:-q8_0}"
export BENCH_NGL="${BENCH_NGL:-99}"
export BENCH_GEN_TOKENS="$GEN"
export BENCH_LOAD_TIMEOUT="${BENCH_LOAD_TIMEOUT:-1200}"
export BENCH_NO_WARMUP=1
export BENCH_TRACE="${BENCH_TRACE:-0}"
export BENCH_RUNS="${BENCH_RUNS:-1}"
export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
export BENCH_NCMOE="${BENCH_NCMOE-}"
export BENCH_EXTRA="${BENCH_EXTRA:---fit off --verbose -lv 4 --reasoning off}"
export BENCH_PROMPTS_FILE="${BENCH_PROMPTS_FILE:-}"
export BENCH_CURL_TIMEOUT="${BENCH_CURL_TIMEOUT:-300}"
export BENCH_SAVE_FULL="${BENCH_SAVE_FULL:-0}"
export BENCH_MULTITURN_KVFILL="${BENCH_MULTITURN_KVFILL:-0}"
export BENCH_KV_TARGET="${BENCH_KV_TARGET:-7500}"
export BENCH_KV_TURN_GEN="${BENCH_KV_TURN_GEN:-64}"
export BENCH_KV_FINAL_GEN="${BENCH_KV_FINAL_GEN:-128}"

echo "=== 2-GPU bench label=$LABEL gen=$GEN ctx=$BENCH_CTX runs=$BENCH_RUNS ==="
host="${BENCH_RPC_ENDPOINT%%:*}"
port="${BENCH_RPC_ENDPOINT##*:}"
nc -zv -w 5 "$host" "$port"

bash ./rpc-patch/scripts/rpc-server-bench.sh pathb "$LABEL"
echo BENCH_DONE "$LABEL"