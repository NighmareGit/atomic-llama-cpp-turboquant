#!/bin/bash
# 5-GPU Config G trace bench from romulus (7900 ROCm client).
set -euo pipefail

LABEL="${1:-trace-g-5gpu-plus}"
GEN="${BENCH_GEN_TOKENS:-128}"
ROOT="/home/hunter/atomic-llama-cpp-turboquant"

cd "$ROOT"
docker rm -f bench-rpc bench-llama 2>/dev/null || true

export LLAMA_TURBOQUANT_ROOT="$ROOT"
export BENCH_MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
export BENCH_RPC_MODE=multi
# Default 5-GPU: primary 4 + spare hop. Omit :50052 (RX6600) unless legacy experiment.
# Future: replace 5th endpoint with 3090+3070 node when online.
export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053}"
export BENCH_TS="${BENCH_TS:-30,24,24,22}"
export BENCH_CTX="${BENCH_CTX:-4096}"
export BENCH_CTK="${BENCH_CTK:-q4_0}"
export BENCH_CTV="${BENCH_CTV:-q4_0}"
export BENCH_NGL="${BENCH_NGL:-99}"
export BENCH_GEN_TOKENS="$GEN"
export BENCH_LOAD_TIMEOUT="${BENCH_LOAD_TIMEOUT:-1800}"
export BENCH_NO_WARMUP=1
export BENCH_TRACE="${BENCH_TRACE:-1}"
export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
export BENCH_NCMOE="${BENCH_NCMOE-8}"
export BENCH_EXTRA="${BENCH_EXTRA:---fit off --verbose -lv 4 --reasoning off}"

echo "=== 5-GPU bench label=$LABEL gen=$GEN ==="
for ep in ${BENCH_RPC_ENDPOINT//,/ }; do
    host="${ep%%:*}"
    port="${ep##*:}"
    if nc -zv -w 5 "$host" "$port"; then
        echo "RPC ok: $host:$port"
    else
        echo "WARN: RPC down: $host:$port" >&2
    fi
done

bash ./rpc-patch/scripts/rpc-server-bench.sh pathb "$LABEL"

TELEM="${ROOT}/rpc-patch/patch/bench-results/rpc-server-bench/${LABEL}/telemetry"
if [[ -d "$TELEM" ]]; then
    bash ./rpc-patch/scripts/pathb-hotpath-summary.sh "$TELEM" || true
fi
echo BENCH_DONE "$LABEL"