#!/usr/bin/env bash
# Run on romulus: bash scripts/b6-gate-run-remote.sh b6-2gpu-f
set -euo pipefail
LABEL="${1:?label required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PROFILER_LOCAL=1
export PROFILER_BIN="${ROOT}/build-rocm-docker/bin/llama-pipeline-profiler"
export BENCH_MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
export BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-384}"
export BENCH_PROMPT_FILE="${BENCH_PROMPT_FILE:-${ROOT}/benches/path-b-plus/prompts/profiler-reasoning-long.txt}"
export PROFILER_MODE=trace
export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"

case "$LABEL" in
    b6-2gpu-f)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        ;;
    b6-2gpu-f-triton)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.23:50054}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        ;;
    b6-4gpu-g)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053}"
        export BENCH_TS="${BENCH_TS:-25,12,25,38}"
        ;;
    b6-4gpu-g-triton)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054}"
        export BENCH_TS="${BENCH_TS:-22,11,34,33}"
        ;;
    b6-5gpu-g)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053,192.168.8.23:50055}"
        export BENCH_TS="${BENCH_TS:-22,11,22,11,34}"
        ;;
    b6-2gpu-f-plus0)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS=0
        ;;
    *)
        echo "unknown label: $LABEL" >&2
        exit 1
        ;;
esac

echo "=== b6-gate remote run: $LABEL ==="
bash scripts/llama-pipeline-profiler-cluster.sh "$LABEL" \
    -ctk q8_0 -ctv q8_0 -ngl 99 --no-warmup \
    --with-gpu-telemetry --trace-sample 5 \
    --regression-file "${ROOT}/benches/path-b-plus/regression.jsonl"