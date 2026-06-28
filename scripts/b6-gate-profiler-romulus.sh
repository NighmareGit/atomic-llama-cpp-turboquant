#!/usr/bin/env bash
# B+6 gate profiler presets on romulus (see rpc-patch/docs/b6-gate/PLAN.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:-}"
shift || true

if [[ -z "$LABEL" || "$LABEL" == "-h" || "$LABEL" == "--help" ]]; then
    echo "usage: $0 <label> [extra profiler args...]"
    echo "labels:"
    echo "  b6-2gpu-f          remus 5060 RPC (gate baseline)"
    echo "  b6-2gpu-f-triton   triton 3090 RPC (spike, same client)"
    echo "  b6-4gpu-g          4-GPU primary"
    echo "  b6-2gpu-f-plus0    2-GPU remus, GGML_PIPELINE_PLUS=0"
    exit 0
fi

export PATHB_ROMULUS_SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-12345}"
export PROFILER_BIN="${PROFILER_BIN:-${ROOT}/build-rocm-docker/bin/llama-pipeline-profiler}"
export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export BENCH_MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
export BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-384}"
export BENCH_PROMPT_FILE="${BENCH_PROMPT_FILE:-${ROOT}/benches/path-b-plus/prompts/profiler-reasoning-long.txt}"
export PROFILER_MODE="${PROFILER_MODE:-trace}"
export PROFILER_LOCAL=1
export BENCH_TRACE=1

case "$LABEL" in
    b6-2gpu-f)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-2gpu-f-triton)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.23:50054}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-4gpu-g)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053}"
        export BENCH_TS="${BENCH_TS:-36,24,24,16}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-2gpu-f-plus0)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS=0
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    *)
        echo "error: unknown label: $LABEL" >&2
        exit 1
        ;;
esac

cd "$ROOT"
exec bash scripts/llama-pipeline-profiler-cluster.sh "$LABEL" \
    -ctk q8_0 -ctv q8_0 -ngl 99 --no-warmup \
    --with-gpu-telemetry --trace-sample 5 \
    --regression-file "${ROOT}/benches/path-b-plus/regression.jsonl" \
    "$@"