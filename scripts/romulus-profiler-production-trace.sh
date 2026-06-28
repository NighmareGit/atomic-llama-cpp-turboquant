#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATHB_ROMULUS_SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-12345}"
export PROFILER_BIN="${PROFILER_BIN:-${ROOT}/build-rocm-docker/bin/llama-pipeline-profiler}"
export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export BENCH_MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053}"
export BENCH_TS="${BENCH_TS:-36,24,24,16}"
export BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-384}"
export BENCH_PROMPT_FILE="${BENCH_PROMPT_FILE:-${ROOT}/benches/path-b-plus/prompts/profiler-reasoning-long.txt}"
export PROFILER_MODE="${PROFILER_MODE:-trace}"
export PROFILER_LOCAL=1
export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/profiler-4gpu-primary-romulus-trace}"
cd "$ROOT"
exec bash scripts/llama-pipeline-profiler-cluster.sh profiler-4gpu-primary-romulus \
    -ctk q8_0 -ctv q8_0 -ngl 99 --no-warmup \
    --with-gpu-telemetry --trace-sample 5 "$@"