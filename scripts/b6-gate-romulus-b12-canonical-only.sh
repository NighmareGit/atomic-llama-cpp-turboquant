#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PROFILER_LOCAL=1
export PROFILER_BIN="${ROOT}/build-rocm-docker/bin/llama-pipeline-profiler"
export PROFILER_SKIP_VALIDATE=1
export BENCH_GEN_TOKENS=384
export GGML_PIPELINE_PLUS=1
export PROFILER_OUT_DIR="${ROOT}/benches/path-b-plus/b6-2gpu-f-triton-n384-romulus-native-b12"
exec bash "${ROOT}/scripts/b6-gate-bisect-run.sh" canonical-romulus