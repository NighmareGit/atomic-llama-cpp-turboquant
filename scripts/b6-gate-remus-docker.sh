#!/usr/bin/env bash
# Run B+6 gate profiler on remus via CUDA docker (no host nvcc/cudart required).
#
# usage: bash scripts/b6-gate-remus-docker.sh <label> [extra profiler args...]
#
# labels: b6-2gpu-jupiter, b6-2gpu-jupiter-plus0, b6-2gpu-f, b6-2gpu-f-plus0, ...
# env: BENCH_MODEL, BENCH_GEN_TOKENS, BENCH_RPC_ENDPOINT, GGML_PIPELINE_PLUS, MODELS_DIR

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:?label required (e.g. b6-2gpu-jupiter)}"
shift || true

MODELS_DIR="${MODELS_DIR:-/home/hunter/models}"
BENCH_MODEL="${BENCH_MODEL:-${MODELS_DIR}/Qwen3.5-9B-MTP-Q4_K_M.gguf}"
BUILD_BIN="${BUILD_BIN:-${ROOT}/build-cuda-b-bin/bin}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.0-devel-ubuntu22.04}"

if [[ ! -x "${BUILD_BIN}/llama-pipeline-profiler" ]]; then
    echo "error: profiler missing: ${BUILD_BIN}/llama-pipeline-profiler" >&2
    echo "hint: docker CUDA build with -DCMAKE_CUDA_ARCHITECTURES=120a-real" >&2
    exit 1
fi

export PROFILER_BIN="/src/build-cuda-b-bin/bin/llama-pipeline-profiler"
export PROFILER_LOCAL=1
export LD_LIBRARY_PATH="/src/build-cuda-b-bin/bin"
export BENCH_MODEL="/models/$(basename "${BENCH_MODEL}")"

docker run --rm --network=host --gpus=all --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "${ROOT}:/src" \
    -v "${MODELS_DIR}:/models:ro" \
    -w /src \
    -e LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -e PROFILER_BIN="${PROFILER_BIN}" \
    -e PROFILER_LOCAL="${PROFILER_LOCAL}" \
    -e PROFILER_SKIP_VALIDATE="${PROFILER_SKIP_VALIDATE:-0}" \
    -e PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-}" \
    -e BENCH_MODEL="${BENCH_MODEL}" \
    -e BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-}" \
    -e BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-}" \
    -e BENCH_TS="${BENCH_TS:-}" \
    -e GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}" \
    -e GGML_PIPELINE_TRACE="${GGML_PIPELINE_TRACE:-1}" \
    -e GGML_RPC_TRACE="${GGML_RPC_TRACE:-0}" \
    -e GGML_SCHED_TRACE="${GGML_SCHED_TRACE:-0}" \
    "${CUDA_IMAGE}" \
    bash -c 'apt-get update -qq && apt-get install -y -qq libgomp1 libopenblas0 >/dev/null && \
        exec bash scripts/b6-gate-profiler-romulus.sh "$@"' _ "${LABEL}" "$@"