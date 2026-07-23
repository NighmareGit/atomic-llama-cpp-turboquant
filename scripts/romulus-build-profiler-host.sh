#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-$HOME/atomic-llama-cpp-turboquant}"
BUILD=build-rocm-docker
cd "$SRC"
cmake -S . -B "$BUILD" \
    -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4 \
    -DLLAMA_BUILD_TOOLS=ON
cmake --build "$BUILD" --target llama-pipeline-profiler -j"$(nproc)"
test -x "${BUILD}/bin/llama-pipeline-profiler"
ls -la "${BUILD}/bin/llama-pipeline-profiler"
echo PROFILER_HOST_BUILD_OK