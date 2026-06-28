#!/usr/bin/env bash
set -euo pipefail
SRC="${1:-$HOME/atomic-llama-cpp-turboquant}"
BUILD=build-rocm-profiler
cd "$SRC"
docker run --rm --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -v "${SRC}:/src" -w /src \
    rocm/dev-ubuntu-22.04:6.4-complete \
    -lc "set -e
if ! command -v cmake >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq cmake ninja-build build-essential git
fi
cmake -S . -B ${BUILD} -G Ninja \
    -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4 \
    -DLLAMA_BUILD_TOOLS=ON
cmake --build ${BUILD} --target llama-pipeline-profiler -j8
test -x ${BUILD}/bin/llama-pipeline-profiler
ls -la ${BUILD}/bin/llama-pipeline-profiler"
echo PROFILER_BUILD_OK