#!/usr/bin/env bash
set -euo pipefail
cd /home/hunter/atomic-llama-cpp-turboquant
BUILD=build-rocm-docker
LOG=/tmp/rocm-docker-build.log

docker run --rm --entrypoint rm \
    -v "/home/hunter/atomic-llama-cpp-turboquant:/src" \
    llama-rocm-patched -rf "/src/${BUILD}" 2>/dev/null || rm -rf "$BUILD"

cmake -S . -B "$BUILD" \
    -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4

cmake --build "$BUILD" --target llama-server -j"$(nproc)" 2>&1 | tee "$LOG"
mkdir -p "$BUILD/bin"
find "$BUILD" -name '*.so*' -exec cp -n {} "$BUILD/bin/" \; 2>/dev/null || true
test -x "$BUILD/bin/llama-server"
echo ROCM_BUILD_OK