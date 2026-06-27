#!/usr/bin/env bash
# Build build-rocm-docker/bin/llama-server on romulus (7900 XTX gfx1100).
set -euo pipefail

SRC="${1:-$HOME/atomic-llama-cpp-turboquant}"
BUILD=build-rocm-docker
LOG="${2:-/tmp/rocm-docker-build.log}"

cd "$SRC"
rm -rf "$BUILD"

if command -v hipcc >/dev/null && sudo apt-get install -y ninja-build 2>/dev/null; then
    cmake -S . -B "$BUILD" -G Ninja \
        -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
        -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4
    cmake --build "$BUILD" --target llama-server -j"$(nproc)"
else
    docker run --rm --entrypoint bash \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        -v "${SRC}:/src" -w /src \
        rocm/dev-ubuntu-22.04:6.4-complete \
        -lc "set -e
apt-get update -qq
apt-get install -y -qq cmake ninja-build build-essential
rm -rf ${BUILD}
cmake -S . -B ${BUILD} -G Ninja \
    -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4
cmake --build ${BUILD} --target llama-server -j\$(nproc)"
fi

mkdir -p "${BUILD}/bin"
find "${BUILD}" -name '*.so*' -exec cp -n {} "${BUILD}/bin/" \; 2>/dev/null || true
ls -la "${BUILD}/bin/llama-server"
test -x "${BUILD}/bin/llama-server"
echo ROCM_BUILD_OK