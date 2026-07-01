#!/usr/bin/env bash
# Package host build-cuda-b-bin rpc-server into atomic-llama-triton-pathb-rpc:latest.
#
# usage (on triton): ./build-from-bin.sh
# env:
#   TRITON_REPO  default ~/projects/atomic-llama-cpp-turboquant
#   BIN_DIR      default ${TRITON_REPO}/build-cuda-b-bin/bin

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${TRITON_REPO:-${HOME}/projects/atomic-llama-cpp-turboquant}"
BIN_DIR="${BIN_DIR:-${REPO}/build-cuda-b-bin/bin}"
STAGING="${DIR}/staging"

if [[ ! -x "${BIN_DIR}/rpc-server" ]]; then
    echo "ERROR: missing ${BIN_DIR}/rpc-server (run b6-gate-triton-sync-rebuild.sh)" >&2
    exit 1
fi

rm -rf "$STAGING"
mkdir -p "$STAGING/lib"
cp "${BIN_DIR}/rpc-server" "$STAGING/"
cp -a "${BIN_DIR}"/libggml*.so* "$STAGING/lib/" 2>/dev/null || true
cp -a "${BIN_DIR}"/libllama*.so* "$STAGING/lib/" 2>/dev/null || true
cp -a "${BIN_DIR}"/libmtmd*.so* "$STAGING/lib/" 2>/dev/null || true

docker build -f "${DIR}/Dockerfile.runtime" -t atomic-llama-triton-pathb-rpc:latest "${DIR}"
echo "BUILD_OK atomic-llama-triton-pathb-rpc:latest"