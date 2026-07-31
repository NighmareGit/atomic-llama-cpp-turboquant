#!/usr/bin/env bash
# Run on triton: git sync + incremental rpc-server rebuild + restart :50054.
#
# usage: b6-gate-triton-sync-rebuild.sh [--no-restart]
#
# env: see b6-gate-triton-git-sync.sh and b6-gate-triton-start-rpc.sh

set -euo pipefail

REPO="${B6_TRITON_REPO:-${HOME}/projects/atomic-llama-cpp-turboquant}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NO_RESTART=0
[[ "${1:-}" == "--no-restart" ]] && NO_RESTART=1

bash "${SCRIPT_DIR}/b6-gate-triton-git-sync.sh"

cd "$REPO"
if [[ -d build-cuda-b-bin/ggml/src/ggml-rpc ]]; then
    rm -rf build-cuda-b-bin/ggml/src/ggml-rpc/CMakeFiles/ggml-rpc.dir
fi
cmake --build build-cuda-b-bin -j"$(nproc)" --target rpc-server

sha="$(git rev-parse --short HEAD)"
if [[ "$NO_RESTART" -eq 0 ]]; then
    bash "${SCRIPT_DIR}/b6-gate-triton-start-rpc.sh"
    echo "TRITON_SYNC_REBUILD_OK sha=${sha} rpc=:50054"
else
    echo "TRITON_SYNC_REBUILD_OK sha=${sha} restart=skipped"
fi