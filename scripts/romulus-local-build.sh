#!/usr/bin/env bash
# Build romulus-local stack: ROCm llama-server + CUDA rpc-server docker image.
#
# usage (on romulus):
#   bash scripts/romulus-local-build.sh
#   bash scripts/romulus-local-build.sh --client-only
#   bash scripts/romulus-local-build.sh --rpc-only
#
# env:
#   CLUSTER_REPO              default ~/projects/atomic-llama-cpp-turboquant
#   ROMULUS_PATHB_DEPLOY_DIR  default ~/docker/Atomic-Llama-Romulus-PathB
#   ROMULUS_BUILD_DIR         default build-rocm-docker

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER_REPO="${CLUSTER_REPO:-$HOME/projects/atomic-llama-cpp-turboquant}"
DEPLOY_DIR="${ROMULUS_PATHB_DEPLOY_DIR:-$HOME/docker/Atomic-Llama-Romulus-PathB}"
BUILD_DIR="${ROMULUS_BUILD_DIR:-build-rocm-docker}"
COMPOSE="${ROOT}/scripts/romulus-pathb/docker-compose.yml"
GIT_BRANCH="${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"

DO_CLIENT=1
DO_RPC=1
DO_START=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --client-only) DO_RPC=0; shift ;;
        --rpc-only) DO_CLIENT=0; shift ;;
        --no-start) DO_START=0; shift ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$(readlink -f "$ROOT")" != "$(readlink -f "$CLUSTER_REPO")" ]]; then
    echo "warn: running from ${ROOT}; canonical repo is ${CLUSTER_REPO}" >&2
fi

cd "$ROOT"
# shellcheck source=/dev/null
source "${ROOT}/rpc-patch/scripts/pathb-cluster-git-remotes.sh"

sync_deploy() {
    local src="${ROOT}/rpc-patch/deploy/Atomic-Llama-Romulus-PathB"
    mkdir -p "$DEPLOY_DIR"
    echo "=== sync deploy -> ${DEPLOY_DIR} ==="
    rsync -av --delete \
        --exclude 'staging/' \
        --exclude 'staging/**' \
        "${src}/" "${DEPLOY_DIR}/"
    rsync -av "${COMPOSE}" "${DEPLOY_DIR}/docker-compose.quick.yml"
    chmod +x "${DEPLOY_DIR}/build.sh"
}

build_rpc() {
    local git_url
    git_url="$(pathb_cluster_clone_url)"
    echo "=== rebuild rpc-server docker (git ${git_url}) ==="
    cd "$DEPLOY_DIR"
    GIT_BRANCH="$GIT_BRANCH" GIT_URL="$git_url" GIT_COMMIT="${GIT_COMMIT:-}" ./build.sh
    if [[ "$DO_START" -eq 1 ]]; then
        docker compose -f docker-compose.quick.yml up -d --force-recreate
        sleep 2
        nc -zv 127.0.0.1 50051
    fi
}

build_client() {
    echo "=== rebuild llama-server (${BUILD_DIR}) ==="
    cd "$ROOT"
    if [[ -f ggml/include/ggml-rpc.h ]]; then
        cp -f ggml/include/ggml-rpc.h ggml/src/ggml-rpc.h
    fi
    touch ggml/src/ggml-backend.cpp ggml/src/ggml-rpc/ggml-rpc.cpp \
        ggml/include/ggml-rpc.h ggml/src/ggml-rpc.h 2>/dev/null || true
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        cmake -S . -B "$BUILD_DIR" \
            -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
            -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4 \
            -DLLAMA_BUILD_TOOLS=ON
    fi
    cmake --build "$BUILD_DIR" --target llama-server ggml-rpc ggml-base -j"$(nproc)"
    test -x "${BUILD_DIR}/bin/llama-server"
    echo "CLIENT_BUILD_OK ${ROOT}/${BUILD_DIR}/bin/llama-server"
}

[[ "$DO_RPC" -eq 1 ]] && sync_deploy && build_rpc
[[ "$DO_CLIENT" -eq 1 ]] && build_client

echo BUILD_OK