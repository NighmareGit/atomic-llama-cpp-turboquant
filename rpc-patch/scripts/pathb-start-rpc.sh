#!/usr/bin/env bash
# Start CUDA rpc-server for Path B Config A (NVIDIA worker).
#
# usage: pathb-start-rpc.sh [--stop]
#
# env:
#   PATHB_RPC_CONTAINER   default pathb-rpc
#   PATHB_BIN_CUDA        default $ROOT/build-cuda-b-bin/bin
#   PATHB_CUDA_IMAGE      default llama-rpc-cuda-a2
#   PATHB_RPC_HOST        default 0.0.0.0
#   PATHB_RPC_PORT        default 50051
#   PATHB_CUDA_DEVICE     default 0  (CUDA_VISIBLE_DEVICES)

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"

PATHB_RPC_CONTAINER="${PATHB_RPC_CONTAINER:-pathb-rpc}"
_sync_bin="${TQ}/build-cuda-b-bin-sync/bin"
if [[ -z "${PATHB_BIN_CUDA:-}" ]] && [[ -x "${_sync_bin}/rpc-server" ]]; then
    PATHB_BIN_CUDA="${_sync_bin}"
else
    PATHB_BIN_CUDA="${PATHB_BIN_CUDA:-${TQ}/build-cuda-b-bin/bin}"
fi
PATHB_CUDA_IMAGE="${PATHB_CUDA_IMAGE:-llama-rpc-cuda-a2}"
PATHB_RPC_HOST="${PATHB_RPC_HOST:-0.0.0.0}"
PATHB_RPC_PORT="${PATHB_RPC_PORT:-50051}"
PATHB_CUDA_DEVICE="${PATHB_CUDA_DEVICE:-0}"

if [[ "${1:-}" == "--stop" ]]; then
    docker rm -f "$PATHB_RPC_CONTAINER" 2>/dev/null || true
    echo "stopped $PATHB_RPC_CONTAINER"
    exit 0
fi

[[ -x "${PATHB_BIN_CUDA}/rpc-server" ]] || {
    echo "rpc-server missing: ${PATHB_BIN_CUDA}/rpc-server" >&2
    echo "Rebuild with Docker volume-mount to $TQ (see docs/rpc-path-b-tracking.md)" >&2
    exit 1
}

docker rm -f "$PATHB_RPC_CONTAINER" 2>/dev/null || true

docker run -d --name "$PATHB_RPC_CONTAINER" \
    --gpus "device=${PATHB_CUDA_DEVICE}" \
    --network host \
    -v "${PATHB_BIN_CUDA}:/app/bin:ro" \
    -e LD_LIBRARY_PATH=/app/bin \
    "$PATHB_CUDA_IMAGE" \
    bash -c "/app/bin/rpc-server -H ${PATHB_RPC_HOST} -p ${PATHB_RPC_PORT} -d CUDA0"

sleep 2
if docker ps -q --filter "name=^${PATHB_RPC_CONTAINER}$" | grep -q .; then
    echo "RPC server started: ${PATHB_RPC_HOST}:${PATHB_RPC_PORT} (container: $PATHB_RPC_CONTAINER)"
else
    echo "WARN: rpc-server container exited. Check: docker logs $PATHB_RPC_CONTAINER" >&2
    exit 1
fi