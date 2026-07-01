#!/usr/bin/env bash
# Lifecycle for triton dual-GPU isolated RPC docker compose.
#
# usage: pathb-triton-rpc.sh start|stop|status|logs|deploy|build|rebuild
#
# env:
#   PATHB_TRITON_SSH      default hunter@192.168.8.23
#   PATHB_TRITON_SSH_PASS default 12345
#   PATHB_TRITON_DOCKER_DIR default ~/docker/Atomic-Llama-Triton-PathB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pathb-node-deploy-sync.sh
source "${SCRIPT_DIR}/pathb-node-deploy-sync.sh"

PATHB_NODE_SSH="${PATHB_NODE_SSH:-${PATHB_TRITON_SSH:-hunter@${TRITON_RPC_IP:-192.168.8.23}}}"
PATHB_NODE_SSH_PASS="${PATHB_NODE_SSH_PASS:-${PATHB_TRITON_SSH_PASS:-}}"
TRITON_DIR="${PATHB_TRITON_DOCKER_DIR:-~/docker/Atomic-Llama-Triton-PathB}"
DEPLOY_SRC="${SCRIPT_DIR}/../deploy/Atomic-Llama-Triton-PathB"
TRITON_REPO="${PATHB_TRITON_REPO:-~/projects/atomic-llama-cpp-turboquant}"

pathb_node_init_ssh

remote() {
    pathb_node_remote "$@"
}

docker_remote() {
    pathb_node_remote "sg docker -c $(printf '%q' "$*")"
}

deploy_sync() {
    pathb_node_deploy_sync "$DEPLOY_SRC" "$TRITON_DIR"
}

stop_native_rpc() {
    remote "pkill -f 'rpc-server.*-p 50054' 2>/dev/null || true; pkill -f 'rpc-server.*-p 50055' 2>/dev/null || true; true"
}

ACTION="${1:-status}"

case "$ACTION" in
    deploy)
        deploy_sync
        ;;
    start)
        stop_native_rpc
        docker_remote "cd $TRITON_DIR && docker compose up -d --force-recreate"
        sleep 3
        remote "ss -tlnp | grep -E '50054|50055' || true"
        ;;
    stop)
        docker_remote "cd $TRITON_DIR && docker compose down" || true
        ;;
    status)
        docker_remote "cd $TRITON_DIR && docker compose ps"
        remote "ss -tlnp | grep -E '50054|50055' || true"
        ;;
    logs)
        tail_n="${PATHB_TRITON_LOG_TAIL:-80}"
        docker_remote "cd $TRITON_DIR && docker compose logs --tail=${tail_n}"
        ;;
    build)
        if [[ "${PATHB_TRITON_SKIP_DEPLOY:-}" != "1" ]]; then
            deploy_sync
        fi
        docker_remote "cd $TRITON_DIR && TRITON_REPO=$TRITON_REPO ./build-from-bin.sh"
        ;;
    rebuild)
        deploy_sync
        remote "cd $TRITON_REPO && B6_TRITON_SKIP_STASH=1 bash scripts/b6-gate-triton-sync-rebuild.sh --no-restart"
        docker_remote "cd $TRITON_DIR && TRITON_REPO=$TRITON_REPO ./build-from-bin.sh"
        stop_native_rpc
        docker_remote "cd $TRITON_DIR && docker compose up -d --force-recreate"
        ;;
    *)
        echo "usage: $0 {deploy|start|stop|status|logs|build|rebuild}" >&2
        exit 1
        ;;
esac