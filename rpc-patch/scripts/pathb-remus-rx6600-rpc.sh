#!/usr/bin/env bash
# Lifecycle for remus RX6600 RPC docker compose (Config F worker on port 50052).
#
# usage: pathb-remus-rx6600-rpc.sh start|stop|status|logs|deploy|build|rebuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pathb-remus-deploy-sync.sh
source "${SCRIPT_DIR}/pathb-remus-deploy-sync.sh"

REMUS_SSH="${PATHB_REMUS_SSH:-hunter@${REMUS_RPC_IP:-192.168.8.176}}"
REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
REMUS_DIR="${PATHB_REMUS_RX6600_DIR:-~/docker/Atomic-Llama-Remus-RX6600}"
RPC_PORT="${PATHB_RX6600_RPC_PORT:-50052}"
DEPLOY_SRC="${SCRIPT_DIR}/../deploy/Atomic-Llama-Remus-RX6600"

pathb_remus_init_ssh

remote() {
    pathb_remus_remote "$@"
}

deploy_sync() {
    pathb_remus_deploy_sync "$DEPLOY_SRC" "$REMUS_DIR"
}

ACTION="${1:-status}"

case "$ACTION" in
    deploy)
        deploy_sync
        ;;
    start)
        remote "cd $REMUS_DIR && RPC_PORT=${RPC_PORT} docker compose up -d rx6600-rpc"
        sleep 2
        nc -zv "${REMUS_RPC_IP:-192.168.8.176}" "${RPC_PORT}"
        ;;
    stop)
        remote "cd $REMUS_DIR && docker compose stop rx6600-rpc" || true
        ;;
    status)
        remote "cd $REMUS_DIR && docker compose ps && ss -tlnp | grep ${RPC_PORT} || true"
        ;;
    logs)
        tail_n="${PATHB_REMUS_LOG_TAIL:-80}"
        remote "cd $REMUS_DIR && docker compose logs --tail=${tail_n} rx6600-rpc"
        ;;
    build)
        if [[ "${PATHB_REMUS_SKIP_DEPLOY:-}" != "1" ]]; then
            deploy_sync
        fi
        remote "cd $REMUS_DIR && ./build-rpc.sh"
        ;;
    rebuild)
        deploy_sync
        remote "cd $REMUS_DIR && ./build-rpc.sh"
        remote "cd $REMUS_DIR && RPC_PORT=${RPC_PORT} docker compose up -d --force-recreate rx6600-rpc"
        sleep 2
        nc -zv "${REMUS_RPC_IP:-192.168.8.176}" "${RPC_PORT}"
        ;;
    *)
        echo "usage: $0 start|stop|status|logs|deploy|build|rebuild" >&2
        echo "  build/rebuild: sync deploy from repo first (set PATHB_REMUS_SKIP_DEPLOY=1 to skip)" >&2
        exit 1
        ;;
esac