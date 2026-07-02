#!/usr/bin/env bash
# Lifecycle for remus Path B RPC docker compose.
#
# usage: pathb-remus-rpc.sh start|stop|status|logs|deploy|build|rebuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pathb-remus-deploy-sync.sh
source "${SCRIPT_DIR}/pathb-remus-deploy-sync.sh"

REMUS_SSH="${PATHB_REMUS_SSH:-hunter@${REMUS_RPC_IP:-192.168.8.176}}"
REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
REMUS_DIR="${PATHB_REMUS_DOCKER_DIR:-~/docker/Atomic-Llama-Remus-PathB}"
DEPLOY_SRC="${SCRIPT_DIR}/../deploy/Atomic-Llama-Remus-PathB"

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
        remus_env=""
        if [[ -n "${PATHB_CUDA_DISABLE_GRAPHS:-${GGML_CUDA_DISABLE_GRAPHS:-}}" ]]; then
            remus_env="GGML_CUDA_DISABLE_GRAPHS=1"
        fi
        remote "cd $REMUS_DIR && ${remus_env} docker compose up -d --force-recreate"
        sleep 2
        nc -zv "${REMUS_RPC_IP:-192.168.8.176}" 50051
        ;;
    stop)
        remote "cd $REMUS_DIR && docker compose down" || true
        ;;
    status)
        remote "cd $REMUS_DIR && docker compose ps && ss -tlnp | grep 50051 || true"
        ;;
    logs)
        tail_n="${PATHB_REMUS_LOG_TAIL:-80}"
        remote "cd $REMUS_DIR && docker compose logs --tail=${tail_n}"
        ;;
    logs-since)
        remote "cd $REMUS_DIR && docker compose logs --since=10m"
        ;;
    build)
        if [[ "${PATHB_REMUS_SKIP_DEPLOY:-}" != "1" ]]; then
            deploy_sync
        fi
        remote "cd $REMUS_DIR && GIT_COMMIT='${GIT_COMMIT:-}' ./build.sh"
        ;;
    rebuild)
        deploy_sync
        remote "cd $REMUS_DIR && GIT_COMMIT='${GIT_COMMIT:-}' ./build.sh"
        remote "cd $REMUS_DIR && docker compose up -d --force-recreate"
        sleep 2
        nc -zv "${REMUS_RPC_IP:-192.168.8.176}" 50051
        ;;
    *)
        echo "usage: $0 start|stop|status|logs|deploy|build|rebuild" >&2
        echo "  build/rebuild: sync deploy from repo first (set PATHB_REMUS_SKIP_DEPLOY=1 to skip)" >&2
        exit 1
        ;;
esac