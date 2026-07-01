#!/usr/bin/env bash
# Lifecycle for romulus Path B RPC docker compose.
#
# usage: pathb-romulus-rpc.sh start|stop|status|logs|deploy|build|rebuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pathb-node-deploy-sync.sh
source "${SCRIPT_DIR}/pathb-node-deploy-sync.sh"
# shellcheck source=pathb-node-git-sync.sh
source "${SCRIPT_DIR}/pathb-node-git-sync.sh"

PATHB_NODE_SSH="${PATHB_NODE_SSH:-${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}}"
PATHB_NODE_SSH_PASS="${PATHB_NODE_SSH_PASS:-${PATHB_ROMULUS_SSH_PASS:-}}"
ROMULUS_DIR="${PATHB_ROMULUS_DOCKER_DIR:-~/docker/Atomic-Llama-Romulus-PathB}"
DEPLOY_SRC="${SCRIPT_DIR}/../deploy/Atomic-Llama-Romulus-PathB"

pathb_node_init_ssh

remote() {
    pathb_node_remote "$@"
}

deploy_sync() {
    pathb_node_deploy_sync "$DEPLOY_SRC" "$ROMULUS_DIR"
}

host_git_sync() {
    if [[ "${PATHB_ROMULUS_SKIP_GIT_SYNC:-}" == "1" ]]; then
        return 0
    fi
    pathb_node_git_sync "${PATHB_ROMULUS_GIT_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
}

remote_build() {
    local branch
    branch="$(pathb_node_git_branch)"
    remote "cd $ROMULUS_DIR && GIT_BRANCH='${branch}' ./build.sh"
}

ACTION="${1:-status}"

case "$ACTION" in
    deploy)
        deploy_sync
        ;;
    start)
        romulus_env=""
        if [[ -n "${PATHB_CUDA_DISABLE_GRAPHS:-${GGML_CUDA_DISABLE_GRAPHS:-}}" ]]; then
            romulus_env="GGML_CUDA_DISABLE_GRAPHS=1"
        fi
        remote "cd $ROMULUS_DIR && ${romulus_env} docker compose up -d --force-recreate"
        sleep 2
        nc -zv "${ROMULUS_RPC_IP:-192.168.8.108}" 50051
        ;;
    stop)
        remote "cd $ROMULUS_DIR && docker compose down" || true
        ;;
    status)
        remote "cd $ROMULUS_DIR && docker compose ps && ss -tlnp | grep 50051 || true"
        ;;
    logs)
        tail_n="${PATHB_ROMULUS_LOG_TAIL:-80}"
        remote "cd $ROMULUS_DIR && docker compose logs --tail=${tail_n}"
        ;;
    logs-since)
        remote "cd $ROMULUS_DIR && docker compose logs --since=10m"
        ;;
    build)
        if [[ "${PATHB_ROMULUS_SKIP_DEPLOY:-}" != "1" ]]; then
            deploy_sync
        fi
        host_git_sync
        remote_build
        ;;
    rebuild)
        deploy_sync
        host_git_sync
        remote_build
        remote "cd $ROMULUS_DIR && docker compose up -d --force-recreate"
        sleep 2
        nc -zv "${ROMULUS_RPC_IP:-192.168.8.108}" 50051
        ;;
    *)
        echo "usage: $0 start|stop|status|logs|deploy|build|rebuild" >&2
        echo "  build/rebuild: sync deploy from repo first (set PATHB_ROMULUS_SKIP_DEPLOY=1 to skip)" >&2
        exit 1
        ;;
esac