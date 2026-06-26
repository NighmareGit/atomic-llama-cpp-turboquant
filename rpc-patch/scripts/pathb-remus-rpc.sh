#!/usr/bin/env bash
# Lifecycle for remus Path B RPC docker compose.
#
# usage: pathb-remus-rpc.sh start|stop|status|logs|build

set -euo pipefail

REMUS_SSH="${PATHB_REMUS_SSH:-hunter@${REMUS_RPC_IP:-192.168.8.176}}"
REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
REMUS_DIR="${PATHB_REMUS_DOCKER_DIR:-~/docker/Atomic-Llama-Remus-PathB}"

SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new)
SCP_CMD=(scp -o StrictHostKeyChecking=accept-new)
if [[ -n "$REMUS_SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_CMD=(sshpass -p "$REMUS_SSH_PASS" ssh -o StrictHostKeyChecking=accept-new)
    SCP_CMD=(sshpass -p "$REMUS_SSH_PASS" scp -o StrictHostKeyChecking=accept-new)
fi

remote() {
    "${SSH_CMD[@]}" "$REMUS_SSH" "$@"
}

ACTION="${1:-status}"

case "$ACTION" in
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
        remote "cd $REMUS_DIR && ./build.sh"
        ;;
    *)
        echo "usage: $0 start|stop|status|logs|build" >&2
        exit 1
        ;;
esac