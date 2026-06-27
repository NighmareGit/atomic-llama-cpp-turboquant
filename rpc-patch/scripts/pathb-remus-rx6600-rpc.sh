#!/usr/bin/env bash
# Lifecycle for remus RX6600 RPC docker compose (Config F worker on port 50052).
#
# usage: pathb-remus-rx6600-rpc.sh start|stop|status|logs|build

set -euo pipefail

REMUS_SSH="${PATHB_REMUS_SSH:-hunter@${REMUS_RPC_IP:-192.168.8.176}}"
REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
REMUS_DIR="${PATHB_REMUS_RX6600_DIR:-~/docker/Atomic-Llama-Remus-RX6600}"
RPC_PORT="${PATHB_RX6600_RPC_PORT:-50052}"

SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new)
if [[ -n "$REMUS_SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_CMD=(sshpass -p "$REMUS_SSH_PASS" ssh -o StrictHostKeyChecking=accept-new)
fi

remote() {
    "${SSH_CMD[@]}" "$REMUS_SSH" "$@"
}

ACTION="${1:-status}"

case "$ACTION" in
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
        remote "cd $REMUS_DIR && ./build-rpc.sh"
        ;;
    *)
        echo "usage: $0 start|stop|status|logs|build" >&2
        exit 1
        ;;
esac