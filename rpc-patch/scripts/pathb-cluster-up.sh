#!/usr/bin/env bash
# Cluster lifecycle: remus + romulus Path B RPC (+ optional local CUDA worker).
#
# usage: pathb-cluster-up.sh start|stop|status [--local]
#
# env:
#   PATHB_CLUSTER_LOCAL=1   also start/stop local pathb-rpc (pathb-start-rpc.sh)
#   PATHB_REMUS_SSH_PASS, PATHB_ROMULUS_SSH_PASS
#   REMUS_RPC_IP (default 192.168.8.176), ROMULUS_RPC_IP (default 192.168.8.108)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "usage: $0 start|stop|status [--local]" >&2
    echo "  --local: include local pathb-rpc (Romulus 3060 Ti via pathb-start-rpc.sh)" >&2
    exit 1
}

ACTION="${1:-status}"
LOCAL=0
[[ "${PATHB_CLUSTER_LOCAL:-}" == "1" ]] && LOCAL=1
if [[ $# -gt 0 ]]; then
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) LOCAL=1 ;;
        *) usage ;;
    esac
    shift
done

case "$ACTION" in
    start)
        "$SCRIPT_DIR/pathb-remus-rpc.sh" start
        "$SCRIPT_DIR/pathb-romulus-rpc.sh" start
        if [[ "$LOCAL" == "1" ]]; then
            "$SCRIPT_DIR/pathb-start-rpc.sh"
        fi
        echo "cluster: remus + romulus pathb-rpc up${LOCAL:+ (+ local pathb-rpc)}"
        ;;
    stop)
        if [[ "$LOCAL" == "1" ]]; then
            "$SCRIPT_DIR/pathb-start-rpc.sh" --stop 2>/dev/null || docker rm -f pathb-rpc 2>/dev/null || true
        fi
        "$SCRIPT_DIR/pathb-romulus-rpc.sh" stop
        "$SCRIPT_DIR/pathb-remus-rpc.sh" stop
        ;;
    status)
        echo "=== remus (${REMUS_RPC_IP:-192.168.8.176}) ==="
        "$SCRIPT_DIR/pathb-remus-rpc.sh" status || true
        echo "=== romulus (${ROMULUS_RPC_IP:-192.168.8.108}) ==="
        "$SCRIPT_DIR/pathb-romulus-rpc.sh" status || true
        if [[ "$LOCAL" == "1" ]]; then
            echo "=== local pathb-rpc ==="
            docker ps --filter name=pathb-rpc --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
        fi
        ;;
    *)
        usage
        ;;
esac