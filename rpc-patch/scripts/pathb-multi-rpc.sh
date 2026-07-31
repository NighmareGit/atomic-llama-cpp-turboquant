#!/usr/bin/env bash
# Start/stop remus 5060 Ti RPC + local Romulus 3060 Ti pathb-rpc (Config C).

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ACTION="${1:-status}"

case "$ACTION" in
    start)
        export PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
        "$SCRIPT_DIR/pathb-remus-rpc.sh" start
        "$SCRIPT_DIR/pathb-start-rpc.sh"
        echo "multi-rpc: remus + local pathb-rpc up"
        ;;
    stop)
        "$SCRIPT_DIR/pathb-start-rpc.sh" --stop 2>/dev/null || docker rm -f pathb-rpc 2>/dev/null || true
        "$SCRIPT_DIR/pathb-remus-rpc.sh" stop
        ;;
    status)
        echo "=== remus ==="
        "$SCRIPT_DIR/pathb-remus-rpc.sh" status || true
        echo "=== local pathb-rpc ==="
        docker ps --filter name=pathb-rpc --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
        ;;
    *)
        echo "usage: $0 start|stop|status" >&2
        exit 1
        ;;
esac