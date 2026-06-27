#!/usr/bin/env bash
# Start/stop remus 5060 Ti RPC (:50051) + RX6600 RPC (:50052) for Config F.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"

ACTION="${1:-status}"

case "$ACTION" in
    start)
        "$SCRIPT_DIR/pathb-remus-rpc.sh" start
        "$SCRIPT_DIR/pathb-remus-rx6600-rpc.sh" start
        echo "multi-rpc-win: remus 5060 Ti :50051 + RX6600 :50052 up"
        ;;
    stop)
        "$SCRIPT_DIR/pathb-remus-rx6600-rpc.sh" stop || true
        "$SCRIPT_DIR/pathb-remus-rpc.sh" stop || true
        ;;
    status)
        echo "=== pathb 5060 Ti ==="
        "$SCRIPT_DIR/pathb-remus-rpc.sh" status || true
        echo "=== rx6600 ==="
        "$SCRIPT_DIR/pathb-remus-rx6600-rpc.sh" status || true
        ;;
    *)
        echo "usage: $0 start|stop|status" >&2
        exit 1
        ;;
esac