#!/usr/bin/env bash
# Orchestrate triton ops from remus/romulus over SSH (Ubuntu triton host).
#
# usage: b6-gate-triton-remote.sh <sync|status|rpc|stop|rebuild|audit>
#
# env:
#   B6_TRITON_HOST     default 192.168.8.23
#   B6_TRITON_USER     default hunter
#   B6_TRITON_PASS     default 12345 (sshpass; prefer key auth)
#   B6_TRITON_REPO     default ~/projects/atomic-llama-cpp-turboquant

set -euo pipefail

HOST="${B6_TRITON_HOST:-192.168.8.23}"
USER="${B6_TRITON_USER:-hunter}"
PASS="${B6_TRITON_PASS:-12345}"
REPO="${B6_TRITON_REPO:-~/projects/atomic-llama-cpp-turboquant}"
ACTION="${1:-status}"

ssh_cmd() {
    local remote="$1"
    if command -v sshpass &>/dev/null && [[ -n "$PASS" ]]; then
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=accept-new "${USER}@${HOST}" "$remote"
    else
        ssh -o StrictHostKeyChecking=accept-new "${USER}@${HOST}" "$remote"
    fi
}

case "$ACTION" in
    sync)
        ssh_cmd "cd ${REPO} && bash scripts/b6-gate-triton-git-sync.sh"
        ;;
    status)
        ssh_cmd "cd ${REPO} && git rev-parse --short HEAD 2>/dev/null || echo GIT_BROKEN; ss -tlnp | grep -E '50054|50055' || echo RPC_DOWN"
        ;;
    rpc)
        ssh_cmd "cd ${REPO} && bash scripts/b6-gate-triton-start-rpc.sh"
        ;;
    stop)
        ssh_cmd "bash ${REPO}/scripts/b6-gate-triton-stop-rpc.sh"
        ;;
    rebuild)
        ssh_cmd "cd ${REPO} && bash scripts/b6-gate-triton-sync-rebuild.sh"
        ;;
    audit)
        ssh_cmd "cd ${REPO} && echo SHA=\$(git rev-parse --short HEAD 2>/dev/null || echo unknown) && pgrep -a rpc-server || echo no-rpc && ss -tlnp | grep -E '50054|50055' || true && tail -5 /tmp/triton-rpc-50054.log 2>/dev/null || true"
        ;;
    *)
        echo "usage: $0 <sync|status|rpc|stop|rebuild|audit>" >&2
        exit 1
        ;;
esac