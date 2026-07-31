#!/usr/bin/env bash
# Remote ops on jupiter (Windows OpenSSH): start Path B RPC :50053.
#
# usage:
#   B6_JUPITER_PASS='...' bash scripts/b6-gate-jupiter-remote.sh start-rpc
#   bash scripts/b6-gate-jupiter-remote.sh probe
#
# env:
#   B6_JUPITER_HOST     default 192.168.8.21
#   B6_JUPITER_USER     default hunter
#   B6_JUPITER_PASS     ssh password (or use key auth)
#   B6_JUPITER_REPO     default D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant

set -euo pipefail

JUPITER_HOST="${B6_JUPITER_HOST:-192.168.8.21}"
JUPITER_USER="${B6_JUPITER_USER:-nightmare}"
JUPITER_REPO="${B6_JUPITER_REPO:-D:\\projects\\atomic-llama-cpp-5070ti\\atomic-llama-cpp-turboquant}"
CMD="${1:-probe}"

ssh_jupiter() {
    local remote_cmd="$1"
    if command -v sshpass &>/dev/null && [[ -n "${B6_JUPITER_PASS:-}" ]]; then
        sshpass -p "$B6_JUPITER_PASS" ssh -o StrictHostKeyChecking=accept-new \
            "${JUPITER_USER}@${JUPITER_HOST}" "$remote_cmd"
    else
        ssh -o StrictHostKeyChecking=accept-new "${JUPITER_USER}@${JUPITER_HOST}" "$remote_cmd"
    fi
}

probe() {
    echo "=== jupiter probe ${JUPITER_HOST} ==="
    ssh_jupiter "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
        echo "SSH failed. Set B6_JUPITER_PASS or configure key auth." >&2
        return 1
    }
    if command -v nc &>/dev/null; then
        if nc -z -w3 "$JUPITER_HOST" 50053 2>/dev/null; then
            echo "RPC :50053 LISTEN"
        else
            echo "RPC :50053 CLOSED"
        fi
    fi
}

start_rpc() {
    local ps1="${JUPITER_REPO}\\scripts\\b6-gate-jupiter-start-rpc-task.ps1"
    echo "=== jupiter start rpc :50053 ==="
    ssh_jupiter "powershell -NoProfile -ExecutionPolicy Bypass -File \"${ps1}\""
    sleep 3
    probe
}

case "$CMD" in
    probe) probe ;;
    start-rpc) start_rpc ;;
    *)
        echo "usage: $0 {probe|start-rpc}" >&2
        exit 2
        ;;
esac