#!/usr/bin/env bash
# Ground-truth GPU inventory: nvidia-smi + rocm-smi on romulus, remus, triton.
#
# usage: b6-gate-cluster-gpu-inventory.sh
#
# env:
#   B6_ROMULUS_HOST   default 192.168.8.108
#   B6_REMUS_HOST     default 192.168.8.176
#   B6_TRITON_HOST    default 192.168.8.23
#   B6_CLUSTER_PASS   default 12345 (sshpass when available)

set -euo pipefail

ROMULUS_HOST="${B6_ROMULUS_HOST:-192.168.8.108}"
REMUS_HOST="${B6_REMUS_HOST:-192.168.8.176}"
TRITON_HOST="${B6_TRITON_HOST:-192.168.8.23}"
CLUSTER_PASS="${B6_CLUSTER_PASS:-12345}"

is_local_host() {
    local host="$1"
    local hn
    hn="$(hostname)"
    case "$host" in
        "$ROMULUS_HOST"|Romulus|romulus) [[ "$hn" == Romulus || "$hn" == romulus ]] ;;
        "$REMUS_HOST"|Remus|remus) [[ "$hn" == Remus || "$hn" == remus ]] ;;
        "$TRITON_HOST"|Triton|triton) [[ "$hn" == Triton || "$hn" == triton ]] ;;
        *) false ;;
    esac
}

ssh_node() {
    local host="$1"
    shift
    if is_local_host "$host"; then
        bash -lc "$*"
        return
    fi
    if command -v sshpass &>/dev/null && [[ -n "$CLUSTER_PASS" ]]; then
        SSHPASS="$CLUSTER_PASS" sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "hunter@${host}" "$@"
    else
        ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "hunter@${host}" "$@"
    fi
}

probe_host() {
    local label="$1" host="$2"
    echo "=== ${label} (${host}) ==="
    ssh_node "$host" '
        echo "hostname: $(hostname)"
        echo "--- nvidia-smi ---"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
            || echo "(no NVIDIA GPUs)"
        echo "--- rocm-smi ---"
        if command -v rocm-smi &>/dev/null; then
            rocm-smi --showproductname --showmeminfo vram 2>/dev/null \
                | grep -E "GPU\[|Card Series|VRAM Total Memory" \
                || rocm-smi 2>/dev/null | head -6
        else
            echo "(no ROCm GPUs)"
        fi
    '
    echo ""
}

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cluster gpu inventory @ ${STAMP}"
echo ""

probe_host "romulus" "$ROMULUS_HOST"
probe_host "remus" "$REMUS_HOST"
probe_host "triton" "$TRITON_HOST"

echo "Active topology (4-GPU gate):"
echo "  RPC0 remus 5060     -> ${REMUS_HOST}:50051"
echo "  RPC1 romulus 3060   -> 127.0.0.1:50051 (from romulus client)"
echo "  RPC2 triton 3090    -> ${TRITON_HOST}:50054"
echo "  ROCm0 romulus 7900  -> local on ${ROMULUS_HOST}"
echo "  remus RX6600 :50052 and triton 3070 :50055 are parked (not in primary gate)"