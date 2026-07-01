#!/usr/bin/env bash
# Stop triton rpc-server on :50054 (and optional :50055 3070 worker).
#
# usage: b6-gate-triton-stop-rpc.sh [port...]
#
# env:
#   B6_TRITON_RPC_PORTS  default "50054 50055"

set -euo pipefail

ports=("$@")
if [[ ${#ports[@]} -eq 0 ]]; then
    read -r -a ports <<< "${B6_TRITON_RPC_PORTS:-50054 50055}"
fi

for port in "${ports[@]}"; do
    pkill -f "rpc-server.*-p ${port}" 2>/dev/null || true
done

sleep 1
echo "TRITON_RPC_STOP_OK ports=${ports[*]}"