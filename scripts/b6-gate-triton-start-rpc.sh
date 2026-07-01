#!/usr/bin/env bash
# Start triton CUDA rpc-server on :50054 (3090, CUDA0).
#
# usage: b6-gate-triton-start-rpc.sh [--restart]
#
# env:
#   B6_TRITON_REPO       default ~/projects/atomic-llama-cpp-turboquant
#   B6_TRITON_RPC_PORT   default 50054
#   B6_TRITON_RPC_DEVICE default CUDA0
#   B6_TRITON_RPC_LOG    default /tmp/triton-rpc-50054.log

set -euo pipefail

REPO="${B6_TRITON_REPO:-${HOME}/projects/atomic-llama-cpp-turboquant}"
PORT="${B6_TRITON_RPC_PORT:-50054}"
DEVICE="${B6_TRITON_RPC_DEVICE:-CUDA0}"
LOG="${B6_TRITON_RPC_LOG:-/tmp/triton-rpc-${PORT}.log}"
RPC_BIN="${REPO}/build-cuda-b-bin/bin/rpc-server"

[[ -x "$RPC_BIN" ]] || {
    echo "rpc-server missing: $RPC_BIN (run b6-gate-triton-sync-rebuild.sh)" >&2
    exit 1
}

pkill -f "rpc-server.*-p ${PORT}" 2>/dev/null || true
sleep 1

export LD_LIBRARY_PATH="${REPO}/build-cuda-b-bin/bin:${LD_LIBRARY_PATH:-}"
nohup "$RPC_BIN" -H 0.0.0.0 -p "$PORT" -d "$DEVICE" >"$LOG" 2>&1 &
pid=$!
sleep 3

if ! kill -0 "$pid" 2>/dev/null; then
    echo "rpc-server exited; log tail:" >&2
    tail -30 "$LOG" >&2 || true
    exit 1
fi

if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "rpc-server pid=${pid} not listening on :${PORT}" >&2
    tail -30 "$LOG" >&2 || true
    exit 1
fi

echo "TRITON_RPC_OK pid=${pid} port=${PORT} device=${DEVICE} log=${LOG}"