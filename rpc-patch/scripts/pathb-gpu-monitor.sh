#!/usr/bin/env bash
# Poll local + remote GPU stats during a bench run.
# usage: pathb-gpu-monitor.sh <logfile> [duration_sec]
# env: REMUS_RPC_IP, PATHB_REMUS_SSH_PASS

set -euo pipefail

LOG="${1:?logfile required}"
DURATION="${2:-600}"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
end=$(( $(date +%s) + DURATION ))

SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_CMD=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
fi

: >"$LOG"
while [[ $(date +%s) -lt $end ]]; do
    {
        echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        echo "[romulus-rocm]"
        timeout 3 rocm-smi --showuse 2>/dev/null | head -3 || timeout 3 rocm-smi 2>/dev/null | head -2 || echo "rocm-smi timeout"
        echo "[romulus-nvidia]"
        timeout 3 nvidia-smi --query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi timeout"
        echo "[remus-nvidia]"
        timeout 8 "${SSH_CMD[@]}" nvidia-smi --query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "remus ssh nvidia-smi failed"
    } >>"$LOG"
    sleep 2
done