#!/usr/bin/env bash
# Poll Windows-host (WSL nvidia-smi) + remus GPU stats during a bench run.
# usage: pathb-gpu-monitor-win.sh <logfile> [duration_sec]
# env: REMUS_RPC_IP, PATHB_REMUS_SSH_PASS, PATHB_MONITOR_REMUS_ROCM=1

set -euo pipefail

LOG="${1:?logfile required}"
DURATION="${2:-600}"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
MON_ROCM="${PATHB_MONITOR_REMUS_ROCM:-0}"
end=$(( $(date +%s) + DURATION ))

SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_CMD=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
fi

: >"$LOG"
while [[ $(date +%s) -lt $end ]]; do
    {
        echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        echo "[win-nvidia]"
        timeout 3 nvidia-smi --query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi timeout"
        echo "[remus-nvidia]"
        timeout 8 "${SSH_CMD[@]}" nvidia-smi --query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "remus ssh nvidia-smi failed"
        if [[ "$MON_ROCM" == "1" ]]; then
            echo "[remus-rocm]"
            timeout 8 "${SSH_CMD[@]}" rocm-smi --showmeminfo vram 2>/dev/null | head -8 || echo "remus rocm-smi failed"
        fi
    } >>"$LOG"
    sleep 2
done