#!/usr/bin/env bash
# Profile-mode remus GPU telemetry (5060 nvidia dmon CSV, 6600 rocm JSONL).
# usage: pathb-profile-gpu-monitor.sh <telemetry_dir> <duration_sec>
# env: REMUS_RPC_IP, PATHB_REMUS_SSH_PASS, PATHB_MONITOR_ROCM=1

set -euo pipefail

TEL_DIR="${1:?telemetry_dir required}"
DURATION="${2:-600}"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
MON_ROCM="${PATHB_MONITOR_REMUS_ROCM:-1}"

mkdir -p "$TEL_DIR"

SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_CMD=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
fi

# Windows 5070 via WSL nvidia-smi (1s min for dmon; use query loop for finer samples)
WIN_CSV="$TEL_DIR/win-5070-dmon.csv"
REMUS_CSV="$TEL_DIR/remus-5060-dmon.csv"
ROCM_JSONL="$TEL_DIR/remus-6600-rocm.jsonl"

# Background: win 5070 query loop ~500ms
(
    echo "timestamp,power_w,util_gpu_pct,util_mem_pct,sm_clock_mhz,mem_clock_mhz,mem_used_mib,mem_total_mib,rxpci_mbs,txpci_mbs"
    end=$(( $(date +%s) + DURATION ))
    while [[ $(date +%s) -lt $end ]]; do
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        vals=$(timeout 2 nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "")
        if [[ -n "$vals" ]]; then
            pcie=$(timeout 2 nvidia-smi dmon -s t -d 1 -c 1 2>/dev/null | tail -1 | awk '{print $17","$18}' || echo "0,0")
            echo "${ts},${vals},${pcie}"
        fi
        sleep 0.5
    done
) >"$WIN_CSV" &

WIN_PID=$!

# Background: remus 5060 dmon CSV
(
    "${SSH_CMD[@]}" "timeout $((DURATION + 10)) nvidia-smi dmon -s pucvmte -d 1 -o DT --format csv nounit -f /tmp/remus-5060-dmon.csv 2>/dev/null; cat /tmp/remus-5060-dmon.csv" >"$REMUS_CSV" 2>/dev/null || echo "remus dmon failed" >"$REMUS_CSV"
) &
REMUS_PID=$!

# Background: remus 6600 rocm JSONL
if [[ "$MON_ROCM" == "1" ]]; then
    (
        end=$(( $(date +%s) + DURATION ))
        while [[ $(date +%s) -lt $end ]]; do
            echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
            timeout 5 "${SSH_CMD[@]}" rocm-smi -d 0 --showuse --showpower --showmemuse --json 2>/dev/null | tr -d '\n' | sed 's/^{//' | sed 's/^/,\"/' || true
            echo "}"
            sleep 1
        done
    ) >"$ROCM_JSONL" &
    ROCM_PID=$!
fi

wait $WIN_PID 2>/dev/null || true
wait $REMUS_PID 2>/dev/null || true
[[ -n "${ROCM_PID:-}" ]] && wait $ROCM_PID 2>/dev/null || true