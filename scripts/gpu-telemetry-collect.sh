#!/usr/bin/env bash
# Collect schema-stable GPU telemetry during a profiler gen window.
#
# usage:
#   gpu-telemetry-collect.sh <telemetry/gpu_dir> <duration_sec>
#
# env:
#   REMUS_RPC_IP, PATHB_REMUS_SSH_PASS, PATHB_MONITOR_ROCM=1
#   GPU_COLLECT_INTERVAL_MS=500  (NVIDIA query loop)
#   GPU_COLLECT_WIN=1            (collect local NVIDIA via WSL/Linux)

set -euo pipefail

GPU_DIR="${1:?telemetry/gpu dir required}"
DURATION="${2:-120}"
INTERVAL_MS="${GPU_COLLECT_INTERVAL_MS:-500}"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
MON_ROCM="${PATHB_MONITOR_ROCM:-1}"
COLLECT_WIN="${GPU_COLLECT_WIN:-1}"

mkdir -p "$GPU_DIR"

SCHEMA_FILE="$GPU_DIR/schema.json"
cat >"$SCHEMA_FILE" <<'EOF'
{
  "version": 1,
  "nvidia_csv": {
    "file_pattern": "nvidia-*.csv",
    "columns": [
      "timestamp_utc", "power_w", "util_gpu_pct", "util_mem_pct",
      "sm_clock_mhz", "mem_clock_mhz", "mem_used_mib", "mem_total_mib",
      "pcie_rx_mbs", "pcie_tx_mbs"
    ],
    "interval_ms": 500
  },
  "rocm_jsonl": {
    "file_pattern": "rocm-*.jsonl",
    "fields": ["ts", "gpu_use_pct", "power_w", "vram_use_pct"]
  },
  "tdp_w": { "5070": 300, "5060": 175, "6600": 140 }
}
EOF

SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_CMD=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
fi

sleep_frac() {
    local ms="$1"
    if command -v python3 >/dev/null; then
        python3 -c "import time; time.sleep(${ms}/1000.0)"
    else
        sleep 1
    fi
}

WIN_CSV="$GPU_DIR/nvidia-local.csv"
REMUS_CSV="$GPU_DIR/nvidia-remus-5060.csv"
ROCM_JSONL="$GPU_DIR/rocm-remus.jsonl"

if [[ "$COLLECT_WIN" == "1" ]] && command -v nvidia-smi >/dev/null; then
    (
        echo "timestamp_utc,power_w,util_gpu_pct,util_mem_pct,sm_clock_mhz,mem_clock_mhz,mem_used_mib,mem_total_mib,pcie_rx_mbs,pcie_tx_mbs"
        end=$(( $(date +%s) + DURATION ))
        while [[ $(date +%s) -lt $end ]]; do
            ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            vals=$(timeout 2 nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
            if [[ -n "$vals" ]]; then
                pcie=$(timeout 2 nvidia-smi dmon -s t -d 1 -c 1 2>/dev/null | tail -1 | awk '{print $17","$18}' || echo "0,0")
                echo "${ts},${vals},${pcie}"
            fi
            sleep_frac "$INTERVAL_MS"
        done
    ) >"$WIN_CSV" &
    WIN_PID=$!
else
    WIN_PID=""
fi

(
    "${SSH_CMD[@]}" "timeout $((DURATION + 10)) nvidia-smi dmon -s pucvmte -d 1 -o DT --format csv nounit -f /tmp/remus-5060-dmon.csv 2>/dev/null; cat /tmp/remus-5060-dmon.csv" >"$REMUS_CSV" 2>/dev/null || echo "remus dmon failed" >"$REMUS_CSV"
) &
REMUS_PID=$!

if [[ "$MON_ROCM" == "1" ]]; then
    (
        end=$(( $(date +%s) + DURATION ))
        while [[ $(date +%s) -lt $end ]]; do
            ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            payload=$(timeout 5 "${SSH_CMD[@]}" rocm-smi -d 0 --showuse --showpower --showmemuse --json 2>/dev/null || echo "{}")
            echo "{\"ts\":\"${ts}\",\"payload\":${payload}}"
            sleep 1
        done
    ) >"$ROCM_JSONL" &
    ROCM_PID=$!
fi

[[ -n "$WIN_PID" ]] && wait "$WIN_PID" 2>/dev/null || true
wait "$REMUS_PID" 2>/dev/null || true
[[ -n "${ROCM_PID:-}" ]] && wait "$ROCM_PID" 2>/dev/null || true

echo "gpu-telemetry-collect done dir=$GPU_DIR duration=${DURATION}s"