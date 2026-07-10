#!/bin/bash
# safety-check.sh — Pre-flight safety check before any resource-intensive operation.
# Aborts if VRAM, RAM, disk, or running instances indicate risk of OOM or overallocation.
#
# Usage: bash scripts/safety-check.sh
# Exit 0 = safe to proceed
# Exit 1 = UNSAFE — abort operation

set -euo pipefail

SAFE=0
WARN=1

# --- GPU VRAM ---
echo "=== GPU VRAM ==="
if command -v nvidia-smi &>/dev/null; then
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  GPU 0: ${FREE_VRAM} MiB free / ${TOTAL_VRAM} MiB total"
    if [ "${FREE_VRAM:-0}" -lt 2000 ]; then
        echo "  WARNING: < 2000 MiB free VRAM. Risk of OOM."
        exit ${WARN}
    fi
elif command -v rocm-smi &>/dev/null; then
    FREE_VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i "free" | head -1 | awk '{print $2}' || echo "unknown")
    echo "  ROCm: ${FREE_VRAM} (check manually)"
else
    echo "  No nvidia-smi or rocm-smi found. Cannot check VRAM."
fi

# --- System RAM ---
echo "=== System RAM ==="
FREE_RAM=$(free -m | awk '/^Mem:/{print $7}')
TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
echo "  ${FREE_RAM} MiB free / ${TOTAL_RAM} MiB total"
if [ "${FREE_RAM:-0}" -lt 4000 ]; then
    echo "  WARNING: < 4000 MiB free RAM. Risk of OOM."
    exit ${WARN}
fi

# --- Disk / ---
echo "=== Disk / ==="
FREE_DISK=$(df -BG / | awk 'NR==2{print $4}' | tr -d 'G')
echo "  ${FREE_DISK} GB free on /"
if [ "${FREE_DISK:-0}" -lt 20 ]; then
    echo "  WARNING: < 20 GB free on /. Risk of build failure."
    exit ${WARN}
fi

# --- Running llama-server instances ---
echo "=== Running llama-server instances ==="
SERVER_COUNT=$(pgrep -c llama-server 2>/dev/null || echo 0)
echo "  ${SERVER_COUNT} instance(s) running"
if [ "${SERVER_COUNT}" -gt 0 ]; then
    echo "  WARNING: llama-server already running. Do not start another (OOM risk)."
    exit ${WARN}
fi

# --- Running llama-cli instances ---
echo "=== Running llama-cli instances ==="
CLI_COUNT=$(pgrep -c llama-cli 2>/dev/null || echo 0)
echo "  ${CLI_COUNT} instance(s) running"
if [ "${CLI_COUNT}" -gt 0 ]; then
    echo "  WARNING: llama-cli running. May indicate fault-start risk."
    exit ${WARN}
fi

# --- Docker disk usage ---
echo "=== Docker disk usage ==="
if command -v docker &>/dev/null; then
    DOCKER_SPACE=$(docker system df 2>/dev/null | tail -1 || echo "unknown")
    echo "  ${DOCKER_SPACE}"
    DOCKER_RECLAIMABLE=$(docker system df 2>/dev/null | grep -i "reclaimable" | awk '{print $3}' || echo "0")
    if [ "${DOCKER_RECLAIMABLE:-0}" != "0" ] && [ "${DOCKER_RECLAIMABLE:-0}" != "0B" ]; then
        echo "  NOTE: Reclaimable docker space: ${DOCKER_RECLAIMABLE}. Consider 'docker system prune -f'."
    fi
else
    echo "  Docker not installed. Skipping."
fi

echo ""
echo "ALL CHECKS PASSED — safe to proceed."
exit ${SAFE}
