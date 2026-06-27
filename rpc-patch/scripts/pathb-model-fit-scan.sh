#!/usr/bin/env bash
# Rank /mnt/models GGUFs by fit against combined GPU VRAM + host RAM.
#
# usage: pathb-model-fit-scan.sh [--config config-a|remus|config-c|config-e|config-f] [--top 15]

set -euo pipefail

CONFIG="${1:-config-c}"
TOP_N=15
MODELS_ROOT="/mnt/models"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --top) TOP_N="$2"; shift 2 ;;
        *) shift ;;
    esac
done

case "$CONFIG" in
    config-a) GPU_GB=32; NAME="Config A (3060+7900)" ;;
    remus)    GPU_GB=40; NAME="Config B (remus 5060+7900)" ;;
    config-c) GPU_GB=48; NAME="Config C (remus 5060+3060+7900)" ;;
    config-e) GPU_GB=31; NAME="Config E (remus 5060+Windows 5070 Ti)" ;;
    config-f) GPU_GB=39; NAME="Config F (remus 5060+6600+Windows 5070 Ti)" ;;
    *) echo "unknown config $CONFIG" >&2; exit 1 ;;
esac

mem_avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
ram_gb=$(python3 -c "print(round($mem_avail_kb/1024/1024, 1))")
budget_gb=$(python3 -c "print(round($GPU_GB + 0.7*$ram_gb, 1))")

echo "=== pathb-model-fit-scan config=$CONFIG ==="
echo "$NAME"
echo "GPU budget:     ${GPU_GB} GB (usable)"
echo "RAM available:  ${ram_gb} GB"
echo "Total budget:   ${budget_gb} GB (GPU + 70% RAM for CPU weights)"
echo ""
printf "%-8s %-10s %s\n" "GB" "FIT" "MODEL"
ls -lhS "$MODELS_ROOT"/*.gguf 2>/dev/null | awk '{print $5, $9}' | while read -r size path; do
    gb=$(python3 -c "
s='$size'
if s.endswith('G'): print(float(s[:-1]))
elif s.endswith('M'): print(float(s[:-1])/1024)
else: print(0)
")
    fit="maybe"
    python3 -c "import sys; sys.exit(0 if float('$gb') <= $budget_gb else 1)" && fit="YES" || fit="no"
    base=$(basename "$path")
    printf "%-8.1f %-10s %s\n" "$gb" "$fit" "$base"
done | head -n "$((TOP_N+1))"