#!/usr/bin/env bash
# Pull key romulus bench artifacts into repo before cluster shutdown.
set -euo pipefail

DEST="${1:-rpc-patch/patch/bench-results/cluster-4gpu-primary/romulus-host}"
H="${PATHB_ROMULUS_SSH:-hunter@192.168.8.108}"
PASS="${PATHB_ROMULUS_SSH_PASS:-12345}"
R="atomic-llama-cpp-turboquant/rpc-patch/patch/bench-results/rpc-server-bench"

mkdir -p "$DEST"
SCP=(scp -o StrictHostKeyChecking=no)
if command -v sshpass >/dev/null && [[ -n "$PASS" ]]; then
    SCP=(sshpass -p "$PASS" scp -o StrictHostKeyChecking=no)
fi

for f in \
    trace-g-4gpu-primary.meta trace-g-4gpu-primary.result \
    trace-g-4gpu-primary-r2.meta trace-g-4gpu-primary-r2.result \
    trace-g-4gpu-primary-r3.meta trace-g-4gpu-primary-r3.result \
    trace-g-4gpu-primary-trace.meta trace-g-4gpu-primary-trace.result \
    trace-g-4gpu-romulus-q8-pp1.meta \
    trace-g-3gpu-primary-r1.meta trace-g-3gpu-primary-r1.result \
    trace-g-3gpu-primary-r3.meta trace-g-3gpu-primary-r3.result; do
    "${SCP[@]}" "${H}:${R}/${f}" "${DEST}/" 2>/dev/null || true
done

"${SCP[@]}" "${H}:${R}/trace-g-4gpu-primary-trace/telemetry/trace-summary.txt" \
    "${DEST}/trace-g-4gpu-primary-trace-summary.txt" 2>/dev/null || true

echo "pulled to ${DEST}"
ls -la "${DEST}"