#!/usr/bin/env bash
# B+13 audit spike: C-full hotpath on 2-GPU triton reference (sync_copy_fallback vs copy_async_ok).
#
# usage:
#   bash scripts/b6-gate-b13-audit-spike.sh
#
# env:
#   B6_B13_PRESET    default b6-2gpu-f-triton
#   BENCH_MODEL      default Qwen3.6-35B-A3B
#   BENCH_GEN_TOKENS default 384

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
PRESET="${B6_B13_PRESET:-b6-2gpu-f-triton}"
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
GEN="${BENCH_GEN_TOKENS:-384}"
OUT="${ROMULUS_REPO}/benches/path-b-plus/b6-b13-audit-spike-${STAMP}"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

echo "OUT=${OUT} PRESET=${PRESET} MODEL=$(basename "$MODEL")"
set +e
ssh_romulus "cd ${ROMULUS_REPO} && \\
  BENCH_MODEL='${MODEL}' BENCH_GEN_TOKENS=${GEN} \\
  GGML_PIPELINE_PLUS=1 BENCH_TRACE=1 \\
  GGML_SCHED_TRACE=1 GGML_PIPELINE_TRACE=1 \\
  PATHB_VRAM_PREFLIGHT=1 B6_GATE_PRESET=${PRESET} \\
  PROFILER_OUT_DIR=${OUT} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
  PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
  LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
  bash scripts/b6-gate-run-remote.sh ${PRESET} --no-warmup --skip-rpc-validate \\
    -ctk q8_0 -ctv turbo3" 2>&1 | tail -10
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${OUT}/result.jsonl'"; then
    echo "FAIL b13-audit rc=$rc"
    exit 1
fi

ssh_romulus "bash ${ROMULUS_REPO}/rpc-patch/scripts/pathb-hotpath-summary.sh '${OUT}/telemetry' 2>&1 | tail -40"
ssh_romulus "bash ${ROMULUS_REPO}/rpc-patch/scripts/pathb-rpc-trace-parse.sh '${OUT}/telemetry' 2>&1 | tail -25"
ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-overlap-serial-audit.py \\
  '${OUT}/telemetry' --label b13 --min-decode 2 \\
  --json '${OUT}/telemetry/serial-overlap-audit.json' 2>/dev/null | tail -4"
ssh_romulus "python3 -c \"
import json
from pathlib import Path
out = Path('${OUT}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
row = {
  'preset': '${PRESET}',
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'blocking_ms': diag.get('blocking_ms'),
  'drain_flush_ms': diag.get('drain_flush_ms'),
  'input_wait_copy_ms': diag.get('input_wait_copy_ms'),
}
print('b13_audit', row)
with open('${OUT}/b13-audit-summary.json', 'w') as f:
    json.dump(row, f, indent=2)
\""
echo "B13_AUDIT_SPIKE_DONE dir=${OUT}"