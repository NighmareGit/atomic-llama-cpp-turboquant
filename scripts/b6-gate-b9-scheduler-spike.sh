#!/usr/bin/env bash
# B+9 scheduler spike: EVENT defer-to-barrier ON (canonical) vs OFF on Tier A reference.
#
# usage:
#   bash scripts/b6-gate-b9-scheduler-spike.sh
#
# env:
#   B6_B9_PRESET     default b6-3gpu-g-triton
#   BENCH_MODEL      default Qwen3.6-35B-A3B (A1 reference)
#   BENCH_GEN_TOKENS default 384

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
PRESET="${B6_B9_PRESET:-b6-3gpu-g-triton}"
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
GEN="${BENCH_GEN_TOKENS:-384}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-b9-scheduler-spike-${STAMP}"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

run_arm() {
    local tag="$1"
    local defer="$2"
    local out="${OUT_BASE}/${tag}"
    echo "=== B+9 arm ${tag} EVENT_DEFER_BARRIER=${defer} ==="
    set +e
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      BENCH_MODEL='${MODEL}' BENCH_GEN_TOKENS=${GEN} \\
      GGML_PIPELINE_PLUS=1 \\
      GGML_RPC_EVENT_DEFER_BARRIER=${defer} \\
      GGML_RPC_DUAL_SOCKET=1 B6_PERF_AUTO=0 \\
      PATHB_VRAM_PREFLIGHT=1 B6_GATE_PRESET=${PRESET} \\
      PROFILER_OUT_DIR=${out} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh ${PRESET} --no-warmup --skip-rpc-validate \\
        -ctk q8_0 -ctv turbo3" 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${out}/result.jsonl'"; then
        echo "FAIL ${tag} rc=$rc"
        return 1
    fi
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-overlap-serial-audit.py \\
      '${out}/telemetry' --label '${tag}' --min-decode 2 \\
      --json '${out}/telemetry/serial-overlap-audit.json' 2>/dev/null | tail -4"
    ssh_romulus "python3 -c \"
import json
from pathlib import Path
out = Path('${out}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
audit = json.loads((out/'telemetry/serial-overlap-audit.json').read_text()) if (out/'telemetry/serial-overlap-audit.json').is_file() else {}
env = {}
for line in (out/'env.txt').read_text().splitlines():
    if '=' in line:
        k,v=line.split('=',1); env[k]=v
row = {
  'arm': '${tag}',
  'event_defer_barrier': ${defer},
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'stall_ratio': diag.get('stall_ratio'),
  'blocking_ms': diag.get('blocking_ms'),
  'drain_flush_ms': diag.get('drain_flush_ms'),
  'serial_dispatch_pct': audit.get('timeline_pct',{}).get('serial_dispatch'),
  'timeline_multi_pct': audit.get('timeline_pct',{}).get('multi'),
  'TS': env.get('TS'),
}
print('arm', row['arm'], 'G', row['G_tps'], 'ov', row['overlap_pct'], 'stall', row['stall_ratio'], 'serial', row['serial_dispatch_pct'])
with open('${OUT_BASE}/summary.jsonl', 'a') as f:
    import json as j; f.write(j.dumps(row)+chr(10))
\""
}

echo "OUT_BASE=${OUT_BASE} PRESET=${PRESET} MODEL=$(basename "$MODEL")"
run_arm "canonical-defer-on" 1
run_arm "no-defer" 0
echo "B9_SCHEDULER_SPIKE_DONE dir=${OUT_BASE}"