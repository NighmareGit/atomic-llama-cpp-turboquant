#!/usr/bin/env bash
# B+14 n=384 gate depth: A1 baseline (A0) vs full wavefront (A3) on 5-GPU prod.
#
# usage:
#   bash scripts/b6-gate-b14-n384-compare.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-user@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-b14-n384-compare-${STAMP}"
MODEL="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
GEN=384

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

run_arm() {
    local arm="$1"
    local wf_env="$2"
    local out="${OUT_BASE}/A1-${arm}-n${GEN}"
    echo "=== A1 ${arm} n=${GEN} ==="
    set +e
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      BENCH_MODEL='${MODEL}' BENCH_GEN_TOKENS=${GEN} \\
      BENCH_CTK=q8_0 BENCH_CTV=turbo3 \\
      GGML_PIPELINE_PLUS=1 B6_5GPU_WAVEFRONT=0 \\
      ${wf_env} \\
      GGML_RPC_DUAL_SOCKET=1 B6_PERF_AUTO=0 \\
      PATHB_VRAM_PREFLIGHT=1 B6_GATE_PRESET=b6-5gpu-g \\
      PROFILER_OUT_DIR=${out} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      GGML_SCHED_TRACE=1 GGML_PIPELINE_TRACE=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh b6-5gpu-g-prod --no-warmup --skip-rpc-validate \\
        -ctk q8_0 -ctv turbo3" 2>&1 | tail -12
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${out}/result.jsonl'"; then
        echo "FAIL ${arm} rc=$rc"
        return 1
    fi
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-overlap-serial-audit.py \\
      '${out}/telemetry' --label 'A1-${arm}' --min-decode 2 \\
      --json '${out}/telemetry/serial-overlap-audit.json' 2>/dev/null | tail -4"
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-phase0-assembly-bounds.py \\
      '${out}/telemetry' --json '${out}/telemetry/phase0-bounds.json' 2>/dev/null | tail -5"
    ssh_romulus "python3 -c \"
import json
from pathlib import Path
out = Path('${out}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
audit = json.loads((out/'telemetry/serial-overlap-audit.json').read_text())
bounds = json.loads((out/'telemetry/phase0-bounds.json').read_text())
rep = bounds['reports'][0]
gl = rep.get('global_timeline_pct', {})
row = {
  'arm': '${arm}',
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'serial_dispatch_pct': audit.get('timeline_pct',{}).get('serial_dispatch'),
  'timeline_multi_pct': audit.get('timeline_pct',{}).get('multi'),
  'global_multi_pct': gl.get('multi_pct'),
  'global_3bk_pct': gl.get('multi_3_pct'),
  'splits_p50': rep.get('split_granularity',{}).get('splits_per_decode_p50'),
}
print('row', row)
with open('${OUT_BASE}/summary.jsonl', 'a') as f:
    import json as j; f.write(j.dumps(row)+chr(10))
\""
}

echo "OUT_BASE=${OUT_BASE}"
ssh_romulus "mkdir -p '${OUT_BASE}'"
run_arm "A0" "GGML_SCHED_WAVEFRONT_DISPATCH=0"
run_arm "A3" "GGML_SCHED_WAVEFRONT_DISPATCH=1 GGML_SCHED_PIPELINE_DEPTH=4"
echo "B14_N384_COMPARE_DONE dir=${OUT_BASE}"