#!/usr/bin/env bash
# Phase 1c L1: GGML_RPC_HASH_DEFER OFF vs ON on A1 @ n=384 (5-GPU prod).
#
# usage:
#   bash scripts/b6-gate-phase1c-l1-hash-defer-spike.sh
#
# env:
#   B6_L1_GEN_TOKENS   default 384
#   B6_L1_MODEL        default A1 Qwen3.6-35B path

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-phase1c-l1-hashdefer-${STAMP}"
MODEL="${B6_L1_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
GEN="${B6_L1_GEN_TOKENS:-384}"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

run_arm() {
    local arm="$1"
    local hash_env="$2"
    local out="${OUT_BASE}/${arm}-n${GEN}"
    echo "=== L1 ${arm} hash_defer=${hash_env} n=${GEN} ==="
    set +e
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      BENCH_MODEL='${MODEL}' BENCH_GEN_TOKENS=${GEN} \\
      BENCH_CTK=q8_0 BENCH_CTV=turbo3 \\
      GGML_PIPELINE_PLUS=1 B6_5GPU_WAVEFRONT=0 \\
      ${hash_env} \\
      GGML_RPC_DUAL_SOCKET=1 B6_PERF_AUTO=0 \\
      PATHB_VRAM_PREFLIGHT=0 \\
      PROFILER_OUT_DIR=${out} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      GGML_SCHED_TRACE=1 GGML_RPC_TRACE=1 \\
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
    ssh_romulus "bash ${ROMULUS_REPO}/rpc-patch/scripts/pathb-hotpath-summary.sh '${out}/telemetry' 2>/dev/null | tail -20"
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-phase0-assembly-bounds.py \\
      '${out}/telemetry' --json '${out}/telemetry/phase0-bounds.json' 2>/dev/null | tail -4"
    ssh_romulus "python3 -c \"
import json, re
from pathlib import Path
out = Path('${out}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
bounds = json.loads((out/'telemetry/phase0-bounds.json').read_text())
rep = bounds['reports'][0]
gl = rep.get('global_timeline_pct', {})
hash_ms = None
blocking_ms = diag.get('blocking_ms')
sumf = out/'telemetry/trace-summary.txt'
if sumf.is_file():
    for line in sumf.read_text().splitlines():
        if 'SET_TENSOR_HASH' in line and 'total_ms' in line:
            m = re.search(r'total_ms=([0-9.]+)', line)
            if m:
                hash_ms = float(m.group(1))
        if 'blocking_events=' in line:
            m = re.search(r'blocking_ms=([0-9.]+)', line)
            if m:
                blocking_ms = float(m.group(1))
row = {
  'arm': '${arm}',
  'hash_defer': '${hash_env}',
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'blocking_ms': diag.get('blocking_ms'),
  'drain_flush_ms': diag.get('drain_flush_ms'),
  'global_multi_pct': gl.get('multi_pct'),
  'global_3bk_pct': gl.get('multi_3_pct'),
  'hash_total_ms': hash_ms,
}
print('row', row)
with open('${OUT_BASE}/summary.jsonl', 'a') as f:
    import json as j; f.write(j.dumps(row)+chr(10))
\""
}

echo "OUT_BASE=${OUT_BASE} MODEL=${MODEL} GEN=${GEN}"
ssh_romulus "mkdir -p '${OUT_BASE}'"
run_arm "hash-off" "B6_5GPU_HASH_DEFER=0"
run_arm "hash-on" "B6_5GPU_HASH_DEFER=1"
echo "PHASE1C_L1_DONE dir=${OUT_BASE}"