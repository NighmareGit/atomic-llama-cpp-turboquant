#!/usr/bin/env bash
# B+14 wavefront spike: W1+W2 factorial on 5-GPU prod (skip jupiter).
#
# usage:
#   bash scripts/b6-gate-b14-wavefront-spike.sh              # n=64 smoke
#   BENCH_GEN_TOKENS=384 bash scripts/b6-gate-b14-wavefront-spike.sh
#
# env:
#   B6_B14_PRESET        default b6-5gpu-g-prod
#   B6_B14_MODELS        default A1,A8,A13
#   B6_B14_ARMS          default A0,A1,A2,A3 (factorial)
#   BENCH_GEN_TOKENS     default 64 (smoke); 384 for gate depth

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
PRESET="${B6_B14_PRESET:-b6-5gpu-g-prod}"
GEN="${BENCH_GEN_TOKENS:-64}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-b14-wavefront-spike-${STAMP}"
MODELS="${B6_B14_MODELS:-A1,A8,A13}"
ARMS="${B6_B14_ARMS:-A0,A1,A2,A3}"

declare -A MODEL_PATH
MODEL_PATH[A1]="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
MODEL_PATH[A8]="/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf"
MODEL_PATH[A13]="/mnt/models/Kimi-Dev-72B-IQ4_XS.gguf"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

wavefront_env() {
    local arm="$1"
    case "$arm" in
        A0)
            echo "GGML_SCHED_WAVEFRONT_DISPATCH=0"
            ;;
        A1)
            echo "GGML_SCHED_WAVEFRONT_DISPATCH=0 GGML_SCHED_WAVEFRONT_INTRA=1 GGML_SCHED_WAVEFRONT_CROSS=0"
            ;;
        A2)
            echo "GGML_SCHED_WAVEFRONT_DISPATCH=0 GGML_SCHED_WAVEFRONT_INTRA=0 GGML_SCHED_WAVEFRONT_CROSS=1 GGML_SCHED_PIPELINE_DEPTH=4"
            ;;
        A3)
            echo "GGML_SCHED_WAVEFRONT_DISPATCH=1 GGML_SCHED_PIPELINE_DEPTH=${B6_5GPU_PIPELINE_DEPTH:-4}"
            ;;
        *)
            echo "error: unknown arm ${arm}" >&2
            return 1
            ;;
    esac
}

run_case() {
    local model_id="$1"
    local arm="$2"
    local path="${MODEL_PATH[$model_id]}"
    local wf_env
    wf_env="$(wavefront_env "$arm")"
    local out="${OUT_BASE}/${model_id}-${arm}-n${GEN}"
    echo "=== B+14 ${model_id} arm=${arm} n=${GEN} ==="
    if ! ssh_romulus "test -f '${path}'"; then
        echo "SKIP ${model_id} missing ${path}"
        return 0
    fi
    set +e
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      BENCH_MODEL='${path}' BENCH_GEN_TOKENS=${GEN} \\
      BENCH_CTK=\${BENCH_CTK:-q8_0} BENCH_CTV=\${BENCH_CTV:-turbo3} \\
      GGML_PIPELINE_PLUS=1 B6_5GPU_WAVEFRONT=0 \\
      ${wf_env} \\
      GGML_RPC_DUAL_SOCKET=1 B6_PERF_AUTO=0 \\
      PATHB_VRAM_PREFLIGHT=1 B6_GATE_PRESET=b6-5gpu-g \\
      PROFILER_OUT_DIR=${out} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      GGML_SCHED_TRACE=1 GGML_PIPELINE_TRACE=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh ${PRESET} --no-warmup --skip-rpc-validate \\
        -ctk q8_0 -ctv turbo3" 2>&1 | tail -10
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${out}/result.jsonl'"; then
        echo "FAIL ${model_id}-${arm} rc=$rc"
        return 1
    fi
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-overlap-serial-audit.py \\
      '${out}/telemetry' --label '${model_id}-${arm}' --min-decode 2 \\
      --json '${out}/telemetry/serial-overlap-audit.json' 2>/dev/null | tail -4"
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-phase0-assembly-bounds.py \\
      '${out}/telemetry' --json '${out}/telemetry/phase0-bounds.json' 2>/dev/null | tail -6"
    ssh_romulus "python3 -c \"
import json
from pathlib import Path
out = Path('${out}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
audit = json.loads((out/'telemetry/serial-overlap-audit.json').read_text()) if (out/'telemetry/serial-overlap-audit.json').is_file() else {}
bounds_raw = json.loads((out/'telemetry/phase0-bounds.json').read_text()) if (out/'telemetry/phase0-bounds.json').is_file() else {}
rep = (bounds_raw.get('reports') or [{}])[0]
gl = rep.get('global_timeline_pct', {})
row = {
  'model_id': '${model_id}',
  'arm': '${arm}',
  'n_gen': ${GEN},
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'serial_dispatch_pct': audit.get('timeline_pct',{}).get('serial_dispatch'),
  'timeline_multi_pct': audit.get('timeline_pct',{}).get('multi'),
  'global_multi_pct': gl.get('multi_pct'),
  'global_3bk_pct': gl.get('multi_3_pct'),
}
print('row', row['model_id'], row['arm'], 'G', row['G_tps'], 'g3bk', row['global_3bk_pct'], 'g_multi', row['global_multi_pct'])
with open('${OUT_BASE}/summary.jsonl', 'a') as f:
    import json as j; f.write(j.dumps(row)+chr(10))
\""
}

echo "OUT_BASE=${OUT_BASE} PRESET=${PRESET} GEN=${GEN}"
ssh_romulus "mkdir -p '${OUT_BASE}'"
IFS=',' read -ra MODEL_ARR <<<"$MODELS"
IFS=',' read -ra ARM_ARR <<<"$ARMS"
for mid in "${MODEL_ARR[@]}"; do
    for arm in "${ARM_ARR[@]}"; do
        run_case "$mid" "$arm" || true
    done
done

echo "=== B+14 gate check (global_3bk >= 5%) ==="
ssh_romulus "python3 - <<'PY' '${OUT_BASE}/summary.jsonl'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print('no summary yet')
    sys.exit(0)
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
pass_rows = [r for r in rows if (r.get('global_3bk_pct') or 0) >= 5.0]
print(f'total={len(rows)} pass_g3bk={len(pass_rows)}')
for r in sorted(rows, key=lambda x: (-(x.get('global_3bk_pct') or 0), x.get('model_id',''), x.get('arm',''))):
    print(f\"  {r.get('model_id')} {r.get('arm')} G={r.get('G_tps')} g3bk={r.get('global_3bk_pct')}% g_multi={r.get('global_multi_pct')}%\")
PY"

echo "B14_WAVEFRONT_SPIKE_DONE dir=${OUT_BASE}"