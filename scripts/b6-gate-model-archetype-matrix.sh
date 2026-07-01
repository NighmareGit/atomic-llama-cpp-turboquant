#!/usr/bin/env bash
# Tier A model archetype baseline matrix (read-only traces + serial audit).
#
# usage:
#   bash scripts/b6-gate-model-archetype-matrix.sh              # Tier A all
#   B6_MATRIX_TIER=A bash scripts/b6-gate-model-archetype-matrix.sh
#   B6_MATRIX_ONLY=A1,A6 bash scripts/b6-gate-model-archetype-matrix.sh
#
# On load failure: retries with PATHB_VRAM_PREFLIGHT=1 (llama-fit plan). TS may differ per model.
#
# env:
#   B6_MATRIX_TOPOLOGY   b6-3gpu-g-triton (default) | b6-5gpu-g-prod | b6-6gpu-g
#   BENCH_CTK / BENCH_CTV  default q8_0 / turbo3

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
TOPO="${B6_MATRIX_TOPOLOGY:-b6-3gpu-g-triton}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-model-archetype-matrix-${STAMP}"
BASELINE_JSONL="${ROMULUS_REPO}/benches/path-b-plus/model-archetype-baseline.jsonl"
ONLY="${B6_MATRIX_ONLY:-}"
GEN="${BENCH_GEN_TOKENS:-384}"

declare -A MATRIX
MATRIX[A1]="qwen3-moe-compact|/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
MATRIX[A2]="qwen3-moe-next|/mnt/models/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf"
MATRIX[A3]="qwen-coder-next|/mnt/models/Qwen3-Coder-Next-APEX-I-Quality.gguf"
MATRIX[A4]="qwen3-dense|/mnt/models/Qwen3.5-27B-Q5_K_M.gguf"
MATRIX[A5]="qwen2-dense|/mnt/toshiba_a/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
MATRIX[A6]="gemma4-moe|/mnt/models/gemma-4-26B-A4B-APEX-I-Compact.gguf"
MATRIX[A7]="gemma-dense|/mnt/models/gemma-3-12b-it-Q5_K_M.gguf"
MATRIX[A8]="llama3-dense-skew|/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf"
MATRIX[A9]="llama3-dense-small|/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
MATRIX[A10]="deepseek-distill|/mnt/toshiba_b/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
MATRIX[A11]="moe-mixtral|/mnt/toshiba_b/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00001-of-00002.gguf"
MATRIX[A12]="glm-moe|/mnt/toshiba_b/models/GLM-4.5-Air-REAP-82B-A12B.i1-IQ4_XS.gguf"
MATRIX[A13]="kimi-moe|/mnt/models/Kimi-Dev-72B-IQ4_XS.gguf"
MATRIX[A14]="nex-lateral|/mnt/models/nex-agi_Nex-N2-mini-Q4_K_S.gguf"
MATRIX[A15]="cohere-moe|/mnt/toshiba_b/models/c4ai-command-r-v01-q4_k_m.gguf"
MATRIX[A16]="mistral|/mnt/toshiba_b/models/Mistral-Nemo-12B-Instruct.Q5_K_M.gguf"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

run_id() {
    local id="$1"
    local arch model path
    IFS='|' read -r arch path <<<"${MATRIX[$id]}"
    model="$(basename "$path")"
    local out="${OUT_BASE}/${id}-${TOPO}"
    echo "=== ${id} ${arch} ${model} ==="
    if ! ssh_romulus "test -f '${path}'"; then
        echo "SKIP ${id} missing ${path}"
        return 0
    fi
    set +e
    local dual_socket=0
    case "${TOPO}" in
        b6-3gpu-g|b6-3gpu-g-triton) dual_socket=1 ;;
    esac
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      BENCH_MODEL='${path}' BENCH_GEN_TOKENS=${GEN} \\
      BENCH_CTK=\${BENCH_CTK:-q8_0} BENCH_CTV=\${BENCH_CTV:-turbo3} \\
      GGML_RPC_DUAL_SOCKET=${dual_socket} B6_PERF_AUTO=0 \\
      PATHB_VRAM_PREFLIGHT=1 B6_GATE_PRESET=${TOPO} \\
      PROFILER_OUT_DIR=${out} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh ${TOPO} --no-warmup --skip-rpc-validate \\
        -ctk q8_0 -ctv turbo3" 2>&1 | tail -6
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${out}/result.jsonl'"; then
        echo "FAIL ${id} (load or run rc=$rc)"
        ssh_romulus "printf '%s\t%s\t%s\t%s\t%s\n' '${TOPO}' '${arch}' '${model}' 'load_fail' 'rc=${rc}' >> ${ROMULUS_REPO}/rpc-patch/docs/b6-gate/placement-broken.tsv"
        return 0
    fi
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-overlap-serial-audit.py \\
      '${out}/telemetry' --label '${id}' --min-decode 2 \\
      --json '${out}/telemetry/serial-overlap-audit.json' 2>/dev/null | tail -3"
    ssh_romulus "python3 -c \"
import json, datetime
from pathlib import Path
out = Path('${out}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
audit = json.loads((out/'telemetry/serial-overlap-audit.json').read_text()) if (out/'telemetry/serial-overlap-audit.json').is_file() else {}
env = {}
for line in (out/'env.txt').read_text().splitlines():
    if '=' in line:
        k,v=line.split('=',1); env[k]=v
row = {
  'version': 1,
  'ts_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
  'id': '${id}',
  'archetype': '${arch}',
  'model': '${model}',
  'topology': '${TOPO}',
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'overlap_efficiency': diag.get('overlap_efficiency'),
  'straggler_backend': diag.get('straggler_backend'),
  'straggler_ms_per_token': diag.get('straggler_ms_per_token'),
  'serial_dispatch_pct': audit.get('timeline_pct',{}).get('serial_dispatch'),
  'timeline_multi_pct': audit.get('timeline_pct',{}).get('multi'),
  'TS': env.get('TS'),
  'GIT_SHA': env.get('GIT_SHA'),
}
with open('${BASELINE_JSONL}', 'a') as f:
    f.write(json.dumps(row)+chr(10))
print('baseline', row['id'], 'G', row['G_tps'], 'ov', row['overlap_pct'])
\""
}

echo "OUT_BASE=${OUT_BASE} TOPO=${TOPO}"
ids=($(printf '%s\n' "${!MATRIX[@]}" | sort -t'A' -k2 -n))
if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra ids <<<"$ONLY"
fi
for id in "${ids[@]}"; do
    run_id "$id"
done
echo "MATRIX_DONE baseline=${BASELINE_JSONL}"