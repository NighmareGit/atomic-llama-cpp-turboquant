#!/usr/bin/env bash
# L4 layer spread spike: equal TS vs VRAM-ratio TS on big models (5-GPU prod).
#
# usage:
#   bash scripts/b6-gate-b15-l4-layer-spread-spike.sh
#
# env:
#   B6_L4_MODELS   default A8,A13
#   BENCH_GEN_TOKENS  default 64

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-user@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-b15-l4-layer-spread-${STAMP}"
GEN="${BENCH_GEN_TOKENS:-64}"
MODELS="${B6_L4_MODELS:-A8,A13}"

declare -A MODEL_PATH
MODEL_PATH[A8]="/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf"
MODEL_PATH[A13]="/mnt/models/Kimi-Dev-72B-IQ4_XS.gguf"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

plan_deploy() {
    local mode="$1"
    local path="$2"
    local out
    out="$(ssh_romulus "cd ${ROMULUS_REPO} && python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \\
      --preset b6-5gpu-g --gguf '${path}' --ts-mode ${mode} 2>/dev/null || true")"
    local ts ngl fitt pass_ngl
    ts="$(printf '%s\n' "$out" | awk -F= '/^  BENCH_TS=/{print $2; exit}')"
    ngl="$(printf '%s\n' "$out" | awk -F= '/^  BENCH_NGL=/{print $2; exit}')"
    fitt="$(printf '%s\n' "$out" | awk -F= '/^  BENCH_FITT=/{print $2; exit}')"
    pass_ngl="$(printf '%s\n' "$out" | sed -n 's/^PASS: ngl=\([0-9]*\).*/\1/p' | head -1)"
    if [[ -n "$pass_ngl" ]]; then
        ngl="$pass_ngl"
    fi
    if [[ -z "$ts" ]]; then
        return 1
    fi
    printf '%s %s %s\n' "$ts" "${ngl:-}" "${fitt:-}"
}

run_case() {
    local model_id="$1"
    local ts_mode="$2"
    local path="${MODEL_PATH[$model_id]}"
    local ts ngl fitt plan
    plan="$(plan_deploy "$ts_mode" "$path")" || true
    ts="$(printf '%s\n' "$plan" | awk 'NR==1{print $1}')"
    ngl="$(printf '%s\n' "$plan" | awk 'NR==1{print $2}')"
    fitt="$(printf '%s\n' "$plan" | awk 'NR==1{print $3}')"
    if [[ -z "$ts" ]]; then
        echo "SKIP ${model_id}-${ts_mode}: no TS from preflight"
        return 0
    fi
    if [[ "$ts_mode" == "equal" ]]; then
        local fit_ok
        fit_ok="$(ssh_romulus "cd ${ROMULUS_REPO} && python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \\
          --preset b6-5gpu-g --gguf '${path}' --ts-mode equal 2>/dev/null | grep -c '^PASS:' || true")"
        if [[ "${fit_ok:-0}" -eq 0 ]]; then
            echo "SKIP ${model_id}-equal: equal TS does not fit (OOM); use vram mode or lighter quant"
            return 0
        fi
    fi
    local ngl_env="" fitt_env=""
    if [[ -n "$ngl" ]]; then
        ngl_env="BENCH_NGL=${ngl}"
    fi
    if [[ -n "$fitt" ]]; then
        fitt_env="BENCH_FIT_TARGET='${fitt}'"
    fi
    local out="${OUT_BASE}/${model_id}-${ts_mode}-n${GEN}"
    echo "=== L4 ${model_id} ts_mode=${ts_mode} TS=${ts} ngl=${ngl:-?} fitt=${fitt:-?} ==="
    set +e
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      BENCH_MODEL='${path}' BENCH_GEN_TOKENS=${GEN} BENCH_TS='${ts}' ${ngl_env} ${fitt_env} \\
      BENCH_CTK=q8_0 BENCH_CTV=turbo3 \\
      GGML_PIPELINE_PLUS=1 B6_5GPU_WAVEFRONT=0 \\
      GGML_RPC_HASH_DEFER=\${B6_L4_HASH_DEFER:-0} \\
      GGML_RPC_DUAL_SOCKET=1 B6_PERF_AUTO=0 \\
      PATHB_VRAM_PREFLIGHT=1 B6_GATE_PRESET=b6-5gpu-g \\
      PROFILER_OUT_DIR=${out} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      GGML_SCHED_TRACE=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh b6-5gpu-g-prod --no-warmup --skip-rpc-validate \\
        -ctk q8_0 -ctv turbo3" 2>&1 | tail -10
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${out}/result.jsonl'"; then
        echo "FAIL ${model_id}-${ts_mode} rc=$rc"
        return 1
    fi
    ssh_romulus "python3 ${ROMULUS_REPO}/scripts/b6-gate-phase0-assembly-bounds.py \\
      '${out}/telemetry' --json '${out}/telemetry/phase0-bounds.json' 2>/dev/null | tail -4"
    ssh_romulus "python3 -c \"
import json
from pathlib import Path
out = Path('${out}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
bounds = json.loads((out/'telemetry/phase0-bounds.json').read_text())
rep = bounds['reports'][0]
sg = rep.get('split_granularity', {})
gl = rep.get('global_timeline_pct', {})
row = {
  'model_id': '${model_id}',
  'ts_mode': '${ts_mode}',
  'TS': '${ts}',
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'splits_p50': sg.get('splits_per_decode_p50'),
  'global_3bk_pct': gl.get('multi_3_pct'),
  'global_multi_pct': gl.get('multi_pct'),
}
print('row', row)
with open('${OUT_BASE}/summary.jsonl', 'a') as f:
    import json as j; f.write(j.dumps(row)+chr(10))
\""
}

echo "OUT_BASE=${OUT_BASE} GEN=${GEN}"
ssh_romulus "mkdir -p '${OUT_BASE}'"
IFS=',' read -ra MODEL_ARR <<<"$MODELS"
for mid in "${MODEL_ARR[@]}"; do
    run_case "$mid" "vram" || true
    run_case "$mid" "equal" || true
done
echo "L4_LAYER_SPREAD_SPIKE_DONE dir=${OUT_BASE}"