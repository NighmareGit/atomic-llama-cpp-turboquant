#!/usr/bin/env bash
# Phase 1c: L4 equal-safe @ n=384 on big models (5-GPU prod load confirm).
#
# usage:
#   bash scripts/b6-gate-phase1c-l4-n384-confirm.sh
#
# env:
#   B6_L4_N384_MODELS   default A8,A13
#   B6_L4_N384_GEN      default 384

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-phase1c-l4-n384-${STAMP}"
GEN="${B6_L4_N384_GEN:-384}"
MODELS="${B6_L4_N384_MODELS:-A8,A13}"

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

plan_ts() {
    local path="$1"
    local out
    out="$(ssh_romulus "cd ${ROMULUS_REPO} && python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \\
      --preset b6-5gpu-g-prod --gguf '${path}' --ts-mode equal --phase load 2>/dev/null || true")"
    local ts ngl fitt pass_ngl
    ts="$(printf '%s\n' "$out" | awk -F= '/^  BENCH_TS=/{print $2; exit}')"
    ngl="$(printf '%s\n' "$out" | awk -F= '/^  BENCH_NGL=/{print $2; exit}')"
    fitt="$(printf '%s\n' "$out" | awk -F= '/^  BENCH_FITT=/{print $2; exit}')"
    pass_ngl="$(printf '%s\n' "$out" | sed -n 's/^PASS: ngl=\([0-9]*\).*/\1/p' | head -1)"
    [[ -n "$pass_ngl" ]] && ngl="$pass_ngl"
    [[ -n "$ts" ]] || return 1
    printf '%s %s %s\n' "$ts" "${ngl:-}" "${fitt:-}"
}

run_model() {
    local mid="$1"
    local path="${MODEL_PATH[$mid]}"
    local plan ts ngl fitt
    plan="$(plan_ts "$path")" || { echo "SKIP ${mid}: preflight failed"; return 0; }
    ts="$(printf '%s\n' "$plan" | awk 'NR==1{print $1}')"
    ngl="$(printf '%s\n' "$plan" | awk 'NR==1{print $2}')"
    fitt="$(printf '%s\n' "$plan" | awk 'NR==1{print $3}')"
    local out="${OUT_BASE}/${mid}-equal-n${GEN}"
    echo "=== L4 n384 ${mid} TS=${ts} ngl=${ngl} ==="
    set +e
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      BENCH_MODEL='${path}' BENCH_GEN_TOKENS=${GEN} BENCH_TS='${ts}' BENCH_NGL=${ngl} \\
      BENCH_FIT_TARGET='${fitt}' BENCH_CTK=q8_0 BENCH_CTV=turbo3 \\
      GGML_PIPELINE_PLUS=1 B6_5GPU_WAVEFRONT=0 B6_5GPU_HASH_DEFER=0 \\
      GGML_RPC_DUAL_SOCKET=1 B6_PERF_AUTO=0 \\
      PATHB_VRAM_PREFLIGHT=0 PROFILER_OUT_DIR=${out} PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      GGML_SCHED_TRACE=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh b6-5gpu-g-prod --no-warmup --skip-rpc-validate \\
        -ctk q8_0 -ctv turbo3" 2>&1 | tail -10
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${out}/result.jsonl'"; then
        echo "FAIL ${mid} rc=$rc"
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
gl = rep.get('global_timeline_pct', {})
row = {
  'model_id': '${mid}',
  'TS': '${ts}',
  'G_tps': diag.get('G_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'global_multi_pct': gl.get('multi_pct'),
  'global_3bk_pct': gl.get('multi_3_pct'),
  'n_gen': ${GEN},
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
    run_model "$mid" || true
done
echo "PHASE1C_L4_N384_DONE dir=${OUT_BASE}"