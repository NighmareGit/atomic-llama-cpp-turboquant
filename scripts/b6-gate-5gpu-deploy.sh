#!/usr/bin/env bash
# 5-GPU prod deploy runbook: live VRAM preflight -> BENCH_* env -> optional smoke.
#
# usage:
#   bash scripts/b6-gate-5gpu-deploy.sh --gguf /mnt/models/foo.gguf
#   bash scripts/b6-gate-5gpu-deploy.sh --gguf foo.gguf --smoke --gen 64
#
# env:
#   B6_DEPLOY_PRESET     default b6-5gpu-g-prod
#   B6_DEPLOY_TS_MODE    default equal (L4 3060-safe spread)
#   B6_DEPLOY_PHASE      default load
#   B6_DEPLOY_SMOKE      1 to run profiler smoke after plan (or pass --smoke)
#   B6_5GPU_HASH_DEFER   passed through to prod env when set

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRESET="${B6_DEPLOY_PRESET:-b6-5gpu-g-prod}"
TS_MODE="${B6_DEPLOY_TS_MODE:-equal}"
PHASE="${B6_DEPLOY_PHASE:-load}"
GGUF=""
DO_SMOKE=0
GEN="${B6_DEPLOY_GEN:-64}"

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gguf) GGUF="${2:?}"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --ts-mode) TS_MODE="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --smoke) DO_SMOKE=1; shift ;;
        --gen) GEN="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "error: unknown arg: $1" >&2; usage ;;
    esac
done

[[ -n "$GGUF" ]] || { echo "error: --gguf required" >&2; exit 1; }

echo "=== 5-GPU deploy preflight: ${PRESET} ==="
echo "  gguf: ${GGUF}"
echo "  ts-mode: ${TS_MODE}  phase: ${PHASE}"
echo

OUT="$(python3 "${ROOT}/rpc-patch/scripts/pathb-rpc-vram-preflight.py" \
    --preset "$PRESET" --gguf "$GGUF" --ts-mode "$TS_MODE" --phase "$PHASE" 2>&1)" || {
    printf '%s\n' "$OUT"
    echo "FAIL: preflight did not find a fitting split" >&2
    exit 1
}
printf '%s\n' "$OUT"

TS="$(printf '%s\n' "$OUT" | awk -F= '/^  BENCH_TS=/{print $2; exit}')"
NGL="$(printf '%s\n' "$OUT" | awk -F= '/^  BENCH_NGL=/{print $2; exit}')"
FITT="$(printf '%s\n' "$OUT" | awk -F= '/^  BENCH_FITT=/{print $2; exit}')"
PASS_NGL="$(printf '%s\n' "$OUT" | sed -n 's/^PASS: ngl=\([0-9]*\).*/\1/p' | head -1)"
[[ -n "$PASS_NGL" ]] && NGL="$PASS_NGL"

if [[ -z "$TS" ]]; then
    echo "FAIL: could not parse BENCH_TS from preflight" >&2
    exit 1
fi

echo
echo "=== deploy env (copy/paste) ==="
cat <<EOF
export BENCH_RPC_ENDPOINT='192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054,192.168.8.23:50055'
export BENCH_MODEL='${GGUF}'
export BENCH_TS='${TS}'
export BENCH_NGL='${NGL}'
export BENCH_FIT_TARGET='${FITT}'
export BENCH_CTK=q8_0
export BENCH_CTV=turbo3
export GGML_PIPELINE_PLUS=1
export B6_5GPU_WAVEFRONT=0
export B6_5GPU_DUAL_SOCKET=1
export B6_5GPU_HASH_DEFER=\${B6_5GPU_HASH_DEFER:-0}
EOF

if [[ "$DO_SMOKE" -eq 1 || "${B6_DEPLOY_SMOKE:-0}" == "1" ]]; then
    echo
    echo "=== smoke run (n=${GEN}) ==="
    # shellcheck source=scripts/b6-gate-5gpu-production-env.sh
    source "${ROOT}/scripts/b6-gate-5gpu-production-env.sh"
    b6_5gpu_production_env
    export BENCH_MODEL="$GGUF"
    export BENCH_TS="$TS"
    export BENCH_NGL="$NGL"
    export BENCH_FIT_TARGET="$FITT"
    export BENCH_GEN_TOKENS="$GEN"
    export BENCH_CTK=q8_0
    export BENCH_CTV=turbo3
    export PATHB_VRAM_PREFLIGHT=0
    export PROFILER_SKIP_VALIDATE=1
    export PROFILER_LOCAL="${PROFILER_LOCAL:-1}"
    STAMP="$(date -u +%Y%m%d-%H%M%S)"
    export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/b6-5gpu-deploy-smoke-${STAMP}}"
    bash "${ROOT}/scripts/b6-gate-run-remote.sh" b6-5gpu-g-prod \
        --no-warmup --skip-rpc-validate -ctk q8_0 -ctv turbo3
    echo "SMOKE_DONE out=${PROFILER_OUT_DIR}"
fi

echo "DEPLOY_PLAN_OK ts=${TS} ngl=${NGL}"