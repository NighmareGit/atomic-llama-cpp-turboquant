#!/usr/bin/env bash
# n=384 / multiturn retest matrix for performance rules + overlap grill prep.
#
# usage (orchestrator -> romulus):
#   bash scripts/b6-gate-retest-matrix.sh [--dry-run]
#
# env:
#   B6_ROMULUS_HOST, B6_ROMULUS_PASS (default user@192.168.8.108 / 12345)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
MATRIX_DIR="${B6_RETEST_MATRIX_DIR:-${ROOT}/benches/path-b-plus/b6-retest-matrix-${STAMP}}"
ROMULUS_HOST="${B6_ROMULUS_HOST:-user@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
MULTITURN="${ROMULUS_REPO}/benches/path-b-plus/prompts/profiler-hard-multiturn.txt"
LONG="${ROMULUS_REPO}/benches/path-b-plus/prompts/profiler-reasoning-long.txt"
DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "error: unknown arg: $1" >&2; exit 1 ;;
    esac
done

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

run_case() {
    local id="$1" label="$2" n_gen="$3" dual="$4" prompt_kind="$5"
    local out="${MATRIX_DIR}/${id}"
    local prompt_file="$LONG"
    if [[ "$prompt_kind" == "multiturn" ]]; then
        prompt_file="$MULTITURN"
    fi
    local cmd
    cmd=$(cat <<EOF
cd ${ROMULUS_REPO} && \\
  GGML_RPC_DUAL_SOCKET=${dual} B6_PERF_AUTO=0 \\
  BENCH_GEN_TOKENS=${n_gen} \\
  BENCH_PROMPT_FILE=${prompt_file} \\
  PROFILER_OUT_DIR=${out} \\
  PROFILER_LOCAL=1 \\
  PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
  LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
  bash scripts/b6-gate-run-remote.sh ${label} --no-warmup --skip-rpc-validate
EOF
)
    echo "=== ${id}: label=${label} n=${n_gen} dual=${dual} prompt=${prompt_kind} ==="
    if [[ "$DRY" -eq 1 ]]; then
        echo "$cmd"
        return 0
    fi
    ssh_romulus "$cmd" 2>&1 | tail -6
    ssh_romulus "python3 -c \"
import json
d=json.load(open('${out}/telemetry/diagnose.json'))
print('${id}', 'G', round(d['G_tps'],2), 'ov', d['overlap_pct'], 'blk', round(d['blocking_ms']), 'dual', ${dual})
\"" || echo "WARN: ${id} diagnose missing" >&2
}

mkdir -p "$MATRIX_DIR"
echo "MATRIX_DIR=${MATRIX_DIR}"

# R1-R2: 3-GPU dual bisect @ n=384
run_case "R1-3gpu-n384-single" "b6-3gpu-g" 384 0 "long"
run_case "R2-3gpu-n384-dual" "b6-3gpu-g" 384 1 "long"

# R3-R5: 5-GPU gate depth
run_case "R3-5gpu-prod-n384" "b6-5gpu-g-prod" 384 1 "long"
run_case "R4-5gpu-prod-n2048-multiturn" "b6-5gpu-g-prod" 2048 1 "multiturn"
run_case "R5-5gpu-n2048-multiturn-single" "b6-5gpu-g" 2048 0 "multiturn"

echo "MATRIX_DONE dir=${MATRIX_DIR}"