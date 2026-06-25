#!/usr/bin/env bash
# Run A1 vs A1+A2 vs Path B matrix on large models with ts=10,90.
#
# usage: rpc-server-bench-matrix.sh [model...]
#   models: 27b 31b 35b 36b  (default: all)
#
# env: same as rpc-server-bench.sh; BENCH_MATRIX_SKIP on failure continues

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
BENCH="${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh"
MODELS_ROOT="/mnt/models"
SUMMARY="${BENCH_MATRIX_SUMMARY:-${RPC_PATCH_ROOT}/patch/bench-results/rpc-server-bench/matrix-summary.txt}"
SKIP_ON_FAIL="${BENCH_MATRIX_SKIP:-1}"

mkdir -p "$(dirname "$SUMMARY")"
if [[ ! -f "$SUMMARY" ]]; then : >"$SUMMARY"; fi

declare -A PRESET_MODEL PRESET_CTX PRESET_CTK PRESET_CTV PRESET_NCMOE PRESET_LABEL

PRESET_MODEL[27b]="${MODELS_ROOT}/Qwen3.5-27B-Q5_K_M.gguf"
PRESET_CTX[27b]=8192
PRESET_CTK[27b]=q8_0
PRESET_CTV[27b]=turbo3
PRESET_NCMOE[27b]=""
PRESET_LABEL[27b]="27b-qwen"

PRESET_MODEL[31b]="${MODELS_ROOT}/gemma-4-31B-it-Q4_K_M.gguf"
PRESET_CTX[31b]=8192
PRESET_CTK[31b]=q8_0
PRESET_CTV[31b]=turbo3
PRESET_NCMOE[31b]=""
PRESET_LABEL[31b]="31b-gemma"

PRESET_MODEL[35b]="${MODELS_ROOT}/Qwen3.5-35B-A3B.i1-Q4_K_M.gguf"
PRESET_CTX[35b]=4096
PRESET_CTK[35b]=q4_0
PRESET_CTV[35b]=q4_0
PRESET_NCMOE[35b]=8
PRESET_LABEL[35b]="35b-a3b"

PRESET_MODEL[36b]="${MODELS_ROOT}/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
PRESET_CTX[36b]=4096
PRESET_CTK[36b]=q4_0
PRESET_CTV[36b]=q4_0
PRESET_NCMOE[36b]=8
PRESET_LABEL[36b]="36b-a3b"

if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=(27b 31b 35b 36b)
fi

VARIANTS=(a1 a1a2 pathb)

log_summary() { echo "$*" | tee -a "$SUMMARY"; }

run_one() {
    local preset="$1" variant="$2"
    local label="${PRESET_LABEL[$preset]}-${variant}-ts1090"
    local ncmoe="${PRESET_NCMOE[$preset]}"

    export BENCH_MODEL="${PRESET_MODEL[$preset]}"
    export BENCH_CTX="${PRESET_CTX[$preset]}"
    export BENCH_CTK="${PRESET_CTK[$preset]}"
    export BENCH_CTV="${PRESET_CTV[$preset]}"
    export BENCH_TS=10,90
    export BENCH_NGL=99
    export BENCH_LOAD_TIMEOUT=600
    export BENCH_GEN_TOKENS=48
    export BENCH_NO_WARMUP=1
    export BENCH_NP=1
    export BENCH_NCMOE="$ncmoe"

    log_summary "--- ${label} $(date -u +%H:%M:%SZ) ---"
    if "$BENCH" "$variant" "$label"; then
        local result="${RPC_PATCH_ROOT}/patch/bench-results/rpc-server-bench/${label}.result"
        if [[ -f "$result" ]]; then
            python3 - <<PY "$result" "$variant" "$preset" | tee -a "$SUMMARY"
import sys, re
path, variant, preset = sys.argv[1:4]
gs = []
for line in open(path):
    m = re.search(r'G=([\d.]+)', line)
    if m: gs.append(float(m.group(1)))
avg = sum(gs)/len(gs) if gs else 0
print(f"PASS {preset} {variant} avg_G={avg:.1f} runs={gs}")
PY
        fi
    else
        log_summary "FAIL ${preset} ${variant}"
        [[ "$SKIP_ON_FAIL" == "1" ]] || exit 1
    fi
    docker rm -f bench-rpc bench-llama 2>/dev/null || true
    sleep 2
}

log_summary "=== RPC path matrix ts=10,90 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for preset in "${TARGETS[@]}"; do
    if [[ -z "${PRESET_MODEL[$preset]:-}" ]]; then
        log_summary "SKIP unknown preset: $preset"
        continue
    fi
    if [[ ! -f "${PRESET_MODEL[$preset]}" ]]; then
        log_summary "SKIP missing model: ${PRESET_MODEL[$preset]}"
        continue
    fi
    for variant in "${VARIANTS[@]}"; do
        run_one "$preset" "$variant"
    done
done

log_summary "=== done ==="