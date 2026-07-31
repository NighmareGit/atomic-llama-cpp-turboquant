#!/usr/bin/env bash
# Run A1 vs A1+A2 vs Path B matrix on large models.
#
# usage: rpc-server-bench-matrix.sh [model...]
#   models: 27b 31b 35b 36b  (default: all)
#
# env:
#   BENCH_CONFIG=local|remus|config-c
#   BENCH_MATRIX_SKIP on failure continues

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
BENCH="${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh"
DISK="${RPC_PATCH_ROOT}/scripts/pathb-disk-preflight.sh"
MODELS_ROOT="/mnt/models"
CONFIG="${BENCH_CONFIG:-local}"
SKIP_ON_FAIL="${BENCH_MATRIX_SKIP:-1}"

case "$CONFIG" in
    local)
        SUMMARY="${BENCH_MATRIX_SUMMARY:-${RPC_PATCH_ROOT}/patch/bench-results/rpc-server-bench/matrix-summary.txt}"
        VARIANTS=(a1 a1a2 pathb)
        export BENCH_RPC_MODE="${BENCH_RPC_MODE:-local}"
        export BENCH_TS="${BENCH_TS:-10,90}"
        LABEL_PREFIX=""
        TS_TAG="ts1090"
        ;;
    remus)
        SUMMARY="${BENCH_MATRIX_SUMMARY:-${RPC_PATCH_ROOT}/patch/bench-results/rpc-server-bench/matrix-summary-remus.txt}"
        VARIANTS=(pathb)
        export BENCH_RPC_MODE=remote
        export BENCH_RPC_HOST="${BENCH_RPC_HOST:-${REMUS_RPC_IP:-192.168.8.176}}"
        export REMUS_RPC_IP="${REMUS_RPC_IP:-192.168.8.176}"
        export BENCH_TS="${BENCH_TS:-15,85}"
        LABEL_PREFIX="remus-5060-"
        TS_TAG="ts1585"
        ;;
    config-c)
        SUMMARY="${BENCH_MATRIX_SUMMARY:-${RPC_PATCH_ROOT}/patch/bench-results/rpc-server-bench/matrix-summary-config-c.txt}"
        VARIANTS=(pathb)
        export BENCH_RPC_MODE=multi
        export REMUS_RPC_IP="${REMUS_RPC_IP:-192.168.8.176}"
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-127.0.0.1:50051,${REMUS_RPC_IP}:50051}"
        export BENCH_TS="${BENCH_TS:-12,8,80}"
        LABEL_PREFIX="config-c-"
        TS_TAG="ts12880"
        ;;
    *)
        echo "unknown BENCH_CONFIG=$CONFIG" >&2
        exit 1
        ;;
esac

mkdir -p "$(dirname "$SUMMARY")"
if [[ ! -f "$SUMMARY" ]]; then : >"$SUMMARY"; fi

PATHB_CONFIG="$CONFIG" "$DISK" --cleanup
if [[ "$CONFIG" == "remus" || "$CONFIG" == "config-c" ]]; then
    PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}" PATHB_CONFIG="$CONFIG" \
        "$DISK" --remote --min-gb 25
fi

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

PRESET_MODEL[9b]="${MODELS_ROOT}/Qwen3.5-9B-MTP-Q4_K_M.gguf"
PRESET_CTX[9b]=8192
PRESET_CTK[9b]=q8_0
PRESET_CTV[9b]=turbo3
PRESET_NCMOE[9b]=""
PRESET_LABEL[9b]="9b-qwen"

if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=(27b 31b 35b 36b)
fi

log_summary() { echo "$*" | tee -a "$SUMMARY"; }

run_one() {
    local preset="$1" variant="$2"
    local label="${LABEL_PREFIX}${PRESET_LABEL[$preset]}-${variant}-${TS_TAG}"
    local ncmoe="${PRESET_NCMOE[$preset]}"

    export BENCH_MODEL="${PRESET_MODEL[$preset]}"
    export BENCH_CTX="${PRESET_CTX[$preset]}"
    export BENCH_CTK="${PRESET_CTK[$preset]}"
    export BENCH_CTV="${PRESET_CTV[$preset]}"
    export BENCH_NGL=99
    export BENCH_LOAD_TIMEOUT=600
    export BENCH_GEN_TOKENS=48
    export BENCH_NO_WARMUP=1
    export BENCH_NP=1
    export BENCH_NCMOE="$ncmoe"

    log_summary "--- ${label} $(date -u +%H:%M:%SZ) config=$CONFIG ---"
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

log_summary "=== RPC path matrix config=$CONFIG ts=${BENCH_TS} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
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

docker rm -f bench-rpc bench-llama pathb-longgen512 2>/dev/null || true
docker image prune -f >/dev/null 2>&1 || true
df -h / | tee -a "$SUMMARY"
log_summary "=== done config=$CONFIG ==="