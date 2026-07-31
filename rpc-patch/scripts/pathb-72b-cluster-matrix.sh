#!/usr/bin/env bash
# S0-lite Config G cluster benches (72B+ MoE/dense) with BENCH_TRACE=1.
#
# usage: pathb-72b-cluster-matrix.sh [preset...]
#   presets: qwen72b coder-next-q4
#
# env:
#   WIN_RPC_IP          Windows 5070 rpc-server LAN IP (:50053)
#   REMUS_RPC_IP        default 192.168.8.176
#   ROMULUS_RPC_IP      default 192.168.8.108
#   BENCH_GEN_TOKENS    default 128
#   GGML_PIPELINE_PLUS  default 1

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
BENCH="${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh"
VRAM_CALC="${RPC_PATCH_ROOT}/scripts/pathb-72b-vram-calc.py"
LOG_DIR="${RPC_PATCH_ROOT}/patch/bench-results/72b-cluster"
SUMMARY="${LOG_DIR}/cluster-summary.txt"
MODELS="${MODELS_ROOT:-/mnt/models}"

REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
ROMULUS_IP="${ROMULUS_RPC_IP:-192.168.8.108}"
WIN_IP="${WIN_RPC_IP:-}"
TS="${BENCH_TS:-28,12,28,32}"
GEN="${BENCH_GEN_TOKENS:-128}"

declare -A P_MODEL P_MOE P_LABEL
P_MODEL[qwen72b]="${MODELS}/Qwen3-72B-Instruct.IQ4_XS.gguf"
P_MOE[qwen72b]=0
P_LABEL[qwen72b]="s0lite-qwen72b"

P_MODEL[coder-next-q4]="${MODELS}/Qwen_Qwen3-Coder-Next-Q4_K_M.gguf"
P_MOE[coder-next-q4]=1
P_LABEL[coder-next-q4]="s0lite-coder-next-q4"

ALL_PRESETS=(qwen72b coder-next-q4)
TARGETS=("${ALL_PRESETS[@]}")
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
fi

mkdir -p "$LOG_DIR"
: >"$SUMMARY"

if [[ -z "$WIN_IP" ]]; then
    echo "WARN: WIN_RPC_IP unset; use 2-RPC Config G subset (remus+romulus only)" | tee -a "$SUMMARY"
    RPC_ENDPOINT="${REMUS_IP}:50051,${ROMULUS_IP}:50051"
else
    RPC_ENDPOINT="${REMUS_IP}:50051,${ROMULUS_IP}:50051,${WIN_IP}:50053"
fi

log() { echo "$*" | tee -a "$SUMMARY"; }

log "=== S0-lite Config G matrix $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
log "endpoint=$RPC_ENDPOINT ts=$TS gen=$GEN"
log "cluster: ./rpc-patch/scripts/pathb-cluster-up.sh start"

export PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
export PATHB_ROMULUS_SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"
export REMUS_RPC_IP="$REMUS_IP"
export ROMULUS_RPC_IP="$ROMULUS_IP"

"${RPC_PATCH_ROOT}/scripts/pathb-cluster-up.sh" start || true

for preset in "${TARGETS[@]}"; do
    model="${P_MODEL[$preset]:-}"
    label="${P_LABEL[$preset]}"
    ncmoe="${P_MOE[$preset]}"
    if [[ ! -f "$model" ]]; then
        log "SKIP $preset: missing $model"
        continue
    fi
    log "--- preset=$preset label=$label ---"
    python3 "$VRAM_CALC" --config config-g --gguf "$model" --ts "$TS" 2>&1 | tee -a "$SUMMARY" || true

    export BENCH_MODEL="$model"
    export BENCH_RPC_MODE=multi
    export BENCH_RPC_ENDPOINT="$RPC_ENDPOINT"
    export BENCH_TS="$TS"
    export BENCH_GEN_TOKENS="$GEN"
    export BENCH_NO_WARMUP=1
    export BENCH_TRACE=1
    export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
    export BENCH_EXTRA="--fit off --verbose -lv 4 --reasoning off"
    if [[ "$ncmoe" == "1" ]]; then
        export BENCH_NCMOE=0
    else
        unset BENCH_NCMOE
    fi

    if "$BENCH" pathb "$label"; then
        log "PASS $label"
    else
        log "FAIL $label"
    fi
done

log "=== done; results in $LOG_DIR and $SUMMARY ==="