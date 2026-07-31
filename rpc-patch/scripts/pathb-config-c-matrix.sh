#!/usr/bin/env bash
# Expanded Path B RPC benchmark matrix (Config B or C).
#
# Config B (2 devices): Romulus 7900 XTX + remus 5060 Ti RPC
# Config C (3 devices): Romulus 7900 XTX + local 3060 Ti RPC + remus 5060 Ti RPC
#
# usage: pathb-config-c-matrix.sh [preset...]
#   presets: 9b 9b-mtp 12b 12b-mtp 27b 31b 31b-mtp 35b 35b-mtp 36b 36b-ngl50 72b-fit
#   default: all available presets in order
#
# env:
#   PATHB_CONFIG=config-b|config-c  (default config-c)
#   REMUS_RPC_IP, PATHB_REMUS_SSH_PASS
#   BENCH_MATRIX_SKIP=1  continue on failure (default)
#   BENCH_FILTER=9b,35b   run subset

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
BENCH="${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh"
MONITOR="${RPC_PATCH_ROOT}/scripts/pathb-gpu-monitor.sh"
DISK="${RPC_PATCH_ROOT}/scripts/pathb-disk-preflight.sh"
MODELS="${MODELS_ROOT:-/mnt/models}"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
PATHB_CONFIG="${PATHB_CONFIG:-config-c}"
case "$PATHB_CONFIG" in
    config-b)
        LOG_DIR="${RPC_PATCH_ROOT}/patch/bench-results/config-b-matrix"
        RPC_MODE=remote
        RPC_ENDPOINT="${REMUS_IP}:50051"
        DEFAULT_TS="25,75"
        DEFAULT_FITT="900,900"
        CONFIG_LABEL="config-b"
        ;;
    config-c)
        LOG_DIR="${RPC_PATCH_ROOT}/patch/bench-results/config-c-matrix"
        RPC_MODE=multi
        RPC_ENDPOINT="127.0.0.1:50051,${REMUS_IP}:50051"
        DEFAULT_TS="10,25,65"
        DEFAULT_FITT="900,900,900"
        CONFIG_LABEL="config-c"
        ;;
    *)
        echo "unknown PATHB_CONFIG=$PATHB_CONFIG (use config-b or config-c)" >&2
        exit 1
        ;;
esac
SUMMARY="${LOG_DIR}/matrix-summary.txt"
SKIP_ON_FAIL="${BENCH_MATRIX_SKIP:-1}"

# CLI uses upstream spec-type names (draft-mtp, not legacy nextn/mtp aliases).
SPEC_MTP_QWEN="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1 -ngld 99"
SPEC_MTP_GEMMA="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1 -ngld 99"

mkdir -p "$LOG_DIR"
export PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
export REMUS_RPC_IP="$REMUS_IP"

declare -A P_MODEL P_CTX P_CTK P_CTV P_NGL P_TS P_FITT P_NCMOE P_EXTRA P_LABEL P_LOAD

# --- dense / small ---
P_MODEL[9b]="${MODELS}/Qwen3.5-9B-MTP-Q4_K_M.gguf"
P_CTX[9b]=4096; P_CTK[9b]=q8_0; P_CTV[9b]=turbo3; P_NGL[9b]=99
P_TS[9b]="$DEFAULT_TS"; P_FITT[9b]="$DEFAULT_FITT"; P_NCMOE[9b]=""
P_EXTRA[9b]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2"
P_LABEL[9b]="9b-qwen"; P_LOAD[9b]=600

P_MODEL[9b-mtp]="${MODELS}/Qwen3.5-9B-MTP-Q4_K_M.gguf"
P_CTX[9b-mtp]=4096; P_CTK[9b-mtp]=q8_0; P_CTV[9b-mtp]=turbo3; P_NGL[9b-mtp]=99
P_TS[9b-mtp]="$DEFAULT_TS"; P_FITT[9b-mtp]="$DEFAULT_FITT"; P_NCMOE[9b-mtp]=""
P_EXTRA[9b-mtp]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2 ${SPEC_MTP_QWEN}"
P_LABEL[9b-mtp]="9b-qwen-mtp"; P_LOAD[9b-mtp]=600

P_MODEL[27b]="${MODELS}/Qwen3.5-27B-Q5_K_M.gguf"
P_CTX[27b]=8192; P_CTK[27b]=q8_0; P_CTV[27b]=turbo3; P_NGL[27b]=99
P_TS[27b]="$DEFAULT_TS"; P_FITT[27b]="$DEFAULT_FITT"; P_NCMOE[27b]=""
P_EXTRA[27b]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2"
P_LABEL[27b]="27b-qwen"; P_LOAD[27b]=600

# --- gemma 12B (assistant GGUF uses gemma4-assistant arch) ---
P_MODEL[12b]="${MODELS}/gemma-4-12b-it-Q4_K_M.gguf"
P_CTX[12b]=8192; P_CTK[12b]=q8_0; P_CTV[12b]=turbo3; P_NGL[12b]=99
P_TS[12b]="$DEFAULT_TS"; P_FITT[12b]="$DEFAULT_FITT"; P_NCMOE[12b]=""
P_EXTRA[12b]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2"
P_LABEL[12b]="12b-gemma"; P_LOAD[12b]=600

P_MODEL[12b-mtp]="${MODELS}/gemma-4-12b-it-Q4_K_M.gguf"
P_CTX[12b-mtp]=8192; P_CTK[12b-mtp]=q8_0; P_CTV[12b-mtp]=turbo3; P_NGL[12b-mtp]=99
P_TS[12b-mtp]="$DEFAULT_TS"; P_FITT[12b-mtp]="$DEFAULT_FITT"; P_NCMOE[12b-mtp]=""
P_EXTRA[12b-mtp]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2 ${SPEC_MTP_GEMMA} -md ${MODELS}/gemma-4-12B-it-assistant-Q8_0.gguf -ctkd q8_0 -ctvd q8_0"
P_LABEL[12b-mtp]="12b-gemma-mtp"; P_LOAD[12b-mtp]=600

# --- gemma dense + MTP assistant ---
P_MODEL[31b]="${MODELS}/gemma-4-31B-it-Q4_K_M.gguf"
P_CTX[31b]=8192; P_CTK[31b]=q8_0; P_CTV[31b]=turbo3; P_NGL[31b]=99
P_TS[31b]="$DEFAULT_TS"; P_FITT[31b]="$DEFAULT_FITT"; P_NCMOE[31b]=""
P_EXTRA[31b]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2"
P_LABEL[31b]="31b-gemma"; P_LOAD[31b]=900

P_MODEL[31b-mtp]="${MODELS}/gemma-4-31B-it-Q4_K_M.gguf"
P_CTX[31b-mtp]=8192; P_CTK[31b-mtp]=q8_0; P_CTV[31b-mtp]=turbo3; P_NGL[31b-mtp]=99
P_TS[31b-mtp]="$DEFAULT_TS"; P_FITT[31b-mtp]="$DEFAULT_FITT"; P_NCMOE[31b-mtp]=""
P_EXTRA[31b-mtp]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2 ${SPEC_MTP_GEMMA} -md ${MODELS}/gemma-4-31B-it-assistant-Q8_0.gguf -ctkd q8_0 -ctvd q8_0"
P_LABEL[31b-mtp]="31b-gemma-mtp"; P_LOAD[31b-mtp]=900

# --- MoE (expert CPU offload) ---
P_MODEL[35b]="${MODELS}/Qwen3.5-35B-A3B.i1-Q4_K_M.gguf"
P_CTX[35b]=4096; P_CTK[35b]=q4_0; P_CTV[35b]=q4_0; P_NGL[35b]=99
P_TS[35b]="$DEFAULT_TS"; P_FITT[35b]="$DEFAULT_FITT"; P_NCMOE[35b]=8
P_EXTRA[35b]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2"
P_LABEL[35b]="35b-a3b"; P_LOAD[35b]=900

P_MODEL[35b-mtp]="${MODELS}/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"
P_CTX[35b-mtp]=4096; P_CTK[35b-mtp]=q8_0; P_CTV[35b-mtp]=turbo3; P_NGL[35b-mtp]=99
P_TS[35b-mtp]="$DEFAULT_TS"; P_FITT[35b-mtp]="$DEFAULT_FITT"; P_NCMOE[35b-mtp]=8
P_EXTRA[35b-mtp]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2 ${SPEC_MTP_QWEN}"
P_LABEL[35b-mtp]="35b-a3b-mtp"; P_LOAD[35b-mtp]=900

P_MODEL[36b]="${MODELS}/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
P_CTX[36b]=4096; P_CTK[36b]=q4_0; P_CTV[36b]=q4_0; P_NGL[36b]=99
P_TS[36b]="$DEFAULT_TS"; P_FITT[36b]="$DEFAULT_FITT"; P_NCMOE[36b]=8
P_EXTRA[36b]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2"
P_LABEL[36b]="36b-a3b"; P_LOAD[36b]=900

# partial GPU offload: more layers on CPU/host RAM
P_MODEL[36b-ngl50]="${MODELS}/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
P_CTX[36b-ngl50]=4096; P_CTK[36b-ngl50]=q4_0; P_CTV[36b-ngl50]=q4_0; P_NGL[36b-ngl50]=50
P_TS[36b-ngl50]="$DEFAULT_TS"; P_FITT[36b-ngl50]="$DEFAULT_FITT"; P_NCMOE[36b-ngl50]=16
P_EXTRA[36b-ngl50]="--fit off --fit-target ${DEFAULT_FITT} --verbose -lv 2"
P_LABEL[36b-ngl50]="36b-a3b-ngl50-ncmoe16"; P_LOAD[36b-ngl50]=900

# --- 72B: fit-assisted split across GPUs + host RAM ---
P_MODEL[72b-fit]="${MODELS}/Qwen3-72B-Instruct.IQ4_XS.gguf"
P_CTX[72b-fit]=2048; P_CTK[72b-fit]=q4_0; P_CTV[72b-fit]=q4_0; P_NGL[72b-fit]=0
if [[ "$PATHB_CONFIG" == "config-c" ]]; then
    P_TS[72b-fit]="10,25,65"; P_FITT[72b-fit]="512,512,512"
else
    P_TS[72b-fit]="30,70"; P_FITT[72b-fit]="512,512"
fi
P_NCMOE[72b-fit]=""
P_EXTRA[72b-fit]="--fit on --fit-target ${P_FITT[72b-fit]} --verbose -lv 2 --reasoning off"
P_LABEL[72b-fit]="72b-qwen-fit"; P_LOAD[72b-fit]=1800

ALL_PRESETS=(9b 9b-mtp 12b 12b-mtp 27b 31b 31b-mtp 35b 35b-mtp 36b 36b-ngl50 72b-fit)

if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
elif [[ -n "${BENCH_FILTER:-}" ]]; then
    IFS=',' read -ra TARGETS <<<"$BENCH_FILTER"
else
    TARGETS=("${ALL_PRESETS[@]}")
fi

rpc_prep() {
    "${RPC_PATCH_ROOT}/scripts/pathb-remus-rpc.sh" start
    if [[ "$PATHB_CONFIG" == "config-c" ]]; then
        local cuda_bin="${TQ}/build-cuda-b-bin-sync/bin"
        [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin-rebuild/bin"
        [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin/bin"
        PATHB_BIN_CUDA="$cuda_bin" "${RPC_PATCH_ROOT}/scripts/pathb-start-rpc.sh"
    else
        docker rm -f pathb-rpc 2>/dev/null || true
    fi
}

log_summary() { echo "$*" | tee -a "$SUMMARY"; }

PATHB_CONFIG="$PATHB_CONFIG" "$DISK" --cleanup
PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}" PATHB_CONFIG="$PATHB_CONFIG" "$DISK" --remote --min-gb 20

log_summary "=== ${CONFIG_LABEL} matrix ts=${DEFAULT_TS} $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown) ==="
if [[ "$PATHB_CONFIG" == "config-c" ]]; then
    log_summary "endpoint=${RPC_ENDPOINT} devices=3060+remus5060+7900XTX"
else
    log_summary "endpoint=${RPC_ENDPOINT} devices=remus5060+7900XTX"
fi

for preset in "${TARGETS[@]}"; do
    model="${P_MODEL[$preset]:-}"
    if [[ -z "$model" ]]; then
        log_summary "SKIP unknown preset: $preset"
        continue
    fi
    if [[ ! -f "$model" ]]; then
        log_summary "SKIP missing $preset: $model"
        continue
    fi
    # gemma mtp needs assistant file
    if [[ "$preset" == "31b-mtp" ]]; then
        # local 31B assistant uses legacy gemma4_mtp arch; needs gemma4_assistant convert
        log_summary "SKIP 31b-mtp: assistant GGUF arch gemma4_mtp (need gemma4_assistant); use 12b-mtp"
        continue
    fi
    if [[ "$preset" == "12b-mtp" ]] && [[ ! -f "${MODELS}/gemma-4-12B-it-assistant-Q8_0.gguf" ]]; then
        log_summary "SKIP 12b-mtp: assistant GGUF missing"
        continue
    fi

    label="${CONFIG_LABEL}-${P_LABEL[$preset]}"
    log_summary "--- ${label} $(date -u +%H:%M:%SZ) preset=$preset ---"

    rpc_prep

    export BENCH_RPC_MODE="$RPC_MODE"
    export BENCH_RPC_ENDPOINT="$RPC_ENDPOINT"
    export BENCH_MODEL="$model"
    export BENCH_CTX="${P_CTX[$preset]}"
    export BENCH_CTK="${P_CTK[$preset]}"
    export BENCH_CTV="${P_CTV[$preset]}"
    export BENCH_NGL="${P_NGL[$preset]}"
    export BENCH_TS="${P_TS[$preset]}"
    export BENCH_NCMOE="${P_NCMOE[$preset]}"
    export BENCH_LOAD_TIMEOUT="${P_LOAD[$preset]}"
    export BENCH_GEN_TOKENS=64
    export BENCH_RUNS=3
    export BENCH_NO_WARMUP=1
    export BENCH_NP=1
    export BENCH_LOG_DIR="$LOG_DIR"
    export BENCH_EXTRA="${P_EXTRA[$preset]}"

    gpu_log="${LOG_DIR}/${label}.gpu"
    "$MONITOR" "$gpu_log" 300 &
    mon_pid=$!

    if "$BENCH" pathb "$label"; then
        result="${LOG_DIR}/${label}.result"
        if [[ -f "$result" ]]; then
            python3 - <<'PY' "$result" "$preset" | tee -a "$SUMMARY"
import sys, re
path, preset = sys.argv[1:3]
gs, ps = [], []
for line in open(path):
    m = re.search(r'P=([\d.]+).*G=([\d.]+)', line)
    if m:
        ps.append(float(m.group(1))); gs.append(float(m.group(2)))
avg_g = sum(gs)/len(gs) if gs else 0
avg_p = sum(ps)/len(ps) if ps else 0
print(f"PASS {preset} avg_P={avg_p:.1f} avg_G={avg_g:.1f} runs_G={gs}")
PY
        else
            log_summary "PASS $preset (no result file)"
        fi
    else
        log_summary "FAIL $preset"
        [[ "$SKIP_ON_FAIL" == "1" ]] || exit 1
    fi

    kill "$mon_pid" 2>/dev/null || true
    docker rm -f pathb-rpc bench-llama 2>/dev/null || true
    sleep 3
done

df -h / | tee -a "$SUMMARY"
log_summary "=== done ${CONFIG_LABEL} matrix ==="