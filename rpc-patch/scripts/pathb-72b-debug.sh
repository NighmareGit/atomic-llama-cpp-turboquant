#!/usr/bin/env bash
# Debug dense 72B RPC inference crash (Config C).
#
# usage: pathb-72b-debug.sh [case...]
#   cases: preflight version-check fitoff-ngl56 fitoff-nographs fitoff-ts391150
#           fiton-ngl0-ts391150 fiton-ngl56-ts391150 configb-fitoff-ngl56
#           swap-order-fitoff-ngl56 27b-fitoff-ngl99 stale-check
#   default: preflight fitoff-ngl56
#
# env: PATHB_CUDA_DISABLE_GRAPHS=1  (sets GGML_CUDA_DISABLE_GRAPHS on both RPC workers)
#
# env: PATHB_REMUS_SSH_PASS, REMUS_RPC_IP=192.168.8.176

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
LOG_DIR="${RPC_PATCH_ROOT}/patch/bench-results/72b-matrix/debug"
BENCH="${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
MODEL="${PATHB_DEBUG_MODEL:-/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf}"
TS="35,15,50"
ENDPOINT="${REMUS_IP}:50051,127.0.0.1:50051"

CASES=()
while [[ $# -gt 0 ]]; do
    CASES+=("$1")
    shift
done
[[ ${#CASES[@]} -eq 0 ]] && CASES=(preflight fitoff-ngl56)

mkdir -p "$LOG_DIR"
export PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
export REMUS_RPC_IP="$REMUS_IP"

log() { echo "$*" | tee -a "${LOG_DIR}/debug-run.log"; }

version_check() {
    local out="${LOG_DIR}/version-check.txt"
    local cuda_bin="${TQ}/build-cuda-b-bin-sync/bin"
    [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin-rebuild/bin"
    [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin/bin"
    local rocm_bin="${TQ}/build-rocm-docker/bin"
    {
        echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) version parity ==="
        echo "[romulus rpc-server host]"
        ls -la "${cuda_bin}/rpc-server" 2>/dev/null || true
        md5sum "${cuda_bin}/rpc-server" 2>/dev/null || true
        echo "[romulus llama-server host]"
        ls -la "${rocm_bin}/llama-server" "${rocm_bin}/libggml-rpc.so" 2>/dev/null || true
        md5sum "${rocm_bin}/llama-server" "${rocm_bin}/libggml-rpc.so" 2>/dev/null || true
        echo "[pathb-rpc container]"
        docker exec pathb-rpc ls -la /app/bin/rpc-server 2>/dev/null || echo "pathb-rpc not running"
        docker exec pathb-rpc md5sum /app/bin/rpc-server 2>/dev/null || true
        echo "[bench image rpc-server embedded]"
        docker run --rm --entrypoint md5sum llama-rpc-cuda-a2 /app/build/bin/rpc-server 2>/dev/null || true
        echo "[remus compose]"
        PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}" \
            "${RPC_PATCH_ROOT}/scripts/pathb-remus-rpc.sh" status 2>/dev/null || true
        if [[ -n "${PATHB_REMUS_SSH_PASS:-}" ]] && command -v sshpass >/dev/null; then
            sshpass -p "${PATHB_REMUS_SSH_PASS}" ssh -o StrictHostKeyChecking=accept-new \
                "hunter@${REMUS_IP}" \
                "docker exec pathb-rpc-remus ls -la /usr/local/bin/rpc-server; docker exec pathb-rpc-remus md5sum /usr/local/bin/rpc-server; docker inspect pathb-rpc-remus --format '{{.Config.Image}} {{.Image}}'" \
                2>/dev/null || true
        fi
    } | tee "$out"
    log "version-check written: $out"
}

rpc_prep() {
    "${RPC_PATCH_ROOT}/scripts/pathb-remus-rpc.sh" start
    local cuda_bin="${TQ}/build-cuda-b-bin-sync/bin"
    [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin-rebuild/bin"
    [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin/bin"
    PATHB_BIN_CUDA="$cuda_bin" "${RPC_PATCH_ROOT}/scripts/pathb-start-rpc.sh"
}

gpu_preflight() {
    local out="${LOG_DIR}/preflight-gpu.txt"
    {
        echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        echo "[docker rpc/llama containers]"
        docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' \
            | rg -i 'bench|pathb|rpc|llama' || true
        echo "[romulus nvidia-smi]"
        nvidia-smi 2>/dev/null || true
        echo "[romulus rocm-smi]"
        rocm-smi --showmeminfo vram 2>/dev/null | head -6 || rocm-smi 2>/dev/null | head -3 || true
        echo "[remus nvidia-smi]"
        PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}" \
            "${RPC_PATCH_ROOT}/scripts/pathb-remus-rpc.sh" status 2>/dev/null || true
    } | tee "$out"
    log "preflight written: $out"
}

stale_cleanup() {
    log "removing stale bench/llama containers (keeping pathb-rpc + remus compose)"
    docker rm -f bench-llama bench-rpc 2>/dev/null || true
    docker ps -a --format '{{.Names}}' | rg '^bench-' | xargs -r docker rm -f 2>/dev/null || true
}

run_case() {
    local case="$1"
    local label="debug-${case}"
    local ngl=56
    local extra="--fit off --verbose -lv 4 --reasoning off"

    case "$case" in
        preflight|stale-check|version-check)
            return 0
            ;;
        fitoff-ngl56)
            ngl=56
            extra="--fit off --verbose -lv 4 --reasoning off"
            ;;
        fitoff-ngl52)
            ngl=52
            extra="--fit off --verbose -lv 4 --reasoning off"
            ;;
        fitoff-faoff)
            ngl=56
            extra="--fit off -fa off --verbose -lv 4 --reasoning off"
            ;;
        fitoff-nographs)
            ngl=56
            extra="--fit off --verbose -lv 4 --reasoning off"
            export PATHB_CUDA_DISABLE_GRAPHS=1
            ;;
        fitoff-ts391150)
            ngl=56
            TS="39,11,50"
            extra="--fit off --verbose -lv 4 --reasoning off"
            export PATHB_CUDA_DISABLE_GRAPHS=1
            ;;
        fiton-ngl0-ts391150)
            ngl=0
            TS="39,11,50"
            extra="--fit on --fit-target 620,1024,880 --verbose -lv 4 --reasoning off"
            export PATHB_CUDA_DISABLE_GRAPHS=1
            ;;
        fiton-ngl56-ts391150)
            ngl=56
            TS="39,11,50"
            extra="--fit on --fit-target 620,1024,880 --verbose -lv 4 --reasoning off"
            export PATHB_CUDA_DISABLE_GRAPHS=1
            ;;
        27b-fitoff-ngl99)
            MODEL="/mnt/models/Qwen3.5-27B-Q5_K_M.gguf"
            ngl=99
            TS="39,11,50"
            extra="--fit off --verbose -lv 4 --reasoning off"
            export PATHB_CUDA_DISABLE_GRAPHS=1
            ;;
        version-check)
            version_check
            return 0
            ;;
        configb-fitoff-ngl56)
            ngl=56
            ENDPOINT="${REMUS_IP}:50051"
            TS="35,65"
            extra="--fit off --verbose -lv 4 --reasoning off"
            export PATHB_CUDA_DISABLE_GRAPHS=1
            docker rm -f pathb-rpc 2>/dev/null || true
            ;;
        swap-order-fitoff-ngl56)
            ngl=56
            ENDPOINT="127.0.0.1:50051,${REMUS_IP}:50051"
            TS="15,35,50"
            extra="--fit off --verbose -lv 4 --reasoning off"
            export PATHB_CUDA_DISABLE_GRAPHS=1
            ;;
        *)
            log "unknown case: $case"
            return 1
            ;;
    esac

    log "=== case ${case} label=${label} ngl=${ngl} ==="
    stale_cleanup
    rpc_prep
    gpu_preflight

    export BENCH_RPC_MODE=multi
    export BENCH_RPC_ENDPOINT="$ENDPOINT"
    export BENCH_MODEL="$MODEL"
    export BENCH_CTX=8192
    export BENCH_CTK=q4_0
    export BENCH_CTV=q4_0
    export BENCH_NGL="$ngl"
    export BENCH_TS="$TS"
    export BENCH_LOAD_TIMEOUT=1800
    export BENCH_GEN_TOKENS=16
    export BENCH_RUNS=1
    export BENCH_NO_WARMUP=1
    export BENCH_NP=1
    export BENCH_LOG_DIR="$LOG_DIR"
    export BENCH_EXTRACT_VRAM=1
    export BENCH_VERBOSE_LV=4
    export BENCH_CURL_TIMEOUT=600
    export BENCH_EXTRA="$extra"

    local rc=0
    "$BENCH" pathb "$label" || rc=$?
    log "case ${case} bench_rc=${rc}"
    docker rm -f bench-llama bench-rpc 2>/dev/null || true
    return "$rc"
}

: >"${LOG_DIR}/debug-run.log"
log "72b debug start cases=${CASES[*]} model=${MODEL}"

for c in "${CASES[@]}"; do
    case "$c" in
        preflight)
            stale_cleanup
            gpu_preflight
            ;;
        stale-check)
            stale_cleanup
            gpu_preflight
            ;;
        version-check)
            version_check
            ;;
        *)
            run_case "$c" || log "FAIL case $c (see ${LOG_DIR}/debug-${c}*)"
            ;;
    esac
done

log "done. artifacts in ${LOG_DIR}/"