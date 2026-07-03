#!/usr/bin/env bash
# Full TSC spike validation: sync code, rebuild romulus, multi-prompt + long ctx/gen matrix.
#
# usage: pathb-plus1-tsc-spike-validate.sh [--skip-rebuild]
# logs:  /tmp/plus1-tsc-spike-validate/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

SKIP_REBUILD=0
for arg in "$@"; do
    [[ "$arg" == "--skip-rebuild" ]] && SKIP_REBUILD=1
done

if [[ -f .scratch/cluster-access.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .scratch/cluster-access.env
    set +a
fi

SSH_HOST="${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}"
SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"
REPO_REMOTE="/home/hunter/atomic-llama-cpp-turboquant"
LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-spike-validate}"
PROMPTS_LOCAL="${ROOT}/rpc-patch/bench-prompts/tsc-spike-validation.json"
PROMPTS_REMOTE="${REPO_REMOTE}/rpc-patch/bench-prompts/tsc-spike-validation.json"
RESULTS_REMOTE="${REPO_REMOTE}/rpc-patch/patch/bench-results/rpc-server-bench"

SSH_BASE=(ssh -o StrictHostKeyChecking=no)
SCP_BASE=(scp -o StrictHostKeyChecking=no)
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_BASE=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no)
    SCP_BASE=(sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no)
fi

mkdir -p "$LOG_DIR"

SPIKE_FILES=(
    ggml/include/ggml-backend.h
    ggml/src/ggml-backend.cpp
    ggml/src/ggml-rpc/ggml-rpc.cpp
    src/llama-context.cpp
    rpc-patch/scripts/rpc-server-bench.sh
    rpc-patch/scripts/pathb-romulus-2gpu-bench.sh
    rpc-patch/scripts/pathb-romulus-2gpu-bench-host.sh
    rpc-patch/scripts/pathb-plus1-tsc-spike-validate.py
    rpc-patch/bench-prompts/tsc-spike-validation.json
)

echo "=== Step 1: sync spike files to romulus ==="
for f in "${SPIKE_FILES[@]}"; do
    "${SCP_BASE[@]}" "$ROOT/$f" "${SSH_HOST}:${REPO_REMOTE}/$f"
done
chmod +x "${SCRIPT_DIR}/pathb-romulus-2gpu-bench-host.sh" 2>/dev/null || true
"${SSH_BASE[@]}" "$SSH_HOST" "chmod +x ${REPO_REMOTE}/rpc-patch/scripts/pathb-romulus-2gpu-bench-host.sh"

if [[ "$SKIP_REBUILD" == "0" ]]; then
    echo "=== Step 2: rebuild llama-server on romulus (incremental) ==="
    "${SSH_BASE[@]}" "$SSH_HOST" bash -s <<'REMOTE'
set -euo pipefail
REPO=/home/hunter/atomic-llama-cpp-turboquant
BUILD=build-rocm-docker
cd "$REPO"
cmake -S . -B "$BUILD" \
    -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4
cmake --build "$BUILD" --target llama-server -j"$(nproc)" 2>&1 | tail -20
test -x "$BUILD/bin/llama-server"
echo ROCM_BUILD_OK
REMOTE
else
    echo "=== Step 2: skipped rebuild ==="
fi

BENCH="${SCRIPT_DIR}/pathb-romulus-2gpu-bench.sh"
BASE_ENV=(
    BENCH_MODEL="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
    BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50
    BENCH_CTX=8192
    BENCH_GEN_TOKENS=768
    BENCH_RUNS=1
    BENCH_LOAD_TIMEOUT=1800
    BENCH_CURL_TIMEOUT=900
    BENCH_SAVE_FULL=1
    BENCH_PROMPTS_FILE="$PROMPTS_REMOTE"
    BENCH_EXTRA="--fit off --verbose -lv 4 --reasoning off"
    BENCH_TRACE=0
)

run_arm() {
    local arm="$1"
    local plus="$2"
    local seq="$3"
    local label="tsc-spike-val-${arm}"
    echo "=== Step 3: arm=${arm} plus=${plus} seq=${seq} ==="
    env "${BASE_ENV[@]}" \
        GGML_PIPELINE_PLUS="$plus" \
        GGML_PIPELINE_MULTI_BACKEND_SEQ="$seq" \
        "$BENCH" "$label" 2>&1 | tee "${LOG_DIR}/${arm}.log" || true

    echo "=== Step 4: fetch results for ${arm} ==="
    mkdir -p "${LOG_DIR}/${arm}"
    "${SCP_BASE[@]}" "${SSH_HOST}:${RESULTS_REMOTE}/${label}.full.jsonl" "${LOG_DIR}/${arm}/" 2>/dev/null || true
    "${SCP_BASE[@]}" "${SSH_HOST}:${RESULTS_REMOTE}/${label}.result" "${LOG_DIR}/${arm}/" 2>/dev/null || true
    "${SCP_BASE[@]}" "${SSH_HOST}:${RESULTS_REMOTE}/${label}.meta" "${LOG_DIR}/${arm}/" 2>/dev/null || true
    if [[ -f "${LOG_DIR}/${arm}/${label}.full.jsonl" ]]; then
        mv -f "${LOG_DIR}/${arm}/${label}.full.jsonl" "${LOG_DIR}/${arm}/full.jsonl"
    fi
    if [[ -f "${LOG_DIR}/${arm}/${label}.result" ]]; then
        mv -f "${LOG_DIR}/${arm}/${label}.result" "${LOG_DIR}/${arm}/result.txt"
    fi
}

# Broken baseline (expect failures)
run_arm "canonical-plus1" 1 0

# Spike repair (expect pass)
run_arm "seq-repair" 1 1

# Known-good reference
run_arm "plus0-ref" 0 0

echo "=== Step 5: validate outputs ==="
python3 "${SCRIPT_DIR}/pathb-plus1-tsc-spike-validate.py" "$PROMPTS_LOCAL" "${LOG_DIR}/canonical-plus1" > "${LOG_DIR}/canonical-plus1-score.txt" 2>&1 || true
python3 "${SCRIPT_DIR}/pathb-plus1-tsc-spike-validate.py" "$PROMPTS_LOCAL" "${LOG_DIR}/seq-repair" > "${LOG_DIR}/seq-repair-score.txt" 2>&1 || true
python3 "${SCRIPT_DIR}/pathb-plus1-tsc-spike-validate.py" "$PROMPTS_LOCAL" "${LOG_DIR}/plus0-ref" > "${LOG_DIR}/plus0-ref-score.txt" 2>&1 || true

echo "=== SUMMARY ==="
for f in canonical-plus1-score.txt seq-repair-score.txt plus0-ref-score.txt; do
    echo "--- $f ---"
    cat "${LOG_DIR}/$f" 2>/dev/null || echo "(missing)"
done
echo "logs: ${LOG_DIR}"