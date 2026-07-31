#!/usr/bin/env bash
# Re-run failed open-points: kvfill (large chunks) + 3-GPU suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .scratch/cluster-access.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .scratch/cluster-access.env
    set +a
fi

LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-open-points}"
PROMPTS_LOCAL="${ROOT}/rpc-patch/bench-prompts/tsc-spike-validation.json"
PROMPTS_REMOTE="/home/hunter/atomic-llama-cpp-turboquant/rpc-patch/bench-prompts/tsc-spike-validation.json"
SSH_HOST="${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}"
SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"
RESULTS_REMOTE="/home/hunter/atomic-llama-cpp-turboquant/rpc-patch/patch/bench-results/rpc-server-bench"

SSH_BASE=(ssh -o StrictHostKeyChecking=no)
SCP_BASE=(scp -o StrictHostKeyChecking=no)
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_BASE=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no)
    SCP_BASE=(sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no)
fi

"${SCP_BASE[@]}" "${ROOT}/rpc-patch/scripts/pathb-plus1-tsc-multiturn-kvfill.py" \
    "${ROOT}/rpc-patch/scripts/pathb-plus1-tsc-spike-validate.py" \
    "${ROOT}/rpc-patch/scripts/pathb-romulus-3gpu-bench.sh" \
    "${SSH_HOST}:/home/hunter/atomic-llama-cpp-turboquant/rpc-patch/scripts/"

BASE=(
    BENCH_CTX=8192 BENCH_CTK=q8_0 BENCH_CTV=q8_0
    BENCH_GEN_TOKENS=512 BENCH_CURL_TIMEOUT=900 BENCH_SAVE_FULL=1
    BENCH_PROMPTS_FILE="$PROMPTS_REMOTE"
    BENCH_RUNS=1 BENCH_LOAD_TIMEOUT=1800
    BENCH_EXTRA="--fit off --verbose -lv 4 --reasoning off"
)

echo "=== kvfill seq-repair (large chunks, target=7200) ==="
env BENCH_MODEL="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf" \
    BENCH_MULTITURN_KVFILL=1 BENCH_KV_TARGET=7200 BENCH_KV_MIN_RATIO=0.90 \
    BENCH_KV_TURN_GEN=96 BENCH_KV_FINAL_GEN=128 BENCH_KV_MAX_TURNS=12 \
    BENCH_CTX=8192 BENCH_TS=50,50 BENCH_CURL_TIMEOUT=1800 \
    GGML_PIPELINE_PLUS=1 GGML_PIPELINE_MULTI_BACKEND_SEQ=1 \
    "${SCRIPT_DIR}/pathb-romulus-2gpu-bench.sh" tsc-kvfill-2gpu-seq-repair-v2 \
    2>&1 | tee "${LOG_DIR}/tsc-kvfill-2gpu-seq-repair-v2.log"

mkdir -p "${LOG_DIR}/tsc-kvfill-2gpu-seq-repair-v2"
"${SCP_BASE[@]}" "${SSH_HOST}:${RESULTS_REMOTE}/tsc-kvfill-2gpu-seq-repair-v2.kvfill.json" \
    "${LOG_DIR}/tsc-kvfill-2gpu-seq-repair-v2/kvfill.json" 2>/dev/null || true

for arm in canonical-plus1 seq-repair; do
    plus=1; seq=0
    [[ "$arm" == "seq-repair" ]] && seq=1
    label="tsc-3gpu-apex-${arm}"
    echo "=== 3gpu ${label} ==="
    env BENCH_MODEL="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf" \
        "${BASE[@]}" GGML_PIPELINE_PLUS="$plus" GGML_PIPELINE_MULTI_BACKEND_SEQ="$seq" \
        "${SCRIPT_DIR}/pathb-romulus-3gpu-bench.sh" primary-5060-3060 "$label" \
        2>&1 | tee "${LOG_DIR}/${label}-rerun.log"
    mkdir -p "${LOG_DIR}/${label}"
    "${SCP_BASE[@]}" "${SSH_HOST}:${RESULTS_REMOTE}/${label}.full.jsonl" \
        "${LOG_DIR}/${label}/full.jsonl" 2>/dev/null || true
    python3 "${SCRIPT_DIR}/pathb-plus1-tsc-spike-validate.py" "$PROMPTS_LOCAL" "${LOG_DIR}/${label}" \
        | tee "${LOG_DIR}/${label}-rerun-score.txt"
done

echo "=== rerun complete ==="