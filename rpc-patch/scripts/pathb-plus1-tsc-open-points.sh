#!/usr/bin/env bash
# Complete open TSC validation points:
#   1) multi-turn single-slot KV fill toward 8192
#   2) llama3 + gemma3 under same prompt suite
#   3) 3-GPU topology (primary-5060-3060)
#
# usage: pathb-plus1-tsc-open-points.sh [--skip-sync]
# logs:  /tmp/plus1-tsc-open-points/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

SKIP_SYNC=0
for arg in "$@"; do
    [[ "$arg" == "--skip-sync" ]] && SKIP_SYNC=1
done

if [[ -f .scratch/cluster-access.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .scratch/cluster-access.env
    set +a
fi

LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-open-points}"
PROMPTS_LOCAL="${ROOT}/rpc-patch/bench-prompts/tsc-spike-validation.json"
PROMPTS_REMOTE="/home/hunter/atomic-llama-cpp-turboquant/rpc-patch/bench-prompts/tsc-spike-validation.json"
RESULTS_REMOTE="/home/hunter/atomic-llama-cpp-turboquant/rpc-patch/patch/bench-results/rpc-server-bench"
SSH_HOST="${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}"
SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-}"

SSH_BASE=(ssh -o StrictHostKeyChecking=no)
SCP_BASE=(scp -o StrictHostKeyChecking=no)
if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_BASE=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no)
    SCP_BASE=(sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no)
fi

mkdir -p "$LOG_DIR"
SUMMARY="${LOG_DIR}/summary.tsv"
echo "phase|arm|model|topology|verdict|detail" > "$SUMMARY"

log_line() {
    echo "$1" | tee -a "${LOG_DIR}/run.log"
}

gate_suite() {
    local dir="$1"
    local name="$2"
    python3 "${SCRIPT_DIR}/pathb-plus1-tsc-spike-validate.py" "$PROMPTS_LOCAL" "$dir" > "${LOG_DIR}/${name}-score.txt" 2>&1 || true
    if grep -q "validation: 8/8 passed" "${LOG_DIR}/${name}-score.txt"; then
        echo "PASS|8/8"
    else
        local score
        score=$(grep -o 'validation: [0-9]*/[0-9]*' "${LOG_DIR}/${name}-score.txt" | tail -1 || echo "unknown")
        echo "FAIL|${score}"
    fi
}

gate_kvfill() {
    local f="$1"
    if [[ ! -f "$f" ]]; then
        echo "FAIL|missing"
        return
    fi
    python3 -c "
import json, sys
d=json.load(open(sys.argv[1]))
ok = d.get('pass', False)
pt = d.get('final_prompt_tokens', 0)
print('PASS' if ok else 'FAIL', f'prompt_tokens={pt} turns={d.get(\"turns\",0)}', sep='|')
" "$f"
}

BASE_2GPU=(
    BENCH_CTX=8192 BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50
    BENCH_GEN_TOKENS=512 BENCH_CURL_TIMEOUT=900 BENCH_SAVE_FULL=1
    BENCH_PROMPTS_FILE="$PROMPTS_REMOTE"
    BENCH_RUNS=1 BENCH_LOAD_TIMEOUT=1800
    BENCH_EXTRA="--fit off --verbose -lv 4 --reasoning off"
    BENCH_TRACE=0
)

if [[ "$SKIP_SYNC" == "0" ]]; then
    log_line "=== sync scripts to romulus ==="
    SYNC_FILES=(
        rpc-patch/scripts/rpc-server-bench.sh
        rpc-patch/scripts/pathb-romulus-2gpu-bench.sh
        rpc-patch/scripts/pathb-romulus-2gpu-bench-host.sh
        rpc-patch/scripts/pathb-romulus-3gpu-bench.sh
        rpc-patch/scripts/pathb-romulus-3gpu-bench-host.sh
        rpc-patch/scripts/pathb-plus1-tsc-multiturn-kvfill.py
        rpc-patch/scripts/pathb-plus1-tsc-spike-validate.py
        rpc-patch/bench-prompts/tsc-spike-validation.json
    )
    for f in "${SYNC_FILES[@]}"; do
        "${SCP_BASE[@]}" "$ROOT/$f" "${SSH_HOST}:/home/hunter/atomic-llama-cpp-turboquant/$f"
    done
    "${SSH_BASE[@]}" "$SSH_HOST" "chmod +x /home/hunter/atomic-llama-cpp-turboquant/rpc-patch/scripts/pathb-romulus-{2,3}gpu-bench-host.sh /home/hunter/atomic-llama-cpp-turboquant/rpc-patch/scripts/pathb-plus1-tsc-multiturn-kvfill.py"
fi

run_bench() {
    local script="$1"
    local label="$2"
    shift 2
    log_line "=== bench ${label} ==="
    env "$@" "$script" "$label" 2>&1 | tee "${LOG_DIR}/${label}.log" || true
}

fetch_full() {
    local label="$1"
    local dest="$2"
    mkdir -p "$dest"
    "${SCP_BASE[@]}" "${SSH_HOST}:${RESULTS_REMOTE}/${label}.full.jsonl" "${dest}/full.jsonl" 2>/dev/null || true
}

fetch_kvfill() {
    local label="$1"
    local dest="$2"
    mkdir -p "$dest"
    "${SCP_BASE[@]}" "${SSH_HOST}:${RESULTS_REMOTE}/${label}.kvfill.json" "${dest}/kvfill.json" 2>/dev/null || true
}

# --- Point 1: multi-turn KV fill (2-GPU, apex) ---
for arm in canonical-plus1 seq-repair; do
    plus=1
    seq=0
    [[ "$arm" == "seq-repair" ]] && seq=1
    label="tsc-kvfill-2gpu-${arm}"
    run_bench "${SCRIPT_DIR}/pathb-romulus-2gpu-bench.sh" "$label" \
        BENCH_MODEL="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf" \
        BENCH_MULTITURN_KVFILL=1 BENCH_KV_TARGET=7500 BENCH_KV_TURN_GEN=64 BENCH_KV_FINAL_GEN=128 \
        BENCH_CURL_TIMEOUT=1200 BENCH_CTX=8192 BENCH_TS=50,50 BENCH_CTK=q8_0 BENCH_CTV=q8_0 \
        BENCH_LOAD_TIMEOUT=1800 BENCH_EXTRA="--fit off --verbose -lv 4 --reasoning off" \
        GGML_PIPELINE_PLUS="$plus" GGML_PIPELINE_MULTI_BACKEND_SEQ="$seq"
    fetch_kvfill "$label" "${LOG_DIR}/${label}"
    IFS='|' read -r verdict detail <<< "$(gate_kvfill "${LOG_DIR}/${label}/kvfill.json")"
    echo "kvfill|${arm}|apex|2gpu|${verdict}|${detail}" >> "$SUMMARY"
done

# --- Point 2: cross-arch suite (2-GPU) ---
declare -A MODELS
MODELS[llama3]="/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
MODELS[gemma3]="/mnt/models/gemma-3-12b-it-Q5_K_M.gguf"

for arch in llama3 gemma3; do
    for arm in canonical-plus1 seq-repair; do
        plus=1
        seq=0
        [[ "$arm" == "seq-repair" ]] && seq=1
        label="tsc-arch-${arch}-2gpu-${arm}"
        run_bench "${SCRIPT_DIR}/pathb-romulus-2gpu-bench.sh" "$label" \
            BENCH_MODEL="${MODELS[$arch]}" \
            "${BASE_2GPU[@]}" \
            GGML_PIPELINE_PLUS="$plus" GGML_PIPELINE_MULTI_BACKEND_SEQ="$seq"
        fetch_full "$label" "${LOG_DIR}/${label}"
        IFS='|' read -r verdict detail <<< "$(gate_suite "${LOG_DIR}/${label}" "${label}")"
        echo "crossarch|${arm}|${arch}|2gpu|${verdict}|${detail}" >> "$SUMMARY"
    done
done

# --- Point 3: 3-GPU topology (apex, fox + full suite) ---
for arm in canonical-plus1 seq-repair; do
    plus=1
    seq=0
    [[ "$arm" == "seq-repair" ]] && seq=1
    label="tsc-3gpu-apex-${arm}"
    log_line "=== bench ${label} (3gpu preset=primary-5060-3060) ==="
    env BENCH_MODEL="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf" \
        "${BASE_2GPU[@]}" \
        GGML_PIPELINE_PLUS="$plus" GGML_PIPELINE_MULTI_BACKEND_SEQ="$seq" \
        "${SCRIPT_DIR}/pathb-romulus-3gpu-bench.sh" primary-5060-3060 "$label" \
        2>&1 | tee "${LOG_DIR}/${label}.log" || true
    fetch_full "$label" "${LOG_DIR}/${label}"
    IFS='|' read -r verdict detail <<< "$(gate_suite "${LOG_DIR}/${label}" "${label}")"
    echo "3gpu|${arm}|apex|3gpu|${verdict}|${detail}" >> "$SUMMARY"
done

log_line "=== OPEN POINTS SUMMARY ==="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
log_line "full summary: ${SUMMARY}"