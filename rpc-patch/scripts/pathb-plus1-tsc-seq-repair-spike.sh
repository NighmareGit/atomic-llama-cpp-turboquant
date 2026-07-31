#!/usr/bin/env bash
# Spike: Plus=1 TSC repair via GGML_PIPELINE_MULTI_BACKEND_SEQ=1 (R1+R2 bundle).
# Compares canonical Plus=1 stutter vs seq-repair vs sched-p0-legacy reference.
#
# usage: pathb-plus1-tsc-seq-repair-spike.sh [label-prefix]
# logs: /tmp/plus1-tsc-seq-repair-spike/

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

BENCH="${SCRIPT_DIR}/pathb-romulus-2gpu-bench.sh"
LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-seq-repair-spike}"
mkdir -p "$LOG_DIR"

PREFIX="${1:-trace-g-2gpu-seq-spike}"
BASE_EXTRA="--fit off --verbose -lv 4 --reasoning off"
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"

gate_b() {
    python3 - <<'PY' "$1"
import re, sys
text = open(sys.argv[1]).read()
previews = re.findall(r"preview='([^']*)'", text)
if not previews:
    print("unknown")
    raise SystemExit(0)
bad = 0
for p in previews:
    low = p.lower()
    if re.search(r'(here){4,}', low, re.I):
        bad += 1
        continue
    if re.search(r'(\w{2,})\1\1', low):
        bad += 1
        continue
    if re.search(r'(!){6,}', p):
        bad += 1
        continue
    if 'pangram' in low or 'alphabet' in low or 'typing practice' in low:
        continue
    if len(p.strip()) < 40:
        bad += 1
        continue
    if p.count('Here') >= 6:
        bad += 1
        continue
print("coherent" if bad == 0 else "stutter")
PY
}

run_arm() {
    local arm="$1"
    local label="${PREFIX}-${arm}"
    local plus="$2"
    local seq="$3"
    local p0="$4"
    local sched="$5"

    echo "=== arm=${arm} plus=${plus} seq=${seq} p0=${p0} sched=${sched} ==="

    BENCH_MODEL="${BENCH_MODEL:-$MODEL}" \
    BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 \
    BENCH_GEN_TOKENS=64 BENCH_RUNS=1 BENCH_TRACE=0 \
    BENCH_LOAD_TIMEOUT=1200 BENCH_NCMOE= BENCH_EXTRA="$BASE_EXTRA" \
    GGML_PIPELINE_PLUS="$plus" \
    GGML_PIPELINE_MULTI_BACKEND_SEQ="$seq" \
    GGML_PIPELINE_P0_FULL_SYNC="$p0" \
    GGML_PIPELINE_SCHED_LEGACY="$sched" \
        "$BENCH" "$label" 2>&1 | tee "${LOG_DIR}/${arm}.log" || true

    local gate
    gate=$(gate_b "${LOG_DIR}/${arm}.log")
    echo "GATE_B=${gate} arm=${arm}"
    echo "${arm}|plus=${plus}|seq=${seq}|p0=${p0}|sched=${sched}|${gate}" >> "${LOG_DIR}/summary.tsv"
}

SUMMARY="${LOG_DIR}/summary.tsv"
echo "arm|config|gate_b" > "$SUMMARY"

# canonical Plus=1 (expect stutter)
run_arm "canonical-plus1" 1 0 0 0

# spike repair (expect coherent - equivalent to sched-p0-legacy)
run_arm "seq-repair" 1 1 0 0

# proven reference
run_arm "sched-p0-legacy" 1 0 1 1

# llama3 generic control (no GDN)
BENCH_MODEL="/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" \
    run_arm "llama3-seq-repair" 1 1 0 0

echo "=== seq-repair spike complete; summary: ${SUMMARY} ==="