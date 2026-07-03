#!/usr/bin/env bash
# Phase 1f: pairwise Plus-surface bisect (2-of-3 surfaces legacy).
# Plus=1 always. Surfaces: llama P0/P1, backend+RPC (SCHED_LEGACY).
#
# usage: pathb-plus1-tsc-pairwise-bisect.sh [arm...]
#
# docs: docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-semantic-collapse.md

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
LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-pairwise-bisect}"
mkdir -p "$LOG_DIR"

coherent_preview() {
    local log="$1"
    python3 - <<'PY' "$log"
import re, sys
text = open(sys.argv[1]).read()
previews = re.findall(r"preview='([^']*)'", text)
if not previews:
    print("unknown")
    sys.exit(0)
bad = 0
for p in previews:
    low = p.lower()
    if re.search(r'(here){4,}', low, re.I):
        bad += 1
        continue
    if re.search(r'(\w{2,})\1\1', low):
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
    local label="trace-g-2gpu-tsc-pair-${arm}"

    unset GGML_PIPELINE_SCHED_LEGACY
    unset GGML_PIPELINE_P0_FULL_SYNC
    unset GGML_PIPELINE_P1_FULL_SYNC

    case "$arm" in
        canonical) ;;
        full-legacy)
            export GGML_PIPELINE_SCHED_LEGACY=1
            export GGML_PIPELINE_P0_FULL_SYNC=1
            export GGML_PIPELINE_P1_FULL_SYNC=1
            ;;
        sched-legacy)
            export GGML_PIPELINE_SCHED_LEGACY=1
            ;;
        llama-legacy)
            export GGML_PIPELINE_P0_FULL_SYNC=1
            export GGML_PIPELINE_P1_FULL_SYNC=1
            ;;
        sched-p0-legacy)
            export GGML_PIPELINE_SCHED_LEGACY=1
            export GGML_PIPELINE_P0_FULL_SYNC=1
            ;;
        sched-p1-legacy)
            export GGML_PIPELINE_SCHED_LEGACY=1
            export GGML_PIPELINE_P1_FULL_SYNC=1
            ;;
        p0-legacy)
            export GGML_PIPELINE_P0_FULL_SYNC=1
            ;;
        p1-legacy)
            export GGML_PIPELINE_P1_FULL_SYNC=1
            ;;
        plus0)
            export GGML_PIPELINE_PLUS=0
            ;;
        *)
            echo "error: unknown arm '$arm'" >&2
            exit 1
            ;;
    esac

    local plus="${GGML_PIPELINE_PLUS:-1}"
    echo "=== arm=${arm} label=${label} ==="
    echo "  GGML_PIPELINE_PLUS=${plus}"
    echo "  SCHED_LEGACY=${GGML_PIPELINE_SCHED_LEGACY:-0}"
    echo "  P0_FULL=${GGML_PIPELINE_P0_FULL_SYNC:-0}"
    echo "  P1_FULL=${GGML_PIPELINE_P1_FULL_SYNC:-0}"

    GGML_PIPELINE_PLUS="${plus}" \
        BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_TRACE=0 \
        "$BENCH" "$label" 2>&1 | tee "${LOG_DIR}/${arm}.log"

    local gate
    gate=$(coherent_preview "${LOG_DIR}/${arm}.log")
    echo "GATE_B=${gate} arm=${arm} log=${LOG_DIR}/${arm}.log"
}

ARMS=()
if [[ $# -gt 0 ]]; then
    ARMS=("$@")
else
    ARMS=(
        canonical
        full-legacy
        sched-legacy
        llama-legacy
        sched-p0-legacy
        sched-p1-legacy
        p0-legacy
        p1-legacy
        plus0
    )
fi

for arm in "${ARMS[@]}"; do
    run_arm "$arm"
done

echo "=== pairwise bisect complete; logs in ${LOG_DIR} ==="