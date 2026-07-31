#!/usr/bin/env bash
# Decompose SCHED_LEGACY (S) with P0_FULL_SYNC=1 fixed (Plus=1).
# Answers: which S component is necessary beyond P0 for coherence?
#
# usage: pathb-plus1-tsc-s-factorial.sh [arm...]
#
# docs: docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-ROOTCAUSE.md

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
LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-s-factorial}"
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
    local label="trace-g-2gpu-tsc-sfact-${arm}"

    unset GGML_PIPELINE_SCHED_LEGACY
    unset GGML_PIPELINE_P0_FULL_SYNC
    unset GGML_PIPELINE_P1_FULL_SYNC
    unset GGML_PIPELINE_BARRIER_PARTIAL
    unset GGML_RPC_EVENT_DEFER_BARRIER
    unset GGML_RPC_GET_TENSOR_DEFER
    unset GGML_SCHED_MOE_ASYNC_COPY

    export GGML_PIPELINE_P0_FULL_SYNC=1

    case "$arm" in
        p0-only) ;;
        sched-legacy)
            export GGML_PIPELINE_SCHED_LEGACY=1
            ;;
        no-partial)
            export GGML_PIPELINE_BARRIER_PARTIAL=0
            ;;
        no-event)
            export GGML_RPC_EVENT_DEFER_BARRIER=0
            ;;
        no-get)
            export GGML_RPC_GET_TENSOR_DEFER=0
            ;;
        no-moe-copy)
            export GGML_SCHED_MOE_ASYNC_COPY=0
            ;;
        no-get-no-event)
            export GGML_RPC_GET_TENSOR_DEFER=0
            export GGML_RPC_EVENT_DEFER_BARRIER=0
            ;;
        no-get-no-partial)
            export GGML_RPC_GET_TENSOR_DEFER=0
            export GGML_PIPELINE_BARRIER_PARTIAL=0
            ;;
        all-s-off)
            export GGML_PIPELINE_BARRIER_PARTIAL=0
            export GGML_RPC_EVENT_DEFER_BARRIER=0
            export GGML_RPC_GET_TENSOR_DEFER=0
            export GGML_SCHED_MOE_ASYNC_COPY=0
            ;;
        *)
            echo "error: unknown arm '$arm'" >&2
            exit 1
            ;;
    esac

    echo "=== arm=${arm} label=${label} ==="
    echo "  P0_FULL=1 SCHED_LEGACY=${GGML_PIPELINE_SCHED_LEGACY:-0}"
    echo "  BARRIER_PARTIAL=${GGML_PIPELINE_BARRIER_PARTIAL:-1}"
    echo "  EVENT_DEFER=${GGML_RPC_EVENT_DEFER_BARRIER:-1}"
    echo "  GET_DEFER=${GGML_RPC_GET_TENSOR_DEFER:-1}"
    echo "  MOE_ASYNC=${GGML_SCHED_MOE_ASYNC_COPY:-1}"

    GGML_PIPELINE_PLUS=1 \
        BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_TRACE=0 \
        "$BENCH" "$label" 2>&1 | tee "${LOG_DIR}/${arm}.log" || true

    local gate
    gate=$(coherent_preview "${LOG_DIR}/${arm}.log")
    echo "GATE_B=${gate} arm=${arm} log=${LOG_DIR}/${arm}.log"
}

ARMS=()
if [[ $# -gt 0 ]]; then
    ARMS=("$@")
else
    ARMS=(
        p0-only
        no-get
        no-event
        no-partial
        no-moe-copy
        no-get-no-event
        no-get-no-partial
        all-s-off
        sched-legacy
    )
fi

for arm in "${ARMS[@]}"; do
    run_arm "$arm"
done

echo "=== S factorial complete; logs in ${LOG_DIR} ==="