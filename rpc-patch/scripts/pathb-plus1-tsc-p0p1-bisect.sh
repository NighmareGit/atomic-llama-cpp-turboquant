#!/usr/bin/env bash
# Phase 1f: Tier-0 Plus P0/P1 split bisect.
# Requires romulus ROCm client rebuilt with GGML_PIPELINE_P0/P1_FULL_SYNC knobs.
#
# usage: pathb-plus1-tsc-p0p1-bisect.sh [arm...]
#   arms: p0-legacy p1-legacy p0-p1-legacy canonical
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
LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-p0p1-bisect}"
mkdir -p "$LOG_DIR"

run_arm() {
    local arm="$1"
    local label="trace-g-2gpu-tsc-p0p1-${arm}"

    unset GGML_PIPELINE_P0_FULL_SYNC
    unset GGML_PIPELINE_P1_FULL_SYNC

    case "$arm" in
        canonical) ;;
        p0-legacy)    export GGML_PIPELINE_P0_FULL_SYNC=1 ;;
        p1-legacy)    export GGML_PIPELINE_P1_FULL_SYNC=1 ;;
        p0-p1-legacy)
            export GGML_PIPELINE_P0_FULL_SYNC=1
            export GGML_PIPELINE_P1_FULL_SYNC=1
            ;;
        *)
            echo "error: unknown arm '$arm'" >&2
            exit 1
            ;;
    esac

    echo "=== arm=${arm} label=${label} ==="
    echo "  GGML_PIPELINE_PLUS=1"
    echo "  GGML_PIPELINE_P0_FULL_SYNC=${GGML_PIPELINE_P0_FULL_SYNC:-0}"
    echo "  GGML_PIPELINE_P1_FULL_SYNC=${GGML_PIPELINE_P1_FULL_SYNC:-0}"

    GGML_PIPELINE_PLUS=1 \
        BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_TRACE=0 \
        "$BENCH" "$label" 2>&1 | tee "${LOG_DIR}/${arm}.log"

    echo "BENCH_DONE arm=${arm} log=${LOG_DIR}/${arm}.log"
}

ARMS=()
if [[ $# -gt 0 ]]; then
    ARMS=("$@")
else
    ARMS=(canonical p0-legacy p1-legacy p0-p1-legacy)
fi

for arm in "${ARMS[@]}"; do
    run_arm "$arm"
done

echo "=== P0/P1 bisect complete; logs in ${LOG_DIR} ==="