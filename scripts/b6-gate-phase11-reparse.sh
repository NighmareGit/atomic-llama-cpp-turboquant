#!/usr/bin/env bash
# Phase 1.1: re-parse existing trace artifacts (no new cluster bench time).
#
# usage: b6-gate-phase11-reparse.sh [bench-label...]
#   default: all in-scope 2-GPU triton ladder dirs

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARSE="${ROOT}/rpc-patch/scripts/pathb-rpc-trace-parse.sh"
HOT="${ROOT}/rpc-patch/scripts/pathb-hotpath-summary.sh"
DIAG="${ROOT}/scripts/llama-pipeline-diagnose.sh"
COMPARE="${ROOT}/scripts/b6-gate-bisect-compare.sh"

DEFAULT_DIRS=(
    b6-2gpu-f-triton-n384-remus-docker
    b6-2gpu-f-triton-n384-romulus-native
    b6-2gpu-f-triton-n384-no-defer
    b6-2gpu-f-triton-n384-no-partial
    b6-2gpu-f-triton-n384-no-async-copy
    b6-2gpu-f-triton-n384-b8-partial
    b6-2gpu-f-triton-guard-n128
)

declare -A GEN_TOKENS=(
    [b6-2gpu-f-triton-n384-remus-docker]=384
    [b6-2gpu-f-triton-n384-romulus-native]=384
    [b6-2gpu-f-triton-n384-no-defer]=384
    [b6-2gpu-f-triton-n384-no-partial]=384
    [b6-2gpu-f-triton-n384-no-async-copy]=384
    [b6-2gpu-f-triton-n384-b8-partial]=384
    [b6-2gpu-f-triton-guard-n128]=128
)

DIRS=("${@:-${DEFAULT_DIRS[@]}}")

resolve_canon_telem() {
    local bench_dir="$1"
    local env="${bench_dir}/env.txt"
    local kind="" out_dir=""
    if [[ -f "$env" ]]; then
        kind="$(grep -E '^client_kind=' "$env" 2>/dev/null | cut -d= -f2- || true)"
        out_dir="$(grep -E '^OUT_DIR=' "$env" 2>/dev/null | cut -d= -f2- || true)"
    fi
    [[ "$kind" == "native" && "$out_dir" == /src/* ]] && kind=cuda-docker
    [[ "$kind" == "native" && "$out_dir" != /src/* && -n "$out_dir" ]] && kind=rocm-native
    [[ -z "$kind" && "$out_dir" == /src/* ]] && kind=cuda-docker
    [[ -z "$kind" && "$bench_dir" == *romulus-native* ]] && kind=rocm-native
    [[ -z "$kind" && "$bench_dir" == *remus-docker* ]] && kind=cuda-docker
    case "$kind" in
        cuda-docker)  echo "${ROOT}/benches/path-b-plus/b6-2gpu-f-triton-n384-remus-docker/telemetry" ;;
        rocm-native)  echo "${ROOT}/benches/path-b-plus/b6-2gpu-f-triton-n384-romulus-native/telemetry" ;;
        *)            echo "" ;;
    esac
}

for label in "${DIRS[@]}"; do
    dir="${ROOT}/benches/path-b-plus/${label}"
    telem="${dir}/telemetry"
    if [[ ! -d "$telem" ]]; then
        echo "SKIP ${label} (no telemetry)" >&2
        continue
    fi
    echo "=== phase1.1 ${label} ==="
    bash "$PARSE" "$telem"
    n="${GEN_TOKENS[$label]:-384}"
    PATHB_GEN_TOKENS="$n" bash "$HOT" "$telem" >/dev/null
    canon="$(resolve_canon_telem "$dir")"
    if [[ -n "$canon" && -d "$canon" ]]; then
        bash "$DIAG" "$telem" --gen-only --overlap-target 5 --baseline "$canon" >/dev/null 2>&1 || true
    else
        bash "$DIAG" "$telem" --gen-only --overlap-target 5 >/dev/null 2>&1 || true
    fi
done

echo ""
echo "=== bisect verdicts ==="
bash "$COMPARE" "${DIRS[@]}"

echo ""
bash "${ROOT}/scripts/b6-gate-diagnose-runs.sh" "${DIRS[@]}"