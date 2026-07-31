#!/usr/bin/env bash
# R5: validate generic multi-GPU RPC layout before a profiler run.
#
# usage:
#   BENCH_RPC_ENDPOINT=host:50051,host2:50052 BENCH_TS=50,50 \
#     ./scripts/llama-pipeline-r5-validate.sh
#
# env:
#   PROFILER_BIN  path to llama-pipeline-profiler (default: build/bin/Release/...)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENDPOINT="${BENCH_RPC_ENDPOINT:-}"
TS="${BENCH_TS:-}"
PROFILER="${PROFILER_BIN:-${ROOT}/build/bin/Release/llama-pipeline-profiler}"

if [[ -z "$ENDPOINT" ]]; then
    echo "error: BENCH_RPC_ENDPOINT required" >&2
    exit 1
fi

if [[ ! -f "$PROFILER" ]] && [[ ! -x "$PROFILER" ]]; then
    PROFILER="${ROOT}/build/bin/llama-pipeline-profiler"
fi
if [[ ! -f "$PROFILER" ]] && [[ ! -x "$PROFILER" ]]; then
    echo "error: llama-pipeline-profiler not found" >&2
    exit 1
fi

ARGS=(--validate-rpc -rpc "$ENDPOINT")
[[ -n "$TS" ]] && ARGS+=(-ts "$TS")

cd "$ROOT"
exec "$PROFILER" "${ARGS[@]}"