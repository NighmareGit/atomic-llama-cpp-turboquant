#!/usr/bin/env bash
# 72B IQ4_XS server test for Config A (24GB ROCm + 8GB RPC).
# ~40GB weights require CPU offload; use --fit on instead of -ngl 99.
#
# usage: pathb-72b-server.sh <kimi72b|qwen72b> [label-suffix]
#
# See: scripts/pathb-72b-vram-calc.py

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
MODELS="${MODELS_ROOT:-/mnt/models}"
PRESET="${1:?kimi72b or qwen72b}"
SUFFIX="${2:-$(date +%H%M%S)}"

case "$PRESET" in
    kimi72b) MODEL="${MODELS}/Kimi-Dev-72B-IQ4_XS.gguf" ;;
    qwen72b) MODEL="${MODELS}/Qwen3-72B-Instruct.IQ4_XS.gguf" ;;
    *) echo "unknown preset: $PRESET" >&2; exit 1 ;;
esac

LABEL="${PRESET}-${SUFFIX}"
echo ">>> 72B server preset=$PRESET label=$LABEL"

BENCH_MODEL="$MODEL" \
BENCH_CTX=1024 \
BENCH_CTK=q4_0 \
BENCH_CTV=q4_0 \
BENCH_TS=10,90 \
BENCH_NGL=0 \
BENCH_GEN_TOKENS=24 \
BENCH_RUNS=1 \
BENCH_LOAD_TIMEOUT=600 \
BENCH_NO_WARMUP=1 \
BENCH_EXTRA="--fit on --fit-target 512,1024 --reasoning off" \
BENCH_LOG_DIR="${RPC_PATCH_ROOT}/patch/bench-results/pathb-runs" \
"${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh" pathb "$LABEL"