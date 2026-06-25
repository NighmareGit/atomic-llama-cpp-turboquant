#!/usr/bin/env bash
# Path B preset test launcher (Config A: ROCm 7900 XTX client / CUDA 3060 Ti RPC worker).
#
# usage:
#   pathb-test.sh <preset> [run-label-suffix]
#
# presets:
#   4b          Qwen3.5-4B-Q4_K_M (primary regression, 3 min load timeout)
#   gemma12b    gemma-4-12b-it-Q4_K_M
#   qwen27b     Qwen3.5-27B-Q5_K_M (tensor-split 2,1)
#   qwen35b     Qwen3.5-35B-A3B Q4_K_M (tensor-split 3,1)
#   gemma31b    gemma-4-31B-it-Q4_K_M (tensor-split 3,1)
#   kimi72b     Kimi-Dev-72B-IQ4_XS via llama-server + --fit on (see pathb-72b-server.sh)
#   qwen72b     Qwen3-72B-Instruct IQ4_XS via llama-server + --fit on
#   Note: 72B IQ4_XS (~40GB) exceeds 32GB GPU VRAM; must offload ~8GB+ to CPU (-ngl / --fit on).
#         Run: python3 scripts/pathb-72b-vram-calc.py for split guidance.
#   matrix-4b   run 3x 4B (q4_0 cache)
#
# Avoid: Qwen_Qwen3.5-9B-* (Unsloth/damaged), unsloth Qwen 3.5 variants.
#
# env: see pathb-run-test.sh and pathb-start-rpc.sh

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
RUNNER="${RPC_PATCH_ROOT}/scripts/pathb-run-test.sh"
MODELS="${MODELS_ROOT:-/mnt/models}"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -ge 1 ]] || usage

PRESET="$1"
SUFFIX="${2:-$(date +%H%M%S)}"

run() {
    local label="$1"
    shift
    echo ">>> preset=$PRESET label=$label"
    "$RUNNER" --label "$label" "$@"
}

case "$PRESET" in
    4b)
        run "4b-${SUFFIX}" \
            --model "${MODELS}/Qwen3.5-4B-Q4_K_M.gguf" \
            --load-timeout 180 --gen-timeout 300 \
            --ctk q4_0 --ctv q4_0 \
            --ngl 99 --split-mode layer \
            --n-predict 64
        ;;
    gemma12b)
        run "gemma12b-${SUFFIX}" \
            --model "${MODELS}/gemma-4-12b-it-Q4_K_M.gguf" \
            --load-timeout 180 --gen-timeout 360 \
            --prompt "Say hello in one short sentence." \
            --ctk q4_0 --ctv q4_0 \
            --ngl 99 --split-mode layer \
            --n-predict 32
        ;;
    qwen27b)
        run "qwen27b-${SUFFIX}" \
            --model "${MODELS}/Qwen3.5-27B-Q5_K_M.gguf" \
            --load-timeout 180 --gen-timeout 420 \
            --ctk q4_0 --ctv q4_0 \
            --ngl 80 --split-mode layer --tensor-split 2,1 \
            --ctx 4096 --n-predict 48
        ;;
    qwen35b)
        run "qwen35b-${SUFFIX}" \
            --model "${MODELS}/Qwen3.5-35B-A3B.i1-Q4_K_M.gguf" \
            --load-timeout 180 --gen-timeout 420 \
            --ctk q4_0 --ctv q4_0 \
            --ngl 70 --split-mode layer --tensor-split 3,1 \
            --ctx 4096 --n-predict 48
        ;;
    gemma31b)
        run "gemma31b-${SUFFIX}" \
            --model "${MODELS}/gemma-4-31B-it-Q4_K_M.gguf" \
            --load-timeout 180 --gen-timeout 480 \
            --ctk q4_0 --ctv q4_0 \
            --ngl 65 --split-mode layer --tensor-split 3,1 \
            --ctx 4096 --n-predict 48
        ;;
    kimi72b|qwen72b)
        # 72B: use server + auto-fit (cli crashes RPC on CUDA graph with manual -ngl)
        exec "${RPC_PATCH_ROOT}/scripts/pathb-72b-server.sh" "$PRESET" "$SUFFIX"
        ;;
    matrix-4b)
        for i in 1 2 3; do
            run "4b-m${i}-${SUFFIX}" \
                --model "${MODELS}/Qwen3.5-4B-Q4_K_M.gguf" \
                --load-timeout 180 --gen-timeout 300 \
                --ctk q4_0 --ctv q4_0 \
                --ngl 99 --n-predict 64 || exit 1
        done
        ;;
    *)
        echo "unknown preset: $PRESET" >&2
        usage
        ;;
esac