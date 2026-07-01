#!/usr/bin/env bash
# Phase 1c assembly line production: deploy preflight + L1 hash-defer + optional L4 n384.
#
# usage:
#   bash scripts/b6-gate-phase1c-assembly-line.sh
#   B6_PHASE1C_SKIP_MOE=1 bash scripts/b6-gate-phase1c-assembly-line.sh
#   B6_PHASE1C_SKIP_L1=1 bash scripts/b6-gate-phase1c-assembly-line.sh
#
# steps:
#   1. Deploy preflight sanity (A1 + A8 gguf paths on romulus)
#   2. L1 GGML_RPC_HASH_DEFER bisect @ n=384
#   3. MoE light CPU expert offload smoke (ncmoe 0/8/16 @ n=128)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"

A1="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
A8="/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

echo "=== Phase 1c assembly line ${STAMP} ==="

echo "--- step 1: deploy preflight (A1 MoE + A8 70B) ---"
for gguf in "$A1" "$A8"; do
    echo ">> ${gguf}"
    ssh_romulus "cd ${ROMULUS_REPO} && python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \\
      --preset b6-5gpu-g-prod --gguf '${gguf}' --ts-mode equal --phase load 2>&1 | \\
      grep -E '^(PASS:|  BENCH_TS=|  BENCH_NGL=)' || true"
done

if [[ "${B6_PHASE1C_SKIP_L1:-0}" != "1" ]]; then
    echo "--- step 2: L1 hash-defer spike ---"
    bash "${ROOT}/scripts/b6-gate-phase1c-l1-hash-defer-spike.sh"
else
    echo "--- step 2: L1 skipped (B6_PHASE1C_SKIP_L1=1) ---"
fi

if [[ "${B6_PHASE1C_SKIP_MOE:-0}" != "1" ]]; then
    echo "--- step 3: MoE light CPU expert offload ---"
    bash "${ROOT}/scripts/b6-gate-phase1c-moe-light-offload-spike.sh"
else
    echo "--- step 3: MoE offload skipped (B6_PHASE1C_SKIP_MOE=1) ---"
fi

echo "PHASE1C_ASSEMBLY_LINE_DONE stamp=${STAMP}"