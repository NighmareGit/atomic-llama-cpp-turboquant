#!/usr/bin/env bash
# 6-GPU frontier spike: smoke load + stability gen with CKV turbo3 / CKD q8_0.
#
# usage:
#   B6_JUPITER_PASS='...' bash scripts/b6-gate-6gpu-frontier-spike.sh
#
# env:
#   B6_FRONTIER_MODEL   default /mnt/models/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf
#   B6_FRONTIER_MODEL_2 optional second model (e.g. MiniMax 139B)
#   B6_ROMULUS_HOST     default user@192.168.8.108

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-user@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-6gpu-frontier-spike-${STAMP}"

MODEL_SMOKE="${B6_FRONTIER_MODEL:-/mnt/models/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf}"
MODEL_HEAVY="${B6_FRONTIER_MODEL_2:-/mnt/models/MiniMax-M2.7-REAP-139B-A10B-Q4_K_M.gguf}"
CTK="${BENCH_CTK:-q8_0}"
CTV="${BENCH_CTV:-turbo3}"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

echo "OUT_BASE=${OUT_BASE}"

echo "=== phase 0: jupiter rpc ==="
if [[ -n "${B6_JUPITER_PASS:-}" ]]; then
    B6_JUPITER_PASS="$B6_JUPITER_PASS" bash "${ROOT}/scripts/b6-gate-jupiter-remote.sh" start-rpc || true
else
    echo "skip jupiter start (set B6_JUPITER_PASS); probing..."
    bash "${ROOT}/scripts/b6-gate-jupiter-remote.sh" probe || true
fi

run_case() {
    local tag="$1"
    local model="$2"
    local n_gen="$3"
    local mode="$4"
    local out="${OUT_BASE}/${tag}"
    echo "=== ${tag}: ${model} n=${n_gen} mode=${mode} ==="
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      export B6_JUPITER_USER='${B6_JUPITER_USER:-nightmare}' B6_JUPITER_PASS='${B6_JUPITER_PASS:-}' && \\
      source scripts/b6-gate-6gpu-env.sh && b6_6gpu_production_env && b6_6gpu_frontier_kv_env && \\
      B6_6GPU_DUAL_SOCKET=0 GGML_RPC_DUAL_SOCKET=0 \\
      BENCH_MODEL='${model}' \\
      BENCH_GEN_TOKENS=${n_gen} \\
      PROFILER_MODE=${mode} \\
      PROFILER_OUT_DIR=${out} \\
      PROFILER_LOCAL=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      PATHB_VRAM_PREFLIGHT=1 B6_GATE_PRESET=b6-6gpu-g \\
      bash scripts/b6-gate-run-remote.sh b6-6gpu-g --no-warmup --skip-rpc-validate \\
        -ctk ${CTK} -ctv ${CTV}" 2>&1 | tail -8
    ssh_romulus "test -f '${out}/result.jsonl' && python3 -c \"
import json
r=open('${out}/result.jsonl').read().strip().splitlines()[-1]
d=json.loads(r)
print('${tag}', 'G', round(d.get('G_tps',0),2), 'tokens', d.get('n_gen', d.get('n_gen_tokens','?')))
\" || echo '${tag} FAIL (no result.jsonl)'"
}

# smoke: short gen proves load + split without long trace cost
run_case "smoke-qwen80b" "$MODEL_SMOKE" 16 "spike-check"
# stability: full trace window
run_case "stable-qwen80b" "$MODEL_SMOKE" 128 "trace"

if ssh_romulus "test -f '${MODEL_HEAVY}'"; then
    run_case "smoke-minimax139b" "$MODEL_HEAVY" 16 "spike-check"
    run_case "stable-minimax139b" "$MODEL_HEAVY" 128 "trace"
else
    echo "skip MiniMax (missing ${MODEL_HEAVY})"
fi

echo ""
echo "SPIKE_DONE dir=${OUT_BASE}"
echo "Review: result.jsonl, telemetry/diagnose.json, env.txt per case"