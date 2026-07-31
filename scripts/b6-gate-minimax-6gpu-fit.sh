#!/usr/bin/env bash
# Exclusive 6-GPU MiniMax load: live VRAM preflight -> adjusted TS/ngl -> smoke + stability.
# Do not run Tier A matrix or other cluster benches while this script is active.
#
# usage:
#   B6_JUPITER_PASS=... bash scripts/b6-gate-minimax-6gpu-fit.sh
#
# env:
#   B6_MINIMAX_MODEL  default /mnt/models/MiniMax-M2.7-REAP-139B-A10B-Q4_K_M.gguf
#   B6_MINIMAX_CTX    default 4096 (lower than 8192 to ease KV on frontier)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-user@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
MODEL="${B6_MINIMAX_MODEL:-/mnt/models/MiniMax-M2.7-REAP-139B-A10B-Q4_K_M.gguf}"
CTX="${B6_MINIMAX_CTX:-4096}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-minimax-6gpu-fit-${STAMP}"
CTK="${BENCH_CTK:-q8_0}"
CTV="${BENCH_CTV:-turbo3}"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

echo "=== MiniMax 6-GPU exclusive fit (cluster reserved) ==="
echo "OUT_BASE=${OUT_BASE}"
echo "MODEL=${MODEL}"

ssh_romulus "cd ${ROMULUS_REPO} && \\
  export B6_JUPITER_USER='${B6_JUPITER_USER:-nightmare}' B6_JUPITER_PASS='${B6_JUPITER_PASS:-}' && \\
  B6_JUPITER_USER='${B6_JUPITER_USER:-nightmare}' B6_JUPITER_PASS='${B6_JUPITER_PASS:-}' \\
    bash scripts/b6-gate-jupiter-remote.sh probe" || true

# Live VRAM plan on romulus (all 6 devices must be idle).
FIT_LOG="${OUT_BASE}/vram-preflight.txt"
ssh_romulus "cd '${ROMULUS_REPO}' && mkdir -p '${OUT_BASE}' && \\
  export B6_JUPITER_USER='${B6_JUPITER_USER:-nightmare}' B6_JUPITER_PASS='${B6_JUPITER_PASS:-}' && \\
  python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py \\
    --preset b6-6gpu-g --gguf '${MODEL}' --ctx ${CTX} --live --strict \\
    2>&1 | tee '${FIT_LOG}'"

# Parse recommended TS / ngl from preflight output on romulus.
read -r BENCH_TS BENCH_NGL < <(ssh_romulus "cd '${ROMULUS_REPO}' && python3 - <<'PY' '${FIT_LOG}'
import re, sys
text = open(sys.argv[1]).read()
ts = re.search(r'BENCH_TS=([\\d,]+)', text)
ngl = re.search(r'BENCH_NGL=(\\d+)', text)
if not ts or not ngl:
    raise SystemExit('preflight parse failed')
print(ts.group(1), ngl.group(1))
PY")

echo "live fit: TS=${BENCH_TS} NGL=${BENCH_NGL} ctx=${CTX}"

run_case() {
    local tag="$1" n_gen="$2" mode="$3"
    local out="${OUT_BASE}/${tag}"
    echo "=== ${tag} n=${n_gen} mode=${mode} ==="
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      export B6_JUPITER_USER='${B6_JUPITER_USER:-nightmare}' B6_JUPITER_PASS='${B6_JUPITER_PASS:-}' && \\
      source scripts/b6-gate-6gpu-env.sh && b6_6gpu_base_env && b6_6gpu_frontier_kv_env && \\
      B6_6GPU_DUAL_SOCKET=0 GGML_RPC_DUAL_SOCKET=0 B6_PERF_AUTO=0 && \\
      BENCH_MODEL='${MODEL}' BENCH_TS='${BENCH_TS}' BENCH_NGL='${BENCH_NGL}' \\
      BENCH_GEN_TOKENS=${n_gen} PROFILER_MODE=${mode} \\
      BENCH_N_BATCH=256 BENCH_N_UBATCH=256 \\
      PROFILER_OUT_DIR='${out}' PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 PATHB_VRAM_PREFLIGHT=0 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh b6-6gpu-g --no-warmup --skip-rpc-validate \\
        -ctk ${CTK} -ctv ${CTV} -b 256 -ub 256 -ngl ${BENCH_NGL}" 2>&1 | tail -10
    ssh_romulus "python3 -c \"
import json
p='${out}/telemetry/diagnose.json'
try:
  d=json.load(open(p))
  print('${tag}', 'G', round(d['G_tps'],2), 'ov', d['overlap_pct'], 'straggler', d['straggler_backend'])
except FileNotFoundError:
  print('${tag} FAIL no diagnose')
\""
}

run_case "smoke" 8 "spike-check"
run_case "stable" 64 "trace"

echo ""
echo "MINIMAX_FIT_DONE dir=${OUT_BASE}"
echo "TS=${BENCH_TS} NGL=${BENCH_NGL} — safe to start Tier A 3-GPU matrix after reviewing results."