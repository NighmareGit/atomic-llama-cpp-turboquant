#!/usr/bin/env bash
# Overlap grill: GPU-class skew experiment (3-GPU A/B).
#
# Swaps romulus 3060 docker (slow straggler) for triton 3090 on the second RPC hop.
# Same client (7900), same n=384, dual-socket ON. Compares overlap vs straggler shift.
#
# usage:
#   bash scripts/b6-gate-overlap-gpu-skew.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-overlap-gpu-skew-${STAMP}"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

summarize_dir() {
    local tag="$1" dir="$2"
    ssh_romulus "python3 -c \"
import json, re, sys
tag, path = sys.argv[1], sys.argv[2]
d = json.load(open(path + '/telemetry/diagnose.json'))
print(f'{tag} G={d[\\\"G_tps\\\"]:.2f} overlap={d[\\\"overlap_pct\\\"]}% stall={d[\\\"stall_ratio\\\"]:.3f} '
      f'straggler=backend{d.get(\\\"straggler_backend\\\")} @{d.get(\\\"straggler_ms_per_token\\\")}ms '
      f'blocking={d[\\\"blocking_ms\\\"]:.0f}')
try:
    txt = open(path + '/telemetry/trace-summary.txt').read()
    for m in re.finditer(r'backend(\\\\d+) splits=(\\\\d+) ms=([\\\\d.]+)', txt):
        print(f'  backend{m.group(1)} split_total_ms={float(m.group(3)):.2f}')
except FileNotFoundError:
    pass
\" \"$tag\" \"$dir\""
}

run_case() {
    local tag="$1"
    local label="$2"
    local out="${OUT_BASE}/${tag}"
    echo "=== ${tag}: ${label} ==="
    ssh_romulus "cd ${ROMULUS_REPO} && \\
      GGML_RPC_DUAL_SOCKET=1 B6_PERF_AUTO=0 \\
      BENCH_GEN_TOKENS=384 \\
      PROFILER_OUT_DIR=${out} \\
      PROFILER_LOCAL=1 \\
      PROFILER_BIN=${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler \\
      LD_LIBRARY_PATH=${ROMULUS_REPO}/build-rocm-docker/bin:/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh ${label} --no-warmup --skip-rpc-validate" 2>&1 | tail -4
    summarize_dir "$tag" "$out"
}

echo "OUT_BASE=${OUT_BASE}"

run_case "A-3gpu-5060-3060" "b6-3gpu-g"
run_case "B-3gpu-5060-3090" "b6-3gpu-g-triton"

echo ""
echo "=== GPU skew verdict ==="
echo "If overlap_pct flat while straggler_ms and per-backend split_ms shift -> GPU class does NOT explain overlap ceiling."
echo "SKEW_DONE dir=${OUT_BASE}"