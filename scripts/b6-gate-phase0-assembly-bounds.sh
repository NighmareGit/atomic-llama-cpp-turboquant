#!/usr/bin/env bash
# Phase 0: Tier A + frontier assembly bound report (read-only).
#
# usage:
#   bash scripts/b6-gate-phase0-assembly-bounds.sh
#
# env:
#   B6_PHASE0_MATRIX   default latest b6-model-archetype-matrix-* on romulus
#   B6_PHASE0_OUT      default benches/path-b-plus/b6-phase0-assembly-bounds-STAMP

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
OUT="${B6_PHASE0_OUT:-${ROMULUS_REPO}/benches/path-b-plus/b6-phase0-assembly-bounds-${STAMP}}"

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

MATRIX="${B6_PHASE0_MATRIX:-}"
if [[ -z "$MATRIX" ]]; then
    MATRIX="$(ssh_romulus "ls -dt ${ROMULUS_REPO}/benches/path-b-plus/b6-model-archetype-matrix-* 2>/dev/null | head -1")"
fi

echo "MATRIX=${MATRIX}"
echo "OUT=${OUT}"

ssh_romulus "mkdir -p '${OUT}'"
ssh_romulus "cd '${ROMULUS_REPO}' && python3 scripts/b6-gate-phase0-assembly-bounds.py \\
  --tier-a-matrix '${MATRIX}' \\
  --json '${OUT}/bounds.json'" 2>&1 | tee /tmp/b6-phase0-bounds.log

# Frontier / 4-GPU A3 + 6-GPU if present
for extra in \
  "${ROMULUS_REPO}/benches/path-b-plus/b6-model-archetype-matrix-20260701-172356/A3-b6-4gpu-g-triton/telemetry" \
  "${ROMULUS_REPO}/benches/path-b-plus/b6-minimax-6gpu-fit-20260701-161232/stable/telemetry" \
  "${ROMULUS_REPO}/benches/path-b-plus/b6-6gpu-frontier-spike-20260701-154500/smoke-qwen80b/telemetry" \
  ; do
    ssh_romulus "test -f '${extra}/sched-trace.jsonl'" 2>/dev/null || continue
    echo "=== extra: ${extra} ==="
    ssh_romulus "cd '${ROMULUS_REPO}' && python3 scripts/b6-gate-phase0-assembly-bounds.py '${extra}'" 2>&1 || true
done

ssh_romulus "python3 - <<'PY' '${OUT}/bounds.json' '${OUT}/work-package.md'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
reports = data['reports']
# Focus archetypes: compact MoE, MoE, dense skew, large MoE
focus = {'A1','A4','A6','A8','A12','A13'}
lines = [
  '# B+14 Phase 0 assembly bounds',
  '',
  'Read-only bound report from Tier A sched traces. Informs W1+W2 work package.',
  '',
  '| ID | G | per_multi | global_multi | global_3bk | intra_max_p50 | cross_ideal | hint |',
  '|----|---|-----------|--------------|------------|---------------|-------------|------|',
]
for r in reports:
    if r['label'] not in focus and len(reports) > 8:
        continue
    pd = r['per_decode_timeline_pct']
    gl = r['global_timeline_pct']
    ib = r['intra_token_bound']
    cb = r['cross_token_bound']
    g = r['diagnose'].get('G_tps', '?')
    lines.append(
        f\"| {r['label']} | {g} | {pd['multi_pct']}% | {gl['multi_pct']}% | {gl['multi_3_pct']}% | \"
        f\"{ib['per_token_max_multi_pct_p50']}% | {cb.get('cross_token_ideal_multi_pct','?')}% | \"
        f\"{r['work_package_hint'][:40]}... |\"
    )
lines += [
  '',
  '## Work package recommendation (cluster-fill / big models)',
  '',
  '1. **W2 cross-decode wavefront** (primary): global_3bk near zero on 3-GPU — must stack tokens to fill remus+triton+7900.',
  '2. **W1 intra-token handoff** (secondary): intra_max_p50 caps W1-only; still required for gather/RPC overlap.',
  '3. **L4 split granularity study** (if splits/decode p50 <= 3): more handoffs for layer-heavy models (A8, A13).',
  '4. **L1 HASH defer** on RPC hot path: parallel with spike; G lever on load+gen blocking.',
  '5. **Phase 1 factorial** A0/A1/A2/A3 on A1 n=64 before n=384 commit.',
  '',
  '**Do not ship W1-only.** Bounds show per-token multi ceiling < cross-token ideal on all PASS rows.',
]
Path(sys.argv[2]).write_text(chr(10).join(lines) + chr(10))
print('wrote', sys.argv[2])
PY"

echo "PHASE0_ASSEMBLY_BOUNDS_DONE out=${OUT}"