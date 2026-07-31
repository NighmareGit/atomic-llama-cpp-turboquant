#!/usr/bin/env bash
# T2 dual-CUDA baselines for AllReduce wayfinder ticket 09.
# Requires: same-process 2x CUDA GPUs, release-ish llama-bench with AR/split traces.
#
# Usage:
#   export LLAMA_BENCH=/path/to/llama-bench
#   export MODEL=/path/to/model.gguf
#   bash scripts/allreduce-t2-baseline.sh
#
# Optional:
#   OUT_DIR=benches/allreduce-baselines/<label>
#   DEV=CUDA0,CUDA1
#   NGL=99  PP=128  TG=64  REPS=3
#   TS=55,45

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/benches/allreduce-baselines/t2-$(hostname)-$(date +%Y%m%d)}"
BIN="${LLAMA_BENCH:-}"
MODEL="${MODEL:-}"
DEV="${DEV:-CUDA0/CUDA1}"
NGL="${NGL:-99}"
PP="${PP:-128}"
TG="${TG:-64}"
REPS="${REPS:-3}"
TS="${TS:-55,45}"

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "Set LLAMA_BENCH to an executable llama-bench (CUDA multi-GPU build)." >&2
  exit 1
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "Set MODEL to a GGUF path (allow-listed: gemma4 / qwen35moe / qwen3next)." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "BIN=$BIN"
echo "MODEL=$MODEL"
echo "DEV=$DEV"

# Device sanity
"$BIN" --list-devices 2>&1 | tee "$OUT_DIR/devices.txt" || true
n_cuda=$("$BIN" --list-devices 2>&1 | grep -cE 'CUDA[0-9]' || true)
if [[ "${n_cuda:-0}" -lt 2 ]]; then
  echo "ERROR: need >=2 CUDA devices visible to llama-bench (found ${n_cuda:-0})." >&2
  exit 1
fi

COMMON=( -m "$MODEL" -ngl "$NGL" -fa on -ctk f16 -ctv f16 -dev "$DEV" -p "$PP" -n "$TG" -r "$REPS" )

run_one() {
  local label="$1"
  shift
  echo "=== $label ==="
  # remaining args are llama-bench flags only (env vars set by caller)
  "$BIN" "${COMMON[@]}" "$@" 2>&1 | tee "$OUT_DIR/${label}.log"
}

# Layer control
GGML_SCHED_TRACE=1 GGML_SCHED_TRACE_FILE="$OUT_DIR/layer-sched.jsonl" \
  run_one layer-sm -sm layer -ts "$TS"

# Tensor provider matrix
for AR in nccl internal none; do
  : > "$OUT_DIR/tensor-${AR}-ar.jsonl"
  : > "$OUT_DIR/tensor-${AR}-sched.jsonl"
  GGML_CUDA_ALLREDUCE="$AR" \
  GGML_ALLREDUCE_TRACE=1 \
  GGML_ALLREDUCE_TRACE_FILE="$OUT_DIR/tensor-${AR}-ar.jsonl" \
  GGML_SCHED_TRACE=1 \
  GGML_SCHED_TRACE_FILE="$OUT_DIR/tensor-${AR}-sched.jsonl" \
  run_one "tensor-${AR}" -sm tensor
done

# Quick AR summary
python3 - <<'PY' "$OUT_DIR" || true
import json, sys, collections, statistics
from pathlib import Path
out = Path(sys.argv[1])
for p in sorted(out.glob("tensor-*-ar.jsonl")):
    rows = []
    for line in p.open():
        if '"event":"allreduce"' not in line and '"event": "allreduce"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("event") != "allreduce":
            continue
        rows.append(o)
    if not rows:
        print(f"{p.name}: no allreduce events")
        continue
    by = collections.defaultdict(list)
    for o in rows:
        by[o.get("provider","?")].append(o.get("duration_us", 0))
    print(f"\n{p.name}: n={len(rows)}")
    for prov, xs in sorted(by.items()):
        xs = sorted(xs)
        p50 = xs[len(xs)//2]
        print(f"  provider={prov} p50_us={p50} mean_us={statistics.mean(xs):.0f} n={len(xs)}")
PY

echo "DONE. Logs in $OUT_DIR"
echo "Append results to docs/research/allreduce-multi-topology-baselines.md and resolve ticket 09."
