#!/usr/bin/env bash
# Seed regression.jsonl from offline telemetry baselines (P0-1 dev fallback).
#
# usage: ./scripts/llama-pipeline-import-baseline.sh [--regression-file PATH]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REGRESSION="${REGRESSION_FILE:-${ROOT}/benches/path-b-plus/regression.jsonl}"
DIAGNOSE="${ROOT}/scripts/llama-pipeline-diagnose.sh"
REGRESS="${ROOT}/scripts/llama-pipeline-regression.sh"
SAMPLE="${ROOT}/scripts/llama-pipeline-trace-sample.sh"

import_telemetry() {
    local label="$1"
    local telemetry="$2"
    local topology="${3:-}"
    local mode="${4:-trace}"
    local plus="${5:-1}"
    local client="${6:-native}"

    if [[ ! -d "$telemetry" ]]; then
        echo "skip ${label}: missing ${telemetry}"
        return 0
    fi

    echo "=== import ${label} ==="
    if [[ -x "$DIAGNOSE" || -f "$DIAGNOSE" ]]; then
        bash "$DIAGNOSE" "$telemetry" --gen-only 2>/dev/null || true
    fi
    if [[ -f "${telemetry}/diagnose.json" ]]; then
        bash "$REGRESS" "$telemetry" --label "$label" --regression-file "$REGRESSION" \
            --out-dir "$(dirname "$telemetry")" --topology "$topology" --mode "$mode" \
            --plus "$plus" --client-kind "$client" || true
        bash "$SAMPLE" "$telemetry" --every 10 2>/dev/null || true
    fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

if [[ "${1:-}" == "--regression-file" ]]; then
    REGRESSION="${2:?}"
fi

mkdir -p "$(dirname "$REGRESSION")"
echo "importing baselines -> ${REGRESSION}"

import_telemetry "trace-f-2gpu-plus" \
    "${ROOT}/docs/cuda-windows-5070ti/benchmarks/trace-f-2gpu-plus/telemetry" \
    "2gpu" "trace" 1 "native"

import_telemetry "trace-f-3gpu-plus" \
    "${ROOT}/docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-plus/telemetry" \
    "3gpu" "trace" 1 "native"

import_telemetry "trace-f-3gpu-legacy" \
    "${ROOT}/docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-legacy/telemetry" \
    "3gpu" "trace" 0 "native"

# P0-1: 4-GPU hotpath summary only (full jsonl on romulus host)
PY_BIN="${PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
    command -v python3 >/dev/null 2>&1 && PY_BIN=python3 || PY_BIN=python
fi
HOTPATH="${ROOT}/rpc-patch/patch/bench-results/cluster-4gpu-primary/romulus-host/trace-g-4gpu-primary-trace-summary.txt"
if [[ -f "$HOTPATH" ]]; then
    echo "=== import trace-g-4gpu-primary-trace (hotpath summary) ==="
    "$PY_BIN" - "$REGRESSION" "$HOTPATH" <<'PY'
import json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path

regression = Path(sys.argv[1])
summary = Path(sys.argv[2]).read_text(encoding="utf-8")
overlap = re.search(r"assembly_overlap_count=(\d+).*overlap_pct=([\d.]+)", summary)
drain = re.search(r"drain_flush_ms=([\d.]+)", summary)
split = re.search(r"avg_us=([\d.]+)", summary)
record = {
    "version": 1,
    "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "label": "trace-g-4gpu-primary-trace",
    "git_sha": "833ad4429",
    "client_kind": "http",
    "trace_dir": "rpc-patch/patch/bench-results/rpc-server-bench/trace-g-4gpu-primary-trace/telemetry",
    "topology": "4gpu-primary",
    "mode": "trace",
    "GGML_PIPELINE_PLUS": 1,
    "G_tps": 37.0,
    "gate_s5": "PASS",
    "gate_b6": "FAIL",
    "overlap_pct": float(overlap.group(2)) if overlap else 0.1,
    "assembly_overlap_count": int(overlap.group(1)) if overlap else 1075,
    "stall_ratio": None,
    "drain_flush_ms": float(drain.group(1)) if drain else None,
    "split_ms_per_token": 20.71,
    "source": "hotpath-summary-offline",
    "diagnose_version": 1,
}
regression.parent.mkdir(parents=True, exist_ok=True)
with regression.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, separators=(",", ":")) + "\n")
print(f"regression append (hotpath) -> {regression}")
PY
fi

echo "IMPORT_BASELINE_DONE file=${REGRESSION}"
wc -l <"$REGRESSION" 2>/dev/null || true