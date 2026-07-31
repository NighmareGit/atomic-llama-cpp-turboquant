#!/usr/bin/env bash
# 4-GPU canonical -ts grid (JUPITER :50053). See rpc-patch/docs/b6-gate/PLAN.md Phase 3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PHASE="grid"
TOKENS=""
SWEEP_ROOT="${ROOT}/benches/path-b-plus/b6-4gpu-ts-sweep"
SUMMARY_TSV="${SWEEP_ROOT}/sweep-summary.tsv"
RANK_FILE="${SWEEP_ROOT}/top2-rows.txt"

usage() {
    echo "usage: $0 --phase grid|confirm [--tokens N]"
    echo "  grid:    5 rows at n=128 (default) or --tokens N"
    echo "  confirm: top 2 rows by overlap_pct at n=384 (default) or --tokens N"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="${2:?}"; shift 2 ;;
        --tokens) TOKENS="${2:?}"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

declare -A GRID_TS=(
    [G0]="25,12,25,38"
    [G1]="28,12,28,32"
    [G2]="36,24,24,16"
    [G3]="22,10,18,50"
    [G4]="30,14,16,40"
)

run_row() {
    local row="$1"
    local ts="$2"
    local ntok="$3"
    local out="${SWEEP_ROOT}/${row}"
    echo "=== ts sweep ${row} ts=${ts} n=${ntok} ==="
    BENCH_GEN_TOKENS="$ntok" \
    BENCH_TS="$ts" \
    BENCH_RPC_ENDPOINT="192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053" \
    PROFILER_OUT_DIR="$out" \
    bash "${ROOT}/scripts/b6-gate-run-remote.sh" b6-4gpu-g
}

read_diagnose() {
    local dir="$1"
    python3 - <<'PY' "$dir"
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "telemetry" / "diagnose.json"
if not p.is_file():
    print("NA\tNA\tNA\tNA\tNA\tNA")
    raise SystemExit(0)
raw = json.loads(p.read_text())
diag = raw.get("diagnose", raw)
def g(k, default="NA"):
    v = diag.get(k, default)
    return v if v is not None else default
print(f"{g('overlap_pct')}\t{g('drain_flush_ms')}\t{g('stall_ratio')}\t{g('straggler_backend')}\t{g('straggler_ms_per_token')}\t{g('G_tps', diag.get('G_tps_wall', 'NA'))}")
PY
}

append_summary_header() {
    if [[ ! -f "$SUMMARY_TSV" ]]; then
        printf 'row\tts\ttokens\toverlap_pct\tdrain_flush_ms\tstall_ratio\tstraggler\tstraggler_ms\tG_tps\tout_dir\n' >"$SUMMARY_TSV"
    fi
}

append_summary_row() {
    local row="$1" ts="$2" ntok="$3" out="$4"
    IFS=$'\t' read -r overlap drain stall straggler straggler_ms g_tps < <(read_diagnose "$out")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$row" "$ts" "$ntok" "$overlap" "$drain" "$stall" "$straggler" "$straggler_ms" "$g_tps" "$out" >>"$SUMMARY_TSV"
}

mkdir -p "$SWEEP_ROOT"
append_summary_header

case "$PHASE" in
    grid)
        N="${TOKENS:-128}"
        for row in G0 G1 G2 G3 G4; do
            if pgrep -af llama-pipeline-profiler >/dev/null 2>&1; then
                echo "error: profiler already running on this host" >&2
                exit 1
            fi
            run_row "$row" "${GRID_TS[$row]}" "$N"
            append_summary_row "$row" "${GRID_TS[$row]}" "$N" "${SWEEP_ROOT}/${row}"
        done
        echo "=== grid complete: ${SUMMARY_TSV} ==="
        ;;
    confirm)
        N="${TOKENS:-384}"
        if [[ ! -f "$SUMMARY_TSV" ]]; then
            echo "error: missing ${SUMMARY_TSV}; run --phase grid first" >&2
            exit 1
        fi
        python3 - <<'PY' "$SUMMARY_TSV" "$RANK_FILE"
import csv, sys
from pathlib import Path
summary, rank_file = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
with summary.open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        try:
            overlap = float(r["overlap_pct"]) if r["overlap_pct"] not in ("NA", "") else -1.0
            drain = float(r["drain_flush_ms"]) if r["drain_flush_ms"] not in ("NA", "") else 1e18
            g = float(r["G_tps"]) if r["G_tps"] not in ("NA", "") else -1.0
        except ValueError:
            overlap, drain, g = -1.0, 1e18, -1.0
        rows.append((overlap, drain, -g, r["row"], r["ts"]))
rows.sort(reverse=True)
top = [t[3] for t in rows[:2]]
rank_file.write_text("\n".join(top) + "\n")
print("top2:", " ".join(top))
PY
        while IFS= read -r row; do
            [[ -z "$row" ]] && continue
            if pgrep -af llama-pipeline-profiler >/dev/null 2>&1; then
                echo "error: profiler already running" >&2
                exit 1
            fi
            confirm_row="${row}-confirm"
            run_row "$confirm_row" "${GRID_TS[$row]}" "$N"
            append_summary_row "$confirm_row" "${GRID_TS[$row]}" "$N" "${SWEEP_ROOT}/${confirm_row}"
        done <"$RANK_FILE"
        echo "=== confirm complete: ${SUMMARY_TSV} ==="
        ;;
    *)
        echo "error: unknown phase: $PHASE" >&2
        usage
        ;;
esac