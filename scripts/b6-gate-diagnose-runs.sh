#!/usr/bin/env bash
# Post-run diagnosis matrix for B+6 gate runs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARSE="${ROOT}/rpc-patch/scripts/pathb-rpc-trace-parse.sh"
HOTPATH="${ROOT}/rpc-patch/scripts/pathb-hotpath-summary.sh"
MATRIX="${ROOT}/benches/path-b-plus/b6-diagnosis-matrix.tsv"

usage() {
    echo "usage: $0 <bench-dir-or-label> [more...]"
    echo "  dirs relative to benches/path-b-plus/ or absolute paths"
    exit 1
}

[[ $# -ge 1 ]] || usage

resolve_dir() {
    local arg="$1"
    if [[ -d "$arg" ]]; then
        echo "$(cd "$arg" && pwd)"
        return
    fi
    local cand="${ROOT}/benches/path-b-plus/${arg}"
    if [[ -d "$cand" ]]; then
        echo "$(cd "$cand" && pwd)"
        return
    fi
    for hit in "${ROOT}/benches/path-b-plus/${arg}"*; do
        if [[ -d "$hit" ]]; then
            echo "$(cd "$hit" && pwd)"
            return
        fi
    done
    echo "error: bench dir not found: $arg" >&2
    return 1
}

classify_verdict() {
    python3 - <<'PY' "$@"
import sys
overlap = float(sys.argv[1]) if sys.argv[1] not in ("NA", "") else 0.0
drain = float(sys.argv[2]) if sys.argv[2] not in ("NA", "") else 0.0
straggler_ms = float(sys.argv[3]) if sys.argv[3] not in ("NA", "") else 0.0
drain_ref = 2300.0
straggler_high = straggler_ms >= 8.0
drain_high = drain >= 10000.0
overlap_low = overlap < 0.5
if straggler_high and drain_high:
    print("MIXED")
elif straggler_high and overlap_low:
    print("STRAGGLER_DOMINANT")
elif drain_high:
    print("DRAIN_DOMINANT")
else:
    print("MIXED")
PY
}

extract_row() {
    local dir="$1"
    local label
    label="$(basename "$dir")"
    local telem="${dir}/telemetry"
    if [[ -d "$telem" ]]; then
        if [[ ! -f "${telem}/trace-summary.txt" ]] || \
           { [[ -f "${telem}/rpc-trace.jsonl" ]] && [[ "${telem}/rpc-trace.jsonl" -nt "${telem}/trace-summary.txt" ]]; }; then
            bash "$PARSE" "$telem" >/dev/null 2>&1 || true
        fi
        bash "$HOTPATH" "$telem" >/dev/null 2>&1 || true
    fi
    python3 - <<'PY' "$dir" "$label"
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1])
label = sys.argv[2]
diag_path = d / "telemetry" / "diagnose.json"
diag = {}
if diag_path.is_file():
    raw = json.loads(diag_path.read_text())
    diag = raw.get("diagnose", raw)
def g(k, default="NA"):
    v = diag.get(k, default)
    return default if v is None else v
ms_tok = []
summary = d / "telemetry" / "trace-summary.txt"
if summary.is_file():
    for line in summary.read_text().splitlines():
        m = re.search(r"backend(\d+).*?(\d+\.?\d*)\s*ms/tok", line)
        if m:
            ms_tok.append(f"b{m.group(1)}={m.group(2)}")
ms_join = ";".join(ms_tok) if ms_tok else "NA"
print("\t".join([
    label,
    str(g("overlap_pct")),
    str(g("drain_flush_ms")),
    str(g("blocking_rpc_count")),
    str(g("stall_ratio")),
    str(g("straggler_backend")),
    str(g("straggler_ms_per_token")),
    str(g("assembly_overlap_count")),
    str(g("G_tps", g("G_tps_wall"))),
    ms_join,
    str(d),
]))
PY
}

printf 'label\toverlap_pct\tdrain_flush_ms\tblocking_rpc\tstall_ratio\tstraggler\tstraggler_ms\tassembly_overlap\tG_tps\tbackend_ms_tok\tpath\tverdict\n' >"$MATRIX"

for arg in "$@"; do
    dir="$(resolve_dir "$arg")" || continue
    IFS=$'\t' read -r label overlap drain blocking stall straggler straggler_ms asm g_tps ms_tok path < <(extract_row "$dir")
    verdict="$(classify_verdict "$overlap" "$drain" "$straggler_ms")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$overlap" "$drain" "$blocking" "$stall" "$straggler" "$straggler_ms" \
        "$asm" "$g_tps" "$ms_tok" "$path" "$verdict"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$overlap" "$drain" "$blocking" "$stall" "$straggler" "$straggler_ms" \
        "$asm" "$g_tps" "$ms_tok" "$path" "$verdict" >>"$MATRIX"
done

echo "=== diagnosis matrix: ${MATRIX} ==="
column -t -s $'\t' "$MATRIX" 2>/dev/null || cat "$MATRIX"