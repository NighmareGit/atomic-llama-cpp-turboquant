#!/usr/bin/env bash
# Phase 1.2B: assembly-line Gantt + HOL tail RTT from sched + rpc traces (no new bench).
#
# usage: b6-gate-phase12b-gantt.sh [bench-label...]
#   default: canonical ladder + B+12 bisect + 4-GPU gate (B+11 HOL proof)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_TSV="${ROOT}/benches/path-b-plus/phase12b-gantt.tsv"
OUT_MD="${ROOT}/benches/path-b-plus/phase12b-gantt.md"

DEFAULT_DIRS=(
    b6-2gpu-f-triton-n384-romulus-native
    b6-2gpu-f-triton-n384-romulus-native-no-get-defer
    b6-2gpu-f-triton-guard-n128
    b6-2gpu-f-triton-n384-no-partial
    b6-2gpu-f-triton-n384-remus-docker
    b6-4gpu-g-n384-romulus-native
)

DIRS=("${@:-${DEFAULT_DIRS[@]}}")

printf 'label\tgen_tokens\toverlap_pct\toverlap_pairs\tpipeline_gap_p50_ms\tpipeline_gap_p95_ms\tcross_backend_overlap_pct\trpc_rtt_p50_ms\trpc_rtt_p95_ms\trpc_rtt_p99_ms\thol_tail_count\thol_tail_ms\tverdict\n' >"$OUT_TSV"

for label in "${DIRS[@]}"; do
    telem="${ROOT}/benches/path-b-plus/${label}/telemetry"
    ref="${ROOT}/docs/cuda-windows-5070ti/benchmarks/${label}/telemetry"
    [[ -d "$telem" ]] || telem="$ref"
    if [[ ! -d "$telem" ]]; then
        echo "SKIP ${label} (no telemetry)" >&2
        continue
    fi

    python3 - <<'PY' "$telem" "$label" "$OUT_TSV"
import json, math, sys
from collections import defaultdict
from pathlib import Path

telem = Path(sys.argv[1])
label = sys.argv[2]
out_tsv = Path(sys.argv[3])

CMD = {
    6: "SET", 7: "HASH", 8: "GET", 9: "COPY", 10: "GRAPH",
    16: "RECOMP", 18: "EVENT", 19: "PEER",
}

BACKEND_ROLE = {
    "0": "CUDA0",
    "1": "RPC",
    "2": "CPU",
    "3": "host",
}

GANTT_WIDTH = 96
GANTT_TOKENS = 24
HOL_TAIL_MULT = 10.0
HOL_TAIL_FLOOR_MS = 5.0


def load_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def load_diag(telem: Path) -> dict:
    p = telem / "diagnose.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text())
    return raw.get("diagnose", raw)


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(p * len(s)) - 1)))
    return round(s[idx], 2)


rpc = load_jsonl(telem / "rpc-trace.jsonl")
sched = load_jsonl(telem / "sched-trace.jsonl")
diag = load_diag(telem)

# Gen window from sched decode_id >= 1
decode_windows: dict[int, tuple[int, int]] = {}
for row in sched:
    if row.get("phase") != "split_total":
        continue
    did = int(row.get("decode_id", 0))
    if did < 1:
        continue
    ts = int(row.get("ts_us", 0))
    end = ts + int(row.get("elapsed_us", 0))
    if did not in decode_windows:
        decode_windows[did] = (ts, end)
    else:
        lo, hi = decode_windows[did]
        decode_windows[did] = (min(lo, ts), max(hi, end))

gen_lo = min(w[0] for w in decode_windows.values()) if decode_windows else 0
gen_hi = max(w[1] for w in decode_windows.values()) if decode_windows else 0
gen_span_us = max(1, gen_hi - gen_lo)

def in_gen(ts: int) -> bool:
    return gen_lo <= ts <= gen_hi


def is_blocking_rpc(row: dict) -> bool:
    phase = row.get("phase")
    if phase in ("send_recv", "drain_event", "flush_get", "drain_copy", "graph_compute"):
        return True
    return str(row.get("blocking")).lower() == "true"


# Split segments for Gantt (lead copy per split per decode)
segments: list[dict] = []
by_decode_split: dict[tuple[int, int], list] = defaultdict(list)

for row in sched:
    if row.get("phase") != "split_total":
        continue
    did = int(row.get("decode_id", 0))
    if did < 1:
        continue
    split = int(row.get("split", -1))
    by_decode_split[(did, split)].append(row)

for (did, split), rows in sorted(by_decode_split.items()):
    lead = min(rows, key=lambda r: (int(r.get("copy", 0)), int(r.get("ts_us", 0))))
    ts = int(lead.get("ts_us", 0))
    el = int(lead.get("elapsed_us", 0))
    backend = str(lead.get("backend", "?"))
    segments.append({
        "decode_id": did, "split": split, "backend": backend,
        "copy": int(lead.get("copy", 0)),
        "start_us": ts, "end_us": ts + el, "ms": el / 1000.0,
    })

# RPC cmd segments (joined)
rpc_segments: list[dict] = []
for row in rpc:
    if row.get("decode_id") is None or row.get("split") is None:
        continue
    did = int(row.get("decode_id"))
    if did < 1 or not in_gen(int(row.get("ts_us", 0))):
        continue
    if not is_blocking_rpc(row):
        continue
    ts = int(row.get("ts_us", 0))
    el = int(row.get("elapsed_us", 0))
    cmd = int(row.get("cmd", -1))
    rpc_segments.append({
        "decode_id": did,
        "split": int(row.get("split", -1)),
        "backend": str(row.get("backend", "?")),
        "cmd": cmd,
        "cmd_name": CMD.get(cmd, str(cmd)),
        "start_us": ts, "end_us": ts + el, "ms": el / 1000.0,
    })

# Pipeline gaps between consecutive decode_ids
sorted_dids = sorted(decode_windows)
pipeline_gaps_ms: list[float] = []
for i in range(len(sorted_dids) - 1):
    d0, d1 = sorted_dids[i], sorted_dids[i + 1]
    gap_us = decode_windows[d1][0] - decode_windows[d0][1]
    pipeline_gaps_ms.append(gap_us / 1000.0)

# Cross-backend overlap (assembly-line metric)
split_ev = [
    row for row in sched
    if row.get("phase") == "split_total" and int(row.get("decode_id", 0)) >= 1
]
overlap = 0
pairs = 0
for i, a in enumerate(split_ev):
    a_start = int(a.get("ts_us", 0))
    a_end = a_start + int(a.get("elapsed_us", 0))
    for j in range(i + 1, len(split_ev)):
        b = split_ev[j]
        if a.get("backend") == b.get("backend"):
            continue
        b_start = int(b.get("ts_us", 0))
        if a_start <= b_start < a_end:
            overlap += 1
        pairs += 1
overlap_pct = round(100.0 * overlap / pairs, 1) if pairs else 0.0

# RPC RTT in gen window (all blocking phases incl. drain/flush)
rtts = [
    int(r.get("elapsed_us", 0))
    for r in rpc
    if is_blocking_rpc(r) and in_gen(int(r.get("ts_us", 0)))
]
rtt_ms = [x / 1000.0 for x in rtts]

# HOL tail: per-backend blocking RPC >> p50 on same backend
rtts_by_backend: dict[str, list[int]] = defaultdict(list)
for r in rpc:
    if not is_blocking_rpc(r) or not in_gen(int(r.get("ts_us", 0))):
        continue
    bk = str(r.get("backend", r.get("fn", "unjoined")))
    rtts_by_backend[bk].append(int(r.get("elapsed_us", 0)))

hol_tails: list[dict] = []
for bk, bus in rtts_by_backend.items():
    if len(bus) < 5:
        continue
    p50 = pct([x / 1000.0 for x in bus], 0.50)
    thresh_ms = max(HOL_TAIL_FLOOR_MS, p50 * HOL_TAIL_MULT)
    for r in rpc:
        if not is_blocking_rpc(r) or str(r.get("backend", r.get("fn", "unjoined"))) != bk:
            continue
        if not in_gen(int(r.get("ts_us", 0))):
            continue
        ms = int(r.get("elapsed_us", 0)) / 1000.0
        if ms >= thresh_ms:
            phase = r.get("phase", "?")
            hol_tails.append({
                "backend": bk,
                "decode_id": r.get("decode_id"),
                "split": r.get("split"),
                "phase": phase,
                "cmd": CMD.get(int(r.get("cmd", -1)), r.get("cmd")),
                "ms": round(ms, 2),
                "p50_ms": p50,
            })

hol_tail_ms = round(sum(t["ms"] for t in hol_tails), 2)

# ASCII Gantt (first N tokens, per backend row)
gantt_lines: list[str] = []
backends = sorted({s["backend"] for s in segments}, key=lambda x: (x == "?", x))
show_dids = sorted({s["decode_id"] for s in segments})[:GANTT_TOKENS]
if show_dids and backends:
    lo = min(s["start_us"] for s in segments if s["decode_id"] in show_dids)
    hi = max(s["end_us"] for s in segments if s["decode_id"] in show_dids)
    span = max(1, hi - lo)
    gantt_lines.append(
        f"gantt decode_id {show_dids[0]}..{show_dids[-1]} "
        f"width={GANTT_WIDTH} span_ms={round(span/1000,1)}"
    )
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for bk in backends:
        row = [" "] * GANTT_WIDTH
        role = BACKEND_ROLE.get(bk, f"b{bk}")
        for seg in segments:
            if seg["backend"] != bk or seg["decode_id"] not in show_dids:
                continue
            rel_start = int((seg["start_us"] - lo) / span * GANTT_WIDTH)
            rel_end = int((seg["end_us"] - lo) / span * GANTT_WIDTH)
            rel_start = max(0, min(GANTT_WIDTH - 1, rel_start))
            rel_end = max(rel_start + 1, min(GANTT_WIDTH, rel_end))
            ch = chars[seg["decode_id"] % len(chars)]
            for c in range(rel_start, rel_end):
                row[c] = ch
        gantt_lines.append(f"  {role:6s} |{''.join(row)}|")

# Cmd-class density per (decode_id, split) for top straggler tokens
cmd_grid: dict[str, dict] = defaultdict(lambda: defaultdict(float))
for seg in rpc_segments:
    key = f"d{seg['decode_id']}s{seg['split']}b{seg['backend']}"
    cmd_grid[key][seg["cmd_name"]] += seg["ms"]

flags = []
if overlap_pct < 1.0:
    flags.append("LOW_OVERLAP")
if pct(pipeline_gaps_ms, 0.50) >= 0:
    flags.append("SERIAL_PIPELINE")
if hol_tails and pct(rtt_ms, 0.99) > pct(rtt_ms, 0.50) * 5:
    flags.append("HOL_TAIL_RTT")
if hol_tails and len(hol_tails) > 20:
    flags.append("HOL_TAIL_FREQUENT")
if label.startswith("b6-4gpu") and pct(rtt_ms, 0.99) > 20:
    flags.append("B11_HOL_CANDIDATE")
verdict = ",".join(flags) if flags else "OK"

gen_tokens = len(decode_windows)
overlap_diag = float(diag.get("overlap_pct", overlap_pct) or overlap_pct)

out_json = {
    "label": label,
    "gen_window_us": {"lo": gen_lo, "hi": gen_hi, "span_ms": round(gen_span_us / 1000.0, 2)},
    "gen_tokens": gen_tokens,
    "overlap_pct": overlap_diag,
    "assembly_overlap": {"count": overlap, "pairs": pairs, "overlap_pct": overlap_pct},
    "pipeline_gap_ms": {
        "p50": pct(pipeline_gaps_ms, 0.50),
        "p95": pct(pipeline_gaps_ms, 0.95),
        "samples": [round(g, 2) for g in pipeline_gaps_ms[:8]],
    },
    "rpc_rtt_ms": {
        "count": len(rtts),
        "p50": pct(rtt_ms, 0.50),
        "p95": pct(rtt_ms, 0.95),
        "p99": pct(rtt_ms, 0.99),
    },
    "hol_tails": {
        "count": len(hol_tails),
        "total_ms": hol_tail_ms,
        "threshold_mult": HOL_TAIL_MULT,
        "top10": sorted(hol_tails, key=lambda t: t["ms"], reverse=True)[:10],
    },
    "gantt_ascii": gantt_lines,
    "segments_count": len(segments),
    "rpc_segments_count": len(rpc_segments),
    "verdict_flags": flags,
}
(telem / "gantt-b.json").write_text(json.dumps(out_json, indent=2) + "\n")

row = [
    label, str(gen_tokens), str(overlap_diag), f"{overlap}/{pairs}",
    str(pct(pipeline_gaps_ms, 0.50)), str(pct(pipeline_gaps_ms, 0.95)),
    str(overlap_pct),
    str(pct(rtt_ms, 0.50)), str(pct(rtt_ms, 0.95)), str(pct(rtt_ms, 0.99)),
    str(len(hol_tails)), str(hol_tail_ms), verdict,
]
with out_tsv.open("a") as fh:
    fh.write("\t".join(row) + "\n")

print(f"=== {label} ===")
print(f"  gen_tokens={gen_tokens} overlap={overlap_diag}% assembly={overlap}/{pairs} ({overlap_pct}%)")
print(f"  pipeline_gap_ms p50={pct(pipeline_gaps_ms,0.5)} p95={pct(pipeline_gaps_ms,0.95)}")
print(f"  rpc_rtt_ms p50={pct(rtt_ms,0.5)} p95={pct(rtt_ms,0.95)} p99={pct(rtt_ms,0.99)}")
print(f"  hol_tails={len(hol_tails)} total_ms={hol_tail_ms}")
for ln in gantt_lines[:6]:
    print(f"  {ln}")
if hol_tails:
    top = sorted(hol_tails, key=lambda t: t["ms"], reverse=True)[:3]
    for t in top:
        print(f"  HOL tail: backend{t['backend']} d{t['decode_id']}s{t['split']} "
              f"{t.get('phase','?')}/{t['cmd']} {t['ms']}ms (p50={t['p50_ms']})")
print(f"  flags: {verdict}")
PY
done

python3 - <<'PY' "$OUT_TSV" "$OUT_MD"
import sys
from pathlib import Path

tsv = Path(sys.argv[1])
md = Path(sys.argv[2])
lines = tsv.read_text().strip().splitlines()
header = lines[0].split("\t")
rows = [ln.split("\t") for ln in lines[1:]]

body = [
    "# Phase 1.2B assembly-line Gantt + HOL tail RTT\n",
    "Staged D step B: `decode_id` x split x cmd class timeline. "
    "Per-dir JSON: `telemetry/gantt-b.json` (ASCII Gantt + HOL tails).\n",
    "## Summary\n",
    "| " + " | ".join(header) + " |",
    "|" + "|".join(["---"] * len(header)) + "|",
]
for r in rows:
    body.append("| " + " | ".join(r) + " |")

body.extend([
    "\n## Interpretation\n",
    "- **pipeline_gap_ms**: `token_start[d+1] - token_end[d]`; >=0 means no cross-token overlap (serial assembly).",
    "- **cross_backend_overlap_pct**: fraction of cross-backend split pairs with temporal overlap (S5 metric).",
    "- **hol_tail_count**: RPC `send_recv` events >= max(5ms, 10x backend p50) — head-of-line / tail RTT suspects.",
    "- **HOL_TAIL_RTT**: p99 >> p50 on wire RTT; motivates B+11 dual-socket RPC.",
    "- **B11_HOL_CANDIDATE**: 4-GPU gate with p99 RTT > 20ms (romulus ladder hypothesis).",
    "- **SERIAL_PIPELINE**: positive median pipeline gap — tokens do not overlap on the assembly line.",
    "\n## Status\n",
    "Phase 1.2 A+B complete. Next work is protocol-level (B+11 dual-socket / proto bump), not more path-b-plus patches.\n",
])
md.write_text("\n".join(body) + "\n")
print(f"summary -> {md}")
PY

echo "TSV -> ${OUT_TSV}"