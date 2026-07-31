#!/usr/bin/env bash
# Phase 1.2A: per-token blocking waterfall from sched + rpc traces (no new bench).
#
# usage: b6-gate-phase12a-waterfall.sh [bench-label...]
#   default: canonical ladder + B+12 bisect + 4-GPU gate

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_TSV="${ROOT}/benches/path-b-plus/phase12a-waterfall.tsv"
OUT_MD="${ROOT}/benches/path-b-plus/phase12a-waterfall.md"

DEFAULT_DIRS=(
    b6-2gpu-f-triton-n384-romulus-native
    b6-2gpu-f-triton-n384-romulus-native-no-get-defer
    b6-2gpu-f-triton-guard-n128
    b6-2gpu-f-triton-n384-no-partial
    b6-2gpu-f-triton-n384-remus-docker
    b6-4gpu-g-n384-romulus-native
)

DIRS=("${@:-${DEFAULT_DIRS[@]}}")

printf 'label\tgen_tokens\ttoken_p50_ms\ttoken_p95_ms\ttoken_p99_ms\tstraggler_backend\tstraggler_ms_per_tok\tinput_wait_ms_per_tok\tevent_record_ms_per_tok\trpc_blocking_ms_per_tok\toverlap_pct\tverdict\n' >"$OUT_TSV"

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
    6: "SET_TENSOR", 7: "SET_TENSOR_HASH", 8: "GET_TENSOR", 9: "COPY_TENSOR",
    10: "GRAPH_COMPUTE", 16: "GRAPH_RECOMPUTE", 18: "EVENT_RECORD", 19: "COPY_TENSOR_PEER",
}

BACKEND_ROLE = {
    "0": "CUDA0",
    "1": "RPC",
    "2": "CPU",
    "3": "CPU/host",
}

WATERFALL_PHASES = (
    "input_wait_copy", "sync_copy_fallback", "graph_compute_async",
    "event_record", "rpc_flush_downloads", "host_h2d_issue",
)


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

# Per decode_id sched aggregation
by_decode: dict[int, dict] = defaultdict(lambda: {
    "splits": defaultdict(float),
    "phases": defaultdict(float),
    "split_rows": [],
    "ts_lo": 10**18,
    "ts_hi": 0,
})

for row in sched:
    did = int(row.get("decode_id", 0))
    if did < 1:
        continue
    ts = int(row.get("ts_us", 0))
    el = int(row.get("elapsed_us", 0))
    d = by_decode[did]
    d["ts_lo"] = min(d["ts_lo"], ts)
    d["ts_hi"] = max(d["ts_hi"], ts + el)
    phase = row.get("phase")
    if phase in WATERFALL_PHASES:
        d["phases"][phase] += el / 1000.0
    if phase == "split_total":
        backend = str(row.get("backend", "?"))
        split = int(row.get("split", -1))
        copy = int(row.get("copy", 0))
        key = (split, backend, copy)
        d["splits"][key] = max(d["splits"].get(key, 0.0), el / 1000.0)
        d["split_rows"].append({
            "split": split, "backend": backend, "copy": copy,
            "ms": el / 1000.0, "ts_us": ts,
        })

# RPC blocking per decode_id (join key or ts window)
rpc_by_decode: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
rpc_total_by_decode: dict[int, float] = defaultdict(float)

for row in rpc:
    did = row.get("decode_id")
    if did is None:
        continue
    did = int(did)
    if did < 1:
        continue
    us = int(row.get("elapsed_us", 0))
    if row.get("phase") != "send_recv" and str(row.get("blocking")).lower() != "true":
        continue
    cmd = int(row.get("cmd", -1))
    rpc_by_decode[did][cmd] += us / 1000.0
    rpc_total_by_decode[did] += us / 1000.0

tokens: list[dict] = []
for did in sorted(by_decode):
    d = by_decode[did]
    wall_ms = (d["ts_hi"] - d["ts_lo"]) / 1000.0 if d["ts_hi"] > d["ts_lo"] else 0.0

    copies_by_split: dict[int, list] = defaultdict(list)
    for row in d["split_rows"]:
        copies_by_split[row["split"]].append(row)
    serial_ms = 0.0
    waterfall_lines = []
    for split in sorted(copies_by_split):
        rows = copies_by_split[split]
        lead = min(rows, key=lambda r: (r["copy"], -r["ms"]))
        backend = lead["backend"]
        ms = lead["ms"]
        serial_ms += ms
        role = BACKEND_ROLE.get(backend, f"backend{backend}")
        phase_bits = []
        for ph in WATERFALL_PHASES:
            # approximate per-split phase from sched rows matching split+backend
            ph_ms = sum(
                int(r.get("elapsed_us", 0)) / 1000.0
                for r in sched
                if int(r.get("decode_id", 0)) == did
                and r.get("phase") == ph
                and int(r.get("split", -1)) == split
                and str(r.get("backend")) == backend
            )
            if ph_ms >= 0.05:
                phase_bits.append(f"{ph}={ph_ms:.1f}")
        rpc_bits = []
        rpc_bits_all = [
            r for r in rpc
            if int(r.get("decode_id", -1)) == did
            and r.get("split") is not None
            and int(r.get("split")) == split
            and str(r.get("backend", "")) == backend
            and (r.get("phase") == "send_recv" or str(r.get("blocking")).lower() == "true")
        ]
        by_cmd_split: dict[int, float] = defaultdict(float)
        for r in rpc_bits_all:
            by_cmd_split[int(r.get("cmd", -1))] += int(r.get("elapsed_us", 0)) / 1000.0
        for cmd, cmd_ms in sorted(by_cmd_split.items()):
            if cmd_ms >= 0.05:
                rpc_bits.append(f"{CMD.get(cmd, cmd)}={cmd_ms:.1f}")
        extra = " ".join(phase_bits + rpc_bits)
        waterfall_lines.append({
            "split": split, "backend": backend, "role": role,
            "ms": round(ms, 2), "detail": extra,
        })

    by_backend_serial = defaultdict(float)
    for ln in waterfall_lines:
        by_backend_serial[ln["backend"]] += ln["ms"]
    straggler_b = max(by_backend_serial, key=by_backend_serial.get) if by_backend_serial else "?"
    straggler_ms = by_backend_serial.get(straggler_b, 0.0)

    tokens.append({
        "decode_id": did,
        "wall_ms": round(wall_ms, 2),
        "serial_ms": round(serial_ms, 2),
        "input_wait_ms": round(d["phases"].get("input_wait_copy", 0.0), 2),
        "event_record_ms": round(d["phases"].get("event_record", 0.0), 2),
        "rpc_blocking_ms": round(rpc_total_by_decode.get(did, 0.0), 2),
        "straggler_backend": straggler_b,
        "straggler_ms": round(straggler_ms, 2),
        "waterfall": waterfall_lines,
        "rpc_by_cmd": {CMD.get(k, str(k)): round(v, 2) for k, v in sorted(rpc_by_decode[did].items())},
    })

gen_tokens = len(tokens)
wall_vals = [t["wall_ms"] for t in tokens]
overlap = float(diag.get("overlap_pct", 0) or 0)

# Sample tokens: first, median, p95
samples = []
if tokens:
    samples.append(tokens[0])
    samples.append(tokens[len(tokens) // 2])
    p95_idx = min(len(tokens) - 1, max(0, int(math.ceil(0.95 * len(tokens)) - 1)))
    if tokens[p95_idx] not in samples:
        samples.append(tokens[p95_idx])

totals = defaultdict(float)
for t in tokens:
    totals["input_wait"] += t["input_wait_ms"]
    totals["event_record"] += t["event_record_ms"]
    totals["rpc_blocking"] += t["rpc_blocking_ms"]
    totals[f"backend_{t['straggler_backend']}"] += t["straggler_ms"]

straggler_counts = defaultdict(int)
for t in tokens:
    straggler_counts[t["straggler_backend"]] += 1
dominant_straggler = max(straggler_counts, key=straggler_counts.get) if straggler_counts else "?"

flags = []
if gen_tokens == 0:
    flags.append("NO_GEN_TOKENS")
avg_iw = totals["input_wait"] / max(gen_tokens, 1)
avg_ev = totals["event_record"] / max(gen_tokens, 1)
avg_rpc = totals["rpc_blocking"] / max(gen_tokens, 1)
if avg_iw > avg_rpc * 0.5 and avg_iw > 1.0:
    flags.append("INPUT_WAIT_DOMINATES")
if avg_ev > avg_rpc * 0.4 and avg_ev > 1.0:
    flags.append("EVENT_RECORD_DOMINATES")
if dominant_straggler == "1":
    flags.append("RPC_STRAGGLER")
if pct(wall_vals, 0.99) > pct(wall_vals, 0.50) * 3 and gen_tokens > 10:
    flags.append("TAIL_TOKEN_SPIKE")
verdict = ",".join(flags) if flags else "OK"

out_json = {
    "label": label,
    "gen_tokens": gen_tokens,
    "overlap_pct": overlap,
    "token_wall_ms": {
        "p50": pct(wall_vals, 0.50),
        "p95": pct(wall_vals, 0.95),
        "p99": pct(wall_vals, 0.99),
    },
    "per_token_avg_ms": {
        "input_wait_copy": round(avg_iw, 2),
        "event_record": round(avg_ev, 2),
        "rpc_blocking": round(avg_rpc, 2),
    },
    "straggler_backend": dominant_straggler,
    "straggler_role": BACKEND_ROLE.get(dominant_straggler, f"backend{dominant_straggler}"),
    "straggler_ms_per_tok": round(totals.get(f"backend_{dominant_straggler}", 0) / max(gen_tokens, 1), 2),
    "sample_waterfalls": samples,
    "verdict_flags": flags,
}
(telem / "waterfall-a.json").write_text(json.dumps(out_json, indent=2) + "\n")

row = [
    label, str(gen_tokens),
    str(pct(wall_vals, 0.50)), str(pct(wall_vals, 0.95)), str(pct(wall_vals, 0.99)),
    dominant_straggler,
    str(round(totals.get(f"backend_{dominant_straggler}", 0) / max(gen_tokens, 1), 2)),
    str(round(avg_iw, 2)), str(round(avg_ev, 2)), str(round(avg_rpc, 2)),
    str(overlap), verdict,
]
with out_tsv.open("a") as fh:
    fh.write("\t".join(row) + "\n")

print(f"=== {label} ===")
print(f"  gen_tokens={gen_tokens} overlap={overlap}%")
print(f"  token_wall_ms p50={pct(wall_vals,0.5)} p95={pct(wall_vals,0.95)} p99={pct(wall_vals,0.99)}")
print(f"  straggler=backend{dominant_straggler} ({BACKEND_ROLE.get(dominant_straggler,'?')}) "
      f"{round(totals.get(f'backend_{dominant_straggler}',0)/max(gen_tokens,1),2)} ms/tok")
print(f"  avg input_wait={round(avg_iw,2)} event_record={round(avg_ev,2)} rpc_blocking={round(avg_rpc,2)} ms/tok")
for samp in samples:
    print(f"  --- decode_id={samp['decode_id']} wall={samp['wall_ms']}ms serial={samp['serial_ms']}ms ---")
    for ln in samp["waterfall"]:
        det = f" ({ln['detail']})" if ln["detail"] else ""
        print(f"    |-- {ln['role']} split{ln['split']} {ln['ms']}ms{det}")
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
    "# Phase 1.2A per-token blocking waterfall\n",
    "Staged D step A: sched + rpc join on `(decode_id, split, backend)`. "
    "Per-dir JSON: `telemetry/waterfall-a.json` (sample waterfalls + distribution).\n",
    "## Summary\n",
    "| " + " | ".join(header) + " |",
    "|" + "|".join(["---"] * len(header)) + "|",
]
for r in rows:
    body.append("| " + " | ".join(r) + " |")

body.extend([
    "\n## Interpretation\n",
    "- **token_wall_ms**: sched wall clock per decode_id (cross-split pipeline included).",
    "- **straggler_backend**: backend with largest serial split_total sum per token (mode across gen).",
    "- **INPUT_WAIT_DOMINATES**: copy-slot wait exceeds half of per-token RPC blocking.",
    "- **EVENT_RECORD_DOMINATES**: EVENT_RECORD sched phase rivals RPC budget (B+9/B+12 lever).",
    "- **RPC_STRAGGLER**: backend1 (RPC worker) is modal straggler — assembly line bound by RPC stage.",
    "- **TAIL_TOKEN_SPIKE**: p99 token wall >> p50 (HOL / cold-path suspect; see Phase 1.2B Gantt).",
    "\n## Next: Phase 1.2B\n",
    "Assembly-line Gantt + HOL tail RTT (`decode_id` x split x cmd class).\n",
])
md.write_text("\n".join(body) + "\n")
print(f"summary -> {md}")
PY

echo "TSV -> ${OUT_TSV}"