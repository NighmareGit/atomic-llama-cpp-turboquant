#!/usr/bin/env bash
# Phase 1.2C: B+13 / blocking hot-path audit on existing traces (no new bench).
#
# usage: b6-gate-phase12c-blocking-audit.sh [bench-label...]
#   default: canonical 2-GPU ladder + guard-n128 + production 2gpu ref

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_TSV="${ROOT}/benches/path-b-plus/phase12c-blocking-audit.tsv"
OUT_MD="${ROOT}/benches/path-b-plus/phase12c-blocking-audit.md"

DEFAULT_DIRS=(
    b6-2gpu-f-triton-n384-romulus-native
    b6-2gpu-f-triton-guard-n128
    b6-2gpu-f-triton-n384-no-partial
    b6-2gpu-f-triton-n384-remus-docker
    b6-4gpu-g-n384-romulus-native
)

DIRS=("${@:-${DEFAULT_DIRS[@]}}")

printf 'label\tgen_tokens\toverlap_pct\tstall_ratio\tinput_wait_copy_ms\tgraph_compute_ms\tsync_copy_fallback_ms\tcopy_async_ok_count\tblocking_ms\tdrain_ms\tcopy_issue\tcopy_tensor_rpc\tcopy_peer_rpc\tget_tensor_rpc\tset_hash_rpc\tevent_record_rpc\tlocal_sync_gap_ms\tc_full\tverdict\n' >"$OUT_TSV"

for label in "${DIRS[@]}"; do
    telem="${ROOT}/benches/path-b-plus/${label}/telemetry"
    ref="${ROOT}/docs/cuda-windows-5070ti/benchmarks/${label}/telemetry"
    [[ -d "$telem" ]] || telem="$ref"
    if [[ ! -d "$telem" ]]; then
        echo "SKIP ${label} (no telemetry)" >&2
        continue
    fi

    python3 - <<'PY' "$telem" "$label" "$OUT_TSV"
import json, re, sys
from collections import defaultdict
from pathlib import Path

telem = Path(sys.argv[1])
label = sys.argv[2]
out_tsv = Path(sys.argv[3])

CMD = {
    6: "SET_TENSOR", 7: "SET_TENSOR_HASH", 8: "GET_TENSOR", 9: "COPY_TENSOR",
    10: "GRAPH_COMPUTE", 16: "GRAPH_RECOMPUTE", 18: "EVENT_RECORD", 19: "COPY_TENSOR_PEER",
}

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

if decode_windows:
    gen_lo = min(w[0] for w in decode_windows.values())
    gen_hi = max(w[1] for w in decode_windows.values())
else:
    gen_lo, gen_hi = 0, 10**18

def in_gen(ts: int) -> bool:
    return gen_lo <= ts <= gen_hi

# RPC gen-window stats
copy_issue = [r for r in rpc if r.get("phase") == "copy_issue" and in_gen(int(r.get("ts_us", 0)))]
defer_copy = [r for r in copy_issue if str(r.get("defer")).lower() == "true"]
peer_copy = [r for r in copy_issue if str(r.get("peer_copy")).lower() == "true"]

by_cmd_ms: dict[int, float] = defaultdict(float)
by_cmd_cnt: dict[int, int] = defaultdict(int)
blocking_ms = 0.0
for r in rpc:
    ts = int(r.get("ts_us", 0))
    if not in_gen(ts):
        continue
    cmd = int(r.get("cmd", -1))
    us = int(r.get("elapsed_us", 0))
    if r.get("phase") == "send_recv" or str(r.get("blocking")).lower() == "true":
        by_cmd_ms[cmd] += us / 1000.0
        by_cmd_cnt[cmd] += 1
        blocking_ms += us / 1000.0

# Sched gen-only phases
sched_gen = defaultdict(float)
for row in sched:
    did = int(row.get("decode_id", 0))
    if did < 1:
        continue
    phase = row.get("phase")
    if phase:
        sched_gen[phase] += int(row.get("elapsed_us", 0)) / 1000.0

input_wait = sched_gen.get("input_wait_copy", 0.0)
graph_compute = sched_gen.get("graph_compute_async", 0.0)
sync_fallback_ms = sched_gen.get("sync_copy_fallback", 0.0)
copy_async_ok_count = int(sched_gen.get("copy_async_ok", 0.0))  # marker rows, elapsed_us=0
if copy_async_ok_count == 0:
    copy_async_ok_count = sum(1 for row in sched if row.get("phase") == "copy_async_ok" and int(row.get("decode_id", 0)) >= 1)

sync_fallback_by_backend: dict[str, float] = defaultdict(float)
sync_fallback_count = 0
for row in sched:
    if row.get("phase") != "sync_copy_fallback":
        continue
    if int(row.get("decode_id", 0)) < 1:
        continue
    sync_fallback_count += 1
    sync_fallback_by_backend[str(row.get("backend", "?"))] += int(row.get("elapsed_us", 0)) / 1000.0

c_full = sync_fallback_count > 0 or copy_async_ok_count > 0 or any(
    row.get("split") is not None and row.get("backend") is not None
    for row in rpc if row.get("phase") in ("send_recv", "copy_issue") and in_gen(int(row.get("ts_us", 0)))
)

copy_tensor_ms = by_cmd_ms.get(9, 0.0)
copy_peer_ms = by_cmd_ms.get(19, 0.0)
get_tensor_ms = by_cmd_ms.get(8, 0.0)
set_hash_ms = by_cmd_ms.get(7, 0.0)
event_ms = by_cmd_ms.get(18, 0.0)

# Local sync gap: input_wait not explained by on-wire COPY RPC in gen window
local_sync_gap = max(0.0, input_wait - copy_tensor_ms - copy_peer_ms)

overlap = float(diag.get("overlap_pct", 0) or 0)
stall = float(diag.get("stall_ratio", 0) or 0)
drain = float(diag.get("drain_flush_ms", 0) or 0)
gen_tokens = int(diag.get("gen_tokens_est", len(decode_windows)) or len(decode_windows))

# Verdict heuristics for Phase 1.2C / C-full
flags = []
if len(copy_issue) == 0:
    flags.append("NO_COPY_ISSUE_TRACE")
if c_full:
    if sync_fallback_ms > 50:
        flags.append("SYNC_COPY_FALLBACK_MEASURED")
    elif copy_async_ok_count > 0 and sync_fallback_count == 0:
        flags.append("COPY_ASYNC_OK_ONLY")
    elif sync_fallback_count == 0 and copy_async_ok_count == 0:
        flags.append("C_FULL_NO_B13_PHASES")
elif local_sync_gap > input_wait * 0.25 and input_wait > 100:
    flags.append("LOCAL_SYNC_FALLBACK_LIKELY")
if sync_fallback_ms > input_wait * 0.5 and input_wait > 100:
    flags.append("B13_DOMINATES_INPUT_WAIT")
if event_ms > 0 and abs(event_ms - drain) < drain * 0.15:
    flags.append("EVENT_RECORD_DOMINATES_DRAIN")
if set_hash_ms > blocking_ms * 0.2:
    flags.append("SET_HASH_RPC_HEAVY")
if get_tensor_ms > copy_tensor_ms + copy_peer_ms:
    flags.append("GET_TENSOR_GT_COPY")
if input_wait > graph_compute * 2:
    flags.append("WAIT_DOMINATES_COMPUTE")

verdict = ",".join(flags) if flags else "MIXED"

out_json = {
    "label": label,
    "gen_window": {"lo_us": gen_lo, "hi_us": gen_hi, "decode_ids": len(decode_windows)},
    "gen_tokens": gen_tokens,
    "overlap_pct": overlap,
    "stall_ratio": stall,
    "sched_gen_ms": dict(sched_gen),
    "rpc_gen_blocking_ms_by_cmd": {CMD.get(k, str(k)): round(v, 2) for k, v in sorted(by_cmd_ms.items())},
    "copy_issue": {
        "count": len(copy_issue),
        "defer_count": len(defer_copy),
        "peer_copy_count": len(peer_copy),
    },
    "local_sync_gap_ms": round(local_sync_gap, 2),
    "c_full_trace": c_full,
    "sync_copy_fallback_ms": round(sync_fallback_ms, 2),
    "sync_copy_fallback_count": sync_fallback_count,
    "sync_copy_fallback_by_backend_ms": {
        k: round(v, 2) for k, v in sorted(sync_fallback_by_backend.items(), key=lambda x: x[0])
    },
    "copy_async_ok_count": copy_async_ok_count,
    "verdict_flags": flags,
}
(telem / "blocking-audit-c.json").write_text(json.dumps(out_json, indent=2) + "\n")

row = [
    label, str(gen_tokens), str(overlap), str(stall),
    f"{input_wait:.1f}", f"{graph_compute:.1f}", f"{sync_fallback_ms:.1f}",
    str(copy_async_ok_count), f"{blocking_ms:.1f}", f"{drain:.1f}",
    str(len(copy_issue)), f"{copy_tensor_ms:.1f}", f"{copy_peer_ms:.1f}",
    f"{get_tensor_ms:.1f}", f"{set_hash_ms:.1f}", f"{event_ms:.1f}",
    f"{local_sync_gap:.1f}", "yes" if c_full else "no", verdict,
]
with out_tsv.open("a") as fh:
    fh.write("\t".join(row) + "\n")

print(f"=== {label} ===")
print(f"  gen_tokens={gen_tokens} overlap={overlap}% stall={stall}")
print(f"  input_wait_copy={input_wait:.0f}ms graph_compute={graph_compute:.0f}ms")
print(f"  blocking_rpc={blocking_ms:.0f}ms drain={drain:.0f}ms")
print(f"  copy_issue={len(copy_issue)} COPY_TENSOR={copy_tensor_ms:.0f}ms GET_TENSOR={get_tensor_ms:.0f}ms SET_HASH={set_hash_ms:.0f}ms EVENT={event_ms:.0f}ms")
print(f"  sync_copy_fallback={sync_fallback_ms:.0f}ms count={sync_fallback_count} copy_async_ok={copy_async_ok_count}")
print(f"  local_sync_gap={local_sync_gap:.0f}ms ({100*local_sync_gap/max(input_wait,1):.0f}% of input_wait) c_full={c_full}")
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

body = ["# Phase 1.2C blocking audit (B+13 preliminary)\n",
        "Staged D step C: existing traces only. Per-dir JSON: `telemetry/blocking-audit-c.json`.\n",
        "## Results\n",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|"]
for r in rows:
    body.append("| " + " | ".join(r) + " |")

body.extend([
    "\n## Interpretation guide\n",
    "- **NO_COPY_ISSUE_TRACE**: `GGML_RPC_TRACE` copy_issue lines absent; wire-level copy deferral not visible in these artifacts.",
    "- **LOCAL_SYNC_FALLBACK_LIKELY**: pre-C-full heuristic; `input_wait_copy_ms` >> wire COPY ms.",
    "- **SYNC_COPY_FALLBACK_MEASURED**: C-full `sync_copy_fallback` phase present (>50ms gen window) — B+13 local sync path proven.",
    "- **B13_DOMINATES_INPUT_WAIT**: measured `sync_copy_fallback_ms` > 50% of `input_wait_copy_ms`.",
    "- **COPY_ASYNC_OK_ONLY**: async copy path succeeded (markers only, no fallback rows).",
    "- **C_FULL_NO_B13_PHASES**: RPC join present but no B+13 phase rows (check trace env + build SHA).",
    "- **EVENT_RECORD_DOMINATES_DRAIN**: B+9/B+12 less likely to move overlap until EVENT path shortened.",
    "- **SET_HASH_RPC_HEAVY**: weight relay still costs gen-window budget (B+4 cache check).",
    "- **WAIT_DOMINATES_COMPUTE**: assembly line starved regardless of straggler ms/tok.",
    "\n## Next: Phase 1.2 A+B\n",
    "Per-token blocking waterfall + assembly-line Gantt (decode_id x split x cmd class).\n",
])
md.write_text("\n".join(body) + "\n")
print(f"summary -> {md}")
PY

echo "TSV -> ${OUT_TSV}"
cat "$OUT_TSV"