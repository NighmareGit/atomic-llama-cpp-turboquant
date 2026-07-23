#!/usr/bin/env bash
# Pipeline diagnose: sched/RPC trace -> diagnose.json + human summary.
#
# usage:
#   llama-pipeline-diagnose.sh <telemetry_dir> [--gen-only] [--overlap-target 5] [--baseline DIR]
#
# env:
#   PATHB_GEN_TOKENS, PATHB_ASSEMBLY_TARGET_PCT

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARSE="${ROOT}/rpc-patch/scripts/pathb-rpc-trace-parse.sh"
HOTPATH="${ROOT}/rpc-patch/scripts/pathb-hotpath-summary.sh"

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

TRACE_DIR=""
GEN_ONLY=0
OVERLAP_TARGET="5"
BASELINE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --gen-only) GEN_ONLY=1; shift ;;
        --overlap-target) OVERLAP_TARGET="${2:?}"; shift 2 ;;
        --baseline) BASELINE_DIR="${2:?}"; shift 2 ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *)
            if [[ -z "$TRACE_DIR" ]]; then
                TRACE_DIR="$1"
            else
                echo "unexpected arg: $1" >&2
                usage
            fi
            shift
            ;;
    esac
done

[[ -n "$TRACE_DIR" ]] || usage
TRACE_DIR="$(cd "$TRACE_DIR" && pwd)"
export PATHB_ASSEMBLY_TARGET_PCT="${OVERLAP_TARGET}"

if [[ -x "$PARSE" ]] || [[ -f "$PARSE" ]]; then
    bash "$PARSE" "$TRACE_DIR" >/dev/null 2>&1 || true
fi

export TRACE_DIR GEN_ONLY OVERLAP_TARGET BASELINE_DIR ROOT
PY_BIN="${PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PY_BIN=python3
    else
        PY_BIN=python
    fi
fi
exec "$PY_BIN" - <<'PY'
from __future__ import annotations

import csv
import json
import os
import re
import statistics
from pathlib import Path

trace_dir = Path(os.environ["TRACE_DIR"]).resolve()
gen_only = os.environ.get("GEN_ONLY", "0") == "1"
overlap_target = float(os.environ.get("OVERLAP_TARGET", "5"))
baseline_dir = os.environ.get("BASELINE_DIR", "").strip()
root = Path(os.environ["ROOT"])

summary_file = trace_dir / "trace-summary.txt"
meta_file = trace_dir.parent / "meta.txt"
if not meta_file.is_file():
    meta_file = trace_dir / "meta.txt"
diagnose_out = trace_dir / "diagnose.json"
human_out = trace_dir / "diagnose-summary.txt"

TDP = {"5070": 300.0, "5060": 175.0, "6600": 140.0}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def parse_summary_kv(text: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^split_total_ms_sum=([\d.]+)", line.strip())
        if m:
            kv["split_total_ms_sum"] = m.group(1)
        for key in (
            "assembly_overlap_count", "overlap_pct", "drain_flush_ms",
            "blocking_ms", "input_wait_copy_ms", "graph_compute_async_ms",
            "event_record_ms", "split_total_count",
        ):
            mkey = re.search(rf"(?:^|\s){re.escape(key)}=([\d.]+)", line)
            if mkey:
                kv[key] = mkey.group(1)
        m2 = re.match(r"^backend(\d+)\s+splits=(\d+)\s+ms=([\d.]+)$", line.strip())
        if m2:
            kv[f"backend{m2.group(1)}_ms"] = m2.group(3)
    return kv


def estimate_gen_tokens(sched_rows: list[dict], summary_kv: dict[str, str]) -> int:
    override = os.environ.get("PATHB_GEN_TOKENS", "").strip()
    if override.isdigit():
        return max(1, int(override))
    if meta_file.is_file():
        for line in meta_file.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"(?:gen[_-]?tokens|n_gen)\s*[=:]\s*(\d+)", line, re.I)
            if m:
                return max(1, int(m.group(1)))
    splits = [r for r in sched_rows if r.get("phase") == "split_total"]
    if not splits:
        return 128
    split_ids = {int(r.get("split", 0)) for r in splits}
    n_splits = len(split_ids) if split_ids else 3
    split_count = int(float(summary_kv.get("split_total_count", len(splits))))
    return max(1, split_count // max(1, n_splits))


def gen_window_start_us(sched_rows: list[dict], rpc_rows: list[dict]) -> int:
    """Heuristic: skip load-phase SET_TENSOR_HASH burst."""
    hash_events = [r for r in rpc_rows if int(r.get("cmd", -1)) == 7]
    if hash_events:
        last_hash = max(int(r.get("ts_us", 0)) for r in hash_events)
        return last_hash
    splits = sorted(sched_rows, key=lambda r: int(r.get("ts_us", 0)))
    if len(splits) > 20:
        return int(splits[10].get("ts_us", 0))
    return 0


def filter_gen(rows: list[dict], start_us: int) -> list[dict]:
    if start_us <= 0:
        return rows
    return [r for r in rows if int(r.get("ts_us", 0)) >= start_us]


def stall_ratio(summary_kv: dict[str, str], gen_tokens: int) -> dict:
    split_ms = float(summary_kv.get("split_total_ms_sum", 0) or 0)
    wait_ms = float(summary_kv.get("input_wait_copy_ms", 0) or 0)
    event_ms = float(summary_kv.get("event_record_ms", 0) or 0)
    compute_ms = float(summary_kv.get("graph_compute_async_ms", 0) or 0)
    stall_ms = wait_ms + event_ms
    per_tok = split_ms / gen_tokens if gen_tokens else 0
    ratio = stall_ms / split_ms if split_ms > 0 else 0.0
    backends = {}
    straggler_id = None
    straggler_ms = 0.0
    for k, v in summary_kv.items():
        m = re.fullmatch(r"backend(\d+)_ms", k)
        if m:
            ms = float(v)
            mpt = ms / gen_tokens if gen_tokens else 0
            backends[m.group(1)] = round(mpt, 3)
            if mpt > straggler_ms:
                straggler_ms = mpt
                straggler_id = m.group(1)
    return {
        "split_ms_per_token": round(per_tok, 3),
        "stall_ratio": round(ratio, 4),
        "stall_ms": round(stall_ms, 2),
        "compute_ms": round(compute_ms, 2),
        "backends_ms_per_token": backends,
        "straggler_backend": straggler_id,
        "straggler_ms_per_token": round(straggler_ms, 3),
    }


def rtt_budget(rpc_rows: list[dict], gen_tokens: int) -> dict:
    blocking = [r for r in rpc_rows if str(r.get("blocking")).lower() == "true"]
    fire = [r for r in rpc_rows if str(r.get("blocking")).lower() != "true" and r.get("cmd") is not None]
    graph = [r for r in rpc_rows if int(r.get("cmd", -1)) in (10, 16)]
    return {
        "topology_class": "client_split",
        "rpc_rtt_per_token": round(len(blocking) / gen_tokens, 2) if gen_tokens else 0,
        "graph_submit_count": len(graph),
        "blocking_rpc_count": len(blocking),
        "fire_and_forget_count": len(fire),
    }


def gate_results(summary_kv: dict[str, str]) -> dict:
    overlap_count = int(float(summary_kv.get("assembly_overlap_count", 0) or 0))
    overlap_pct = float(summary_kv.get("overlap_pct", 0) or 0)
    s5 = overlap_count > 0
    b6 = overlap_pct >= overlap_target
    return {
        "gate_s5": "PASS" if s5 else "FAIL",
        "gate_b6": "PASS" if b6 else "FAIL",
        "assembly_overlap_count": overlap_count,
        "overlap_pct": overlap_pct,
        "overlap_target_pct": overlap_target,
    }


def overlap_efficiency(summary_kv: dict[str, str], g_tps: float | None, gen_tokens: int) -> float | None:
    split_ms = float(summary_kv.get("split_total_ms_sum", 0) or 0)
    if not g_tps or g_tps <= 0 or gen_tokens <= 0:
        return None
    wall_ms_per_token = 1000.0 / g_tps
    serial_ms = split_ms / gen_tokens
    if wall_ms_per_token <= 0:
        return None
    return round(serial_ms / wall_ms_per_token, 4)


def parse_g_tps() -> float | None:
    for name in ("result.jsonl", "../result.jsonl"):
        p = trace_dir / name
        if not p.is_file():
            p = trace_dir.parent / "result.jsonl"
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    tps = row.get("avg_ts") or row.get("G_tps")
                    if tps:
                        return float(tps)
                except json.JSONDecodeError:
                    continue
    return None


def gpu_smells(gpu_dir: Path) -> list[str]:
    smells: list[str] = []
    csv_path = gpu_dir / "nvidia-local.csv"
    if not csv_path.is_file():
        return smells
    powers: list[float] = []
    utils: list[float] = []
    clocks: list[float] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                p = float(row.get("power_w", 0) or 0)
                u = float(row.get("util_gpu_pct", 0) or 0)
                c = float(row.get("sm_clock_mhz", 0) or 0)
                if p > 0:
                    powers.append(p)
                utils.append(u)
                if c > 0:
                    clocks.append(c)
            except ValueError:
                continue
    if not powers or not utils:
        return smells
    tdp = TDP["5070"]
    max_u = max(utils)
    max_p_pct = 100.0 * max(powers) / tdp
    avg_u = statistics.mean(utils)
    if max_u >= 40 and max_p_pct < 10:
        smells.append("GPU_METRIC_MISMATCH")
    if max_u < 15 and max_p_pct < 10:
        smells.append("ORCHESTRATION_STALL")
    if clocks and max(clocks) > 0 and min(clocks) < 0.85 * max(clocks):
        smells.append("CLOCK_THROTTLE")
    if avg_u < 5 and max_p_pct < 8:
        smells.append("LOW_DUTY_CYCLE")
    return smells


sched_rows = load_jsonl(trace_dir / "sched-trace.jsonl")
rpc_rows = load_jsonl(trace_dir / "rpc-trace.jsonl")
summary_text = summary_file.read_text(encoding="utf-8") if summary_file.is_file() else ""
summary_kv = parse_summary_kv(summary_text)

if gen_only and sched_rows:
    start_us = gen_window_start_us(sched_rows, rpc_rows)
    sched_rows = filter_gen(sched_rows, start_us)
    rpc_rows = filter_gen(rpc_rows, start_us)

gen_tokens = estimate_gen_tokens(sched_rows, summary_kv)
g_tps = parse_g_tps()
stall = stall_ratio(summary_kv, gen_tokens)
gates = gate_results(summary_kv)
rtt = rtt_budget(rpc_rows, gen_tokens)
eff = overlap_efficiency(summary_kv, g_tps, gen_tokens)
gpu_dir = trace_dir / "gpu"
smells = gpu_smells(gpu_dir)

diagnose = {
    "version": 1,
    "trace_dir": str(trace_dir),
    "gen_only": gen_only,
    "gen_tokens_est": gen_tokens,
    "G_tps": g_tps,
    "overlap_efficiency": eff,
    "drain_flush_ms": float(summary_kv.get("drain_flush_ms", 0) or 0),
    "blocking_ms": float(summary_kv.get("blocking_ms", 0) or 0),
    **gates,
    **stall,
    **rtt,
    "gpu_smell_flags": smells,
    "path_c_reserved": {
        "cross_endpoint_copy_count": None,
        "server_local_copy_count": None,
    },
}

if baseline_dir:
    base_path = Path(baseline_dir) / "diagnose.json"
    if base_path.is_file():
        try:
            base = json.loads(base_path.read_text(encoding="utf-8"))
            diagnose["baseline_delta"] = {
                "G_tps": (g_tps or 0) - float(base.get("G_tps") or 0),
                "overlap_pct": gates["overlap_pct"] - float(base.get("overlap_pct") or 0),
                "stall_ratio": stall["stall_ratio"] - float(base.get("stall_ratio") or 0),
            }
        except (json.JSONDecodeError, TypeError):
            pass

diagnose_out.write_text(json.dumps(diagnose, indent=2) + "\n", encoding="utf-8")

lines = [
    "=== llama-pipeline-diagnose ===",
    f"diagnose.json -> {diagnose_out}",
    f"gate_s5={gates['gate_s5']} gate_b6={gates['gate_b6']} overlap_pct={gates['overlap_pct']}",
    f"stall_ratio={stall['stall_ratio']} straggler=backend{stall['straggler_backend']} ({stall['straggler_ms_per_token']} ms/tok)",
    f"overlap_efficiency={eff}",
    f"gpu_smells={smells or 'none'}",
]
human_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if hotpath := root / "rpc-patch/scripts/pathb-hotpath-summary.sh":
    if hotpath.is_file():
        os.system(f'PATHB_GEN_TOKENS={gen_tokens} bash "{hotpath}" "{trace_dir}" 2>/dev/null || true')
PY