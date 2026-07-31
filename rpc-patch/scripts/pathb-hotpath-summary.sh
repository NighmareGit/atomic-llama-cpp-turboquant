#!/usr/bin/env bash
# Hot-path stall report from Path B telemetry: sync waterfall, RPC-WAIT-MAP mapping,
# assembly-line gap, SET_TENSOR_HASH bucket, top blockers.
#
# usage:
#   pathb-hotpath-summary.sh <telemetry_dir>
#
# env:
#   PATHB_ASSEMBLY_TARGET_PCT  design overlap target (default 5.0; B+1 pass is >0)
#   PATHB_GEN_TOKENS           override gen token count for per-token waterfall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARSE="${SCRIPT_DIR}/pathb-rpc-trace-parse.sh"

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -eq 1 ]] || usage

TRACE_DIR="$(cd "$1" && pwd)"
if [[ ! -d "$TRACE_DIR" ]]; then
    echo "error: telemetry dir not found: $1" >&2
    exit 1
fi

RPC_FILE="${TRACE_DIR}/rpc-trace.jsonl"
SCHED_FILE="${TRACE_DIR}/sched-trace.jsonl"
SUMMARY_FILE="${TRACE_DIR}/trace-summary.txt"

need_parse=0
if [[ ! -f "$SUMMARY_FILE" ]]; then
    need_parse=1
elif [[ -f "$RPC_FILE" && "$RPC_FILE" -nt "$SUMMARY_FILE" ]]; then
    need_parse=1
elif [[ -f "$SCHED_FILE" && "$SCHED_FILE" -nt "$SUMMARY_FILE" ]]; then
    need_parse=1
fi

if [[ "$need_parse" -eq 1 ]]; then
    echo ">>> parsing traces (trace-summary missing or stale)" >&2
    bash "$PARSE" "$TRACE_DIR" >/dev/null
fi

export TRACE_DIR
export SUMMARY_FILE
export PATHB_ASSEMBLY_TARGET_PCT="${PATHB_ASSEMBLY_TARGET_PCT:-5.0}"
export PATHB_GEN_TOKENS="${PATHB_GEN_TOKENS:-}"

exec python3 - <<'PY'
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

trace_dir = Path(os.environ["TRACE_DIR"]).resolve()
summary_file = Path(os.environ["SUMMARY_FILE"])
assembly_target = float(os.environ.get("PATHB_ASSEMBLY_TARGET_PCT", "5.0"))
gen_tokens_override = os.environ.get("PATHB_GEN_TOKENS", "").strip()

CMD_NAMES = {
    0: "ALLOC_BUFFER", 1: "GET_ALIGNMENT", 2: "GET_MAX_SIZE", 3: "BUFFER_GET_BASE",
    4: "FREE_BUFFER", 5: "BUFFER_CLEAR", 6: "SET_TENSOR", 7: "SET_TENSOR_HASH",
    8: "GET_TENSOR", 9: "COPY_TENSOR", 10: "GRAPH_COMPUTE", 11: "GET_DEVICE_MEMORY",
    12: "INIT_TENSOR", 13: "GET_ALLOC_SIZE", 14: "HELLO", 15: "DEVICE_COUNT",
    16: "GRAPH_RECOMPUTE", 17: "SET_TENSOR_BATCH", 18: "EVENT_RECORD",
    19: "COPY_TENSOR_PEER",
}

BACKEND_ROLE = {
    "0": "CUDA0 local compute",
    "1": "RPC worker (e.g. :50051)",
    "2": "RPC worker / CPU tail (topology-dependent)",
    "3": "CPU / host split",
}

WAIT_MAP = {
    "input_wait_copy": (
        "sched event_wait + tensor_copy",
        "ggml/src/ggml-backend.cpp:1562-1590",
        "RPC-WAIT-MAP: split loop input drain",
    ),
    "sync_copy_fallback": (
        "B+13 local synchronize + tensor_copy (async copy failed)",
        "ggml/src/ggml-backend.cpp:1865-1884",
        "C-full: measured sync fallback",
    ),
    "copy_async_ok": (
        "B+13 cpy_tensor_async succeeded",
        "ggml/src/ggml-backend.cpp:1885-1887",
        "C-full: async copy marker",
    ),
    "graph_compute_async": (
        "GRAPH_RECOMPUTE / local graph_compute_async",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:944-979",
        "RPC-WAIT-MAP: RPC graph stage",
    ),
    "event_record": (
        "EVENT_RECORD deferred recv on drain",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:304-336",
        "RPC-WAIT-MAP: drain_pending_event_response",
    ),
    "split_total": (
        "serial split loop per token",
        "ggml/src/ggml-backend.cpp:1549-1722",
        "RPC-WAIT-MAP: ggml_backend_sched_compute_splits",
    ),
    "drain_flush": (
        "send_rpc_cmd drain preamble",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:413-461",
        "RPC-WAIT-MAP: flush before blocking cmd",
    ),
    "blocking_events": (
        "blocking RPC RTT budget",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:457-475",
        "RPC-WAIT-MAP: send_recv path",
    ),
    "SET_TENSOR_HASH": (
        "cross-endpoint weight relay (hash path)",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:832+",
        "RPC-WAIT-MAP: COPY fallback GET+SET",
    ),
    "COPY_TENSOR": (
        "sync cross-port COPY (no cpy_tensor_async)",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:709,989",
        "RPC-WAIT-MAP: cpy_tensor_async NULL",
    ),
    "GRAPH_RECOMPUTE": (
        "per-RPC split graph recompute",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:1662",
        "RPC-WAIT-MAP: server graph_compute serial",
    ),
    "EVENT_RECORD": (
        "client EVENT_RECORD + drain",
        "ggml/src/ggml-rpc/ggml-rpc.cpp:1990+",
        "RPC-WAIT-MAP: EVENT ack RTT",
    ),
}


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


def parse_summary_kv(summary_text: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in summary_text.splitlines():
        line = line.strip()
        m = re.match(r"^(\S+)\s+count=(\d+)\s+total_ms=([\d.]+)\s+avg_us=([\d.]+)$", line)
        if m:
            kv[f"rpc:{m.group(1)}"] = line
            kv[f"rpc:{m.group(1)}:count"] = m.group(2)
            kv[f"rpc:{m.group(1)}:total_ms"] = m.group(3)
            kv[f"rpc:{m.group(1)}:avg_us"] = m.group(4)
            continue
        for key in (
            "blocking_events", "drain_flush_ms", "drain_copy_ms",
            "split_total_count", "input_wait_copy_ms",
            "graph_compute_async_ms", "event_record_ms",
            "sync_copy_fallback_ms", "copy_async_ok_count",
            "assembly_overlap_count",
            "pairs",
        ):
            mkey = re.search(rf"(?:^|\s){re.escape(key)}=([\d.]+)", line)
            if mkey:
                kv[key] = mkey.group(1)
        mblock = re.search(r"blocking_ms=([\d.]+)", line)
        if mblock:
            kv["blocking_ms"] = mblock.group(1)
        mpct = re.search(r"overlap_pct=([\d.]+)", line)
        if mpct:
            kv["overlap_pct"] = mpct.group(1)
        msum = re.match(r"^split_total_ms_sum=([\d.]+)", line)
        if msum:
            kv["split_total_ms_sum"] = msum.group(1)
        m2 = re.match(r"^backend(\d+)\s+splits=(\d+)\s+ms=([\d.]+)$", line)
        if m2:
            kv[f"backend{m2.group(1)}_ms"] = m2.group(3)
            kv[f"backend{m2.group(1)}_splits"] = m2.group(2)
        m3 = re.match(r"^copy(\d+)\s+splits=(\d+)$", line)
        if m3:
            kv[f"copy{m3.group(1)}_splits"] = m3.group(2)
    return kv


def estimate_gen_tokens(sched_rows: list[dict], summary_kv: dict[str, str]) -> int:
    if gen_tokens_override.isdigit():
        return max(1, int(gen_tokens_override))
    meta = trace_dir.parent / "meta.txt"
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"(?:gen[_-]?tokens|GenTokens)\s*[=:]\s*(\d+)", line, re.I)
            if m:
                return max(1, int(m.group(1)))
    splits = [r for r in sched_rows if r.get("phase") == "split_total"]
    if not splits:
        return 128
    split_ids = {int(r.get("split", 0)) for r in splits}
    n_splits = len(split_ids) if split_ids else 3
    split_count = int(summary_kv.get("split_total_count", len(splits)))
    return max(1, split_count // max(1, n_splits))


def ms_per_token(total_ms: float, tokens: int) -> float:
    return round(total_ms / tokens, 2)


def main() -> None:
    rpc_rows = load_jsonl(trace_dir / "rpc-trace.jsonl")
    sched_rows = load_jsonl(trace_dir / "sched-trace.jsonl")
    summary_text = summary_file.read_text(encoding="utf-8") if summary_file.is_file() else ""
    summary_kv = parse_summary_kv(summary_text)
    gen_tokens = estimate_gen_tokens(sched_rows, summary_kv)

    # trace_id join (preferred over ts heuristic for long-running / perf_reset cases)
    rpc_by_tid = {}
    for r in rpc_rows:
        tid = int(r.get("trace_id", 0) or 0)
        if tid:
            rpc_by_tid[tid] = rpc_by_tid.get(tid, 0) + 1
    trace_joins = 0
    total_tid = 0
    for r in sched_rows:
        tid = int(r.get("trace_id", 0) or 0)
        if tid:
            total_tid += 1
            if tid in rpc_by_tid:
                trace_joins += 1
    trace_join_rate = round(trace_joins / total_tid, 4) if total_tid > 0 else 0.0

    lines: list[str] = []
    lines.append("=== pathb hotpath summary ===")
    lines.append(f"dir={trace_dir}")
    lines.append(f"trace_summary={summary_file}")
    lines.append(f"gen_tokens_est={gen_tokens}")
    lines.append(f"trace_id_join_rate={trace_join_rate} (joins={trace_joins} total_tid={total_tid}; target >0.95)")
    lines.append("")

    # --- sync stall waterfall ---
    lines.append("[sync-stall-waterfall] per-token critical path (serial splits)")
    split_total_ms = float(summary_kv.get("split_total_ms_sum", "0") or 0)
    lines.append(f"  sched split_total_ms_sum={split_total_ms} ms_per_token={ms_per_token(split_total_ms, gen_tokens)}")
    backend_ids = sorted(
        (k[len("backend") : -len("_ms")] for k in summary_kv if re.fullmatch(r"backend\d+_ms", k)),
        key=int,
    )
    for bid in backend_ids:
        ms = summary_kv.get(f"backend{bid}_ms", "0")
        splits = summary_kv.get(f"backend{bid}_splits", "?")
        role = BACKEND_ROLE.get(bid, "backend split")
        ms_f = float(ms)
        lines.append(
            f"  |-- backend{bid} ({role}) "
            f"total_ms={ms} splits={splits} ms_per_token={ms_per_token(ms_f, gen_tokens)}"
        )
    for phase in ("input_wait_copy", "sync_copy_fallback", "graph_compute_async", "event_record"):
        key = f"{phase}_ms"
        if key in summary_kv:
            ms_f = float(summary_kv[key])
            site, ref, note = WAIT_MAP[phase]
            lines.append(
                f"  |-- sched {phase}={summary_kv[key]} ms "
                f"({ms_per_token(ms_f, gen_tokens)} ms/tok) -> {site}"
            )
            lines.append(f"      ref={ref} ({note})")
    if "copy_async_ok_count" in summary_kv:
        lines.append(
            f"  |-- sched copy_async_ok_count={summary_kv['copy_async_ok_count']} "
            f"(C-full async copy markers)"
        )
    lines.append("")

    # --- RPC-WAIT-MAP mapping ---
    lines.append("[rpc-wait-map] trace buckets -> static wait sites")
    mapping_rows = []
    for phase in ("input_wait_copy", "sync_copy_fallback", "graph_compute_async", "event_record"):
        key = f"{phase}_ms"
        if key in summary_kv:
            site, ref, note = WAIT_MAP[phase]
            mapping_rows.append((float(summary_kv[key]), phase, site, ref, note))
    if "drain_flush_ms" in summary_kv:
        site, ref, note = WAIT_MAP["drain_flush"]
        mapping_rows.append((float(summary_kv["drain_flush_ms"]), "drain_flush", site, ref, note))
    if "blocking_ms" in summary_kv:
        site, ref, note = WAIT_MAP["blocking_events"]
        mapping_rows.append((float(summary_kv["blocking_ms"]), "blocking_events", site, ref, note))
    for cmd in ("SET_TENSOR_HASH", "COPY_TENSOR", "GRAPH_RECOMPUTE", "EVENT_RECORD"):
        key = f"rpc:{cmd}:total_ms"
        if key in summary_kv:
            site, ref, note = WAIT_MAP[cmd]
            mapping_rows.append((float(summary_kv[key]), cmd, site, ref, note))
    mapping_rows.sort(reverse=True, key=lambda r: r[0])
    for total_ms, label, site, ref, note in mapping_rows:
        lines.append(f"  {label} total_ms={total_ms} -> {site}")
        lines.append(f"    {ref} | {note}")
    lines.append("  doc=docs/cuda-windows-5070ti/RPC-WAIT-MAP.md")
    lines.append("")

    # --- assembly-line gap ---
    overlap_count = int(float(summary_kv.get("assembly_overlap_count", "0") or 0))
    overlap_pct = float(summary_kv.get("overlap_pct", "0") or 0)
    pairs = summary_kv.get("pairs", "?")
    b1_pass = overlap_count > 0
    gap = round(assembly_target - overlap_pct, 1)
    lines.append("[assembly-line] token-pipeline overlap (S5 metric)")
    lines.append(f"  assembly_overlap_count={overlap_count} pairs={pairs} overlap_pct={overlap_pct}")
    lines.append(f"  design_target_pct={assembly_target} (Tier 2 aspirational; B+1 pass criterion >0%)")
    lines.append(f"  gap_pct={gap} status={'PASS' if b1_pass else 'FAIL'} (B+1 S5: count>0)")
    copy_slots = sorted(
        (k, summary_kv[k])
        for k in summary_kv
        if k.startswith("copy") and k.endswith("_splits")
    )
    if copy_slots:
        slot_str = " ".join(f"{k.replace('_splits', '')}={v}" for k, v in copy_slots)
        lines.append(f"  copy_rotation: {slot_str}")
        counts = [int(v) for _, v in copy_slots]
        if counts and max(counts) - min(counts) > max(counts) * 0.5:
            lines.append("  copy_rotation_status=WARN (uneven slot use; check Plus=1)")
        else:
            lines.append("  copy_rotation_status=OK")
    lines.append("  note=splits within one token remain serial; overlap is cross-token pipeline")
    lines.append("")

    # --- SET_TENSOR_HASH bucket ---
    lines.append("[set_tensor_hash] cross-RPC weight relay bucket")
    if "rpc:SET_TENSOR_HASH:total_ms" in summary_kv:
        lines.append(
            f"  count={summary_kv.get('rpc:SET_TENSOR_HASH:count', '?')} "
            f"total_ms={summary_kv['rpc:SET_TENSOR_HASH:total_ms']} "
            f"avg_us={summary_kv.get('rpc:SET_TENSOR_HASH:avg_us', '?')}"
        )
        avg_us = float(summary_kv.get("rpc:SET_TENSOR_HASH:avg_us", "0") or 0)
        if avg_us > 10000:
            lines.append("  severity=HIGH (dominant RPC stall class on CUDA<->RPC topologies)")
        elif avg_us > 1000:
            lines.append("  severity=MEDIUM")
        else:
            lines.append("  severity=LOW (hash path cheap or peer COPY active)")
    else:
        lines.append("  (no SET_TENSOR_HASH events in trace)")
    copy_tensor_ms = float(summary_kv.get("rpc:COPY_TENSOR:total_ms", "0") or 0)
    if copy_tensor_ms > 0:
        lines.append(
            f"  COPY_TENSOR total_ms={copy_tensor_ms} "
            f"count={summary_kv.get('rpc:COPY_TENSOR:count', '?')} "
            "(legacy GET+SET path)"
        )
    peer_count = summary_kv.get("rpc:COPY_TENSOR_PEER:count", "0")
    if peer_count and peer_count != "0":
        lines.append(
            f"  COPY_TENSOR_PEER count={peer_count} "
            f"total_ms={summary_kv.get('rpc:COPY_TENSOR_PEER:total_ms', '?')}"
        )
    lines.append("")

    # --- C-full B+13 (measured) ---
    lines.append("[c-full-b13] sync fallback vs async copy (requires C-full build)")
    if "sync_copy_fallback_ms" in summary_kv or "copy_async_ok_count" in summary_kv:
        fb_ms = float(summary_kv.get("sync_copy_fallback_ms", "0") or 0)
        ok_n = int(float(summary_kv.get("copy_async_ok_count", "0") or 0))
        iw_ms = float(summary_kv.get("input_wait_copy_ms", "0") or 0)
        lines.append(f"  sync_copy_fallback_ms={fb_ms} copy_async_ok_count={ok_n}")
        if iw_ms > 0 and fb_ms > 0:
            lines.append(
                f"  sync_fallback_pct_of_input_wait={round(100.0 * fb_ms / iw_ms, 1)}"
            )
        if fb_ms > 50:
            lines.append("  verdict=SYNC_COPY_FALLBACK_MEASURED (B+13 fix target)")
        elif ok_n > 0 and fb_ms == 0:
            lines.append("  verdict=COPY_ASYNC_OK_ONLY")
        else:
            lines.append("  verdict=C_FULL_PHASES_PRESENT")
    else:
        lines.append("  (no C-full phases; rebuild with 6dc504bce+ and re-bench)")
    lines.append("")

    # --- top blockers ---
    lines.append("[top-blockers] ranked RPC/sched wall-time (load+gen window)")
    blockers: list[tuple[float, str]] = []
    for cmd_id, name in CMD_NAMES.items():
        key = f"rpc:{name}:total_ms"
        if key in summary_kv:
            blockers.append((float(summary_kv[key]), f"rpc {name}"))
    for phase in ("input_wait_copy", "sync_copy_fallback", "graph_compute_async", "event_record"):
        key = f"{phase}_ms"
        if key in summary_kv:
            blockers.append((float(summary_kv[key]), f"sched {phase}"))
    if "drain_flush_ms" in summary_kv:
        blockers.append((float(summary_kv["drain_flush_ms"]), "rpc drain_flush"))
    if "blocking_ms" in summary_kv:
        blockers.append((float(summary_kv["blocking_ms"]), "rpc blocking_events"))
    blockers.sort(reverse=True, key=lambda x: x[0])
    for rank, (total_ms, label) in enumerate(blockers[:8], start=1):
        lines.append(f"  {rank}. {label} total_ms={total_ms}")
    if not blockers:
        lines.append("  (no trace buckets; run with GGML_RPC_TRACE=1 GGML_SCHED_TRACE=1)")
    lines.append("")
    lines.append("see also: rpc-patch/docs/pathb-sync-site-audit.md")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
PY