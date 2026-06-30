#!/usr/bin/env bash
# Parse GGML_RPC_TRACE + GGML_SCHED_TRACE jsonl into stall budget summary.
# Ports scripts/cuda-windows-5070ti/pathb-rpc-trace-parse.ps1 (cmd map, rpc/sched
# aggregation, assembly_overlap_count, drain_copy, hello, copy_issue).
#
# usage:
#   pathb-rpc-trace-parse.sh <telemetry_dir>

set -euo pipefail

usage() {
    sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -eq 1 ]] || usage

TRACE_DIR="$1"
if [[ ! -d "$TRACE_DIR" ]]; then
    echo "error: telemetry dir not found: $TRACE_DIR" >&2
    exit 1
fi

exec python3 - "$TRACE_DIR" <<'PY'
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

trace_dir = Path(sys.argv[1]).resolve()
rpc_file = trace_dir / "rpc-trace.jsonl"
sched_file = trace_dir / "sched-trace.jsonl"
out_file = trace_dir / "trace-summary.txt"

CMD_NAMES = {
    0: "ALLOC_BUFFER",
    1: "GET_ALIGNMENT",
    2: "GET_MAX_SIZE",
    3: "BUFFER_GET_BASE",
    4: "FREE_BUFFER",
    5: "BUFFER_CLEAR",
    6: "SET_TENSOR",
    7: "SET_TENSOR_HASH",
    8: "GET_TENSOR",
    9: "COPY_TENSOR",
    10: "GRAPH_COMPUTE",
    11: "GET_DEVICE_MEMORY",
    12: "INIT_TENSOR",
    13: "GET_ALLOC_SIZE",
    14: "HELLO",
    15: "DEVICE_COUNT",
    16: "GRAPH_RECOMPUTE",
    17: "SET_TENSOR_BATCH",
    18: "EVENT_RECORD",
    19: "COPY_TENSOR_PEER",
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


def round_ms(us_sum: int) -> float:
    return round(us_sum / 1000.0, 2)


def round_avg(us_values: list[int]) -> float:
    if not us_values:
        return 0.0
    return round(sum(us_values) / len(us_values), 0)


def main() -> None:
    lines: list[str] = []
    lines.append("=== trace summary ===")
    lines.append(f"dir={trace_dir}")

    rpc_rows = load_jsonl(rpc_file)
    if rpc_file.is_file():
        lines.append("")
        lines.append(f"[rpc] events={len(rpc_rows)}")
        if rpc_rows:
            by_cmd: dict[int, list[dict]] = defaultdict(list)
            for row in rpc_rows:
                cmd = row.get("cmd")
                if cmd is None:
                    continue
                by_cmd[int(cmd)].append(row)

            for cmd in sorted(by_cmd, key=str):
                group = by_cmd[cmd]
                name = CMD_NAMES.get(cmd, f"cmd{cmd}")
                us = [int(row.get("elapsed_us", 0)) for row in group]
                lines.append(
                    f"  {name} count={len(group)} total_ms={round_ms(sum(us))} "
                    f"avg_us={round_avg(us)}"
                )

            blocking = [row for row in rpc_rows if str(row.get("blocking")).lower() == "true"]
            bus = [int(row.get("elapsed_us", 0)) for row in blocking]
            lines.append(
                f"  blocking_events={len(blocking)} blocking_ms={round_ms(sum(bus))}"
            )

            # extend for per-split / RPC RTT histogram when Plus=1 (plan 1.1)
            rtts = [int(row.get("elapsed_us", 0)) for row in rpc_rows if row.get("phase") == "send_recv"]
            if rtts:
                lines.append(f"  rpc_rtt_count={len(rtts)} rpc_rtt_ms={round_ms(sum(rtts))}")
                bins = {}
                for us in rtts:
                    b = us // 1000  # ms bins
                    bins[b] = bins.get(b, 0) + 1
                h = " ".join(f"{k}ms:{v}" for k, v in sorted(bins.items())[:5])
                lines.append(f"  rpc_rtt_hist={h}")

            drain = [
                row
                for row in rpc_rows
                if "fn" in row and (
                    "drain" in str(row["fn"]).lower() or "flush" in str(row["fn"]).lower()
                )
            ]
            if drain:
                dus = [int(row.get("elapsed_us", 0)) for row in drain]
                lines.append(f"  drain_flush_ms={round_ms(sum(dus))}")

            drain_copy = [row for row in rpc_rows if row.get("phase") == "drain_copy"]
            if drain_copy:
                dcus = [int(row.get("elapsed_us", 0)) for row in drain_copy]
                lines.append(
                    f"  drain_copy_count={len(drain_copy)} "
                    f"drain_copy_ms={round_ms(sum(dcus))}"
                )

            hello = [row for row in rpc_rows if row.get("phase") == "hello"]
            for row in hello:
                ep = row.get("endpoint") or "?"
                lines.append(
                    f"  hello endpoint={ep} minor={row.get('minor')} "
                    f"peer_copy={row.get('peer_copy')}"
                )

            copy_issue = [row for row in rpc_rows if row.get("phase") == "copy_issue"]
            if copy_issue:
                defer = [row for row in copy_issue if str(row.get("defer")).lower() == "true"]
                peer = [row for row in copy_issue if str(row.get("peer_copy")).lower() == "true"]
                lines.append(
                    f"  copy_issue_count={len(copy_issue)} defer_count={len(defer)} "
                    f"peer_copy_count={len(peer)}"
                )
    else:
        lines.append("")
        lines.append(f"[rpc] missing {rpc_file}")

    sched_rows = load_jsonl(sched_file)
    if sched_file.is_file():
        lines.append("")
        lines.append(f"[sched] events={len(sched_rows)}")
        if sched_rows:
            splits = [row for row in sched_rows if row.get("phase") == "split_total"]
            lines.append(f"  split_total_count={len(splits)}")
            if splits:
                sus = [int(row.get("elapsed_us", 0)) for row in splits]
                lines.append(
                    f"  split_total_ms_sum={round_ms(sum(sus))} "
                    f"avg_us={round_avg(sus)}"
                )
                by_backend: dict[str, list[dict]] = defaultdict(list)
                backend_order: list[str] = []
                for row in splits:
                    backend = str(row.get("backend"))
                    if backend not in by_backend:
                        backend_order.append(backend)
                    by_backend[backend].append(row)
                for backend in backend_order:
                    group = by_backend[backend]
                    bus = [int(row.get("elapsed_us", 0)) for row in group]
                    lines.append(
                        f"    backend{backend} splits={len(group)} "
                        f"ms={round_ms(sum(bus))}"
                    )

            for phase in ("input_wait_copy", "graph_compute_async", "event_record"):
                phase_rows = [row for row in sched_rows if row.get("phase") == phase]
                if phase_rows:
                    pus = [int(row.get("elapsed_us", 0)) for row in phase_rows]
                    lines.append(f"  {phase}_ms={round_ms(sum(pus))}")

            split_ev = [row for row in sched_rows if row.get("phase") == "split_total"]
            if len(split_ev) >= 2:
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
                pct = round(100.0 * overlap / pairs, 1) if pairs else 0.0
                lines.append(
                    f"  assembly_overlap_count={overlap} pairs={pairs} overlap_pct={pct}"
                )

            by_copy: dict[str, list[dict]] = defaultdict(list)
            copy_order: list[str] = []
            for row in split_ev:
                copy = str(row.get("copy"))
                if copy not in by_copy:
                    copy_order.append(copy)
                by_copy[copy].append(row)
            if len(by_copy) > 1:
                for copy in copy_order:
                    lines.append(f"    copy{copy} splits={len(by_copy[copy])}")
    else:
        lines.append("")
        lines.append(f"[sched] missing {sched_file}")

    text = "\n".join(lines) + "\n"
    out_file.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"summary -> {out_file}")


if __name__ == "__main__":
    main()
PY