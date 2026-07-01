#!/usr/bin/env python3
"""Read-only sched-trace audit: serial vs concurrent backend activity per decode.

Sweeps split_total intervals per decode_id and measures wall time with exactly one
backend active vs two or more.  Intended to prove (or refute) serial split dispatch
before scheduler overlap work.

usage:
  python3 scripts/b6-gate-overlap-serial-audit.py [telemetry_dir]
  python3 scripts/b6-gate-overlap-serial-audit.py path/to/sched-trace.jsonl --json out.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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
    raw = json.loads(p.read_text(encoding="utf-8"))
    return raw.get("diagnose", raw)


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(p * len(s)) - 1)))
    return round(s[idx], 2)


def split_intervals(rows: Iterable[dict], *, min_decode: int = 1) -> dict[int, list[tuple[int, int, int]]]:
    """Per decode_id: (start_us, end_us, backend) from split_total."""
    by_decode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "split_total":
            continue
        did = int(row.get("decode_id", 0))
        if did < min_decode:
            continue
        start = int(row.get("ts_us", 0))
        end = start + int(row.get("elapsed_us", 0))
        backend = int(row.get("backend", -1))
        by_decode[did].append((start, end, backend))
    return by_decode


def sweep_timeline(intervals: list[tuple[int, int, int]]) -> dict[str, int]:
    """Return single_us, multi_us, idle_us, and multi_by_n (concurrent backend count)."""
    if not intervals:
        return {
            "single_us": 0,
            "multi_us": 0,
            "idle_us": 0,
            "multi_2_us": 0,
            "multi_3_us": 0,
            "multi_4p_us": 0,
        }

    events: list[tuple[int, str, int]] = []
    for start, end, backend in intervals:
        events.append((start, "start", backend))
        events.append((end, "end", backend))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

    active: set[int] = set()
    prev = events[0][0]
    single_us = multi_us = idle_us = 0
    multi_2_us = multi_3_us = multi_4p_us = 0

    for t, kind, backend in events:
        dt = t - prev
        if dt > 0:
            n = len(active)
            if n == 0:
                idle_us += dt
            elif n == 1:
                single_us += dt
            else:
                multi_us += dt
                if n == 2:
                    multi_2_us += dt
                elif n == 3:
                    multi_3_us += dt
                else:
                    multi_4p_us += dt
        if kind == "start":
            active.add(backend)
        else:
            active.discard(backend)
        prev = t

    return {
        "single_us": single_us,
        "multi_us": multi_us,
        "idle_us": idle_us,
        "multi_2_us": multi_2_us,
        "multi_3_us": multi_3_us,
        "multi_4p_us": multi_4p_us,
    }


def assembly_overlap_pct(rows: list[dict], *, min_decode: int = 1) -> tuple[int, int, float]:
    """Phase 1.2B cross-backend pair metric from split_total rows."""
    split_ev = [
        row
        for row in rows
        if row.get("phase") == "split_total" and int(row.get("decode_id", 0)) >= min_decode
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
    pct_val = round(100.0 * overlap / pairs, 1) if pairs else 0.0
    return overlap, pairs, pct_val


def audit(telem: Path, *, min_decode: int = 1, label: str = "") -> dict:
    sched_path = telem if telem.name.endswith(".jsonl") else telem / "sched-trace.jsonl"
    sched = load_jsonl(sched_path)
    diag = load_diag(telem if not telem.name.endswith(".jsonl") else telem.parent)
    by_decode = split_intervals(sched, min_decode=min_decode)

    totals = defaultdict(int)
    per_decode_multi_pct: list[float] = []
    per_decode_serial_pct: list[float] = []
    split_counts: list[int] = []

    for did, intervals in sorted(by_decode.items()):
        sw = sweep_timeline(intervals)
        local_total = sw["single_us"] + sw["multi_us"] + sw["idle_us"]
        if local_total <= 0:
            continue
        for k, v in sw.items():
            totals[k] += v
        split_counts.append(len(intervals))
        active = sw["single_us"] + sw["multi_us"]
        if active > 0:
            per_decode_multi_pct.append(round(100.0 * sw["multi_us"] / active, 2))
            per_decode_serial_pct.append(round(100.0 * sw["single_us"] / active, 2))

    wall_lo = min(min(iv[0] for iv in ivs) for ivs in by_decode.values()) if by_decode else 0
    wall_hi = max(max(iv[1] for iv in ivs) for ivs in by_decode.values()) if by_decode else 0
    wall_us = max(1, wall_hi - wall_lo)
    serial_sum_us = sum(end - start for ivs in by_decode.values() for start, end, _ in ivs)

    total_us = totals["single_us"] + totals["multi_us"] + totals["idle_us"]
    active_us = totals["single_us"] + totals["multi_us"]
    asm_count, asm_pairs, asm_pct = assembly_overlap_pct(sched, min_decode=min_decode)

    def pct_of(num: int, den: int) -> float:
        return round(100.0 * num / den, 2) if den else 0.0

    serial_dispatch_pct = pct_of(totals["single_us"] + totals["idle_us"], total_us)
    overlap_timeline_pct = pct_of(totals["multi_us"], total_us)
    single_of_active_pct = pct_of(totals["single_us"], active_us)

    verdict = "SERIAL_DISPATCH"
    if overlap_timeline_pct > 10.0:
        verdict = "MATERIAL_OVERLAP"
    elif serial_dispatch_pct < 90.0:
        verdict = "MIXED"

    out = {
        "label": label or telem.parent.name,
        "telemetry": str(telem if not telem.name.endswith(".jsonl") else telem.parent),
        "sched_trace": str(sched_path),
        "min_decode_id": min_decode,
        "gen_tokens": len(by_decode),
        "split_total_per_decode": {
            "min": min(split_counts) if split_counts else 0,
            "max": max(split_counts) if split_counts else 0,
            "p50": pct([float(x) for x in split_counts], 0.50),
        },
        "timeline_us": {
            "single": totals["single_us"],
            "multi": totals["multi_us"],
            "idle": totals["idle_us"],
            "total": total_us,
        },
        "timeline_pct": {
            "single": pct_of(totals["single_us"], total_us),
            "multi": overlap_timeline_pct,
            "idle": pct_of(totals["idle_us"], total_us),
            "single_of_active": single_of_active_pct,
            "serial_dispatch": serial_dispatch_pct,
        },
        "multi_concurrency_us": {
            "two_backends": totals["multi_2_us"],
            "three_backends": totals["multi_3_us"],
            "four_plus_backends": totals["multi_4p_us"],
        },
        "per_decode_active_multi_pct": {
            "p50": pct(per_decode_multi_pct, 0.50),
            "p95": pct(per_decode_multi_pct, 0.95),
            "max": max(per_decode_multi_pct) if per_decode_multi_pct else 0.0,
        },
        "wall_us": wall_us,
        "serial_sum_us": serial_sum_us,
        "overlap_efficiency": round(serial_sum_us / wall_us, 4),
        "assembly_overlap": {
            "count": asm_count,
            "pairs": asm_pairs,
            "pct": asm_pct,
        },
        "diagnose_overlap_pct": diag.get("overlap_pct"),
        "diagnose_G_tps": diag.get("G_tps"),
        "verdict": verdict,
        "notes": [
            "serial_dispatch_pct = (single + idle) / total; >90% means concurrent backends are rare on the wall clock.",
            "timeline multi_pct = wall time with 2+ backends active (split_total intervals only).",
            "assembly_overlap pct matches phase12b pair-start metric; low values are consistent with brief tail overlaps.",
        ],
    }
    return out


def print_report(r: dict) -> None:
    tp = r["timeline_pct"]
    print(f"=== serial overlap audit: {r['label']} ===")
    print(f"  gen_tokens={r['gen_tokens']} splits/decode p50={r['split_total_per_decode']['p50']}")
    print(
        f"  timeline: single={tp['single']}% multi={tp['multi']}% idle={tp['idle']}% "
        f"(single_of_active={tp['single_of_active']}%)"
    )
    print(f"  serial_dispatch_pct={tp['serial_dispatch']}% (target >90% => serial dispatch)")
    mc = r["multi_concurrency_us"]
    print(
        f"  multi_us: 2bk={mc['two_backends']} 3bk={mc['three_backends']} "
        f"4+bk={mc['four_plus_backends']}"
    )
    print(
        f"  per_decode active multi p50={r['per_decode_active_multi_pct']['p50']}% "
        f"p95={r['per_decode_active_multi_pct']['p95']}%"
    )
    ao = r["assembly_overlap"]
    print(f"  assembly_overlap={ao['count']}/{ao['pairs']} ({ao['pct']}%)")
    print(
        f"  wall_ms={r['wall_us']/1000:.1f} serial_sum_ms={r['serial_sum_us']/1000:.1f} "
        f"eff={r['overlap_efficiency']}"
    )
    if r.get("diagnose_overlap_pct") is not None:
        print(f"  diagnose overlap_pct={r['diagnose_overlap_pct']}% G_tps={r.get('diagnose_G_tps')}")
    print(f"  verdict: {r['verdict']}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    path = Path(args[0])
    json_out: Path | None = None
    min_decode = 1
    label = ""
    i = 1
    while i < len(args):
        if args[i] == "--json" and i + 1 < len(args):
            json_out = Path(args[i + 1])
            i += 2
        elif args[i] == "--min-decode" and i + 1 < len(args):
            min_decode = int(args[i + 1])
            i += 2
        elif args[i] == "--label" and i + 1 < len(args):
            label = args[i + 1]
            i += 2
        else:
            print(f"unknown arg: {args[i]}", file=sys.stderr)
            sys.exit(2)

    telem = path
    if telem.is_dir():
        if not (telem / "sched-trace.jsonl").is_file():
            print(f"missing sched-trace.jsonl under {telem}", file=sys.stderr)
            sys.exit(1)
    elif not telem.is_file():
        print(f"not found: {telem}", file=sys.stderr)
        sys.exit(1)

    report = audit(telem, min_decode=min_decode, label=label)
    print_report(report)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {json_out}")


if __name__ == "__main__":
    main()