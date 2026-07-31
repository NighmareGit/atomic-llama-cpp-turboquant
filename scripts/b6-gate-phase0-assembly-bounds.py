#!/usr/bin/env python3
"""Phase 0: assembly-line bound analysis from sched-trace (read-only).

Computes per-decode vs global timeline concurrency, theoretical intra/cross
token overlap ceilings, split granularity, and duration-weighted overlap.

usage:
  python3 scripts/b6-gate-phase0-assembly-bounds.py path/to/telemetry
  python3 scripts/b6-gate-phase0-assembly-bounds.py --tier-a-matrix DIR [--json OUT]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Reuse sweep helpers from serial audit (inline to keep one entrypoint).
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


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(p * len(s)) - 1)))
    return round(s[idx], 2)


def split_rows(rows: Iterable[dict], *, min_decode: int = 1) -> list[tuple[int, int, int, int, int]]:
    """(start, end, backend, decode_id, split_id-ish from order)."""
    out: list[tuple[int, int, int, int, int]] = []
    per_decode_idx: dict[int, int] = defaultdict(int)
    for row in rows:
        if row.get("phase") != "split_total":
            continue
        did = int(row.get("decode_id", 0))
        if did < min_decode:
            continue
        start = int(row.get("ts_us", 0))
        end = start + int(row.get("elapsed_us", 0))
        backend = int(row.get("backend", -1))
        sid = per_decode_idx[did]
        per_decode_idx[did] += 1
        out.append((start, end, backend, did, sid))
    return out


def sweep(intervals: list[tuple[int, int, int]]) -> dict[str, int]:
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


def timeline_pcts(sw: dict[str, int]) -> dict[str, float]:
    total = sw["single_us"] + sw["multi_us"] + sw["idle_us"]
    active = sw["single_us"] + sw["multi_us"]

    def p(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    return {
        "single_pct": p(sw["single_us"], total),
        "multi_pct": p(sw["multi_us"], total),
        "idle_pct": p(sw["idle_us"], total),
        "serial_dispatch_pct": p(sw["single_us"] + sw["idle_us"], total),
        "multi_2_pct": p(sw["multi_2_us"], total),
        "multi_3_pct": p(sw["multi_3_us"], total),
        "multi_4p_pct": p(sw["multi_4p_us"], total),
        "single_of_active_pct": p(sw["single_us"], active),
    }


def duration_weighted_overlap(intervals: list[tuple[int, int, int, int, int]]) -> dict:
    """Wall-time overlap: fraction of timeline where 2+ backends active."""
    iv2 = [(a, b, c) for a, b, c, _, _ in intervals]
    sw = sweep(iv2)
    tp = timeline_pcts(sw)
    return {
        "multi_wall_pct": tp["multi_pct"],
        "multi_3_wall_pct": tp["multi_3_pct"],
        "serial_dispatch_pct": tp["serial_dispatch_pct"],
    }


def intra_token_bound(by_decode: dict[int, list[tuple[int, int, int]]]) -> dict:
    """Ideal pipeline within token: adjacent splits overlap maximally."""
    bounds: list[float] = []
    for did, ivs in sorted(by_decode.items()):
        if len(ivs) < 2:
            continue
        ivs = sorted(ivs, key=lambda x: x[0])
        token_wall = max(e for _, e, _ in ivs) - min(s for s, _, _ in ivs)
        if token_wall <= 0:
            continue
        serial_sum = sum(e - s for s, e, _ in ivs)
        # Perfect overlap lower bound: wall cannot be less than longest split.
        longest = max(e - s for s, e, _ in ivs)
        max_multi = serial_sum - longest
        bounds.append(round(100.0 * max_multi / token_wall, 2))

    return {
        "per_token_max_multi_pct_p50": pct(bounds, 0.50),
        "per_token_max_multi_pct_p95": pct(bounds, 0.95),
        "note": "Theoretical ceiling if splits pipeline with zero gap (W1 perfect).",
    }


def cross_token_bound(rows: list[tuple[int, int, int, int, int]], n_backends: int) -> dict:
    """Stack tokens: ideal wall = max backend busy time across overlapping tokens."""
    if not rows:
        return {}
    by_backend: dict[int, int] = defaultdict(int)
    for start, end, backend, _, _ in rows:
        by_backend[backend] += end - start

    wall = max(end for start, end, _, _, _ in rows) - min(start for start, _, _, _, _ in rows)
    serial_sum = sum(end - start for start, end, _, _, _ in rows)
    busy_max = max(by_backend.values()) if by_backend else 0
    # If n_backends work in parallel across tokens, wall approaches busy_max not serial_sum.
    ideal_multi_frac = 1.0 - (busy_max / serial_sum) if serial_sum > 0 else 0.0
    return {
        "wall_us": wall,
        "serial_sum_us": serial_sum,
        "busy_max_backend_us": busy_max,
        "n_backends_seen": len(by_backend),
        "cross_token_ideal_multi_pct": round(100.0 * max(0.0, ideal_multi_frac), 2),
        "note": "Upper bound if tokens stack across backends (W2 perfect); needs global_multi_3 to realize.",
    }


def split_granularity(rows: list[tuple[int, int, int, int, int]]) -> dict:
    by_backend_us: dict[int, list[int]] = defaultdict(list)
    decode_split_count: dict[int, int] = defaultdict(int)
    for start, end, backend, did, _sid in rows:
        by_backend_us[backend].append(end - start)
        decode_split_count[did] += 1

    backend_share: dict[str, float] = {}
    total_us = sum(end - start for start, end, _, _, _ in rows)
    for b, times in sorted(by_backend_us.items()):
        s = sum(times)
        backend_share[f"backend{b}"] = round(100.0 * s / total_us, 1) if total_us else 0.0

    return {
        "splits_per_decode_p50": pct([float(x) for x in decode_split_count.values()], 0.50),
        "splits_per_decode_max": max(decode_split_count.values()) if decode_split_count else 0,
        "backend_time_share_pct": backend_share,
        "backend_split_ms_p50": {
            f"backend{b}": round(pct([float(x) for x in t], 0.50) / 1000.0, 3)
            for b, t in sorted(by_backend_us.items())
        },
    }


def analyze_telemetry(telem: Path, *, label: str = "", min_decode: int = 1) -> dict:
    sched_path = telem / "sched-trace.jsonl" if telem.is_dir() else telem
    sched = load_jsonl(sched_path)
    rows = split_rows(sched, min_decode=min_decode)

    by_decode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for start, end, backend, did, _sid in rows:
        by_decode[did].append((start, end, backend))

    per_decode_sw = {"single_us": 0, "multi_us": 0, "idle_us": 0, "multi_2_us": 0, "multi_3_us": 0, "multi_4p_us": 0}
    for ivs in by_decode.values():
        sw = sweep(ivs)
        for k, v in sw.items():
            per_decode_sw[k] += v

    global_iv = [(s, e, b) for s, e, b, _, _ in rows]
    global_sw = sweep(global_iv)

    diag_path = telem / "diagnose.json" if telem.is_dir() else telem.parent / "diagnose.json"
    diag = {}
    if diag_path.is_file():
        raw = json.loads(diag_path.read_text())
        diag = raw.get("diagnose", raw)

    per_tp = timeline_pcts(per_decode_sw)
    glob_tp = timeline_pcts(global_sw)

    return {
        "label": label or telem.parent.name,
        "telemetry": str(telem),
        "gen_tokens": len(by_decode),
        "per_decode_timeline_pct": per_tp,
        "global_timeline_pct": glob_tp,
        "duration_weighted": duration_weighted_overlap(rows),
        "intra_token_bound": intra_token_bound(by_decode),
        "cross_token_bound": cross_token_bound(rows, n_backends=len({b for _, _, b, _, _ in rows})),
        "split_granularity": split_granularity(rows),
        "gap_global_vs_per_decode_multi_pct": round(
            glob_tp["multi_pct"] - per_tp["multi_pct"], 2
        ),
        "diagnose": {
            "overlap_pct": diag.get("overlap_pct"),
            "G_tps": diag.get("G_tps"),
            "straggler_backend": diag.get("straggler_backend"),
        },
        "work_package_hint": classify_work_package(per_tp, glob_tp),
    }


def classify_work_package(per: dict[str, float], glob: dict[str, float]) -> str:
    intra_max = per.get("multi_pct", 0)
    global_multi = glob.get("multi_pct", 0)
    global_3 = glob.get("multi_3_pct", 0)
    if global_3 < 1.0 and intra_max < 15.0:
        return "W1+W2+depth: global 3-backend concurrency near zero; bundle wavefront + cross-decode launch"
    if global_multi > intra_max + 5.0:
        return "W2-first: global multi exceeds per-decode; cross-token dispatch is main gap"
    if intra_max < 12.0:
        return "W1+W2: intra ceiling low; need cross-token stack for cluster fill"
    return "W1+W2: mixed; factorial still required"


def find_tier_a_telemetry(matrix_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for child in sorted(matrix_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("A") or "-b6-" not in name:
            continue
        telem = child / "telemetry"
        if (telem / "sched-trace.jsonl").is_file():
            aid = name.split("-")[0]
            out.append((aid, telem))
    return out


def print_row(r: dict) -> None:
    pd = r["per_decode_timeline_pct"]
    gl = r["global_timeline_pct"]
    ib = r["intra_token_bound"]
    cb = r["cross_token_bound"]
    g = r["diagnose"].get("G_tps")
    g_str = f"{g:.4g}" if isinstance(g, (int, float)) else "?"
    print(
        f"{r['label']:6} G={g_str:>7} "
        f"ov={r['diagnose'].get('overlap_pct','?')}% "
        f"per_multi={pd['multi_pct']:>5}% glob_multi={gl['multi_pct']:>5}% "
        f"glob_3bk={gl['multi_3_pct']:>5}% "
        f"intra_max_p50={ib['per_token_max_multi_pct_p50']:>5}% "
        f"cross_ideal={cb.get('cross_token_ideal_multi_pct','?')}% "
        f"| {r['work_package_hint']}"
    )


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    json_out: Path | None = None
    tier_a: Path | None = None
    paths: list[Path] = []
    i = 0
    while i < len(args):
        if args[i] == "--json" and i + 1 < len(args):
            json_out = Path(args[i + 1])
            i += 2
        elif args[i] == "--tier-a-matrix" and i + 1 < len(args):
            tier_a = Path(args[i + 1])
            i += 2
        else:
            paths.append(Path(args[i]))
            i += 1

    reports: list[dict] = []
    if tier_a:
        for aid, telem in find_tier_a_telemetry(tier_a):
            reports.append(analyze_telemetry(telem, label=aid))
    for p in paths:
        telem = p if p.name == "telemetry" else (p / "telemetry" if (p / "telemetry").is_dir() else p)
        reports.append(analyze_telemetry(telem, label=p.parent.name))

    if not reports:
        print("no telemetry found", file=sys.stderr)
        sys.exit(1)

    print("=== Phase 0 assembly bounds ===")
    for r in reports:
        print_row(r)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({"reports": reports}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {json_out}")


if __name__ == "__main__":
    main()