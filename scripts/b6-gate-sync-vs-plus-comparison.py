#!/usr/bin/env python3
"""Refresh path-b-plus-vs-sync-comparison results from sync-vs-plus-comparison.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fmt_g(v) -> str:
    if v is None:
        return "TBD"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "TBD"


def delta_pct(sync_g, plus_g) -> str:
    try:
        s, p = float(sync_g), float(plus_g)
        if s <= 0:
            return "TBD"
        return f"{(p - s) / s * 100:+.1f}%"
    except (TypeError, ValueError):
        return "TBD"


def render_table(
    by_key: dict[tuple, dict[str, dict]],
    topo: str,
    workload: str,
    model_ids: tuple[str, ...],
    model_names: dict[str, str],
) -> list[str]:
    lines = [
        f"### {topo} — workload `{workload}`",
        "",
        "| Model ID | Model | SYNC G | PLUS G | Delta% | Status |",
        "|----------|-------|--------|--------|--------|--------|",
    ]
    for mid in model_ids:
        if topo == "b6-3gpu-g-triton" and mid in ("M6", "M7", "M8"):
            continue
        arms = by_key.get((topo, workload, mid), {})
        sync = arms.get("sync", {})
        plus = arms.get("plus", {})
        sg, pg = sync.get("G_tps"), plus.get("G_tps")
        st = sync.get("status", "-"), plus.get("status", "-")
        status = f"S:{st[0]} P:{st[1]}"
        lines.append(
            f"| {mid} | {model_names.get(mid, mid)} | {fmt_g(sg)} | {fmt_g(pg)} | "
            f"{delta_pct(sg, pg)} | {status} |"
        )
    lines.append("")
    return lines


def build_markdown(rows: list[dict]) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in ok:
        key = (r.get("topology"), r.get("workload", "n384"), r.get("model_id"))
        by_key[key][r.get("arm", "")] = r

    model_names = {
        "M1": "Qwen3.6-35B-A3B",
        "M3": "Gemma-4-26B-A4B",
        "M5": "Qwen3.5-27B",
        "M6": "Llama-3-70B Q4",
        "M7": "Qwen3-Next-80B",
        "M8": "Kimi-Dev-72B",
    }
    model_ids = ("M1", "M3", "M5", "M6", "M7", "M8")
    workloads = []
    for wl in ("n384", "n2048-mt"):
        if any(k[1] == wl for k in by_key):
            workloads.append(wl)
    if not workloads:
        workloads = ["n384", "n2048-mt"]

    campaigns = sorted({r.get("campaign") for r in rows if r.get("campaign")})
    camp_s = campaigns[-1] if campaigns else "-"

    lines = [
        "# Path-B+ vs SYNC — generated results",
        "",
        f"Rows: {len(ok)} ok / {len(rows)} total | last campaign: `{camp_s}`",
        "",
        "Regenerate:",
        "```bash",
        "python3 scripts/b6-gate-sync-vs-plus-comparison.py",
        "```",
        "",
    ]

    for wl in workloads:
        wl_title = "n=384 gate depth" if wl == "n384" else "n=2048 multi-turn"
        lines.append(f"## Workload: {wl} ({wl_title})")
        lines.append("")
        for topo in ("b6-3gpu-g-triton", "b6-5gpu-g-prod"):
            lines.extend(
                render_table(by_key, topo, wl, model_ids, model_names)
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=root / "benches/path-b-plus/sync-vs-plus-comparison.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root
        / "docs/rpc-multi-backend-pipeline-plus/BENCHMARKS/path-b-plus-vs-sync-results.md",
    )
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    md = build_markdown(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()