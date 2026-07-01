#!/usr/bin/env python3
"""Summarize sync_copy_detail rows from sched-trace.jsonl (B+13 instrumentation)."""
import json
import sys
from collections import Counter, defaultdict

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "telemetry/sched-trace.jsonl"
    heavy_us = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    by_key = Counter()
    heavy = []
    rejects = Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("phase") != "sync_copy_detail":
                continue
            reject = row.get("reject", "?")
            rejects[reject] += 1
            key = (
                row.get("tensor", ""),
                row.get("src_buft", ""),
                row.get("dst_buft", ""),
                row.get("input_backend", ""),
                reject,
            )
            by_key[key] += 1
            if row.get("elapsed_us", 0) > heavy_us:
                heavy.append(row)

    print(f"=== sync_copy_detail audit: {path} ===")
    print(f"heavy (>{heavy_us}us): {len(heavy)}")
    print("reject counts:")
    for k, v in rejects.most_common():
        print(f"  {k}: {v}")
    print("top tensor/buft/reject (all sync):")
    for key, n in by_key.most_common(15):
        print(f"  {n:5d}  tensor={key[0]!r} src={key[1]!r} dst={key[2]!r} in_b={key[3]!r} reject={key[4]}")
    if heavy:
        print(f"heavy sample (first 5 of {len(heavy)}):")
        for row in heavy[:5]:
            print(
                f"  decode={row.get('decode_id')} split={row.get('split')} "
                f"tensor={row.get('tensor')!r} nbytes={row.get('nbytes')} "
                f"src={row.get('src_buft')!r} dst={row.get('dst_buft')!r} "
                f"rpc={row.get('rpc_src')}/{row.get('rpc_dst')} "
                f"reject={row.get('reject')} elapsed_us={row.get('elapsed_us')}"
            )

if __name__ == "__main__":
    main()