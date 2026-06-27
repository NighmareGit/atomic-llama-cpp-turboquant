#!/usr/bin/env python3
"""Estimate VRAM split for Path B RPC configs (72B+ load planning).

Usage:
  pathb-72b-vram-calc.py --gguf /mnt/models/Qwen3-72B-Instruct.IQ4_XS.gguf
  pathb-72b-vram-calc.py --config config-c --model-gb 38 --layers 80 --ctx 8192

Device index order with --rpc: RPC workers first (index 0..N-1), local CUDA last.
Config C remus-first: RPC0=5060 Ti, RPC1=3060 Ti, ROCm0=7900 XTX.
Config E: RPC0=remus 5060 Ti, CUDA0=Windows 5070 Ti.
Config F: RPC0=remus 5060 Ti, RPC1=remus RX 6600, CUDA0=Windows 5070 Ti.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

CONFIGS = {
    "config-a": {
        "name": "Config A (3060 Ti + 7900 XTX)",
        "vrams": [7.0, 22.0],
        "ts_default": [15, 85],
        "rpc_order": "3060,7900",
    },
    "remus": {
        "name": "Config B (remus 5060 Ti + 7900 XTX)",
        "vrams": [15.5, 22.0],
        "ts_default": [35, 65],
        "rpc_order": "5060,7900",
    },
    "config-c": {
        "name": "Config C remus-first (5060 + 3060 + 7900 XTX)",
        "vrams": [15.5, 7.0, 22.0],
        "ts_default": [35, 15, 50],
        "rpc_order": "5060,3060,7900",
    },
    "config-e": {
        "name": "Config E (remus 5060 Ti + Windows 5070 Ti)",
        "vrams": [15.5, 15.5],
        "ts_default": [50, 50],
        "rpc_order": "5060,5070",
    },
    "config-f": {
        "name": "Config F (remus 5060 + remus RX6600 + Windows 5070 Ti)",
        "vrams": [15.5, 7.5, 15.5],
        "ts_default": [30, 12, 58],
        "rpc_order": "5060,6600,5070",
    },
}

DEFAULT_MODEL_GB = 38.0
DEFAULT_LAYERS = 80
MARGIN_GB = 1.0
CPU_OFFLOAD_FRACS = (0.25, 0.30, 0.35, 0.40)


def _gguf_skip_value(f, vtype: int) -> object | None:
    """Skip or read one GGUF KV value (ggml/include/gguf.h enum)."""
    if vtype in (0, 1):
        f.read(1)
        return None
    if vtype in (2, 3):
        f.read(2)
        return None
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:
        f.read(4)
        return None
    if vtype == 7:
        f.read(1)
        return None
    if vtype == 8:
        vlen = struct.unpack("<Q", f.read(8))[0]
        return f.read(vlen).decode("utf-8", errors="replace")
    if vtype == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        esize = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: 0, 10: 8, 11: 8, 12: 8}.get(atype)
        if esize is None:
            raise ValueError(f"unknown GGUF array element type {atype}")
        if atype == 8:
            for _ in range(alen):
                slen = struct.unpack("<Q", f.read(8))[0]
                f.read(slen)
        else:
            f.read(alen * esize)
        return None
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:
        f.read(8)
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


def gguf_read_metadata(path: str) -> dict[str, object]:
    """Read GGUF KV metadata (no numpy). Returns arch, block_count if present."""
    out: dict[str, object] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        _n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8", errors="replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            val = _gguf_skip_value(f, vtype)
            if key == "general.architecture" and isinstance(val, str):
                out["arch"] = val
            if key.endswith(".block_count") and isinstance(val, int):
                out["block_count"] = int(val)

    return out


def layers_from_filename(path: str) -> int | None:
    low = path.lower()
    if "9b" in low or "-9-" in low:
        return 36
    if "12b" in low or "12-b" in low:
        return 42
    if "27b" in low:
        return 64
    if "31b" in low:
        return 64
    if "35b" in low or "36b" in low:
        return 64
    if "70b" in low or "72b" in low or "80b" in low:
        return 80
    if "4b" in low or "e4b" in low:
        return 34
    return None


def model_gb_from_path(path: str) -> float:
    return os.path.getsize(path) / (1024**3)


def kv_gb(layers: float, ctx: int, n_embd: int = 8192, kv_heads: int = 8, bytes_per_el: float = 0.5) -> float:
    head_dim = n_embd // 64
    per_layer = 2 * kv_heads * head_dim * ctx * bytes_per_el
    return layers * per_layer / 1e9


def ngl_candidates(n_layer: int) -> list[tuple[int, float]]:
    rows = []
    for frac in CPU_OFFLOAD_FRACS:
        ngl = int(n_layer * (1.0 - frac))
        ngl = max(1, min(n_layer, ngl))
        rows.append((ngl, frac))
    return sorted(set(rows), key=lambda x: -x[0])


def check_split(
    model_gb: float,
    layers: int,
    ctx: int,
    ngl: int,
    ts_parts: list[int],
    vrams: list[float],
) -> tuple[bool, list[float], float]:
    if ngl > layers:
        return False, [], 0.0
    gpu_frac = ngl / layers
    w_gpu = model_gb * gpu_frac
    w_cpu = model_gb * (1 - gpu_frac)
    s = sum(ts_parts)
    totals = []
    for ts, vram in zip(ts_parts, vrams):
        w_i = w_gpu * ts / s
        layers_i = ngl * ts / s
        kv_i = kv_gb(layers_i, ctx)
        tot = w_i + kv_i + MARGIN_GB
        totals.append(tot)
        if tot > vram:
            return False, totals, w_cpu
    return True, totals, w_cpu


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=list(CONFIGS), default="config-c")
    p.add_argument("--gguf", type=str, default="", help="read block_count + size from GGUF")
    p.add_argument("--model-gb", type=float, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ts", type=str, default="", help="override ts, e.g. 35,15,50")
    p.add_argument("--vrams", type=str, default="", help="override vrams GB, e.g. 15.5,7,22")
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    if args.vrams:
        vrams = [float(x) for x in args.vrams.split(",")]
    else:
        vrams = list(cfg["vrams"])

    if args.ts:
        ts_default = [int(x) for x in args.ts.split(",")]
    else:
        ts_default = list(cfg["ts_default"])

    model_gb = args.model_gb
    layers = args.layers
    arch = "unknown"
    if args.gguf:
        model_gb = model_gb if model_gb is not None else model_gb_from_path(args.gguf)
        try:
            meta = gguf_read_metadata(args.gguf)
            arch = str(meta.get("arch", "unknown"))
            if "block_count" in meta:
                layers = int(meta["block_count"])
        except Exception as e:
            print(f"WARN: GGUF metadata read failed: {e}", file=sys.stderr)
            guess = layers_from_filename(args.gguf)
            if guess is not None:
                layers = guess
            if "qwen" in args.gguf.lower():
                arch = "qwen35"
            elif "gemma" in args.gguf.lower():
                arch = "gemma4"
    if model_gb is None:
        model_gb = DEFAULT_MODEL_GB
    if layers is None:
        layers = DEFAULT_LAYERS

    total_vram = sum(vrams)
    cpu_min = max(0.0, model_gb - total_vram)

    print(f"{cfg['name']}")
    if args.gguf:
        print(f"  gguf:               {args.gguf}")
        print(f"  architecture:       {arch}")
    print(f"  weights (file):       {model_gb:.2f} GB")
    print(f"  layers:             {layers}")
    print(f"  ts (default):       {','.join(str(t) for t in ts_default)}")
    print(f"  rpc order:          {cfg['rpc_order']}")
    for i, v in enumerate(vrams):
        label = "RPC" if i < len(vrams) - 1 else "CUDA"
        print(f"  dev{i} ({label}):         {v:.2f} GB usable")
    print(f"  combined GPU:       {total_vram:.2f} GB")
    print(f"  CPU offload (min):  {cpu_min:.2f} GB weights (before KV)")
    print()
    print("Load strategy:")
    print("  1. Prefer --fit off + manual -ngl + -ts (better split than auto-fit)")
    print("  2. If load fails, retry --fit on -ngl 0 with same -ts")
    print("  3. Use --verbose -lv 4 (or higher) to capture per-device MiB during load")
    print()

    ts_candidates = [ts_default]
    if len(vrams) == 2:
        ts_candidates += [(50, 50), (45, 55), (40, 60), (35, 65), (30, 70)]
    else:
        ts_candidates += [(12, 8, 80), (10, 25, 65)]

    seen_ts: set[tuple[int, ...]] = set()
    uniq_ts: list[list[int]] = []
    for t in ts_candidates:
        key = tuple(t)
        if key not in seen_ts:
            seen_ts.add(key)
            uniq_ts.append(list(t))

    print("ngl candidates (25-40% CPU weight offload):")
    for ngl, frac in ngl_candidates(layers):
        print(f"  ngl={ngl:3d}  cpu_offload={frac*100:.0f}%")
    print()

    print(f"{'ngl':>4} {'cpu%':>5} {'ts':>12} {'ctx':>5} | per-dev GB (weights+KV+margin) | ok")
    for ngl, frac in ngl_candidates(layers):
        for ts_parts in uniq_ts:
            if len(ts_parts) != len(vrams):
                continue
            ok, totals, w_cpu = check_split(model_gb, layers, args.ctx, ngl, ts_parts, vrams)
            ts_s = ",".join(str(t) for t in ts_parts)
            tot_s = " ".join(f"{t:.1f}" for t in totals) if totals else "-"
            flag = "OK" if ok else "OOM"
            print(f"{ngl:4d} {frac*100:4.0f}% {ts_s:>12} {args.ctx:5d} | {tot_s} cpu_w={w_cpu:.1f}GB | {flag}")

    print()
    ts_s = ",".join(str(t) for t in ts_default)
    fit_targets = ",".join(str(int(v * 40)) for v in vrams)
    print("Recommended bench env (manual split, fit off):")
    ngl_rec = ngl_candidates(layers)[1][0] if len(ngl_candidates(layers)) > 1 else ngl_candidates(layers)[0][0]
    print(f"  BENCH_NGL={ngl_rec} BENCH_CTX={args.ctx} BENCH_TS={ts_s} BENCH_CTK=q4_0 BENCH_CTV=q4_0 \\")
    print("  BENCH_EXTRA='--fit off --verbose -lv 4 --reasoning off' \\")
    print("  BENCH_RPC_ENDPOINT='192.168.8.176:50051,127.0.0.1:50051' \\")
    print("  ./rpc-patch/scripts/pathb-72b-matrix.sh --phase 1 qwen72b")
    print()
    print("Fallback (fit on, same ts):")
    print(f"  BENCH_NGL=0 BENCH_EXTRA='--fit on --fit-target {fit_targets} --verbose -lv 4 --reasoning off'")


if __name__ == "__main__":
    main()