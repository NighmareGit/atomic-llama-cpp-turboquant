#!/usr/bin/env python3
"""Live VRAM preflight for Path B RPC topologies.

Probes cluster GPUs (nvidia-smi, rocm-smi, --validate-rpc) and plans tensor-split
/ -ngl before bench runs or deployments.

Usage:
  pathb-rpc-vram-preflight.py --preset b6-4gpu-g-triton --gguf /mnt/models/foo.gguf
  pathb-rpc-vram-preflight.py --preset b6-2gpu-f-triton --model-gb 21 --layers 64
  pathb-rpc-vram-preflight.py --config config-c --gguf foo.gguf --no-live

Presets match scripts/b6-gate-profiler-romulus.sh labels. Static planning only
(without --live) defers to pathb-72b-vram-calc.py budgets.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_vram_calc():
    spec = importlib.util.spec_from_file_location("pathb_vram_calc", SCRIPT_DIR / "pathb-72b-vram-calc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VRAM = _load_vram_calc()

CLUSTER_PASS = os.environ.get("B6_CLUSTER_PASS", os.environ.get("PATHB_ROMULUS_SSH_PASS", "12345"))
ROMULUS_HOST = os.environ.get("B6_ROMULUS_HOST", "192.168.8.108")
REMUS_HOST = os.environ.get("B6_REMUS_HOST", "192.168.8.176")
TRITON_HOST = os.environ.get("B6_TRITON_HOST", "192.168.8.23")
JUPITER_HOST = os.environ.get("B6_JUPITER_HOST", "192.168.8.21")
SSH_USER = os.environ.get("B6_CLUSTER_USER", "hunter")
JUPITER_USER = os.environ.get("B6_JUPITER_USER", "nightmare")
JUPITER_PASS = os.environ.get("B6_JUPITER_PASS", CLUSTER_PASS)

PROFILER_CANDIDATES = [
    os.environ.get("PATHB_ROMULUS_PROFILER", ""),
    f"/home/{SSH_USER}/atomic-llama-cpp-turboquant/build-rocm-docker/bin/llama-pipeline-profiler",
    str(SCRIPT_DIR.parent.parent / "build-rocm-docker/bin/llama-pipeline-profiler"),
]


@dataclass
class DeviceSpec:
    label: str
    kind: str  # rpc | nvidia | rocm
    host: str = ""
    endpoint: str = ""
    gpu_index: int = 0
    via_host: str = ""  # ssh host to run validate-rpc (romulus for 127.0.0.1)
    fallback_host: str = ""  # nvidia-smi host when validate-rpc unavailable


@dataclass
class PresetSpec:
    name: str
    rpc: str
    devices: list[DeviceSpec]
    ts_default: list[int]
    vrams_static: list[float]
    config: str = ""


PRESETS: dict[str, PresetSpec] = {
    "b6-2gpu-f": PresetSpec(
        name="b6-2gpu-f (romulus 7900 client + remus 5060 RPC)",
        rpc=f"{REMUS_HOST}:50051",
        devices=[
            DeviceSpec("RPC0 remus 5060", "rpc", endpoint=f"{REMUS_HOST}:50051", fallback_host=REMUS_HOST),
            DeviceSpec("ROCm0 romulus 7900", "rocm", host=ROMULUS_HOST),
        ],
        ts_default=[50, 50],
        vrams_static=[15.5, 22.0],
        config="remus",
    ),
    "b6-2gpu-f-triton": PresetSpec(
        name="b6-2gpu-f-triton (romulus 7900 client + triton 3090 RPC)",
        rpc=f"{TRITON_HOST}:50054",
        devices=[
            DeviceSpec("RPC0 triton 3090", "rpc", endpoint=f"{TRITON_HOST}:50054", fallback_host=TRITON_HOST),
            DeviceSpec("ROCm0 romulus 7900", "rocm", host=ROMULUS_HOST),
        ],
        ts_default=[50, 50],
        vrams_static=[23.0, 22.0],
        config="remus",
    ),
    "b6-3gpu-g-triton": PresetSpec(
        name="b6-3gpu-g-triton (remus 5060 + triton 3090 + romulus 7900)",
        rpc=f"{REMUS_HOST}:50051,{TRITON_HOST}:50054",
        devices=[
            DeviceSpec("RPC0 remus 5060", "rpc", endpoint=f"{REMUS_HOST}:50051", fallback_host=REMUS_HOST),
            DeviceSpec("RPC1 triton 3090", "rpc", endpoint=f"{TRITON_HOST}:50054", fallback_host=TRITON_HOST),
            DeviceSpec("ROCm0 romulus 7900", "rocm", host=ROMULUS_HOST),
        ],
        ts_default=[30, 35, 35],
        vrams_static=[15.5, 23.0, 22.0],
        config="config-g",
    ),
    "b6-4gpu-g-triton": PresetSpec(
        name="b6-4gpu-g-triton (remus 5060 + romulus 3060 + triton 3090 + romulus 7900)",
        rpc=f"{REMUS_HOST}:50051,127.0.0.1:50051,{TRITON_HOST}:50054",
        devices=[
            DeviceSpec("RPC0 remus 5060", "rpc", endpoint=f"{REMUS_HOST}:50051", fallback_host=REMUS_HOST),
            DeviceSpec(
                "RPC1 romulus 3060 docker",
                "rpc",
                endpoint="127.0.0.1:50051",
                via_host=ROMULUS_HOST,
                fallback_host=ROMULUS_HOST,
            ),
            DeviceSpec("RPC2 triton 3090", "rpc", endpoint=f"{TRITON_HOST}:50054", fallback_host=TRITON_HOST),
            DeviceSpec("ROCm0 romulus 7900", "rocm", host=ROMULUS_HOST),
        ],
        ts_default=[22, 11, 34, 33],
        vrams_static=[15.5, 7.0, 23.0, 22.0],
        config="config-g",
    ),
    "b6-5gpu-g": PresetSpec(
        name="b6-5gpu-g Linux (remus 5060 + romulus 3060/7900 + triton 3090/3070)",
        rpc=f"{REMUS_HOST}:50051,127.0.0.1:50051,{TRITON_HOST}:50054,{TRITON_HOST}:50055",
        devices=[
            DeviceSpec("RPC0 remus 5060", "rpc", endpoint=f"{REMUS_HOST}:50051", fallback_host=REMUS_HOST),
            DeviceSpec(
                "RPC1 romulus 3060 docker",
                "rpc",
                endpoint="127.0.0.1:50051",
                via_host=ROMULUS_HOST,
                fallback_host=ROMULUS_HOST,
            ),
            DeviceSpec("RPC2 triton 3090", "rpc", endpoint=f"{TRITON_HOST}:50054", fallback_host=TRITON_HOST),
            DeviceSpec("RPC3 triton 3070", "rpc", endpoint=f"{TRITON_HOST}:50055", fallback_host=TRITON_HOST),
            DeviceSpec("ROCm0 romulus 7900", "rocm", host=ROMULUS_HOST),
        ],
        ts_default=[20, 10, 30, 10, 30],
        vrams_static=[15.5, 7.0, 23.0, 7.5, 22.0],
        config="config-g",
    ),
    "b6-6gpu-g": PresetSpec(
        name="b6-6gpu-g Linux (5-GPU prod + jupiter 5070 Ti :50053)",
        rpc=f"{REMUS_HOST}:50051,127.0.0.1:50051,{TRITON_HOST}:50054,{TRITON_HOST}:50055,192.168.8.21:50053",
        devices=[
            DeviceSpec("RPC0 remus 5060", "rpc", endpoint=f"{REMUS_HOST}:50051", fallback_host=REMUS_HOST),
            DeviceSpec(
                "RPC1 romulus 3060 docker",
                "rpc",
                endpoint="127.0.0.1:50051",
                via_host=ROMULUS_HOST,
                fallback_host=ROMULUS_HOST,
            ),
            DeviceSpec("RPC2 triton 3090", "rpc", endpoint=f"{TRITON_HOST}:50054", fallback_host=TRITON_HOST),
            DeviceSpec("RPC3 triton 3070", "rpc", endpoint=f"{TRITON_HOST}:50055", fallback_host=TRITON_HOST),
            DeviceSpec("RPC4 jupiter 5070", "rpc", endpoint="192.168.8.21:50053", fallback_host="192.168.8.21"),
            DeviceSpec("ROCm0 romulus 7900", "rocm", host=ROMULUS_HOST),
        ],
        ts_default=[16, 10, 30, 10, 16, 22],
        vrams_static=[15.5, 7.0, 23.0, 7.5, 15.5, 22.0],
        config="config-g",
    ),
    "config-c": PresetSpec(
        name="Config C (remus 5060 + romulus 3060 docker + romulus 7900 client)",
        rpc=f"{REMUS_HOST}:50051,127.0.0.1:50051",
        devices=[
            DeviceSpec("RPC0 remus 5060", "rpc", endpoint=f"{REMUS_HOST}:50051", fallback_host=REMUS_HOST),
            DeviceSpec(
                "RPC1 romulus 3060 docker",
                "rpc",
                endpoint="127.0.0.1:50051",
                via_host=ROMULUS_HOST,
                fallback_host=ROMULUS_HOST,
            ),
            DeviceSpec("ROCm0 romulus 7900", "rocm", host=ROMULUS_HOST),
        ],
        ts_default=[35, 15, 50],
        vrams_static=[15.5, 7.0, 22.0],
        config="config-c",
    ),
}

# Prod labels share topology with base presets (dual-socket env is separate).
for _alias, _base in (
    ("b6-5gpu-g-prod", "b6-5gpu-g"),
    ("b6-6gpu-g-prod", "b6-6gpu-g"),
):
    PRESETS[_alias] = PRESETS[_base]


def _ssh_user(host: str) -> str:
    if host == JUPITER_HOST:
        return JUPITER_USER
    return SSH_USER


def _ssh_pass(host: str) -> str:
    if host == JUPITER_HOST:
        return JUPITER_PASS
    return CLUSTER_PASS


def _ssh(host: str, cmd: str, timeout: int = 20) -> str:
    target = f"{_ssh_user(host)}@{host}"
    base = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", f"ConnectTimeout={min(timeout, 8)}", target, cmd]
    passwd = _ssh_pass(host)
    if passwd and _has_sshpass():
        run = ["sshpass", "-p", passwd, *base]
    else:
        run = base
    out = subprocess.run(run, capture_output=True, text=True, timeout=timeout, check=False)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or out.stdout.strip() or f"ssh failed ({host})")
    return out.stdout


def _has_sshpass() -> bool:
    return subprocess.run(["which", "sshpass"], capture_output=True).returncode == 0


def _find_profiler() -> str:
    for cand in PROFILER_CANDIDATES:
        if cand and Path(cand).is_file():
            return cand
    return ""


def _local_hostname() -> str:
    import socket

    return socket.gethostname()


def _is_local_host(host: str) -> bool:
    hn = _local_hostname()
    return host in (hn, hn.lower(), hn.capitalize())


def _run_local(cmd: str, timeout: int = 20) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "local command failed")
    return proc.stdout


def _probe_nvidia(host: str, index: int = 0) -> tuple[int, int]:
    cmd = (
        f"nvidia-smi --query-gpu=index,memory.total,memory.free "
        f"--format=csv,noheader,nounits -i {index}"
    )
    out = _run_local(cmd) if _is_local_host(host) else _ssh(host, cmd)
    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        raise RuntimeError(f"unexpected nvidia-smi line: {line}")
    total_mb = int(float(parts[1]))
    free_mb = int(float(parts[2]))
    return free_mb, total_mb


def _probe_rocm(host: str) -> tuple[int, int]:
    cmd = "rocm-smi --showmeminfo vram --showproductname 2>/dev/null | head -30"
    out = _run_local(cmd) if _is_local_host(host) else _ssh(host, cmd)
    total_b = used_b = None
    for line in out.splitlines():
        if "VRAM Total Memory (B):" in line:
            total_b = int(line.split(":")[-1].strip())
        if "VRAM Total Used Memory (B):" in line:
            used_b = int(line.split(":")[-1].strip())
    if total_b is None or used_b is None:
        raise RuntimeError(f"rocm-smi parse failed on {host}")
    total_mb = total_b // (1024 * 1024)
    free_mb = max(0, (total_b - used_b) // (1024 * 1024))
    return free_mb, total_mb


def _probe_rpc(endpoint: str, via_host: str = "") -> tuple[int, int]:
    profiler = _find_profiler()
    if not profiler:
        raise RuntimeError("llama-pipeline-profiler not found for --validate-rpc")
    remote_cmd = (
        f"timeout 30 {profiler} --validate-rpc -rpc '{endpoint}' -ts 50 2>&1"
    )
    if via_host:
        out = _ssh(via_host, remote_cmd, timeout=45)
    else:
        proc = subprocess.run(remote_cmd, shell=True, capture_output=True, text=True, timeout=45, check=False)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0 and "=== rpc validate ===" not in out:
            raise RuntimeError(out.strip() or "validate-rpc failed")
    m = re.search(r"=== rpc validate ===\s*(\{.*\})", out, re.DOTALL)
    if not m:
        raise RuntimeError(f"validate-rpc JSON not found for {endpoint}")
    report = json.loads(m.group(1))
    rows = report.get("endpoints", [])
    if not rows:
        raise RuntimeError(f"no endpoints in validate-rpc report for {endpoint}")
    row = rows[0]
    if not row.get("ok"):
        raise RuntimeError(row.get("error", f"RPC unreachable: {endpoint}"))
    free_mb = int(row.get("mem_free_mb", 0))
    total_mb = int(row.get("mem_total_mb", 0))
    if total_mb <= 0:
        raise RuntimeError(f"invalid mem_total for {endpoint}")
    return free_mb, total_mb


def probe_device(dev: DeviceSpec) -> tuple[int, int, str]:
    if dev.kind == "rpc":
        try:
            free_mb, total_mb = _probe_rpc(dev.endpoint, dev.via_host)
            return free_mb, total_mb, "validate-rpc"
        except Exception as rpc_err:
            if dev.fallback_host:
                free_mb, total_mb = _probe_nvidia(dev.fallback_host, dev.gpu_index)
                return free_mb, total_mb, f"nvidia-smi (rpc fallback: {rpc_err})"
            raise
    if dev.kind == "nvidia":
        free_mb, total_mb = _probe_nvidia(dev.host, dev.gpu_index)
        return free_mb, total_mb, "nvidia-smi"
    if dev.kind == "rocm":
        free_mb, total_mb = _probe_rocm(dev.host)
        return free_mb, total_mb, "rocm-smi"
    raise ValueError(f"unknown device kind: {dev.kind}")


def fitt_mib_for_preset(preset: PresetSpec) -> list[int]:
    """Per-device fit-target MiB aligned with tensor-split device order (RPC0..ROCm)."""
    n = len(preset.devices) if preset.devices else len(preset.ts_default)
    return [VRAM.FITT_MIB_DEFAULT] * n


def small_device_index(preset: PresetSpec) -> int:
    """Romulus 3060 docker is index 1 on b6-5gpu-g; else smallest VRAM slot."""
    if preset.devices:
        for i, dev in enumerate(preset.devices):
            if "3060" in dev.label:
                return i
    return 1 if len(preset.ts_default) > 1 else 0


def normalize_ts_sum(ts: list[int], floor_pct: int = 4) -> list[int]:
    n = len(ts)
    if n == 0:
        return []
    s = sum(ts)
    if s <= 0:
        base = max(floor_pct, 100 // n)
        return normalize_ts_sum([base] * n, floor_pct=floor_pct)
    raw = [max(floor_pct, int(round(100 * t / s))) for t in ts]
    delta = 100 - sum(raw)
    i = 0
    while delta > 0:
        raw[i % n] += 1
        delta -= 1
        i += 1
    i = 0
    while delta < 0:
        idx = i % n
        if raw[idx] > floor_pct:
            raw[idx] -= 1
            delta += 1
        i += 1
        if i > n * 200:
            break
    return raw


def max_ts_small_fits(
    model_gb: float,
    layers: int,
    ctx: int,
    ngl: int,
    small_idx: int,
    budget_vrams: list[float],
    margins_gb: list[float],
    floor_pct: int = 4,
) -> int:
    n = len(budget_vrams)
    best = floor_pct
    lo, hi = floor_pct, 45
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = [0] * n
        ts[small_idx] = mid
        rem = 100 - mid
        others = [i for i in range(n) if i != small_idx]
        ov = sum(budget_vrams[i] for i in others) or 1.0
        for i in others:
            ts[i] = max(floor_pct, int(round(rem * budget_vrams[i] / ov)))
        ts = normalize_ts_sum(ts, floor_pct=floor_pct)
        ok, _, _ = VRAM.check_split(model_gb, layers, ctx, ngl, ts, budget_vrams, margins_gb)
        if ok:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def ts_l4_spread_cap_small(
    preset: PresetSpec,
    budget_vrams: list[float],
    model_gb: float,
    layers: int,
    ctx: int,
    ngl: int,
    margins_gb: list[float],
    floor_pct: int = 4,
) -> list[int]:
    """L4 spread: cap romulus 3060 share, redistribute by planning budget."""
    n = len(budget_vrams)
    small_idx = small_device_index(preset)
    cap = max_ts_small_fits(
        model_gb, layers, ctx, ngl, small_idx, budget_vrams, margins_gb, floor_pct=floor_pct
    )
    rem = 100 - cap
    others = [i for i in range(n) if i != small_idx]
    ov = sum(budget_vrams[i] for i in others) or 1.0
    ts = [0] * n
    ts[small_idx] = cap
    for i in others:
        ts[i] = max(floor_pct, int(round(rem * budget_vrams[i] / ov)))
    return normalize_ts_sum(ts, floor_pct=floor_pct)


def ts_layer_spread_equal(n_devices: int, floor_pct: int = 4) -> list[int]:
    """L4: equal tensor-split to spread layers across all cluster GPUs."""
    if n_devices <= 0:
        return []
    base = max(floor_pct, 100 // n_devices)
    raw = [base] * n_devices
    delta = 100 - sum(raw)
    i = 0
    while delta > 0:
        raw[i % n_devices] += 1
        delta -= 1
        i += 1
    i = 0
    while delta < 0:
        idx = i % n_devices
        if raw[idx] > floor_pct:
            raw[idx] -= 1
            delta += 1
        i += 1
        if i > n_devices * 200:
            break
    return raw


def ts_from_free_mb(free_mibs: list[int], floor_pct: int = 4) -> list[int]:
    if not free_mibs:
        return []
    total = sum(free_mibs)
    if total <= 0:
        return [100 // len(free_mibs)] * len(free_mibs)
    raw = [max(floor_pct, int(round(100 * f / total))) for f in free_mibs]
    delta = 100 - sum(raw)
    i = 0
    while delta != 0 and raw:
        idx = i % len(raw)
        if delta > 0:
            raw[idx] += 1
            delta -= 1
        elif raw[idx] > floor_pct:
            raw[idx] -= 1
            delta += 1
        i += 1
    return raw


def resolve_model(args) -> tuple[float, int, str]:
    model_gb = args.model_gb
    layers = args.layers
    arch = "unknown"
    if args.gguf:
        if model_gb is None:
            if not os.path.isfile(args.gguf):
                print(f"WARN: gguf not found locally: {args.gguf}", file=sys.stderr)
                model_gb = VRAM.DEFAULT_MODEL_GB
            else:
                model_gb = VRAM.model_gb_from_path(args.gguf)
        try:
            meta = VRAM.gguf_read_metadata(args.gguf)
            arch = str(meta.get("arch", "unknown"))
            if "block_count" in meta:
                layers = int(meta["block_count"])
        except Exception as e:
            print(f"WARN: GGUF metadata read failed: {e}", file=sys.stderr)
            guess = VRAM.layers_from_filename(args.gguf)
            if guess is not None:
                layers = guess
    if model_gb is None:
        model_gb = VRAM.DEFAULT_MODEL_GB
    if layers is None:
        layers = VRAM.DEFAULT_LAYERS
    return model_gb, layers, arch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), default="", help="gate/bench topology preset")
    p.add_argument("--config", choices=list(VRAM.CONFIGS), default="", help="legacy static config (no --preset)")
    p.add_argument("--gguf", default="", help="GGUF path for size + layer count")
    p.add_argument("--model-gb", type=float, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ts", default="", help="override tensor-split to evaluate")
    p.add_argument(
        "--ts-mode",
        choices=("vram", "equal"),
        default="vram",
        help="vram=live ratio (default); equal=L4 cap romulus 3060 + spread (fitt 1024 MiB)",
    )
    p.add_argument(
        "--phase",
        choices=(VRAM.VRAM_PHASE_LOAD, VRAM.VRAM_PHASE_DECODE),
        default=VRAM.VRAM_PHASE_LOAD,
        help="load=conservative (load surge + pipeline); decode=post-prefill steady state",
    )
    p.add_argument("--rpc", default="", help="override RPC endpoint list")
    p.add_argument("--live", action="store_true", default=True, help="probe live VRAM (default)")
    p.add_argument("--no-live", action="store_true", help="use static budgets from --config")
    p.add_argument("--strict", action="store_true", help="exit 1 when default ts cannot load model")
    args = p.parse_args()

    if args.no_live:
        args.live = False

    preset = PRESETS.get(args.preset) if args.preset else None
    if preset is None and args.config:
        cfg = VRAM.CONFIGS[args.config]
        preset = PresetSpec(
            name=cfg["name"],
            rpc="",
            devices=[],
            ts_default=list(cfg["ts_default"]),
            vrams_static=list(cfg["vrams"]),
            config=args.config,
        )
    if preset is None:
        p.error("specify --preset or --config")

    model_gb, layers, arch = resolve_model(args)

    live_rows: list[tuple[DeviceSpec, int, int, str]] = []
    vrams_gb: list[float] = []
    if args.live and preset.devices:
        print(f"=== live VRAM probe: {preset.name} ===")
        for dev in preset.devices:
            try:
                free_mb, total_mb, source = probe_device(dev)
            except Exception as e:
                print(f"ERROR: {dev.label}: {e}", file=sys.stderr)
                return 2
            live_rows.append((dev, free_mb, total_mb, source))
            vrams_gb.append(round(free_mb / 1024.0, 2))
            print(
                f"  {dev.label:28s} free={free_mb:5d} MiB  total={total_mb:5d} MiB  "
                f"({vrams_gb[-1]:.2f} GB) [{source}]"
            )
        print()
    else:
        vrams_gb = list(preset.vrams_static)
        print(f"=== static VRAM budget: {preset.name} ===")
        for i, gb in enumerate(vrams_gb):
            print(f"  dev{i}: {gb:.2f} GB")
        print()

    free_mibs = [r[1] for r in live_rows] if live_rows else []
    ts_live = ts_from_free_mb(free_mibs) if free_mibs else []
    n_dev = len(preset.devices) if preset.devices else len(preset.ts_default)
    fitt_mib = fitt_mib_for_preset(preset)
    budget_vrams, reserve_details = VRAM.budget_vrams_from_live(
        vrams_gb, free_mibs, fitt_mib, phase=args.phase
    )
    margins_gb = [0.0] * n_dev
    ts_equal_naive = ts_layer_spread_equal(n_dev) if n_dev else []
    ngl_rows = VRAM.ngl_candidates(layers)
    ngl_rec = ngl_rows[1][0] if len(ngl_rows) > 1 else ngl_rows[0][0]
    ts_equal_safe: list[int] = []
    if args.ts_mode == "equal" and n_dev:
        for ngl, _frac in ngl_rows:
            ts_equal_safe = ts_l4_spread_cap_small(
                preset, budget_vrams, model_gb, layers, args.ctx, ngl, margins_gb
            )
            ok, _, _ = VRAM.check_split(
                model_gb, layers, args.ctx, ngl, ts_equal_safe, budget_vrams, margins_gb
            )
            if ok:
                ngl_rec = ngl
                break
    if args.ts:
        ts_eval = [int(x) for x in args.ts.split(",")]
    elif args.ts_mode == "equal" and ts_equal_safe:
        ts_eval = ts_equal_safe
    elif ts_live:
        ts_eval = ts_live
    else:
        ts_eval = list(preset.ts_default)

    rpc = args.rpc or preset.rpc
    model_line = f"  gguf: {args.gguf}" if args.gguf else ""
    print("=== model ===")
    if model_line:
        print(model_line)
        print(f"  architecture: {arch}")
    print(f"  weights: {model_gb:.2f} GB  layers: {layers}  ctx: {args.ctx}")
    print(f"  combined GPU free: {sum(vrams_gb):.2f} GB")
    print(f"  combined planning budget ({args.phase}): {sum(budget_vrams):.2f} GB")
    print()
    if reserve_details and preset.devices:
        print(f"=== per-device reserve ({args.phase}, fitt + process + surge + pipeline) ===")
        for i, (dev, rd) in enumerate(zip(preset.devices, reserve_details)):
            print(
                f"  dev{i} {dev.label:28s} free={rd['free_gb']:.2f} GB  "
                f"budget={rd['budget_gb']:.2f} GB  class={rd['class']}  "
                f"reserve={rd['total_reserve_mib']} MiB "
                f"(fitt={rd['fitt_mib']} proc={rd['process_mib']} "
                f"surge={rd['surge_mib']} pipe={rd['pipeline_mib']})"
            )
        print("  # future: --probe-fit via llama_memory_breakdown for measured compute/graph")
        print()
    print("=== recommended deploy flags ===")
    print(f"  BENCH_RPC_ENDPOINT='{rpc}'")
    print(f"  BENCH_TS={','.join(str(t) for t in ts_eval)}")
    small_idx = small_device_index(preset)
    small_label = preset.devices[small_idx].label if preset.devices and small_idx < len(preset.devices) else f"dev{small_idx}"
    if ts_equal_naive:
        print(f"  # L4 naive equal ts: {','.join(str(t) for t in ts_equal_naive)}")
    if ts_equal_safe:
        print(
            f"  # L4 3060-safe spread (dev{small_idx}={small_label} cap, fitt 1024 MiB): "
            f"{','.join(str(t) for t in ts_equal_safe)}"
        )
    if ts_live and ts_eval != preset.ts_default:
        print(f"  # live ratio ts: {','.join(str(t) for t in ts_live)}")
        print(f"  # preset default: {','.join(str(t) for t in preset.ts_default)}")
    if args.ts_mode == "equal":
        print(f"  # ts-mode: equal (L4 cap-small @ {small_label})")
    print(f"  BENCH_FITT={','.join(str(m) for m in fitt_mib)}")
    print(f"  BENCH_NGL={ngl_rec}")
    fitt_s = ",".join(str(m) for m in fitt_mib)
    print(
        f"  BENCH_EXTRA='--fit off --fit-target {fitt_s} --verbose -lv 4 --reasoning off'"
    )
    print()

    ok_default = False
    best: tuple[int, list[int], list[float]] | None = None
    print(f"{'ngl':>4} {'cpu%':>5} {'ts':>16} | per-dev GB (w+KV vs budget) | ok")
    for ngl, frac in ngl_rows:
        ok, totals, w_cpu = VRAM.check_split(
            model_gb, layers, args.ctx, ngl, ts_eval, budget_vrams, margins_gb
        )
        ts_s = ",".join(str(t) for t in ts_eval)
        tot_s = " ".join(f"{t:.1f}" for t in totals) if totals else "-"
        flag = "OK" if ok else "OOM"
        print(f"{ngl:4d} {frac*100:4.0f}% {ts_s:>16} | {tot_s} cpu_w={w_cpu:.1f}GB | {flag}")
        if ok and best is None:
            best = (ngl, ts_eval, totals)
        if ngl == ngl_rec and ok:
            ok_default = True

    print()
    if best:
        print(f"PASS: ngl={best[0]} ts={','.join(str(t) for t in best[1])} fits all devices")
    else:
        print("FAIL: no ngl/ts combo fits on current VRAM budgets")
        print("hints: lower ctx, smaller quant, higher CPU offload (-ngl), or drop a device")

    if args.strict and not ok_default:
        return 1
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())