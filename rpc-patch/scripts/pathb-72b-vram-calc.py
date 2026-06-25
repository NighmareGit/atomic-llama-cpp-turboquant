#!/usr/bin/env python3
"""Estimate 72B IQ4_XS VRAM split for Config A (RPC0=3060 Ti 8GB, ROCm0=7900 XTX 24GB).

Usage:
  pathb-72b-vram-calc.py [--model-gb 40.16] [--layers 80] [--ctx 1024]

Device index order with --rpc: RPC0=index0, ROCm0=index1 (percentage-style -ts).
"""
from __future__ import annotations

import argparse

DEFAULT_MODEL_GB = 40.16
DEFAULT_LAYERS = 80
VRAM_RPC_GB = 7.67
VRAM_ROCM_GB = 24.5
MARGIN_GB = 1.0


def kv_gb(layers: float, ctx: int, n_embd: int = 8192, kv_heads: int = 8, bytes_per_el: float = 0.5) -> float:
    head_dim = n_embd // 64
    per_layer = 2 * kv_heads * head_dim * ctx * bytes_per_el
    return layers * per_layer / 1e9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-gb", type=float, default=DEFAULT_MODEL_GB)
    p.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--ctx", type=int, default=1024)
    args = p.parse_args()

    total_vram = VRAM_RPC_GB + VRAM_ROCM_GB
    cpu_min = max(0.0, args.model_gb - total_vram)

    print("72B IQ4_XS Config A VRAM budget")
    print(f"  weights (file):     {args.model_gb:.2f} GB")
    print(f"  layers (assumed):   {args.layers}")
    print(f"  RPC0 (3060 Ti):     {VRAM_RPC_GB:.2f} GB usable")
    print(f"  ROCm0 (7900 XTX):   {VRAM_ROCM_GB:.2f} GB usable")
    print(f"  combined GPU:       {total_vram:.2f} GB")
    print(f"  CPU offload (min):  {cpu_min:.2f} GB weights (before KV)")
    print()
    print("Rules:")
    print("  - Never use -ngl 99 on 72B; use -ngl 0 + --fit on OR manual -ngl 24-36")
    print("  - Use percentage -ts (10,90), NOT ratio (4,1) which puts 80% on RPC0")
    print("  - --n-cpu-moe only helps MoE models (35B-A3B), not dense 72B")
    print("  - llama-cli may crash RPC during CUDA graph capture; prefer llama-server + --fit on")
    print()
    print(f"{'ngl':>4} {'ts':>8} {'ctx':>5} | {'w_rpc':>5} {'kv_r':>5} {'tot_r':>6} | {'w_rocm':>6} {'kv_ro':>5} {'tot_ro':>6} | {'cpu_w':>6}")
    for ngl in range(20, 49, 4):
        for ts_rpc, ts_rocm in ((10, 90), (8, 92), (5, 95), (3, 97)):
            if ngl > args.layers:
                continue
            gpu_frac = ngl / args.layers
            w_gpu = args.model_gb * gpu_frac
            w_cpu = args.model_gb * (1 - gpu_frac)
            s = ts_rpc + ts_rocm
            w_rpc = w_gpu * ts_rpc / s
            w_rocm = w_gpu * ts_rocm / s
            layers_rpc = ngl * ts_rpc / s
            layers_rocm = ngl * ts_rocm / s
            kv_rpc = kv_gb(layers_rpc, args.ctx)
            kv_rocm = kv_gb(layers_rocm, args.ctx)
            tot_rpc = w_rpc + kv_rpc + MARGIN_GB
            tot_rocm = w_rocm + kv_rocm + MARGIN_GB
            ok = tot_rpc <= VRAM_RPC_GB and tot_rocm <= VRAM_ROCM_GB
            mark = "OK" if ok else "OOM"
            if ok:
                print(
                    f"{ngl:4d} {ts_rpc:2d},{ts_rocm:2d} {args.ctx:5d} | "
                    f"{w_rpc:5.2f} {kv_rpc:5.2f} {tot_rpc:6.2f} | "
                    f"{w_rocm:6.2f} {kv_rocm:5.2f} {tot_rocm:6.2f} | "
                    f"{w_cpu:6.2f}  {mark}"
                )
    print()
    print("Recommended server command (auto-fit):")
    print("  BENCH_NGL=0 BENCH_CTX=1024 BENCH_TS=10,90 BENCH_CTK=q4_0 BENCH_CTV=q4_0 \\")
    print("  BENCH_EXTRA='--fit on --fit-target 512,1024 --reasoning off' \\")
    print("  ./scripts/rpc-server-bench.sh pathb qwen72b-fit")


if __name__ == "__main__":
    main()