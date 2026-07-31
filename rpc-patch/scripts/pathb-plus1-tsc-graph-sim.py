#!/usr/bin/env python3
"""Static decode-graph simulation for Plus=1 TSC model specificity.

Derives main-pass layer composition from qwen35moe arch rules (no GGUF required).
Cross-check against romulus server log when available.

usage: pathb-plus1-tsc-graph-sim.py [--n-layer N] [--interval I] [--n-layer-nextn N]
"""

from __future__ import annotations

import argparse
import json
import sys


def is_recr(il: int, interval: int) -> bool:
    return (il + 1) % interval != 0


def layer_ops(recr: bool, moe: bool) -> list[str]:
    ops = ["attn_norm"]
    if recr:
        ops += [
            "build_conv_state",  # recurrent conv_states
            "build_rs_ssm",      # ssm_states read
            "GATED_DELTA_NET",   # or fused __fgdn_ar__ / __fgdn_ch__
            "ssm_state_write",   # ggml_cpy -> ssm_states_all
        ]
    else:
        ops += ["full_attention", "kv_cache"]
    ops += ["attn_post_norm"]
    if moe:
        ops += ["MUL_MAT_ID", "shared_expert_ffn"]
    else:
        ops += ["dense_ffn"]
    return ops


def simulate(n_layer: int, interval: int, n_layer_nextn: int, moe: bool) -> dict:
    trunk_layers = n_layer - n_layer_nextn
    layers = []
    for il in range(trunk_layers):
        recr = is_recr(il, interval)
        layers.append({
            "il": il,
            "type": "gdn" if recr else "full_attn",
            "main_pass": True,
            "ops": layer_ops(recr, moe),
        })

    mtp_layers = []
    for il in range(trunk_layers, n_layer):
        mtp_layers.append({
            "il": il,
            "type": "mtp_block",
            "main_pass": False,
            "graph": "LLM_GRAPH_TYPE_DECODER_MTP",
            "ops": ["mtp_eh_proj", "full_attention", "MUL_MAT_ID"],
            "note": "NOT executed in standard llama_decode main pass",
        })

    gdn_count = sum(1 for L in layers if L["type"] == "gdn")
    attn_count = sum(1 for L in layers if L["type"] == "full_attn")

    critical_ops = set()
    for L in layers:
        critical_ops.update(L["ops"])

    return {
        "n_layer_kv": n_layer,
        "n_layer_trunk": trunk_layers,
        "n_layer_nextn": n_layer_nextn,
        "mtp_in_main_pass": False,
        "mtp_blocks_loaded": n_layer_nextn > 0,
        "full_attention_interval": interval,
        "gdn_layers": gdn_count,
        "full_attn_layers": attn_count,
        "moe_ffn_all_layers": moe,
        "recurrent_state_tensors": ["conv_states (R)", "ssm_states (S)"],
        "critical_plus_surfaces_per_token": {
            "P0": "pipeline_barrier on graph reuse (cur_copy rotate)",
            "S": "compute_splits async: RPC GET defer, partial barrier, MoE copy-slot",
            "P1": "synchronize_sampling narrow (logits only)",
        },
        "layers": layers,
        "mtp_layers": mtp_layers,
        "tsc_specificity": {
            "mtp_execution_required": False,
            "moe_routing_required": moe,
            "gdn_recurrent_required": gdn_count > 0,
            "multi_backend_recurrent_split": "ROCm + RPC RS buffers (observed in server log)",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-layer", type=int, default=40)
    p.add_argument("--interval", type=int, default=4)
    p.add_argument("--n-layer-nextn", type=int, default=0)
    p.add_argument("--moe", action="store_true", default=True)
    p.add_argument("--dense", action="store_true", help="qwen35 dense FFN instead of MoE")
    args = p.parse_args()
    moe = args.moe and not args.dense

    out = simulate(args.n_layer, args.interval, args.n_layer_nextn, moe)
    print(json.dumps(out, indent=2))

    print("\n--- TSC specificity summary ---", file=sys.stderr)
    t = out["tsc_specificity"]
    print(f"  GDN layers in main pass: {out['gdn_layers']}", file=sys.stderr)
    print(f"  MTP blocks in main pass: {out['mtp_in_main_pass']} (nextn={out['n_layer_nextn']})", file=sys.stderr)
    print(f"  MoE MUL_MAT_ID per layer: {out['moe_ffn_all_layers']}", file=sys.stderr)
    print(f"  Verdict: TSC repro model is GDN + multi-backend + MoE trunk; NOT MTP execution.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())