# Wayfinder Plan — Qwen3.6 35B MoE Pipeline Optimization

## Current State (2026-07-21)

4-GPU RPC pipeline: ROCm 7900 XTX + 3060 Ti + 3070 + RTX 3090
Model: Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf (Q6_K, 22 GB)
26 main layers, 14 MTP head layers, 256 experts (8 active), FA every 4th layer

## Findings

### D1: --tensor-split semantics
- **Resolved.** First value = local GPU, then RPC backends in --rpc order.
- Proportional split of 42 slots, MTP layers force-pinned to device[0].
- Opaque mapping — always verify with --verbose.
- Doc: TENSOR-SPLIT-SEMANTICS.md

### D2: Per-layer compute variance (ROOT CAUSE of imbalance)
- FA (full-attention) layers are 4-6x more expensive than non-FA layers.
- Later FA layers (20, 24) are ~4.6x more expensive than early FA layers (0, 4).
- Contiguous layer assignment guarantees imbalance — one GPU always gets hot FA layers.
- **Pipeline efficiency is 98%** — GPUs are utilized, but workload is inherently unbalanced.

### D3: VRAM budget
- 3060 Ti / 3070: 8 GB each → max ~5-6 main layers (~3-4 GB)
- 3090 / ROCm: 24 GB each → can hold 15+ layers + MTP
- Per-layer VRAM varies: early layers ~693 MiB, mid ~539 MiB, late ~1721 MiB (MoE experts grow)

### D4: Throughput results

| Experiment | Layout | Bottleneck | Tok/s |
|-----------|--------|-----------|-------|
| Baseline | default proportional | 3060Ti (80%) | 0.31 |
| Exp H | -ts 25,1,1,73 (3060Ti dead) | ROCm (82%) | 1.80 |
| Exp J | -ts 10,12,12,66 (balanced) | 3060Ti+3070 (42% each) | ~1.9* |
| Exp K | explicit: 3090 gets late FA | 3090 (78%) | ~1.93 |
| Exp L | interleaved: all FA on ROCm | TBD | TBD |

*estimated from sched-trace

### D5: Interleaved layer placement WORKS
- KV cache properly follows interleaved layer-device assignment.
- Graph splits increase (5-7 → 16), but pipeline handles it.
- No correctness issues observed.

### D6: Placement plan mechanism
- --placement <json> gives full per-layer control.
- Backend IDs: `rpc://host:port#0`, `local:ROCm0`, `local:DEVICE_NAME`.
- Plan covers 26 main layers; MTP layers (26-39) are force-pinned to device[0].
- Layer 40 (nextn head) and output layer follow default tensor split (not in plan).

### D7: Open — Per-layer VRAM-aware split (HIGH PRIORITY)
- Current --tensor-split operates on tensor count, not VRAM or compute cost.
- Each layer has different VRAM (500-1700 MiB) and compute cost (10-200 ms).
- A proper split would consider: VRAM budget per device, compute cost per layer, FA layer count.
- **Blocked by:** no simple CLI for VRAM-aware layer assignment.
- **Fix:** expose layer_devices via simple CLI (--layer-map), or add VRAM-aware split mode.

### D7.1: Open — Per-tensor splitting (row-split) for MoE experts
- MoE experts (256 per layer) are the dominant compute cost in late layers.
- Splitting experts across GPUs (row-split) would balance even intra-layer.
- Requires GGML_SPLIT_MODE_ROW backend support.
- **Effort:** large. New backend split mode, expert dispatch logic.

### D8: Open — KV cache cross-device transfer cost
- With 16 graph splits (interleaved), hidden states cross devices more frequently.
- Need to measure transfer overhead (peer_copy vs RPC copy).
- sched-trace "copy" phase already tracks this — needs analysis.

## Next Steps

1. Complete Exp L benchmark (interleaved FA on ROCm)
2. Run Exp M: interleaved FA on 3090 (reverse)
3. Merge KV cache research (vanilla vs Path-B-Plus)
4. Implement --layer-map CLI for per-layer device assignment (D7)
5. Profile copy/transfer overhead with interleaved layout (D8)
