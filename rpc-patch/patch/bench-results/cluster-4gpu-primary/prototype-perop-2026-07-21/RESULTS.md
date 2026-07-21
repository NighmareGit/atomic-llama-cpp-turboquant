# Benchmarks: Pipeline Stage Count vs Overlap (2026-07-21)

All runs: Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf, 512 ctx, q8_0 KV, pipeline-plus flags, P2P single-endpoint on triton.

## Leaderboard

| Exp | tok/s | Stages | 3060Ti | 3070 | 3090 | ROCm | Bottleneck split |
|-----|-------|--------|--------|------|------|------|------------------|
| H | 1.80 | ~3 | ~0% | 73% | 1% | 25% | (--tensor-split, auto MTP) |
| M | 1.15 | 16 | L0-2,16-18,24-25 | L4-6,12-14,20-22 | L7,15,19 | L3,8-11,23+MTP | Wrong FA mapping |
| Q | 1.10 | 13 | L0-2,16-18,24-25 (8L) | L4-6,12-14,20-22 (9L) | L7,15,19 (3L) | L3,8-11,23+MTP (17L) | 3060Ti 197ms |
| T | 0.95 | 14 | L0-2 (3L) | L4-6,12-14,16-17,20-22 (11L) | L7,15,18-19,24-25 (6L) | L3,8-11,23+MTP (20L) | 3070 214ms |
| R | 0.85 | 13 | L0-2 (3L) | L4-6,12-14,20-22 (9L) | L7,15-19,24-25 (8L) | L3,8-11,23+MTP (17L) | 3090 340ms |
| U | 0.46 | 3 | (none) | L0-7 (8L) | L8-18 (11L) | L19-39 (7+14MTP) | 3090 1382ms |
| S | 0.33 | 4 | L0-2 (3L) | L3-10 (8L) | L11-25 (15L) | MTP (14L) | 3090 2266ms |

## Pipeline Overlap Analysis

```
Stage count vs effective overlap:

3-stage (Exp U):  trace=2283ms x 34 / 69.7s = 1.11x overlap  -> 0.46 tok/s
4-stage (Exp S):  trace=3264ms x 34 / 96.2s = 1.15x overlap  -> 0.33 tok/s
13-stage (Exp Q): trace=1200ms x 34 / 29.0s = 1.41x overlap  -> 1.10 tok/s
14-stage (Exp T): trace=1162ms x 34 / 33.7s = 1.17x overlap  -> 0.95 tok/s
```

**More stages = more sync overhead BUT more pipeline overlap.** The sweet spot is ~13 stages.

## RPC Overhead Deep-Dive (Exp Q)

| Component | Time | % of trace |
|-----------|------|------------|
| GPU compute (graph_compute_async) | 29,572ms | 45.2% |
| Tensor copies (input_copy_slow) | 1,981ms | 3.0% |
| GPU idle within splits | 1,997ms | 3.1% |
| Pipeline sync gap (overlap lost) | 31,931ms | 48.8% |

**The moneyshot**: 49% of total split time is lost to pipeline sync/bubble — not RPC copies. Copies are only 3%.

Per-backend copy costs (Exp Q):
- 3060Ti: 170 copies, 810ms, 1.0MB
- 3070: 170 copies, 858ms, 1.0MB
- 3090: 136 copies, 252ms, 1.3MB
- ROCm: 184 copies, 60ms, 1.3MB

## Key Findings

1. **Pipeline overlap is the real bottleneck**: 49% of trace time is sync waste, not compute or copies
2. **Cold 3060Ti validated**: 8->3 layers drops split time from 197->72ms, but bottleneck shifts
3. **Stage count has a sweet spot**: ~3 stages = no overlap (0.46 tok/s), ~13 stages = best overlap (1.10 tok/s), 14 stages = diminishing returns (0.95 tok/s)
4. **RPC tensor copies are negligible** at 3% — the pplus barrier deferral flags are the right lever
5. **Per-layer compute is cheap at 512-ctx q8_0**: GPU brand matters more than FA vs MoE
