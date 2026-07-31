# Quick-Reference: Best Placement Plan

**Model**: Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf (35B, MoE, MTP)
**Throughput**: **1.20 tok/s** (32 tokens, 3-request median)
**Experiment**: Exp W, 2026-07-21

## Plan: `best-plan.json`

7-block interleaved placement with cold 3060Ti:

```
Block  Backend              Layers   Role
-----  -------------------  -------  ---------------------
 0     ROCm (7900XTX)       L0-3     embedding + early norm
 1     3070 (triton #0)     L3-7     first MoE block (4L)
 2     3090 (triton #1)     L7-8     flash attention (1L)
 3     ROCm (7900XTX)       L8-13    norms + attn
 4     3070 (triton #0)     L13-16   second MoE block (3L)
 5     3090 (triton #1)     L16-20   flash attention (4L)
 6     ROCm (7900XTX)       L20-40   remaining + MTP + output
```

## Flags

```
--pipeline-plus --pplus-rpc-defer-barrier --pplus-rpc-get-defer --pplus-rpc-flush
-c 512 -b 512 -ub 512 -fa on --cache-type-k q8_0 --cache-type-v q8_0
--no-warmup -np 1 --placement-plan best-plan.json
```

## GPU Topology

| GPU       | Backend                          | VRAM  |
|-----------|----------------------------------|-------|
| 7900 XTX  | `local:ROCm0`                    | 24 GB |
| RTX 3060Ti| `rpc://127.0.0.1:50051#0`        | 8 GB  |
| RTX 3070  | `rpc://192.168.8.23:50052#0`     | 8 GB  |
| RTX 3090  | `rpc://192.168.8.23:50052#1`     | 24 GB |

**Single-endpoint P2P**: 3070 and 3090 share `:50052` (same rpc-server instance).

## Quick Launch

```bash
cd rpc-patch/patch/quick-ref
chmod +x launch-best.sh
./launch-best.sh
```

Results land in `runs/<label>/`.

## Profiling

```bash
# With sched-trace + per-node timing
GGML_SCHED_TRACE=1 GGML_SCHED_PER_NODE_TIMING=1 \
  GGML_RPC_SERVER_TELEMETRY=1 ./launch-best.sh profiled

# Server-side telemetry (run on triton before launch)
GGML_RPC_SERVER_TELEMETRY=1 GGML_RPC_NODE_SAMPLE_INTERVAL=2 \
  rpc-server -H 0.0.0.0 -p 50052 -d CUDA0 -d CUDA1
```

## Source

Full results: `rpc-patch/patch/bench-results/cluster-4gpu-primary/prototype-perop-2026-07-21/`
- Plan: `plan-7block-2x3070-cold3060-p2p.json`
- Log: `exp-w-7block-2x3070-p2p/llama-server-stderr.log`
- Results: `RESULTS.md`
