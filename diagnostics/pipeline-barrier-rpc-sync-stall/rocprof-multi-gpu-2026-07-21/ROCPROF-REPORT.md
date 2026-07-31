# Multi-GPU Pipeline Profiling Report

**Date:** 2026-07-21  
**Profiler:** rocprofv3 (AMD GPU kernel trace) + GGML_SCHED_TRACE (per-backend split timing)  
**Goal:** Identify pipeline bubbles, stalls, and sync overhead in multi-GPU assembly-line overlap

---

## 1. Methodology

### Profiling Stack

| Layer | Tool | Output | Scope |
|-------|------|--------|-------|
| GPU kernel trace | `rocprofv3 --kernel-trace --stats --summary` | per-kernel dispatch timestamps on 7900 XTX | Hardware-level: kernel start/end, inter-kernel gaps |
| Backend split trace | `GGML_SCHED_TRACE=1` | per-split compute_us / idle_us by backend | Software-level: which backend waits for what |
| Server log | llama-server stderr (verbosity 3) | request lifecycle, token timing | Correlation: trace events -> HTTP responses |

### Command Pattern

```bash
# Required for rocprofv3: disable CUDA graphs
GGML_CUDA_GRAPHS=0 \
GGML_SCHED_TRACE=1 \
GGML_SCHED_TRACE_FILE=/tmp/trace.jsonl \
rocprofv3 --kernel-trace --stats --summary \
  -d <output-dir> -o <prefix> -f csv \
  -- <llama-server ...>
```

rocprofv3 wraps the server process via `sudo` (ptrace required) and is terminated with `timeout --signal=SIGINT N` to trigger clean output flush.

### Benchmark Requests

After model load, 3 sequential non-streaming completion requests:
- Prompt: "Write a space story." (5 tokens)
- Generation: 64 tokens, temperature 0.3
- Single slot (`-np 1`), no warmup

---

## 2. Test Configurations

### Hardware

| GPU | Role | Interface | VRAM |
|-----|------|-----------|------|
| Radeon RX 7900 XTX | ROCm client (local) | PCIe 4.0 x16 | 24,560 MiB |
| RTX 3060 Ti | CUDA RPC worker | Docker, localhost:50051 | 7,841 MiB |
| RTX 3090 | CUDA RPC worker | Network, triton:50054 | 24,123 MiB |
| RTX 3070 | CUDA RPC worker | Network, triton:50055 | 7,839 MiB |

### Model

- **File:** `/mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf`
- **Architecture:** 40-layer MoE (Qwen3.6 35B A3B), MTP enabled
- **Size:** ~22 GB
- **KV cache:** q8_0 / q8_0, 512 context window

### Pipeline Flags (both configs)

```
--pipeline-plus
--pplus-rpc-defer-barrier
--pplus-rpc-get-defer
--pplus-rpc-flush
-fa on --no-warmup -np 1
```

### Config A: 2-GPU Local

```
--rpc 127.0.0.1:50051 --device-order gpu,rpc
Backends: ROCm0 (7900 XTX) + RPC0 (3060 Ti)
```

### Config B: 4-GPU (2 local + 2 network)

```
--rpc 127.0.0.1:50051,192.168.8.23:50054,192.168.8.23:50055
Backends: ROCm0 (7900 XTX) + RPC0 (3060 Ti) + RPC1 (3090) + RPC2 (3070)
```

---

## 3. Results: 2-GPU Local (Baseline)

### Kernel Trace — 7900 XTX

| Metric | Value |
|--------|-------|
| Trace span | 76,323 ms |
| Total kernels | 257,551 |
| Kernel compute | 1,959.7 ms (2.6%) |
| Idle (span - compute) | 74,650 ms (97.8%) |
| Inter-kernel gaps | P50=7 us, P90=19 us, P99=121 us |

> Note: 97.8% idle includes ~50s post-inference wait (tee buffering delayed curl requests). Inference-only window is ~14.5s.

### Kernel Breakdown

| Category | Calls | Time (ms) | % Compute |
|----------|-------|-----------|-----------|
| matvec | 66,992 | 1,070.2 | 54.6% |
| quantize | 57,680 | 181.4 | 9.3% |
| element-wise | 50,368 | 160.4 | 8.2% |
| norm | 29,488 | 126.4 | 6.4% |
| flash_attn | 1,552 | 92.4 | 4.7% |
| ssm/gate | 20,336 | 83.0 | 4.2% |
| MoE | 6,112 | 77.3 | 3.9% |
| copy/fill | 6,207 | 28.0 | 1.4% |
| rope | 3,104 | 23.8 | 1.2% |

### Throughput

| Metric | With rocprofv3 | Without profiler |
|--------|---------------|-----------------|
| Generation speed | 13.5 tok/s | 64 tok/s |
| Profiler slowdown | 4.7x | — |

### Assessment

The 2-GPU pipeline is healthy. Inter-kernel gaps are tight (P99=121 us). The 7900 XTX is doing real GPU work — matvec dominates at 54.6%, which is expected for the local GPU's layer share. The profiler overhead is significant (4.7x slowdown) but the pipeline structure itself shows no pathological stalls at the kernel level.

---

## 4. Results: 4-GPU (2 Local + 2 Network)

### Kernel Trace — 7900 XTX

| Metric | Value |
|--------|-------|
| Trace span | 178,421 ms |
| Total kernels | 23,081 |
| Kernel compute | 182.7 ms (0.1%) |
| Inter-kernel gaps | P50=31 us, P90=45 us, P99=113 us |

### Kernel Breakdown

| Category | Calls | Time (ms) | % Compute |
|----------|-------|-----------|-----------|
| matvec | 4,474 | 106.5 | 58.3% |
| element-wise | 9,365 | 33.3 | 18.2% |
| quantize | 3,974 | 13.5 | 7.4% |
| norm | 1,675 | 6.4 | 3.5% |
| flash_attn | 100 | 5.1 | 2.8% |
| ssm/gate | 1,200 | 4.7 | 2.6% |
| copy/fill | 519 | 3.3 | 1.8% |
| softmax | 350 | 1.8 | 1.0% |
| rope | 200 | 1.5 | 0.8% |

> The 7900 XTX does 11x fewer kernel calls and 10.7x less compute than in 2-GPU mode. GPU utilization drops from meaningful to 0.1%.

### GGML_SCHED_TRACE — Per-Backend Split Timing

| Backend | Splits | Compute (ms) | Idle (ms) | Util% |
|---------|--------|-------------|-----------|-------|
| ROCm0 (7900 XTX) | 365 | 1,473.4 | 514.5 | 74.1% |
| RPC0 (3090, triton) | 341 | 52,823.0 | 342.1 | 99.4% |
| RPC1 (3070, triton) | 350 | 8,239.2 | 292.7 | 96.6% |
| RPC2 (3060 Ti, local) | 367 | 938.5 | 23.5 | 97.6% |
| CPU | 130 | 0.3 | 0.2 | 53.3% |

### Per-Token Split Pattern (26 tokens observed)

```
Each token → 5 splits across backends (average):

Split  Backend            Compute    Role
 .0    CPU                ~0 ms      Embedding
 .1    ROCm0 (7900 XTX)   ~73 ms     First layer group
 .2    RPC0  (3090)      ~2,135 ms   ← CRITICAL PATH
 .3    RPC1  (3070)       ~341 ms    Middle layer group
 .4    RPC2  (3060 Ti)     ~36 ms    Last layer group

Wall-clock per token: ~2,135 ms (gated by RPC0/3090)
7900 XTX idle per token: 2,062 ms (96.6% of token cycle)
```

### Data Movement Issues

| Issue | Count | Affected Backends |
|-------|-------|-------------------|
| `host_xfer_rpc` rejections | 400 | All RPC backends |
| `iface_cpy_input` rejections | 133 | RPC0 (3090), RPC1 (3070), RPC2 (3060 Ti) |
| `host_h2d_issue` events | 50 | RPC2 (3060 Ti, local Docker) |
| Slow copies (>1ms) | 280 | Distributed across all backends |

### Slow Copy Destinations

| Destination | Count | Total Time |
|-------------|-------|------------|
| RPC0 (3060 Ti, localhost:50051) | 79 | — |
| RPC0 (3070, triton:50055) | 75 | — |
| ROCm0 (7900 XTX) | 67 | — |
| RPC0 (3090, triton:50054) | 59 | — |

### Throughput

| Metric | With rocprofv3 | Without profiler |
|--------|---------------|-----------------|
| Generation speed | ~0.5 tok/s (27 tok/60s, then cancelled) | 24 tok/s |
| Profiler slowdown | ~48x (network RPC amplifies) | — |

---

## 5. Cross-Configuration Comparison

| Metric | 2-GPU | 4-GPU | Delta |
|--------|-------|-------|-------|
| 7900 XTX kernel compute | 1,960 ms | 183 ms | **-90.7%** |
| 7900 XTX kernel calls | 257,551 | 23,081 | **-91.0%** |
| 7900 XTX matvec calls | 66,992 | 4,474 | **-93.3%** |
| Bottleneck backend | 3060 Ti (local) | 3090 (network) | — |
| Speed (no profiler) | 64 tok/s | 24 tok/s | **-62.5%** |
| `host_xfer_rpc` rejections | few | 400 | — |

---

## 6. Bubble Identification

### Bubble 1: 3090 Split Imbalance (CRITICAL)

**Evidence:**
- The 3090 (RPC0) does 52,823 ms of compute vs 1,473 ms for the 7900 XTX — a 36:1 ratio
- Per-token: 3090 = 2,135 ms, 7900 XTX = 73 ms (29:1 ratio)
- Auto-fit assigned most of the 40 MoE layers to the 3090 due to its 24 GB VRAM

**Impact:** The 7900 XTX completes its work in 73 ms then idles for 2,062 ms waiting for the 3090. This is the primary bottleneck.

**Root cause:** The auto-fit algorithm allocates layers proportional to available VRAM without accounting for network RPC latency. The 3090 (24 GB) gets the lion's share while the 7900 XTX (24 GB local) gets a tiny slice because it also hosts the KV cache.

### Bubble 2: Network RPC Data Movement (HIGH)

**Evidence:**
- 400 `host_xfer_rpc` rejections across all RPC backends
- Pipeline defer/flush flags attempt to overlap compute with data movement but network round-trips defeat this
- Slow copies total 1,177 ms, distributed across all RPC destinations
- The 3090 is accessed over 192.168.8.23 — every kernel call under rocprofv3 traps and amplifies network latency

**Impact:** Data cannot flow from CPU host memory to network RPC backends at the rate the pipeline needs. Each rejected transfer forces a sync fallback path.

### Bubble 3: Host-to-Device Issues on Local RPC (MEDIUM)

**Evidence:**
- 50 `host_h2d_issue` events on RPC2 (3060 Ti, Docker on localhost:50051)
- Even the local Docker RPC has DMA problems — the docker network bridge adds latency

**Impact:** The pipeline's fastest backend (3060 Ti, 36 ms/split) is slowed by unnecessary H2D stalls.

### Bubble 4: 7900 XTX Kernel Gaps During Inference (MEDIUM)

**Evidence (4-GPU kernel trace, top gaps):**
```
2,784 ms: rms_norm -> k_bin_bcast (element-wise op)
2,624 ms: unary_op(sigmoid) -> k_bin_bcast
2,617 ms: quantize_q8_1 -> mul_mat_vec_q
2,611 ms: unary_gated(silu) -> quantize_q8_1
2,605 ms: rms_norm -> k_bin_bcast
```

**Pattern:** After every norm/quantize/activation kernel, there is a ~2.6s gap before the next kernel. These are the 7900 XTX waiting for other pipeline stages (primarily the 3090 on the critical path).

### Bubble 5: 7900 XTX Under-Utilization (MODERATE)

**Evidence:**
- 2-GPU: 257,551 kernel calls, 1,960 ms compute — GPU is doing real work
- 4-GPU: 23,081 kernel calls, 183 ms compute — GPU is 99.9% idle
- The 7900 XTX has 24 GB VRAM but only a few layers assigned

**Impact:** A powerful local GPU sits nearly idle while the network GPU (3090) bottlenecks the entire pipeline.

---

## 7. Root Cause Summary

```
                    ┌──────────────────────────────────────┐
                    │  AUTO-FIT LAYER ASSIGNMENT            │
                    │  3090 (24 GB network) gets most layers│
                    │  7900 XTX (24 GB local) gets few      │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  NETWORK RPC LATENCY                  │
                    │  400 host_xfer_rpc rejections         │
                    │  Every kernel call traps over network │
                    │  rocprofv3 amplifies 4.7x -> 48x      │
                    └──────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────┐
        │                          ▼                       │
        │  ┌─────────────────────────────────────────┐     │
        │  │  7900 XTX: 73ms work, 2062ms idle       │     │
        │  │  GPU utilization: 0.1%                  │     │
        │  └─────────────────────────────────────────┘     │
        │                                                  │
        │  ┌─────────────────────────────────────────┐     │
        │  │  3090: 2135ms work, 0ms idle            │     │
        │  │  CRITICAL PATH — gates entire pipeline  │     │
        │  └─────────────────────────────────────────┘     │
        │                                                  │
        │  RESULT: 64 tok/s -> 24 tok/s (-62.5%)           │
        └──────────────────────────────────────────────────┘
```

---

## 8. Recommendations

### Immediate

1. **Explicit tensor splits (`-ts`)** — Manually balance layer distribution. The 7900 XTX should get a larger share to reduce dependency on the network 3090. A starting point: give the 7900 XTX 30-40% of layers instead of the ~10% it gets from auto-fit.

2. **Test without 3090** — Run 3-GPU (7900 + 3060 Ti + 3070) to isolate whether the 3090 specifically or network RPC in general is the bottleneck.

### Medium-Term

3. **Profile the 3090 directly** — If possible, run nvprof/nsys on the triton 3090 to get its kernel trace and see if the bottleneck is compute, PCIe bandwidth, or network round-trips.

4. **Investigate `host_xfer_rpc` rejections** — The pipeline defer/flush flags should allow overlapping data movement. The 400 rejections suggest the overlap window is too narrow or the network latency defeats it. Consider increasing the defer buffer or batching RPC transfers.

### Long-Term

5. **Network RPC optimization** — For multi-node pipelines, network RPC needs either RDMA, GPU-direct RDMA, or at minimum TCP_NODELAY + larger buffer windows. The current implementation treats network RPC the same as local RPC.

6. **Auto-fit awareness** — The layer assignment algorithm should penalize network backends or cap their share to prevent a single remote GPU from becoming the critical path.

---

## 9. Raw Data

| File | Description | Size |
|------|-------------|------|
| `2gpu/roc-kernel_kernel_trace.csv.gz` | 7900 XTX per-kernel timestamps (2-GPU) | 5.6 MB (92 MB raw) |
| `2gpu/roc-kernel_kernel_stats.csv` | Kernel aggregate stats (2-GPU) | 16 KB |
| `4gpu/roc-4gpu_kernel_trace.csv.gz` | 7900 XTX per-kernel timestamps (4-GPU) | 518 KB (8.3 MB raw) |
| `4gpu/roc-4gpu_kernel_stats.csv` | Kernel aggregate stats (4-GPU) | 13 KB |
| `4gpu/sched-trace.jsonl.gz` | GGML_SCHED_TRACE per-split timing (4-GPU) | 504 KB (3.7 MB raw) |

Decompress with `gunzip` before analysis. Small CSV files are stored uncompressed.

### Quick Analysis Commands

```bash
# Decompress trace files first:
#   gunzip 2gpu/roc-kernel_kernel_trace.csv.gz
#   gunzip 4gpu/roc-4gpu_kernel_trace.csv.gz
#   gunzip 4gpu/sched-trace.jsonl.gz

# Kernel trace summary (7900 XTX)
sudo python3 -c "
import csv; from collections import defaultdict
kernels = []
with open('<trace.csv>') as f:
    for row in csv.DictReader(f):
        kernels.append({'start': int(row['Start_Timestamp']), 'end': int(row['End_Timestamp']), 'name': row['Kernel_Name']})
kernels.sort(key=lambda k: k['start'])
total_t = kernels[-1]['end'] - kernels[0]['start']
total_k = sum(k['end'] - k['start'] for k in kernels)
gaps = [kernels[i]['start'] - kernels[i-1]['end'] for i in range(1, len(kernels)) if kernels[i]['start'] > kernels[i-1]['end']]
print(f'Kernels: {len(kernels)} | Compute: {total_k/1e6:.1f}ms | Span: {total_t/1e6:.0f}ms')
g = sorted(gaps)
print(f'Gaps P50={g[len(g)//2]/1e3:.0f}us P99={g[int(len(g)*.99)]/1e3:.0f}us max={max(g)/1e6:.0f}ms')
"

# Sched trace backend summary
python3 -c "
import json
lines = [json.loads(l) for l in open('sched-trace.jsonl') if l.strip()]
by_backend = {}
for e in lines:
    b = str(e.get('backend','?'))
    by_backend.setdefault(b,{'idle_us':0,'compute_us':0,'count':0})
    by_backend[b]['idle_us'] += e.get('idle_us',0)
    by_backend[b]['compute_us'] += e.get('compute_us',0)
    by_backend[b]['count'] += 1
for b in sorted(by_backend):
    d = by_backend[b]
    util = d['compute_us']/max(d['compute_us']+d['idle_us'],1)*100
    print(f'{b}: splits={d[\"count\"]} compute={d[\"compute_us\"]/1000:.1f}ms idle={d[\"idle_us\"]/1000:.1f}ms util={util:.1f}%')
"
```

---

## 10. Related Documents

- [Pipeline Barrier RPC Sync Stall README](../README.md) — original diagnosis and fix verification
- [ISSUE.md](../ISSUE.md) — ticket with symptom, root cause, fix proposal
- [SCRATCHPAD.md](../SCRATCHPAD.md) — analysis notes
- [CLUSTER-NODE-LAYOUT.md](../../../rpc-patch/patch/CLUSTER-NODE-LAYOUT.md) — cluster topology
- [HANDOVER-CLUSTER-OPS.md](../../../rpc-patch/patch/HANDOVER-CLUSTER-OPS.md) — Docker lifecycle and bench launchers
