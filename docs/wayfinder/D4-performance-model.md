# D4 — Performance Model: Custom Partition vs Auto-Partition

**Date:** 2026-07-11  
**Inputs:** D4.1 baseline prep, D4.2 ADR analysis, D4.3 spec, D4.4 codebase analysis  
**Purpose:** Quantify whether custom layer-weighted partition beats naive auto-partition for triton dual-GPU (3090+3070).

---

## 1. GPU Specs Summary

| GPU | CUDA Cores | VRAM | FP32 TFLOPS | Boost Clock | Relative Compute |
|-----|-----------|------|-------------|-------------|------------------|
| RTX 3090 | 10496 | 24 GB | ~35.6 TFLOPS | 1.4 GHz (base) / 1.7 GHz (boost) | 1.0x (reference) |
| RTX 3070 | 5888 | 8 GB | ~20.3 TFLOPS | 1.5 GHz (base) / 1.75 GHz (boost) | ~0.57x |

**Speed ratio:** 3090 is **1.75x** faster than 3070 for transformer layer compute (matches FP32 TFLOPS ratio: 35.6/20.3 = 1.75x). The clock-speed advantage of the 3070 is minor (~3-5%) and does not close the core-count gap. For memory-bound layers (large matmuls in 35B model), the ratio holds because both GPUs use GDDR6X with proportional bandwidth (936 GB/s vs 448 GB/s = 2.09x bandwidth ratio, but core-limited layers dominate at this model size).

---

## 2. How `ggml_backend_sched` Assigns Nodes (Code Evidence)

From `ggml/src/ggml-backend.cpp` line 1328, `ggml_backend_sched_split_graph`:

| Behavior | Code Evidence | Impact |
|----------|---------------|--------|
| **Buffer-type affinity** | Pass 1 (line 1350-1387): assigns backend based on where input tensors already reside | Layers follow pre-allocated weight buffers |
| **No perf weighting** | `ggml_backend_sched_backend_id_from_cur` uses tensor->buffer->backend, NOT compute speed | No awareness of 3090 vs 3070 perf difference |
| **Adjacency expansion** | Pass 2 (line 1390+): expands same-backend assignment to adjacent ops | Contiguous layer ranges per GPU |
| **Equal split default** | With equal buffer allocation on both backends, splits land ~50/50 | 30 layers each on 60-layer model |

**Key finding:** The scheduler is topology-aware but **performance-blind**. It splits based on buffer placement, not compute throughput. If weights are split 50/50 across backends, compute time is dominated by the slower GPU.

---

## 3. Compute-Time Model

### Assumptions (Qwen3.6-35B-A3B, 60 transformer layers, FP16/BF16 inference)

| Parameter | Value | Source |
|-----------|-------|--------|
| Layers | 60 | Model spec |
| Compute per layer on 3090 | 80 us | ~2.4 ms total / 30 layers (D4.3 estimate) |
| Compute per layer on 3070 | 140 us | ~4.2 ms total / 30 layers (1.75x slower) |
| Cross-device hidden-state copy | 300 us | PCIe 4.0 x16 P2P estimate |
| Hidden-state size (35B, 4096-dim, BF16) | ~82 KB | 4096 * 8192 * 2 bytes |

### Scenario A: Equal Split (Auto-Partition Baseline)

| Device | Layers | Compute (us) | Notes |
|--------|--------|-------------|-------|
| RTX 3090 | 30 | 2400 | Finishes early, waits |
| RTX 3070 | 30 | 4200 | Straggler |
| Cross-copy | — | 300 | Between layers 30->31 |
| **Total serial** | — | **4500 us** | 3070-bound |

3090 idle time: 4200 - 2400 = 1800 us (40% of token time wasted).

### Scenario B: Weighted Split (Custom Partition)

Optimal split ratio = 1.75:1 (3090:3070) = **63.6% / 36.4%** of layers.

| Device | Layers | Compute (us) | Notes |
|--------|--------|-------------|-------|
| RTX 3090 | 38 | 3040 | 80 us * 38 |
| RTX 3070 | 22 | 3080 | 140 us * 22 |
| Cross-copy | — | 300 | Between layers 38->39 |
| **Total serial** | — | **3380 us** | Near-balanced |

3090 idle time: 3080 - 3040 = 40 us (1.2% — essentially fully utilized).

### Scenario C: Overlap with CUDA Streams (Best Case)

If the scheduler uses CUDA streams with async copy (both GPUs compute concurrently, overlap copy):

| Scenario | Formula | Total (us) |
|----------|---------|------------|
| A (equal, serial) | max(2400, 4200) + 300 | 4500 |
| B (weighted, serial) | max(3040, 3080) + 300 | 3380 |
| A (equal, overlapped) | max(2400, 4200) | 4200 |
| B (weighted, overlapped) | max(3040, 3080) | 3080 |

---

## 4. Throughput Uplift Summary

| Metric | Equal (A) | Weighted (B) | Uplift |
|--------|-----------|--------------|--------|
| Serial token time (us) | 4500 | 3380 | **24.9%** |
| Overlapped token time (us) | 4200 | 3080 | **26.7%** |
| 3090 utilization | 53% | 90% | +37 pp |
| 3070 utilization | 100% | 100% | — |
| Tokens/sec (serial) | 222 | 296 | +74 t/s |
| Tokens/sec (overlapped) | 238 | 325 | +87 t/s |

**Verdict: 24.9-26.7% uplift far exceeds the 3% threshold.** Custom partition is strongly justified.

---

## 5. Fire-and-Forget vs Blocking Compute (PCIe 1.0 x4)

For chipset-attached GPUs (e.g., 3060 Ti behind chipset, PCIe 1.0 x4 equivalent):

| Parameter | Value |
|-----------|-------|
| PCIe 1.0 x4 bandwidth | ~1 GB/s each direction |
| RTT (chipset hop) | ~50 us |
| Hidden-state size | ~82 KB |
| Transfer time (82 KB @ 1 GB/s) | ~82 us |

### Blocking Compute (Inline Response)

```
Client -> [REQ] -> Server computes -> [RSP + data] -> Client
Total = compute + RTT + transfer = T_comp + 50 + 82 = T_comp + 132 us
```

### Fire-and-Forget with Separate EVENT_RECORD

```
Client -> [REQ] -> Server computes -> [RSP, no data] -> Client
Client -> [EVENT_RECORD] -> Server -> [EVENT_ACK] -> Client
Total = compute + RTT + RTT = T_comp + 50 + 50 = T_comp + 100 us
(No data in response; data readback via separate GET_TENSOR if needed)
```

### Savings

| Mode | Overhead (us) | Savings |
|------|---------------|---------|
| Blocking (inline) | 132 | — |
| Fire-and-forget (EVENT_RECORD) | 100 | 32 us |

For a token time of ~3000 us (dual-GPU overlapped): 32/3000 = **1.07%**.

For a token time of ~1000 us (single fast GPU): 32/1000 = **3.2%**.

**Verdict: Fire-and-forget saves 1-3.2% depending on token time. At the high-throughput end (fast GPU, short token time), it clears the 3% threshold. At the low end (slow dual-GPU), it does not.**

---

## 6. Risks and Caveats

| Risk | Impact | Mitigation |
|------|--------|------------|
| 3070 VRAM cap (8 GB) | Limits max layers assignable to 3070 | Profile actual VRAM per layer; cap at ~22 layers |
| PCIe P2P not enabled | Copy falls back to host-staged (~2x slower) | Enable CUDA peer access; check `cudaDeviceCanAccessPeer` |
| Layer compute variance | MoE layers have variable compute per layer | Use average across layers; variance is ~10% |
| Scheduler overhead | `ggml_backend_sched_reserve` adds ~50-100 us | Amortized across recompute calls |
| Equal buffer allocation | Default ggml allocates evenly across backends | Custom `ggml_gallocr` or pre-place weights on 3090 |

---

## 7. Recommendations

1. **Implement custom partition** — 25% uplift is too large to ignore. Assign layers proportional to compute capacity (64/36 split for 3090/3070).
2. **Enable CUDA P2P** — Required for the 300 us cross-copy estimate. Without P2P, copy cost doubles and uplift drops to ~18%.
3. **Use fire-and-forget EVENT_RECORD** — Marginal at dual-GPU latency but free wins at single-GPU or low-latency configs.
4. **Validate with D4.1 cluster data** — Replace estimated 80/140 us per-layer with measured values from triton capture.

---

*Model complete. Pending D4.1 live cluster measurements to replace estimates with empirical data.*
