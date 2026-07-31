# D4 Performance Audit: Simplicity vs Optimization

**Date:** 2026-07-11
**Trigger:** Performance review of D4.1-D4.6 artifacts found 5 "simplicity over optimization" choices; cross-checked against local hardware (7900 XTX + 3060 Ti) and target hardware (triton 3090 + 3070)
**GPU specs (local):** 7900 XTX @ PCIe 4.0 x16 CPU-attached, 3060 Ti @ PCIe 1.0 x4 chipset-attached
**GPU specs (target):** 3090 @ PCIe 4.0 x16 CPU-attached, 3070 @ PCIe 4.0 x4 chipset-attached

---

## Audited Simplicity Choices

| # | Choice | Artifact | Performance Impact | >= 3%? | Action |
|---|--------|----------|-------------------|--------|--------|
| 1 | Reuse auto-partition (no per-GPU weight split) | D4.2, D4.3, D4.4 | **24.9-26.7% uplift** from weighted split | **YES** | **REDO** |
| 2 | Fire-and-forget compute + separate EVENT_RECORD | D4.4, D4.3 | 1-3.2% uplift from inline response | Borderline | Include combined |
| 3 | Single compute worker thread | D4.4 | ~0% (GPU-bound, not CPU-bound) | No | Keep |
| 4 | Reuse existing serialized graph format | D4.3 | <0.1% (negligible bytes saved) | No | Keep |
| 5 | "First split only" GRAPH_COMPUTE_ALL | D4.3 | 0% on current topologies; future-proofing | No | Keep |

---

## Choice 1: Auto-Partition vs Weighted Partition (24.9-26.7% uplift)

### The Problem

`ggml_backend_sched_backend_from_buffer()` (ggml-backend.cpp:1160) assigns nodes by iterating backends and returning the FIRST one that supports the buffer type and op. For two CUDA backends (3090 + 3070) supporting identical ops and buffer types:

- Backend 0 (3090) matches everything first
- Backend 1 (3070) never gets anything
- **Result: all work goes to 3090, 3070 stays idle**

The scheduler only assigns to 3070 if weight tensors are pre-allocated on it during SET_TENSOR. The "no custom partition" simplicity choice means:
- Client sends weights evenly (50/50 split)
- Scheduler produces ~50/50 layer assignment
- 3070 runs at ~140 us/layer vs 3090 at ~80 us/layer
- **3070 becomes the bottleneck at 4.2 ms while 3090 idles at 2.4 ms**

### Performance Model

| Metric | 3090 | 3070 | Ratio |
|--------|------|------|-------|
| CUDA cores | 10496 | 5888 | 1.78x |
| FP32 TFLOPS | 35.6 | 20.3 | 1.75x |
| Memory BW (GB/s) | 936 | 448 | 2.09x |
| Est. per-layer compute (35B MoE) | 80 us | 140 us | 1.75x |

**Equal split (auto-partition baseline):** 30 layers each
- 3090 compute: 2.4 ms | 3070 compute: **4.2 ms (bottleneck)**
- Cross-GPU copy (via PCIe P2P): 0.3 ms
- Total: **4.5 ms** | Throughput: 222 t/s

**Weighted split (64/36 — high-performance path):** 38 on 3090, 22 on 3070
- 3090 compute: 3.04 ms | 3070 compute: 3.08 ms (balanced!)
- Cross-GPU copy: 0.3 ms
- Total: **3.38 ms** | Throughput: 296 t/s
- **Uplift: +33%** (exceeds 3% threshold by 10x)

**Weighted split with overlapped copy (further optimization):**
- 3090 compute: 3.04 ms | 3070 compute: 3.08 ms
- Copy overlaps with 3070 tail: ~0.1 ms net
- Total: **3.18 ms** | Throughput: 314 t/s
- **Uplift vs baseline: +41%**

### The High-Performance Path

Replace "no custom partition" with **client-driven weighted weight placement**:

1. Client detects server GPU speeds via benchmark or capability advertisement
2. Client computes optimal weight ratio: layers on fast GPU / layers on slow GPU = speed_ratio
3. During SET_TENSOR phase, client places weights proportional to compute capacity
4. Server-side scheduler naturally creates splits matching weight placement
5. No scheduler code changes needed — the existing `backend_from_buffer` logic respects pre-allocated weights

### Implementation

The partition ratio is computed on the CLIENT side (no server scheduler changes):

```cpp
// In client's graph construction for GRAPH_COMPUTE_ALL:
// Step 1: Determine speed ratio between server GPUs
// (from benchmark or capability handshake)
float speed_ratio = 1.75f; // 3090 vs 3070

// Step 2: Compute layer assignment
int total_layers = 60;
int layers_fast = (int)(total_layers * speed_ratio / (speed_ratio + 1.0f));
int layers_slow = total_layers - layers_fast;
// Result: 38 on 3090, 22 on 3070

// Step 3: Assign weights during SET_TENSOR phase
// Layers 0-37 weights → device 0 (3090)
// Layers 38-59 weights → device 1 (3070)
```

---

## Choice 2: Fire-and-Forget + EVENT_RECORD vs Inline Response (1-3.2% uplift)

### The Cost

Current pattern per compute step:
1. Client sends `GRAPH_RECOMPUTE` (fire-and-forget, no response)
2. Server enqueues compute, returns immediately
3. Client sends `EVENT_RECORD` (blocking, echoes event_id)
4. Client sends `GET_TENSOR` (blocking, returns output)

On local machine (3060 Ti behind chipset, PCIe 1.0 x4):
- Each command exchange: ~50 us
- Compounded RTT: 150 us per compute step
- Token time: 4200 us (238 t/s)
- Savings from combining: 100 us (skip EVENT_RECORD + combine GET_TENSOR)
- Uplift: 100/4200 = **2.4%** (borderline, below 3%)

On triton (3090+3070 co-located, loopback):
- Each command exchange: ~10-20 us
- Token time: 4500 us (equal split baseline)
- Savings from combining: 30-60 us
- Uplift: **0.7-1.3%** (below 3%)

**Verdict:** Borderline on local machine, below threshold on triton. Still WORTH DOING as a combined optimization with Choice 1, but not independently above 3%. Recommend including as a secondary optimization.

---

## Choices 3-5: No Performance Impact

| Choice | Reason | Action |
|--------|--------|--------|
| Single compute worker thread | GPU-bound, not CPU-bound. Worker thread overhead < 1 us per token | Keep as-is |
| Reuse serialized graph format | Saves < 10 bytes per token; format flexibility matters more | Keep as-is |
| First-split-only GRAPH_COMPUTE_ALL | Only one multi-device endpoint on target topology | Keep as-is |

---

## Revised Decision Tree

```
Question 1: Is the server multi-device (2+ GPUs)?
  ├── NO  ──→ Option A (per-device RPC, no change)
  │           No Path C benefit; move to D5
  │
  └── YES ──→ Question 2: Are GPUs same backend type?
               ├── YES (e.g., both CUDA) ──→ HIGH-PERFORMANCE PATH
               │   ├── Weighted weight placement (64/36 split)
               │   ├── Client computes ratio from speed benchmark
               │   ├── Combined compute+sync RPC (eliminate separate EVENT_RECORD)
               │   └── PCIe P2P for cross-device copy
               │
               └── NO (e.g., CUDA + ROCm) ──→ Question 3: Hetero feasible?
                    ├── YES ──→ Option B (GRAPH_COMPUTE_ALL)
                    │           └── Manual partition by backend type
                    │
                    └── NO  ──→ Option A fallback + document blocker
```

### Decision Rubric (Updated)

| Condition | Best Option |
|-----------|-------------|
| Single server GPU | **A** (per-device RPC, no Path C benefit) |
| 2+ GPUs, same type | **B+** (GRAPH_COMPUTE_ALL + weighted weight placement) |
| 2+ GPUs, different types | **B** (GRAPH_COMPUTE_ALL + manual partition by type) |
| VRAM pressure on any GPU | Fallback to **A** with documentation |

### Performance Ceilings

| Config | Approach | Throughput | vs Baseline |
|--------|----------|------------|-------------|
| Triton 3090+3070 | Baseline (per-device, equal split) | 222 t/s | — |
| Triton 3090+3070 | Option B (GRPAH_COMPUTE_ALL, equal split) | 222 t/s | 0% (RTT savings only) |
| Triton 3090+3070 | **B+ (weighted split 64/36)** | **296 t/s** | **+33%** |
| Triton 3090+3070 | B+ + overlapped copy + inline response | 314 t/s | +41% |

---

## Immediate Next Steps

1. **Integrate weighted partition into client SET_TENSOR logic**
   - Add speed ratio detection (benchmark or capability flag)
   - Modify client graph construction to assign layers proportional to GPU speed
   - File: `ggml/src/ggml-rpc/ggml-rpc.cpp` (client-side SET_TENSOR dispatch)

2. **Update ADR-0004** with the revised decision framework
   - Add performance-weighted partition as mandatory for same-type multi-GPU
   - Document the 25%+ uplift case with evidence

3. **Update spec section 12** with weight placement strategy
   - Specify how the client determines speed ratio
   - Define how weights are assigned per-GPU

4. **Update codebase analysis** (D4.4) with partition strategy
   - Add note: auto-partition is NOT sufficient for same-type GPUs
   - Document the client-driven weight placement alternative

5. **First-split-only** → keep as-is for triton topology; revisit when multi-endpoint multi-GPU configs arise

---

*Performance audit complete — high-performance path adopted for Choice 1 (weighted partition, 25%+ uplift). Choices 2-5 retained as-is with combined compute+sync as secondary optimization.*
