# ADR-0004: Server-Side Scheduling Model for Co-Localized GPUs

## Status

**ACCEPTED (Option B+: GRAPH_COMPUTE_ALL + weighted weight placement)** — 2026-07-11

## Context

Path C explores server-side scheduling on triton's co-localized dual-GPU (`:50054` 3090 + `:50055` 3070). Currently, the client `ggml_backend_sched` drives every split; each `rpc-server` is a passive compute endpoint. For co-localized GPUs on the same node, server-side scheduling could reduce client RTT overhead and enable local-PCIe cross-device copies.

This ADR establishes the model for server-side scheduling on co-localized GPUs.

### Problem

Current model:
- Client scheduler dispatches splits serially to each RPC endpoint
- Each `rpc-server` is passive — waits for client commands
- For co-localized GPUs (triton 3090+3070), client RTT adds overhead per split
- No server-side parallelism between local GPUs

Path C question:
- Can the server schedule its own GPUs in parallel?
- What changes to the RPC protocol are needed?
- What is the blast radius on existing Path-B+ behavior?

### Inputs (gathered during D4.2 analysis)

| Analysis | Source | Finding |
|----------|--------|---------|
| Per-device RPC splits | C1 baseline (trace-f-3gpu-plus) | 4 splits for 3-server-GPU config |
| Scheduler partition behavior | `ggml-backend.cpp` code analysis | Same-type backends: auto-partition assigns ALL to GPU 0 |
| Performance model | Speed ratio analysis (3090 vs 3070) | 1.75x TFLOPS ratio; optimal split is 64/36 |
| Local hardware benchmark | 7900 XTX + 3060 Ti | GPipe OFF: 237.83 t/s; single GPU: 264-270 t/s; RPC overhead: ~410 us/tok |

---

## Options

### Option A: Client-driven (no change)

- No server-side scheduling
- Client continues to drive all splits
- Simplest; no RPC protocol changes
- Does not reduce client RTT overhead
- **Uplift: 0%**

### Option B: Server-side `GRAPH_COMPUTE_ALL` (simple)

- Server receives full graph, schedules its own GPUs via `ggml_backend_sched`
- Requires new RPC command
- **Critical finding:** `ggml_backend_sched` cannot auto-partition same-type backends. `ggml_backend_sched_backend_from_buffer()` (line 1160) returns the FIRST backend that supports the buffer type and op. For two CUDA backends, ALL nodes go to GPU 0.
- Without explicit weight placement, GPU 1 stays idle.
- **Uplift with equal weight placement: 0%** (single GPU does all work)
- **Uplift with weighted weight placement: +33%** (see Option B+)

### Option B+: Server-side `GRAPH_COMPUTE_ALL` + weighted weight placement (RECOMMENDED)

- Client sends full graph via `RPC_CMD_GRAPH_COMPUTE_ALL`
- Client assigns layer weights proportional to GPU compute capacity during SET_TENSOR phase
- Server-side scheduler follows weight placement — GPU 0 gets 64% of layers (3090), GPU 1 gets 36% (3070)
- Cross-GPU copy via PCIe P2P (no client involvement)
- Combined compute+sync response eliminates separate EVENT_RECORD round-trip
- **Uplift: +33%** (balanced compute time: 3.04 ms on 3090 vs 3.08 ms on 3070)
- **Stretch target: +41%** with overlapped copy

### Option C: Server-side parallel dispatch

- Server dispatches to its GPUs in parallel (CUDA streams)
- Client still specifies split assignment
- Moderate complexity; no graph partitioning needed
- Partial RTT reduction (parallel vs serial)
- Does NOT reduce cross-device copy overhead (still client-mediated)
- **Uplift: ~5-10%** (saves dispatch latency only)
- Rejected: Option B+ provides higher uplift with less server-side threading risk

---

## Decision

**Adopt Option B+ (`GRAPH_COMPUTE_ALL` + weighted weight placement) for Path C.**

### Rationale

1. **Performance evidence:** Performance model (docs/wayfinder/D4-performance-audit.md, docs/wayfinder/D4-performance-model.md) confirms 25-33% throughput uplift over equal split, far exceeding the 3% threshold that would justify the added complexity.

2. **Same-type GPU partition is a solved problem:** The scheduler naturally follows weight placement. No server-side scheduler changes needed — all optimization happens in client SET_TENSOR logic.

3. **Client-side weight assignment is low-risk:** Weight placement during SET_TENSOR is already a per-device concept. Extending it to proportional assignment adds minimal complexity.

4. **RTT reduction is real but secondary:** Combined compute+sync response saves 1-2 RTTs per token. On triton (loopback, ~10-20 us RTT), this is ~1% uplift. On local machine (chipset-bound, ~50 us RTT), this is ~2.4%. Worth doing but not the primary driver.

5. **Path C unlocks D5:** Server-side scheduling on triton validates the combined-graph approach, which D5 needs for per-backend sub-stages. Punting this would leave D5 without a foundational building block.

### Consequences

| Positive | Negative |
|----------|----------|
| +33% throughput on same-type multi-GPU servers | New RPC commands (RPC_CMD_GRAPH_COMPUTE_ALL, RPC_CMD_GRAPH_RECOMPUTE_ALL) |
| Server-local PCIe copies (no client RTT for cross-GPU transfer) | Capability negotiation in HELLO handshake needed |
| Balanced GPU utilization (both GPUs busy simultaneously) | Client must know GPU speed ratio (benchmark or capability flag) |
| One server endpoint simplifies client split topology | Old client + new server: graceful fallback works (per-device code path) |
| Foundation for D5 per-backend sub-stages | New client + old server: graceful fallback (HELLO capability check) |

### What Changes

| Component | Change |
|-----------|--------|
| `ggml-rpc.cpp` server | New command handlers for RPC_CMD_GRAPH_COMPUTE_ALL (18) and RPC_CMD_GRAPH_RECOMPUTE_ALL (19) |
| `ggml-rpc.cpp` server | `create_multi_device_sched()` helper to build server-side ggml_backend_sched across server GPUs |
| `ggml-rpc.cpp` server | Capability flag in HELLO response (`RPC_CAP_MULTI_DEVICE`) |
| `ggml-rpc.cpp` client | Weighted weight placement: assign layers proportional to GPU speed (e.g., 64/36 for 3090+3070) |
| `ggml-rpc.cpp` client | Combined compute+sync: skip separate EVENT_RECORD when using GRAPH_COMPUTE_ALL with sync_mode=1 |
| `ggml-rpc.cpp` client | Capability detection: check HELLO response before using new commands |
| `ggml-rpc.cpp` client | Split logic: first multi-device endpoint sends ALL; subsequent splits on same endpoint become no-ops |

### What Does NOT Change

- `ggml_backend_sched` itself (no changes needed — it naturally follows weight placement)
- Single-GPU RPC paths (no regression for existing configs)
- Per-device GRAPH_COMPUTE path (old clients continue to work)
- Backward compatibility with Path-B+ (GPipe OFF, per-device RPC)

### Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CUDA P2P not enabled between 3090 and 3070 | Medium | Fall back to system-memory copy; uplift drops to ~18% |
| 3070 VRAM insufficient for extra layers | Low | Skip Path C for VRAM-constrained configs; use Option A |
| Weight ratio mis-estimated | Low | Autotune: run 1-token benchmark to measure per-GPU latency, adjust ratio dynamically |
| Server-side scheduler OOM on intermediate buffers | Low | `ggml_backend_sched_reserve()` returns error; client falls back to per-device |

---

## Required D4.1 Data to Validate

| Data Point | Threshold for Confidence |
|-----------|------------------------|
| 3090 vs 3070 layer compute time ratio | >= 1.5x (justifies >60/40 weighted split) |
| CUDA P2P bandwidth between 3090 and 3070 | >= 20 GB/s (fast copy, <0.3 ms overhead) |
| 3070 VRAM headroom | >= 2 GB after model load (intermediate buffers) |
| RTT reduction measurement | >= 1 RTT saved per token |

---

*Decision finalized 2026-07-11 after performance audit. See docs/wayfinder/D4-performance-audit.md for the full analysis and D4-performance-model.md for the performance model.*
