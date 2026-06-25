# Path C: RPC Server Multi-GPU Aggregation — Detailed Implementation Plan

**Version:** 1.0  
**Date:** 2026-06-25  
**Status:** Planning Phase  
**Owner:** TurboQuant Team  
**Related:** Path A (complete), Path B (complete, v4.2.2)

---

## 1. Executive Summary

Path C aims to **dramatically reduce RPC round-trips** by allowing a **single `rpc-server` process** to own and schedule across **multiple GPUs** (on the same machine) and present them to the client as **one logical backend**.

After Path B successfully enabled pipeline parallelism across client + remote worker, the next bottleneck is the **client still splitting the model** and sending many small `SET_TENSOR` / `GRAPH_COMPUTE` commands across the network.

**Target Gain:** +80–200% generation throughput on multi-GPU servers (especially 70B+ models) by collapsing multiple GPUs behind one RPC endpoint.

**Risk Level:** High (significant server-side complexity)  
**Estimated Effort:** 12–18 days for a production-grade implementation

---

## 2. Motivation — Why Path C Now?

### Current State After Path A + B

- Client still sees **N separate RPC devices** when there are N GPUs behind `rpc-server`.
- Every layer transition between GPUs on the **same physical machine** still goes through the client (unnecessary network hop).
- For 70B+ models, the client must manage complex tensor splits and many small RPC calls.
- Path B already gave us good event + drain infrastructure — we can reuse it.

### The Opportunity

If one `rpc-server` can internally manage 2–4 GPUs (e.g. 2× 3090 or 3090 + 3060 Ti), the client only needs to talk to **one endpoint**. This eliminates:
- Cross-RPC `SET_TENSOR` traffic between GPUs on the same box
- Multiple `GRAPH_COMPUTE` round-trips per token
- Complex client-side layer splitting logic

**Expected Impact:**
- Large models (70B+) become much more practical over RPC
- Better GPU utilization (internal scheduler can do smarter placement)
- Foundation for future true multi-node aggregation

---

## 3. High-Level Architecture

### Before (Current State)

```
llama-server (client)
    ├── RPC0 (GPU A on machine X)
    └── RPC1 (GPU B on machine X)
```

### After (Path C Target)

```
llama-server (client)
    └── Single RPC endpoint (machine X)
            ├── Internal Scheduler
            ├── GPU A (local)
            └── GPU B (local)
```

The `rpc-server` becomes a **multi-device backend** that hides internal GPUs from the client.

---

## 4. Implementation Phases

### Phase C1: Multi-Device RPC Server Foundation (3–4 days)

**Goal:** Allow one `rpc-server` to initialize and manage multiple CUDA/ROCm devices.

**Tasks:**
1. Extend `rpc-server` startup to accept multiple devices (e.g. `--devices CUDA0,CUDA1` or auto-detect).
2. Create internal `ggml_backend` array + one unified `ggml_backend_sched` inside the server.
3. Modify device enumeration in `RPC_CMD_DEVICE_COUNT` and `RPC_CMD_GET_DEVICE_PROPS` to report a **single logical multi-GPU device** (or keep multiple but add aggregation metadata).
4. Add basic memory query aggregation (`get_device_memory` across all local GPUs).

**Key Files:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp` (server side)
- New file or section: `rpc_multi_device.cpp` (recommended for cleanliness)

**Deliverable:** `rpc-server` can start with 2 GPUs and report them as managed internally.

---

### Phase C2: New Multi-GPU Protocol Commands (3–4 days)

**Goal:** Introduce commands that let the client send work for **multiple devices in one message**.

**New Commands:**

| Command                    | Purpose                                      | Response |
|---------------------------|----------------------------------------------|----------|
| `RPC_CMD_GRAPH_COMPUTE_MULTI` | Submit graph that spans multiple local GPUs | Deferred or immediate |
| `RPC_CMD_GRAPH_RECOMPUTE_MULTI` | Recompute cached multi-GPU graph           | Deferred |
| `RPC_CMD_SET_TENSOR_MULTI`   | Batch SET_TENSOR targeting specific devices | Immediate |

**Wire Format Ideas:**

```cpp
struct rpc_msg_graph_compute_multi_req {
    uint32_t n_devices;
    uint32_t device_ids[4];           // which local GPUs
    uint64_t layer_ranges[4][2];      // start/end layer per device
    // followed by serialized graph
};
```

**Tasks:**
1. Define new message structs in `ggml-rpc.cpp`.
2. Implement server-side handlers that dispatch to the internal multi-device scheduler.
3. Add client-side support to detect when multiple RPC endpoints resolve to the **same physical `rpc-server`** (via new HELLO metadata or configuration).

**Backward Compatibility:** Old single-device clients must continue to work.

---

### Phase C3: Internal Multi-GPU Scheduler on Server (4–5 days)

**Goal:** The `rpc-server` must intelligently split and schedule graphs across its local GPUs.

**Key Work:**
1. Reuse or extend `ggml_backend_sched` inside `rpc-server`.
2. Implement layer-wise or tensor-parallel splitting logic **inside the server** (instead of forcing the client to do it).
3. Add event-based synchronization between local GPUs (using CUDA events or ROCm equivalents — much cheaper than RPC).
4. Handle heterogeneous GPUs (different VRAM sizes, different compute capability).

**Challenges to Solve:**
- Memory allocation across devices
- KV cache placement strategy
- Graph splitting heuristics (reuse existing llama.cpp split logic where possible)
- Error handling when one GPU fails

**Deliverable:** Server can run a model split across 2 local GPUs with good performance.

---

### Phase C4: Client-Side Aggregation & Discovery (3–4 days)

**Goal:** Make the client treat a multi-GPU `rpc-server` as a single device when possible.

**Tasks:**
1. Add discovery mechanism (new field in `RPC_CMD_HELLO` response: `multi_gpu_aggregate = true`, `local_device_count`).
2. Modify client device initialization to collapse multiple endpoints that point to the same multi-GPU server into one logical device.
3. Update tensor split and layer offload logic in `llama-context.cpp` / `llama-model.cpp` to be aware of aggregated devices.
4. Add configuration option: `--rpc-aggregate` or auto-detect.

**Optional but powerful:**
- Allow the server to suggest optimal tensor splits back to the client during HELLO.

---

### Phase C5: Advanced Features & Hardening (2–3 days)

- Multi-device graph caching (`GRAPH_RECOMPUTE_MULTI`)
- Better memory defragmentation across GPUs
- Telemetry / profiling hooks inside `rpc-server` (per-GPU utilization)
- Graceful degradation when one GPU runs out of memory
- Full testing on 2-GPU and 3-GPU configurations

---

## 5. Protocol Versioning

- Bump `RPC_PROTO_MINOR_VERSION` to **3**
- All new commands must be negotiated via HELLO capability flags
- Old clients continue to see multiple devices (fallback behavior)

---

## 6. Testing Strategy

### Required Test Matrix

| # | Configuration                  | Model Size | Expected Outcome |
|---|--------------------------------|------------|------------------|
| 1 | 2 GPUs on same machine (e.g. 3090 + 3060 Ti) | 70B+      | Major reduction in RPC calls, high throughput |
| 2 | 3–4 GPUs on one server         | 70B–120B  | Good scaling     |
| 3 | Heterogeneous GPUs             | 70B       | Correct placement |
| 4 | Fallback to single-GPU mode    | Any       | No regression    |
| 5 | Mixed with Path B client       | 30B–70B   | Full compatibility |

**Key Metrics to Measure:**
- Number of RPC round-trips per token (should drop significantly)
- Generation tokens/sec vs current Path B baseline
- Memory efficiency across GPUs
- Stability under long generation

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|----------|
| Server-side scheduler complexity explodes | High | High | Start with simple layer-wise split, reuse existing `ggml_backend_sched` |
| Heterogeneous GPU support is buggy | Medium | High | Begin with homogeneous GPUs only |
| Client detection of aggregated servers is fragile | Medium | Medium | Use explicit configuration first (`--rpc-aggregate`) |
| Performance regression on single-GPU path | Low | High | Extensive regression testing |
| Increased memory fragmentation | Medium | Medium | Add periodic defrag + good allocation strategy |

---

## 8. Dependencies & Prerequisites

- Path B must be complete and stable (it is)
- Good understanding of current `ggml_backend_sched` and `llama_split_mode`
- Access to multi-GPU machines for testing (at least 2× high-VRAM cards)

---

## 9. Success Criteria

**Minimum Success (MVP):**
- One `rpc-server` can manage 2 GPUs on the same machine
- Client can offload a model using a single RPC endpoint
- Throughput improves vs current Path B on 70B+ models

**Full Success:**
- Clean heterogeneous GPU support
- Automatic or suggested optimal splitting from server
- No regression on single-GPU or Path B fallback paths
- Production-grade stability and error handling

---

## 10. Estimated Timeline (Aggressive but Realistic)

| Phase | Duration | Cumulative |
|-------|----------|------------|
| C1: Multi-Device Foundation     | 3–4 days | Week 1     |
| C2: Protocol Commands           | 3–4 days | Week 1–2   |
| C3: Internal Scheduler          | 4–5 days | Week 2–3   |
| C4: Client Aggregation          | 3–4 days | Week 3     |
| C5: Hardening & Testing         | 3–4 days | Week 3–4   |
| **Total**                       | **12–18 days** | ~3–4 weeks |

---

## 11. Recommended Approach & Next Steps

**Recommended Scope for First Iteration (MVP):**
1. Focus only on **homogeneous GPUs** on the same machine.
2. Use explicit configuration on both client and server rather than complex auto-discovery.
3. Reuse as much existing `ggml_backend_sched` logic as possible.
4. Prioritize `GRAPH_COMPUTE_MULTI` + internal layer splitting.

**Immediate Next Actions:**
1. Create `rpc-path-c-tracking.md` (following the same format as Path B).
2. Prototype C1 (multi-device `rpc-server` startup) in a branch.
3. Define the exact wire format for `RPC_CMD_GRAPH_COMPUTE_MULTI`.
4. Run initial micro-benchmarks measuring current RPC call count on a 70B model split across two local GPUs.

---

## 12. Open Questions for Decision

Before starting heavy implementation, we should align on:

1. **Scope** — Homogeneous GPUs only for v1, or attempt heterogeneous from the start?
2. **Discovery** — Explicit config (`--rpc-aggregate`) or automatic detection?
3. **Splitting Strategy** — Let the server decide optimal split, or keep client in control?
4. **Priority** — How important is 70B+ support vs improving current 30–40B workloads?

---

**This document establishes the foundation for Path C.**  
Once we align on scope and open questions, we can move to implementation and create the corresponding tracking + handover documents.

Ready to begin **Phase C1 prototyping** when you give the green light.