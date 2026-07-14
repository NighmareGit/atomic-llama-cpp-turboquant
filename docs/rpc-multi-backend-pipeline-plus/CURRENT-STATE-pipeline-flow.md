# CURRENT STATE: Pipeline flow (as implemented, 2026-07-14)

**Status:** Factual — describes what the code actually does today. Not aspirational.
**Contrast with:** [DESIGN-path-d-layer-pipeline.md](DESIGN-path-d-layer-pipeline.md) (GPipe vision, D0 design), [MISSION.md](MISSION.md) (goals and targets).

---

## 1. End-to-end flow: token decode

Each generated token triggers one `llama_decode()` call. The decode path is:

```text
llama_decode()
  |
  +-- build graph (ggml_cgraph for this token)
  |
  +-- ggml_backend_sched_graph_compute_async(sched, graph)
        |
        +-- splits graph into n_splits (one per backend)
        +-- ggml_backend_sched_compute_splits()   [serial loop]
        |     for each split_id:
        |       input_wait_copy  (copy tensors from producer to consumer backend)
        |       graph_compute_async (dispatch to backend)
        |       event_record     (signal completion)
        |
        +-- pipeline_barrier()   [rotate copy slots, wait on reused slots]
```

The **scheduler** (`ggml-backend.cpp`) owns the split dispatch. `llama-server` does not orchestrate individual RPC hops — it calls `sched_graph_compute_async` once per token and the scheduler handles everything.

### 1.1 Single-seq fast path (default)

When there is one sequence (no multi-seq server mode), a shortcut path skips the per-stage state machine:

```cpp
// llama-context.cpp (~line 82)
if (gf) {
    ggml_backend_sched_graph_compute_async(sched, gf);
}
```

The full graph is submitted once. The scheduler splits it and runs the serial loop. After all splits complete, the sampler runs on the output logits and the next token is known.

### 1.2 Multi-seq path (Mode B, GPipe)

When multiple sequences are in flight, each decode step dispatches per-seq graphs through a stage-available loop. This uses `GRAPH_COMPUTE_STAGE` RPC commands to filter splits per stage (D6.9). This is the `llama_decode_gpipe_multi_impl` path and is gated behind multi-seq server mode.

---

## 2. The serial split dispatch loop (within one token)

This is the core of the current system. `ggml_backend_sched_compute_splits()` runs splits **serially**:

```text
for split_id = 0 to n_splits-1:
  +-- input_wait_copy:   copy all input tensors from producer backends
  |                       to the current split's backend.
  |                       This includes RPC GET_TENSOR downloads,
  |                       MoE expert weight copies, and prefetch
  |                       optimizations (B+13b, B+15).
  |
  +-- graph_compute_async: launch the split's subgraph on the backend.
  |                          For local backends: CUDA/ROCm kernel launch.
  |                          For RPC backends: GRAPH_COMPUTE wire command.
  |
  +-- event_record:        record a backend event for this split's
  |                          completion (used by pipeline_barrier later).
  |
  +-- advance to next split
```

Each split **waits for its inputs** before launching compute. There is no concurrent execution of splits within a single token. Split N+1 does not start until split N's compute is *launched* (not necessarily finished — that's where events come in).

### 2.1 Split topology (typical 2-GPU romulus-local)

For a 35B MoE model with `-ngl 99 -ts 50,50`:

| Split | Backend | Content |
|-------|---------|---------|
| 0 | CPU | Token embedding |
| 1 | RPC (CUDA0, 3060 Ti) | ~50% of transformer layers |
| 2 | ROCm (7900 XTX) | Remaining layers + output head |

The RPC split is typically the **dominant straggler** — it holds most of the layer compute. The ROCm gather split waits for RPC outputs (activations) before it can run its layers.

### 2.2 Why pipeline_barrier is not full sync

After all splits are dispatched, `pipeline_barrier()` rotates copy slots and waits **only on backends whose copy slot is being reused for the next token**:

```text
pipeline_barrier():
  new_copy = (cur_copy + 1) % n_copies
  wait_mask = backends where slot `new_copy` is still in flight

  for each backend b where wait_mask has bit b:
    wait on events[b][new_copy]

  cur_copy = new_copy
```

This means: if backend A finished its split early, and backend B is still computing its tail, the next token's RPC head can start on backend A's *next* copy slot while B is still busy. This is the **cross-token tail overlap** that Path-B+ achieves.

---

## 3. What Path-B+ actually pipelines

### 3.1 Cross-token tail overlap (delivered)

```text
Token N:   [RPC split done]  [local ROCm tail still running]
Token N+1:                    [RPC head starts on next copy slot]
```

Path B events (v4.2.2) introduced per-backend events so the scheduler could track which backends are still busy. Path-B+ (Plus) added copy-slot rotation (P0), narrowed post-decode sync to sampling backends (P1), and scoped RPC drains (P2). Together these allow the RPC head for token N+1 to launch while token N's local tail is still finishing.

The result: `overlap_pct` of 0.1–1.3% (best case), with `global_multi` (any two backends concurrent) reaching 14–23% on some topologies. This is the pipeline that the operator sees in trace output.

### 3.2 What it does NOT pipeline (structural ceiling)

**RPC-to-RPC overlap is impossible** within the current architecture:

```text
RPC0(T+1) while RPC1(T)   ← NEVER HAPPENS
```

Why: the split loop is serial within a token. RPC1 for token T hasn't even been dispatched when RPC0 finishes. And even if it had, token T+1 doesn't exist until T is fully decoded and sampled.

This is the structural ceiling documented in [DESIGN-path-d-layer-pipeline.md](DESIGN-path-d-layer-pipeline.md) section 2. The `global_3bk_pct` metric (3+ backends concurrent) is <1% on all measured topologies.

### 3.3 B+ mitigation ladder (shipped, default ON with Plus=1)

| Flag | What it does |
|------|-------------|
| `GGML_PIPELINE_BARRIER_PARTIAL` (B+8) | Wait only on backends whose slot is being reused, not all backends |
| `GGML_RPC_EVENT_DEFER_BARRIER` (B+9) | Defer EVENT recv to barrier instead of draining on every RPC send |
| `GGML_SCHED_MOE_ASYNC_COPY` (B+10) | Use event_wait instead of full synchronize for MoE weight copies |
| `GGML_RPC_DUAL_SOCKET` (B+11) | Separate cmd/rsp TCP sockets (proto 4.4); default OFF (bisect NULL) |
| `GGML_RPC_GET_TENSOR_DEFER` (B+12) | Batch RPC GET downloads to graph boundary |
| B+13 | Async `cpy_tensor_async` with sync_copy_fallback when async fails |
| B+14 W1 | Wavefront intra + RPC gather via producer event_wait instead of full slot wait |
| B+15 | Prefetch next-split RPC GETs at RPC split start |
| B+16 | Flush current-split RPC downloads at gather-split entry |

All defaults with `GGML_PIPELINE_PLUS=1`. The ladder is exhausted — M3 overlap (>=5%) cannot be reached on Path-B+.

---

## 4. RPC server: passive compute endpoint

Each `rpc-server` process (typically in Docker) is a **passive** compute endpoint:

```text
Client (llama-server)                    rpc-server
  |                                         |
  |-- GRAPH_COMPUTE (cmd 10) -------------->|  deserialize graph
  |                                         |  ggml_backend_graph_compute()
  |<-- response ---------------------------|  (result data)
  |
  |-- GRAPH_RECOMPUTE (cmd 16) ----------->|  recompute previous graph
  |                                         |  (Path B events: respects ordering)
  |
  |-- SET_TENSOR / GET_TENSOR ------------>|  weight upload / activation download
  |
  |-- EVENT_RECORD (cmd 18) -------------->|  pipeline event for barrier
```

The rpc-server:
- Has **no knowledge** of the overall token pipeline
- Does not coordinate with other rpc-servers
- Receives per-split subgraphs, computes them, returns results
- The client drives all orchestration: split ordering, event synchronization, copy scheduling

### 4.1 Server-side scheduling (Path C, deferred)

Path C proposes moving the scheduler inside the rpc-server for co-located GPUs (e.g., triton's 3090+3070). Instead of the client dispatching individual `GRAPH_COMPUTE` per GPU, it would send `GRAPH_COMPUTE_ALL` and let the server run its own `ggml_backend_sched`. This would eliminate client-side TCP RTTs between co-located GPU splits.

Status: **Phase D1 stepping stone** (triton only), not in production. See [rpc-path-c-plan.md](../../rpc-patch/docs/rpc-path-c-plan.md).

---

## 5. Server telemetry: per-device, not per-layer

The rpc-server telemetry (`GGML_RPC_SERVER_TELEMETRY=1`) captures:

| Field | Granularity | Meaning |
|-------|------------|---------|
| `device_timings_us` | Per device per graph call | Total compute time for one `GRAPH_COMPUTE` invocation |
| `layer_assignments` | Per device | Device index (0 = first GPU), not layer-to-device mapping |
| `copy_times_us` | Per device | Copy operation time |
| `device_meta` | Per device | Name, VRAM, backend type, PCIe info |
| `kv_read_times_us` | Per KV slot | KV cache read time |
| `kv_write_times_us` | Per KV slot | KV cache write time |

**There is no per-layer timing.** The arrays are sized by device count, not layer count. A 24-layer model on one device produces `layer_assignments: [0]` (one entry — device 0), not 24 entries.

The profiler's heatmap `layers: []` is empty because `layer_assignments.size()` equals device count (1-2 for romulus-local), which is far fewer than the actual layer count. For a 5-GPU topology with one split per GPU, you'd get 5 entries — still per-device, not per-layer.

**For standalone llama-server (no RPC):** `ggml_backend_sched_get_backend_timing_us()` gives per-backend compute time — same granularity as RPC telemetry but without KV timing or device metadata. The profiler's `sched-trace.jsonl` captures this for all modes.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for trace schema details.

---

## 6. Split-to-backend mapping (how layers land on GPUs)

The scheduler's `ggml_backend_sched_split_graph()` assigns each graph node (tensor operation) to a backend based on:
- Where the node's inputs live (data locality)
- `-ts` / `--tensor-split` ratio for weight distribution
- `-ngl` / `--n-gpu-layers` for layer offloading
- Backend capabilities (which ops each backend supports)

For a MoE model on romulus-local (2-GPU):

```text
Graph nodes (simplified):
  inp_embd (CPU)
  layer_0  (RPC/CUDA)
  layer_1  (RPC/CUDA)
  ...
  layer_N  (local/ROCm)
  output   (local/ROCm)
```

The scheduler groups consecutive nodes running on the same backend into one split. Nodes that cross backend boundaries create a new split (because inputs need to be copied across backends). This is why the split count is typically 2-3 for romulus-local, not 1 split per layer.

---

## 7. Throughput model

For a single token decode on a 2-GPU topology:

```text
T_token = max(
    input_wait_copy(RPC_split) + graph_compute(RPC_split),
    input_wait_copy(local_split) + graph_compute(local_split)
) + overlap_loss
```

The `overlap_loss` term captures the serialization cost when one split waits for the other. The `global_multi` metric (14-23%) measures what fraction of wall time has both backends active. The remaining 77-86% is serial dispatch overhead.

Adding a third GPU (e.g., RX6600) increases serialization cost because each additional split adds input_wait_copy time. For 35-36B A3B MoE, the third hop adds ~16 ms/token — a net throughput loss despite adding compute capacity.

---

## 8. Where the docs diverge from reality

| Doc | Describes | Status |
|-----|-----------|--------|
| [DESIGN-path-d-layer-pipeline.md](DESIGN-path-d-layer-pipeline.md) | GPipe assembly line: T+1 enters stage 0 while T at stage 1 | **Aspirational** — D0 design, no code |
| [MISSION.md](MISSION.md) | Overlap gate targets, Path C bridge, success criteria | **Aspirational** — M3 overlap closed on this branch |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | B+8–B+13 mitigation flags, hotpath instrumentation | **Factual** — shipped code |
| [RPC-PROTOCOL.md](RPC-PROTOCOL.md) | Wire format, command table, version history | **Factual** — protocol as implemented |
| [rpc-path-b-plus-overview.md](../../rpc-patch/docs/rpc-path-b-plus-overview.md) | Mixed: shipped features + deferred items | **Mostly factual** with "What's next" aspirational section |
| **This file** | End-to-end flow, split dispatch, rpc-server role | **Factual** — current state |

---

## 9. Key takeaway

The current pipeline is a **serial split dispatcher with cross-token tail overlap via copy-slot rotation**. It is not a GPipe assembly line. The rpc-server is a passive compute endpoint — all orchestration lives in the client-side scheduler. Per-layer hot-path analysis is not available from either RPC telemetry or standalone backend timing. The structural ceiling (no RPC-to-RPC overlap, no per-layer timing) requires Path D (GPipe client scheduler) or Path C (server-side scheduling) to break through.
