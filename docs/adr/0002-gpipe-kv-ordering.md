# ADR-0002: GPipe KV Ordering Model

## Status

**Accepted** — 2026-07-10. Informs GPipe client scheduler design (D2.x).

## Context

Path D seeks to achieve assembly-line saturation: concurrent RPC layer compute across backends for different tokens. This requires relaxing the serial dispatch loop in `ggml_backend_sched_compute_splits()` so that token T+1 can enter the pipeline before token T has fully completed.

The critical ordering constraint is **KV cache write ordering**: token T+1 cannot read KV entries for attention until token T's KV cells are fully written. This ADR establishes the KV ordering model for the GPipe scheduler.

### Inputs

This decision is shaped by four grilling analyses (Q1-Q4):

| Analysis | Source | Key Finding |
|----------|--------|-------------|
| **Q1 -- KV Write Overlap** | `ggml-backend.cpp` `compute_splits()`, `llama-kv-cache.cpp` | KV writes happen in gather split graph compute; no inter-token overlap today |
| **Q2 -- MoE Expert Routing** | `ggml-backend.cpp` MUL_MAT_ID handling | Expert IDs determined per-token by routing FFN; read via blocking `tensor_get` after compute |
| **Q3 -- MTP Coupling** | `common/speculative.cpp`, `llama-context.cpp` | MTP draft uses separate model context; can overlap pipeline stages |
| **Q4 -- GPipe Ordering Model** | `ggml-backend.cpp` event model, `ggml-rpc.cpp` | `event_record`/`event_wait` provide strong ordering; server waits compute idle |

### Prior constraints

From D0.2 split topology analysis:
- Serial split dispatch accounts for ~91% of wall time (`overlap_pct` 0.1-0.2%)
- Gather split (ROCm0) writes KV cache and extracts `t_h_nextn` hidden states
- RPC RTT p50=0.35ms is acceptable; p99=71.55ms tail spikes need mitigation
- Sample gate blocks cross-token overlap for single-sequence decode without MTP

## Decision

### Model: Producer-Consumer Event Signaling with KV-Ready Release

The GPipe ordering model uses **per-split event records** to signal "stage complete" rather than relying on the serial dispatch loop. The key release point is **KV-ready**, not gather-complete.

#### 1. KV Write Location

KV cache writes occur within the gather split (ROCm0, final split). The exact sequence within that split is:

```
gather_split compute (includes KV store ops as graph nodes)
  -> ggml_backend_event_record (signals "split done")
  -> t_h_nextn extraction (hidden state read after compute)
  -> Copy-slot advance for next token
```

The KV cache store operations (graph nodes that write to K/V tensors) are embedded in the graph compute of the final gather split. They complete asynchronously on the gather backend within `graph_compute_async`.

#### 2. Release Ordering

```
Token T:
  Split 0 (embed) -> Split 1 (RPC0 layers) -> Split 2 (RPC1 layers)
    -> Split 3 (RPC2 layers) -> Split 4 (RPC3 layers) -> Split 5 (gather + KV write)

Token T+1:
  Split 0 (embed) -> ... [can start when KV write for T completes]
```

**Release condition for token T+1:** KV cache write for token T must be complete AND the event_record from the gather split must have been consumed.

**Not required for T+1 start:**
- Logits computation for T (can overlap with T+1 compute)
- Sampling for T (can overlap with T+1 compute)
- `t_h_nextn` extraction for T (can overlap with T+1 compute)

#### 3. Event Signaling Protocol

The existing `ggml_backend_event_record`/`ggml_backend_event_wait` API provides the right primitives:

- **Producer (gather split):** After KV graph compute, `event_record` on gather backend signals "KV stage complete for token T"
- **Consumer (embed split for T+1):** `event_wait` on the event before dispatching token T+1's embed split

The current RPC event model provides strong ordering guarantees:
- `rpc_backend_event_record` sends `RPC_CMD_EVENT_RECORD` to the server (fire-and-forget with pending response)
- On the server side, `RPC_CMD_EVENT_RECORD` calls `server.wait_compute_idle()` **before** sending the response
- `rpc_finish_event_response` (via `event_wait`) reads the response, confirming compute is idle
- This provides end-to-end ordering: the response confirms KV writes are committed before the consumer proceeds

#### 4. Split Boundaries in the GPipe Pipeline

```
Stage 0 (compute pipeline): embed(T) -> RPC0(T) -> RPC1(T) -> RPC2(T) -> RPC3(T)
Stage 1 (gather+release):   gather(T) -> event_record(T) -> [event_wait consumed]
Stage 2 (next token):  embed(T+1) -> RPC0(T+1) -> ... [overlaps with Stage 1]
```

The natural 2-stage pipeline:

```
Token T:   [embed + RPC compute layers  ...................] [gather + KV write]
Token T+1:                                                   [embed + RPC compute layers ...................]
```

Overlap achieved: RPC compute for T+1 executes while gather + KV write for T executes.

#### 5. MoE Expert Routing Independence

MoE expert routing is per-token and self-contained:
- Expert IDs are determined by the routing FFN within each token's graph compute
- The `ids` tensor is read via blocking `ggml_backend_tensor_get()` inside the scheduler's copy loop
- There is no inter-token dependency on expert routing
- Expert weight loading for token T+1 can proceed independently from token T's KV write

No additional ordering constraints arise from MoE routing.

#### 6. MTP Coupling

MTP draft contexts (`ctx_dft`) operate independently from the target context (`ctx_tgt`):
- Each has its own scheduler instance (`ggml_backend_sched`)
- Draft tokens are produced by `common_speculative_draft()` on ctx_dft
- Target verification runs on ctx_tgt
- The draft/target relationship does not create additional KV ordering constraints
- `h_prev` (hidden state from target) flows through `embd_nextn` and is readable after the target's gather split completes

For GPipe mode A, MTP provides known-ahead draft tokens that break the autoregressive sample gate dependency.

## Consequences

### Positive

1. **KV-ordering constraint is loose enough for GPipe.** The gather split's event_record signals KV-ready before logits are fully processed, meaning T+1 can start as soon as KV writes commit.

2. **Existing event API is sufficient.** `event_record`/`event_wait` already provide the producer-consumer semantics needed. No new synchronization primitives needed.

3. **MoE routing is independent per token.** No inter-token ordering constraints from expert routing.

4. **MTP draft contexts enable decoupling.** Separate model contexts for draft/verify mean MTP can overlap naturally with GPipe stages.

5. **Gather split is the only KV write point.** In the 5-GPU topology, only ROCm0 (gather split) writes KV cache. The RPC compute splits (RPC0-RPC3) produce activations but do not write KV directly.

### Negative

1. **Event signaling adds latency to the hot path.** Each release requires an RPC round trip for event_record + event_wait (p50=0.35ms, tail spikes to 71.55ms). Mitigation: batch event consumption or use local event signaling for the gather split when it's on the same host (ROCm native).

2. **The gather split is a serialization point.** Even with GPipe, the gather split for T and embed split for T+1 cannot overlap on the same GPU (both run on ROCm0). The pipeline's cycle time is bounded by max(gather(T), embed(T+1) + RPC0(T+1)).

3. **KV write confirmation cost is multiplicative.** In a depth-d pipeline, each token's release incurs event signaling cost. At n_copies>2, the event management overhead grows linearly with copy count.

4. **Straggler sensitivity increases.** If the gather split stalls (e.g., CPU memory bandwidth contention on romulus), it blocks the entire pipeline release chain. The straggler becomes the critical path with amplified impact.

### Neutral

1. **pipeline_barrier remains for the default Path-B+ path.** The `GGML_SCHED_GPIPE=0` path is unchanged; `pipeline_barrier` and copy-slot management continue to work identically.

2. **event_record already includes compute-idle confirmation.** The RPC server's `wait_compute_idle()` ensures no outstanding async compute when the event response is sent. This is a stronger guarantee than needed but doesn't hurt performance significantly.

3. **`t_h_nextn` extraction** (MTP hidden state) happens after event_record and can overlap with T+1's compute. This is a minor optimization opportunity, not a constraint.

## References

### D0.2 Split Topology Analysis

- `docs/wayfinder/D0.2-split-topology-map.md` -- Split boundaries, blocking dependencies, RPC hop timing
- Section 6.3: KV Write Ordering as a blocking dependency
- Section 6.5: Blocking dependency summary table
- Section 8.2: Suggested GPipe stage splitting

### Code Locations Reviewed

| File | Lines | Content |
|------|-------|---------|
| `ggml/src/ggml-backend.cpp` | 2293-2759 | `ggml_backend_sched_compute_splits()` -- serial dispatch loop, event_record, prefetch |
| `ggml/src/ggml-backend.cpp` | 2958-3059 | `ggml_backend_sched_pipeline_barrier()` -- copy-slot synchronization |
| `ggml/src/ggml-backend.cpp` | 1854-1955 | `ggml_backend_sched_wait_copy_slot()` -- stream wait, barrier src mask |
| `ggml/src/ggml-backend.cpp` | 2460-2590 | MoE expert copy -- `MUL_MAT_ID` expert routing, `tensor_get` of ids |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 640-710 | RPC event record protocol -- `rpc_msg_event_record_req`, `rpc_finish_event_response` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 2016-2044 | `rpc_backend_event_record()` and `rpc_backend_event_wait()` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 3365-3380 | Server-side `RPC_CMD_EVENT_RECORD` -- calls `wait_compute_idle()` before response |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 2936-2940 | `rpc_server::wait_compute_idle()` -- waits compute queue empty + inflight == 0 |
| `src/llama-kv-cache.cpp` | 633-680 | `llama_kv_cache::seq_rm()` -- KV cell removal |
| `src/llama-kv-cache-iswa.cpp` | 1-120 | Sliding window KV cache, init_batch, cross-attention setup |
| `src/llama-context.cpp` | 1380-1420 | `process_ubatch()` -- `pipeline_barrier` call site at graph reuse |
| `src/llama-context.cpp` | 1588-1595 | `t_h_nextn` extraction after graph compute (MTP hidden state) |
| `src/llama-context.cpp` | 2514-2520 | `graph_compute()` -- calls `ggml_backend_sched_graph_compute_async` |
| `common/speculative.cpp` | 2050-2095 | `common_speculative_accept()` -- MTP accept flow; draft/target separation |

### Prior Decisions

- `docs/adr/0001-b6-ladder-execution-post-b9-null.md` -- Prior ADR on B+6 ladder; established Path-B+ structural ceiling
- `docs/rpc-multi-backend-pipeline-plus/DESIGN-path-d-layer-pipeline.md` -- Design hypotheses, GPipe modes A/B/C
- `docs/rpc-multi-backend-pipeline-plus/CONTEXT.md` -- Domain language

### Alternatives Considered

| Alternative | Reason Rejected |
|-------------|-----------------|
| **Full pipeline_barrier-based release** | Retains serial dispatch; no GPipe gains |
| **KV-dual-buffered per copy-slot** | Too invasive; KV cache layout change across all model architectures |
| **Gather split on separate GPU** | No free GPU in 5-GPU topology; would require re-split |
| **Deferred KV write with async flush** | KV cache integrity requires ordered writes; async flush adds RTT-dependent lag |

## Domain Language Additions

The following terms are added to the GPipe KV ordering domain:

| Term | Definition |
|------|------------|
| **KV-ready release** | The point at which a gather split's KV writes are committed and the next token's embed split can safely dispatch |
| **Event release chain** | The sequence of `event_record` -> `event_wait` that propagates the "KV ready" signal from gather split to embed split |
| **Compute-confirmed event** | An RPC event_record whose server-side handler calls `wait_compute_idle()` before responding, guaranteeing completion |
| **Stage decoupling** | The separation of gather (KV write) from embed (next token start) via event signaling rather than serial dispatch |
