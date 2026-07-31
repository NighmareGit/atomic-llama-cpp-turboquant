# Path D Specification — GPipe Client Scheduler

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-13  
**Status:** Specification Phase — Mode A complete, Mode B multi-seq complete (section 10.3), Path C (section 12) defined  
**Inputs:** D0.2, D0.3 (ADR-0002), D0.4, D0.5

---

## 1. Problem Statement

Current Path-B+ pipeline achieves cross-token overlap (RPC(T+1) vs local tail(T)) but cannot overlap **within-token** RPC layer compute across backends. This manifests as:

- `global_3bk_pct` < 1% (target: >= 25%)
- `overlap_pct` 0.1-0.2% (target: >= 5% for M3)
- Serial split dispatch: each split must complete before the next starts

**Goal:** Implement GPipe-style layer pipeline where RPC0(T+1) can compute concurrently with RPC1(T).

---

## 2. Architecture Overview

### 2.1 Pipeline Stages

```
Stage 0 (compute): embed(T) + RPC0(T) + RPC1(T) + RPC2(T) + RPC3(T)
Stage 1 (gather):  gather(T) + KV write + sample(T)
```

### 2.2 Stage Overlap Model

```
Token T:   [Stage 0 compute ....................] [Stage 1 gather+KV+sample]
Token T+1:                                                [Stage 0 compute ...........]
```

### 2.3 Producer-Consumer Signaling

Based on ADR-0002:
- **Producer:** gather split records `kv_ready[T]` event after KV write
- **Consumer:** embed split for T+1 waits on `kv_ready[T]` before dispatch
- Uses existing `ggml_backend_event_record`/`ggml_backend_event_wait` API

---

## 3. API Contracts

### 3.1 New Types

```cpp
// llama-context.h
struct llama_gpipe_state {
    int  n_stages;            // Number of pipeline stages (2 initial)
    int  cur_stage;           // Current stage index
    int  microbatch_size;     // Tokens in flight
    bool enabled;             // GPipe mode active
    std::vector<llama_seq_id> stage_tokens;
    int64_t stage_start_us[GGML_SCHED_MAX_SPLITS];
};
```

### 3.2 New Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `llama_decode_gpipe` | `int32_t(llama_context*, llama_batch, const float*)` | GPipe mode decode entry point |
| `ggml_sched_gpipe_init` | `void(ggml_backend_sched_t, int)` | Initialize GPipe event state |
| `ggml_sched_gpipe_wait` | `void(ggml_backend_sched_t, int)` | Wait on stage completion event |
| `ggml_sched_gpipe_advance` | `int(llama_context*)` | Advance pipeline stage |

### 3.3 New Environment Variable

| Variable | Default | Purpose |
|----------|---------|---------|
| `GGML_SCHED_GPIPE` | 0 (off) | Enable GPipe client scheduler |

---

## 4. Implementation Requirements

### 4.1 llama_context Changes

| Location | Change |
|----------|--------|
| `src/llama-context.h` | Add `llama_gpipe_state gpipe;` member |
| `src/llama-context.cpp` | Add `llama_gpipe_enabled()` helper |
| `src/llama-context.cpp` | Modify `decode()` to call `llama_decode_gpipe()` when enabled |
| `src/llama-context.cpp` | Implement `llama_decode_gpipe_impl()` with stage state machine |

### 4.2 ggml_backend_sched Changes

| Location | Change |
|----------|--------|
| `ggml/src/ggml-backend.cpp` | Add `ggml_sched_gpipe_init()` |
| `ggml/src/ggml-backend.cpp` | Add `ggml_sched_gpipe_wait()` |
| `ggml/src/ggml-backend.cpp` | Add `ggml_sched_gpipe_event_record()` |

### 4.3 RPC Event Model

No changes to RPC protocol. Use existing `RPC_CMD_EVENT_RECORD` (proto 4.2.2+) with `wait_compute_idle()` confirmation.

---

## 5. Test Requirements

### 5.1 Correctness Tests

| Test | Description |
|------|-------------|
| Logits hash smoke | Token T+1 logits match between GPipe ON and OFF |
| Multi-turn KV fill | 8149 tokens multi-turn without corruption |
| MoE correctness | Expert routing produces same outputs |
| MTP coupling | Draft tokens + target verify work together |

### 5.2 Performance Tests

| Metric | Target | Method |
|--------|--------|--------|
| `global_3bk_pct` | >= 25% | Profiler trace analysis |
| G (t/s) | Within 5% of Path-B+ baseline | `b6-2gpu-f-romulus-local` comparison |
| Overlap_pct | >= 5% | `assembly_overlap_count` analysis |

### 5.3 Regression Tests

| Test | Description |
|------|-------------|
| Path-B+ compatibility | Existing benchmarks unchanged when GPipe OFF |
| Single GPU | No performance regression on single-backend configs |
| Dense models | 70B/72B runs without OOM |

---

## 6. Performance Targets

| Target | Value | Measurement |
|--------|-------|-------------|
| `global_3bk_pct` | >= 25% | Profiler telemetry |
| `overlap_pct` | >= 5% | Assembly overlap analysis |
| G (A1 2-GPU) | >= 180 t/s | Throughput comparison |
| G (A8 5-GPU) | >= 15 t/s | Dense 70B baseline |

---

## 7. Integration Points

| Component | Change Summary |
|-----------|----------------|
| `llama_context` | New `gpipe` member, decode dispatch change |
| `ggml_backend_sched` | New GPipe event functions |
| `llama.h` | New `llama_decode_gpipe()` public API |
| RPC protocol | No changes (use existing EVENT_RECORD) |

---

## 8. Compatibility

| Aspect | Policy |
|--------|--------|
| Path-B+ default | Unchanged (`GGML_PIPELINE_PLUS=1`, `GGML_SCHED_GPIPE=0`) |
| Upstream merge | GPipe OFF by default; flag-gated changes |
| Model loading | No changes |
| KV cache logic | No changes |

---

## 10. Beyond Mode A — Staged Extension

The current Mode A (2-stage GPipe: compute + gather) is the foundation. This section defines the staged extension beyond Mode A, to be executed only after D2 testing validates the 2-stage pipeline.

### 10.1 Phase D4 — Path C Stepping Stone

**Goal:** Prove server-side scheduling on triton's co-localized dual-GPU before deeper client-side pipelining. See [section 12](#12-path-c--server-side-scheduling) for the full spec.

| Step | What | Output | Status |
|------|------|--------|--------|
| Research | C1 baseline: per-device RPC splits, RTT count, server GPU util | `D4.1-triton-baseline-analysis.md` | Prep ready (cluster access needed) |
| Design | ADR-004: Server-side scheduling model | `docs/adr/0004-server-side-scheduling.md` | **ACCEPTED** (Option B+) |
| Spec | Path C spec section | `docs/path-d-spec.md` (section 12) | ✅ Complete |
| Prototype | Throwaway: test `GRAPH_COMPUTE_ALL` concept | Findings for implement | Pending |
| Implement | C2: server-side sched with `GRAPH_COMPUTE_ALL` | Code in `ggml-rpc.cpp` | Pending |
| Test | Verify server GPU duty improvement | Metrics in TRACKING.md | Pending |

### 10.2 Phase D5 — Deeper Pipelining

**Goal:** Extend GPipe from 2-stage to n_stages > 2 with finer sub-stages within RPC compute.

| Step | What | Output |
|------|------|--------|
| Research | Per-backend split timing analysis | `D5.1-split-timing-analysis.md` |
| Design | ADR-003: Adaptive pipeline depth | `docs/adr/0003-adaptive-pipeline-depth.md` |
| Spec | Deeper pipelining spec section | `docs/path-d-spec.md` (this file) |
| Prototype | Throwaway: per-backend sub-stage dispatch | Findings for implement |
| Implement | Per-backend sub-stages (embed, RPC0, RPC1, RPC2, RPC3) | Code in `llama-context.cpp`, `ggml-backend.cpp` |
| Implement | Dynamic stage assignment (adaptive depth) | Code in `ggml-backend.cpp` |
| Test | Verify `global_3bk_pct >= 25%` | Metrics in TRACKING.md |

### 10.3 Phase D6 -- Mode B Microbatch / Multi-Seq

**Goal:** Support multiple sequences at different pipeline positions (server multi-slot).

**Phases:**

| Step | What | Output |
|------|------|--------|
| Research | Multi-slot requirements, KV cache interaction | `D6.1-multi-seq-requirements.md` |
| Design | ADR-005: Multi-seq GPipe scheduling | `docs/adr/0005-multi-seq-gpipe.md` |
| Spec | Multi-seq spec section | `docs/path-d-spec.md` (this file) |
| Prototype | Throwaway: multi-seq token tracking | Findings for implement |
| Implement | Extend state machine for multi-seq | Code in `llama-context.cpp` |
| Implement | Server multi-slot dispatch | Code in `ggml-rpc.cpp` |
| Test | Verify concurrent multi-seq decode | Metrics in TRACKING.md |

#### 10.3.1 Multi-Seq Pipeline Model

Mode B extends the Mode A single-sequence pipeline to support multiple concurrent sequences occupying different stages:

```
Sequence A: Stage 0 (embed)     -> Stage 1 (RPC0)         -> Stage 2 (ROCm0+gather+KV)
Sequence B:          Stage 0 (embed)     -> Stage 1 (RPC0)         -> Stage 2 (ROCm0+gather+KV)
```

Each sequence advances through stages independently. The dispatch loop fills available stages from any active sequence whose next position matches the stage.

**Scheduling model (ADR-005):** Stage-available. Any sequence can claim a free stage. No fixed ordering between sequences.

**Event signaling model (ADR-005):** Double-buffered. Two alternating event banks prevent timestamp overwrite when sequences share event slots.

#### 10.3.2 API Contracts

**Extended `llama_gpipe_state` (in `src/llama-context.h`):**

```cpp
struct llama_gpipe_state {
    // Existing fields (Mode A)
    int  n_stages;
    int  microbatch_size;     // Target concurrent sequences (2 default)
    bool enabled;

    // Mode B: per-sequence stage tracking
    std::vector<llama_seq_id> stage_tokens;  // stage[s] -> owning seq_id (-1 = free)
    std::map<llama_seq_id, int> seq_stage;  // seq_id -> current stage position
    int active_sequences;                    // Count of sequences in pipeline

    // Adaptive depth (unchanged from D5)
    bool adaptive_enabled;
    int  adaptive_warmup_count;
    static constexpr int ADAPTIVE_WARMUP = 5;
    int64_t stage_timing_sum_us[LLAMA_GPIPE_MAX_STAGES];
    int     stage_timing_count[LLAMA_GPIPE_MAX_STAGES];
    bool    adaptive_finalized;

    int64_t stage_start_us[LLAMA_GPIPE_MAX_STAGES];
};
```

**Double-buffered events (in `ggml/src/ggml-backend.cpp`):**

```cpp
// Scheduler struct extension:
ggml_backend_event_t gpipe_events[2][GGML_SCHED_MAX_STAGES];  // bank 0/1
int gpipe_event_bank;  // current bank index, toggled per dispatch call
```

**New functions:**

```cpp
// Reserve a sequence slot in the pipeline. Returns sequence ID, or -1 if full.
int32_t llama_gpipe_seq_reserve(llama_context * ctx);

// Release a sequence from the pipeline (e.g., on generation complete).
void llama_gpipe_seq_release(llama_context * ctx, llama_seq_id seq_id);

// Multi-seq dispatch: iterate stages, fill from available sequences.
// Returns number of stages dispatched this call, or -1 on error.
int32_t llama_decode_gpipe_multi_impl(llama_context * ctx, llama_batch batch);
```

**Event API extensions:**

```cpp
// Initialize double-buffered GPipe events (n_banks × n_stages events).
void ggml_sched_gpipe_init_multi(ggml_backend_sched_t sched, int n_stages, int n_banks);

// Record stage event for the current bank.
void ggml_sched_gpipe_record_bank(ggml_backend_sched_t sched, int stage_id, int bank);

// Wait on stage event for the given bank.
void ggml_sched_gpipe_wait_bank(ggml_backend_sched_t sched, int stage_id, int bank);

// Toggle event bank (call once per multi-seq dispatch cycle).
void ggml_sched_gpipe_toggle_bank(ggml_backend_sched_t sched);
```

#### 10.3.3 Stage-Ownership Protocol

```
On dispatch:
  for each stage s in [0, n_stages):
    if stage_tokens[s] == -1 (free):
      find seq where seq_stage[seq] == s
      if found:
        dispatch seq at stage s
        stage_tokens[s] = seq
        if s == n_stages - 1 (KV write complete):
          stage_tokens[s] = -1        // release stage
          seq_stage[seq] = 0          // wrap to next token
        else:
          seq_stage[seq] = s + 1      // advance
```

Per-sequence KV-ready release follows ADR-0002: a sequence's next token cannot start Stage 0 until its previous token completes Stage n-1 (gather+KV write). Different sequences have independent release chains.

#### 10.3.4 Implementation Sequence

| Ticket | What changes | Where |
|--------|-------------|-------|
| D6.4 | Throwaway prototype: test multi-seq token tracking | `tests/test-gpipe-multi-seq-prototype.cpp` (deleted after) |
| D6.5 | Double-buffered events + per-sequence stage tracking | `ggml/src/ggml-backend.cpp`, `src/llama-context.h`, `src/llama-context.cpp` |
| D6.6 | Server multi-slot dispatch: accept per-sequence graphs | `ggml/src/ggml-rpc/ggml-rpc.cpp` |
| D6.7 | Integration test: concurrent multi-seq decode | `tests/test-gpipe-multi-seq.cpp` |

#### 10.3.5 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|-------------|
| AC-D6.1 | Two sequences decode concurrently at different pipeline stages | Test: multi-seq test with 2 sequences, verify stages occupied simultaneously |
| AC-D6.2 | No KV cache corruption across sequences | Test: logits match single-seq baseline for each sequence |
| AC-D6.3 | `global_3bk_pct` improves vs single-seq baseline | Metric: pipeline fill > 50% (2 of 3 stages occupied) |
| AC-D6.4 | Event signaling correct: no deadlocks or missed signals | Test: stress test with 1000+ tokens across 2 sequences |
| AC-D6.5 | No regression: single-seq Mode A still works | Test: existing GPipe tests pass (16 assertions as of D5.7) |
| AC-D6.6 | Server handles per-sequence dispatch | Test: server accepts graphs from different sequences concurrently |

#### 10.3.6 Known Limitations

| ID | Limitation | Workaround | Fix Ticket |
|----|-----------|------------|------------|
| KL-D6.1 | GPU backends fully drained between multi-seq stages (`ggml_backend_sched_synchronize`). Event-based pipelining non-functional because CPU gather backend does not support events (`event_new`/`event_record`/`event_wait` all NULL), making all gpipe_events NULL. | Use single-seq Mode A (no per-stage dispatch). Multi-seq correctness is maintained, just without GPU pipelining between stages. | D6.10 |

### 10.4 Phase R3 — Advanced Optimization

**Goal:** Adaptive depth refinement, deprecation of superseded flags. RDMA deferred.

| Step | What | Output |
|------|------|--------|
| Research | Adaptive depth analysis from D5 | `R3.1-adaptive-depth-analysis.md` |
| Design | ADR-006: Finalize adaptive depth | `docs/adr/0006-finalize-adaptive-depth.md` |
| Implement | Deprecation warnings for B+11/B+14/B+7f | Code in `ggml-backend.cpp` |
| Implement | Adaptive depth refinement | Code in `ggml-backend.cpp` |
| Test | Verify no regression | Metrics in TRACKING.md |

### 10.5 Extension Acceptance Criteria

| Gate | Metric | Target |
|------|--------|--------|
| G1 | Cluster fill | `global_3bk_pct >= 25%` @ n=384 5-GPU |
| G2 | Serial dispatch | `serial_dispatch_pct <= 50%` (stretch) |
| G3 | Throughput scaling | Measurable G gain when adding RPC stage |
| G4 | Correctness | Logits hash / generation completes on all gate archetypes |
| G5 | Multi-seq | Concurrent multi-seq decode without KV corruption |

---

## 11. References

- `docs/wayfinder/D0.2-split-topology-map.md`
- `docs/adr/0002-gpipe-kv-ordering.md`
- `docs/wayfinder/D0.5-implementation-seam.md`
- `docs/pipeline-plus/DESIGN-path-d-layer-pipeline.md`

---

## 12. Path C — Server-Side Scheduling

Path C extends Path D by moving multi-GPU RPC dispatch from the client to the server. For co-localized GPUs (triton 3090+3070), the server runs an internal `ggml_backend_sched` to distribute a single graph across its GPUs, eliminating per-device client RTTs.

ADR-0004: **Accepted (Option B+)** — Server-side `GRAPH_COMPUTE_ALL` with client-driven weighted weight placement. Full decision at `docs/adr/0004-server-side-scheduling.md`.

---

### 12.1 Problem Statement

Mode A (section 10) overlaps compute **across tokens** but cannot overlap **within-token** RPC dispatch. The client serializes splits: each GPU must finish before the next starts. This manifests as:

- **7-22 RTTs per token** on multi-GPU RPC (one SET_TENSOR, one GRAPH_COMPUTE, one GET_TENSOR per device)
- **Sequential server GPU duty cycle**: GPU 1 idle while GPU 0 computes
- **Client-mediated tensor copies**: cross-GPU transfers route through client (`RPC_CMD_COPY_TENSOR`)

Path C targets co-localized GPUs first (triton) where PCIe latency is negligible. Cross-host multi-GPU (remus) deferred to C2+.

**Why triton first:**
- 3090 + 3070 on same node, shared PCIe root complex
- Lower latency validates architecture before tackling cross-host
- C1 baseline already shows 4 splits on Config F; triton is simpler (2 GPUs)

---

### 12.2 Architecture

#### Current Flow (per-device RPC)

```
Client                                         RPC Server (triton)
  |-- SET_TENSOR (embeddings) ---------------->|
  |-- GRAPH_COMPUTE (device 0) --------------->|  layers 0-19 on 3090
  |<-- (fire-and-forget) ----------------------|
  |-- COPY_TENSOR ---------------------------->|  3090 -> 3070 (server-local PCIe)
  |<-- (copy done) ----------------------------|
  |-- GRAPH_COMPUTE (device 1) --------------->|  layers 20-39 on 3070
  |<-- (fire-and-forget) ----------------------|
  |-- GET_TENSOR (output) -------------------->|
  |<-- (output data) --------------------------|
  |-- [local compute on client]               |
  TOTAL: 5 RTTs + 2 compute steps
```

#### Target Flow (GRAPH_COMPUTE_ALL)

```
Client                                         RPC Server (triton)
  |-- SET_TENSOR_BATCH ----------------------->|
  |-- GRAPH_COMPUTE_ALL (full graph) -------->|  server schedules internally
  |<-- (result + output_device) --------------|
  |                                           |  Server's internal scheduler:
  |                                           |    Split 0: layers 0-19 on 3090
  |                                           |    Split 1: layers 20-39 on 3070
  |                                           |    PCIe copy (server-local)
  |-- GET_TENSOR (output from device N) ----->|
  |<-- (output data) --------------------------|
  |-- [local compute on client]               |
  TOTAL: 3 RTTs + 1 compute step
```

#### Key Difference

| Aspect | Per-device (current) | GRAPH_COMPUTE_ALL (Path C) |
|--------|---------------------|---------------------------|
| Scheduler location | Client `ggml_backend_sched` | Server `ggml_backend_sched` |
| Split boundary | Client-defined, per-device | Server-defined, full graph |
| Cross-GPU copy | Client-mediated (`COPY_TENSOR`) | Server-local PCIe P2P |
| RTTs per token | 2N+1 (N = server GPUs) | 3 (fixed) |
| Server GPU duty cycle | Serial | Parallel (overlapped) |

---

### 12.3 API Contracts

#### New RPC Commands

| Command | Value | Direction | Purpose |
|---------|-------|-----------|---------|
| `RPC_CMD_GRAPH_COMPUTE_ALL` | 18 | Client -> Server | Submit full graph for server-side multi-GPU compute |
| `RPC_CMD_GRAPH_RECOMPUTE_ALL` | 19 | Client -> Server | Re-execute cached graph (same graph_hash) |

#### Request/Response Payloads

```cpp
// GRAPH_COMPUTE_ALL request
struct rpc_msg_graph_compute_all_req {
    uint32_t n_devices;                          // number of server GPUs
    uint32_t devices[GGML_RPC_MAX_DEVICES];      // device indices (max 8)
    uint32_t sync_mode;                          // 0=fire-and-forget, 1=blocking with response
    uint32_t output_requested;                   // 0=no output, 1=include output in response
    // Followed by serialized graph (same format as GRAPH_COMPUTE, device field ignored)
};

// GRAPH_COMPUTE_ALL response
struct rpc_msg_graph_compute_all_rsp {
    uint32_t result;        // 0 = success
    uint32_t output_device; // which GPU holds the output tensor
    // If output_requested==1, followed by output tensor data
};

// GRAPH_RECOMPUTE_ALL request
struct rpc_msg_graph_recompute_all_req {
    uint32_t n_devices;
    uint32_t devices[GGML_RPC_MAX_DEVICES];
    uint64_t graph_hash;    // identifies cached graph
    uint32_t sync_mode;     // 0=fire-and-forget, 1=blocking with response
    uint32_t output_requested;
};

// GRAPH_RECOMPUTE_ALL response
struct rpc_msg_graph_recompute_all_rsp {
    uint32_t result;
    uint32_t output_device;
};
```

#### Serialized Graph Format

```
| n_devices (4) | device_ids (4 * n_devices) | serialized_graph_data |
```

Where `serialized_graph_data` reuses existing `GRAPH_COMPUTE` format (device + n_nodes + nodes + n_tensors + tensors). The `device` field is ignored in ALL mode.

#### Fallback Behavior

- Old client + new server: client sends per-device `GRAPH_COMPUTE` (existing behavior unchanged)
- New client + old server: client detects missing capability in HELLO response, falls back to per-device
- No capability negotiation failure mode: client retries with per-device after first failed ALL attempt

---

### 12.4 Server-Side Scheduler

#### Scheduler Creation

```cpp
ggml_backend_sched_t create_multi_device_sched(
    const uint32_t * devices, uint32_t n_devices,
    const ggml_cgraph * graph);
```

- Collects `ggml_backend_t` for each device index
- Appends CPU backend as fallback
- Calls `ggml_backend_sched_new(..., false, false)` — no pipeline parallelism server-side (handled by client)
- `graph_size` defaults to 4096 if graph is null

#### Graph Partition Rule

**CRITICAL:** The server-side `ggml_backend_sched` CANNOT auto-partition same-type GPUs.
`ggml_backend_sched_backend_from_buffer()` (ggml-backend.cpp:1160) returns the FIRST backend
matching the buffer type and op. For two CUDA backends, all nodes go to GPU 0.
The scheduler ONLY assigns to GPU 1 if weight tensors are pre-allocated there.

**Partition strategy: client-driven weighted weight placement (NOT auto-partition).**

1. **Client detects GPU speed ratio** during HELLO handshake or via offline benchmark
2. **Client computes optimal split:** layers on fast GPU / layers on slow GPU = speed_ratio
   - 3090 vs 3070: 36 layers on 3090, 22 layers on 3070 (64/36 split, 1.75x ratio)
   - Verified via performance model (docs/wayfinder/D4-performance-model.md): 25-33% throughput uplift over equal split
3. **Client assigns weights during SET_TENSOR phase:**
   - Fast GPU gets proportionally more layers' weights
   - Slow GPU gets fewer layers (balanced compute time)
4. **Server scheduler follows weight placement** — no custom partition code needed
   - `backend_from_buffer` naturally assigns nodes to the device holding their weights
   - Cross-GPU copy inserted at the boundary between fast-GPU and slow-GPU layers

```cpp
// Client-side weight assignment for GRAPH_COMPUTE_ALL:
float speed_ratio = 1.75f;  // 3090 TFLOPS / 3070 TFLOPS
int total_layers = 60;
int layers_fast = (int)(total_layers * speed_ratio / (speed_ratio + 1.0f));
int layers_slow = total_layers - layers_fast;

// SET_TENSOR: fast GPU gets layers 0-37, slow GPU gets layers 38-59
send_set_tensor(device_3090, layers[0..37], weights_3090);
send_set_tensor(device_3070, layers[38..59], weights_3070);
```

#### Cross-GPU Tensor Copy

- Server-local PCIe P2P (no client involvement)
- `ggml_backend_sched` inserts `ggml_backend_tensor_copy` between splits
- For CUDA: uses `cudaMemcpyAsync` or CUDA peer access if enabled
- No `RPC_CMD_COPY_TENSOR` round-trip

#### Response Assembly

- Server computes final output on `output_device` (last GPU that ran output layer)
- Response includes `output_device` field
- Client reads output via existing `GET_TENSOR` from specified device
- Graph cached on server for recompute (keyed by `graph_hash`)

---

### 12.5 Configuration

#### New Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GGML_RPC_MULTIDEVICE` | 0 | Enable GRAPH_COMPUTE_ALL on server (0 = per-device only) |

#### Capability Advertisement

Server advertises multi-device support during HELLO handshake:

```cpp
struct rpc_msg_hello_rsp {
    // ... existing fields ...
    uint8_t server_caps;  // NEW: bit 0 = RPC_CAP_MULTI_DEVICE
};

#define RPC_CAP_MULTI_DEVICE (1 << 0)
```

Client stores capability per-endpoint:

```cpp
struct ggml_backend_rpc_context {
    // ... existing fields ...
    uint32_t n_devices_on_endpoint;
    bool     is_multi_device_capable;
};
```

---

### 12.6 Compatibility

#### Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Old client + old server | Unchanged (per-device RPC) |
| Old client + New server | Server handles per-device commands (existing code path) |
| New client + Old server | Client detects no capability, falls back to per-device |
| New client + New server | Uses GRAPH_COMPUTE_ALL when beneficial |

#### Interaction with GPipe (Mode A)

Path C server-side scheduling is **orthogonal** to Mode A client-side pipelining:

- Mode A overlaps token T compute with token T+1 gather
- Path C reduces RTTs within token T's compute stage
- Both can be active simultaneously
- Server-side sched runs inside each Mode A pipeline stage

**Configuration:** `GGML_SCHED_GPIPE=1` (Mode A) + `GGML_RPC_MULTIDEVICE=1` (Path C) are independent flags.

---

### 12.7 Performance Targets

| Metric | C1 Baseline | Path C Target (Simple) | Path C Target (Weighted) | Measurement |
|--------|-------------|----------------------|------------------------|-------------|
| RTTs per token (2-GPU RPC) | 5 (est.) | 3 | **3** (combined compute+sync) | Wire protocol capture |
| Server GPU duty cycle | ~50% (serial) | ~90% (parallel) | **~95%** (balanced compute) | `nvidia-smi` utilization |
| G (triton 2-GPU, equal split) | ~222 t/s (est.) | >=222 t/s (non-regression) | — | `b6-2gpu-f-triton` |
| G (triton 2-GPU, weighted 64/36) | — | — | **~296 t/s (+33% est.)** | `b6-2gpu-f-triton` |
| G (triton 2-GPU, weighted + inline sync) | — | — | **~314 t/s (+41%, stretch)** | Same |

**Note:** C1 baseline G values are model-based estimates for 35B MoE on 3090+3070. Actual D4.1 cluster measurement will replace estimates. The 42.8 t/s value from trace-f-3gpu-plus was cross-host (remus Config F) and is NOT the triton baseline.

**Non-regression gate:** Path C must not decrease throughput vs measured C1 baseline (D4.1). Weighted partition target: >= +25% uplift over equal-split baseline.

#### Performance Model for Weighted Partition

For 3090+3070 with 1.75x speed ratio (full analysis at `docs/wayfinder/D4-performance-model.md`):
- 60 total layers, optimal split = 38 on 3090 + 22 on 3070
- 3090 compute: 38 * 80 us = 3.04 ms
- 3070 compute: 22 * 140 us = 3.08 ms (balanced)
- Cross-device copy (PCIe P2P): 0.3 ms
- Total: 3.38 ms -> 296 t/s
- Uplift vs equal split (4.5 ms -> 222 t/s): **+33%**

**Requires CUDA P2P** — without peer access, copy doubles to ~0.6 ms and target drops to 272 t/s (+22.5%).

---

### 12.8 Acceptance Criteria

#### From D4.1 (Research)

- [ ] Per-device RPC splits documented for triton topology
- [ ] RTT count measured: baseline 5 for 2-GPU config
- [ ] Server GPU utilization profiled: serial duty cycle confirmed

#### From D4.2 (Design)

- [x] ADR-0004 status: **ACCEPTED** (Option B+: GRAPH_COMPUTE_ALL + weighted weight placement)
- [x] Protocol changes documented (section 12.3)
- [x] Fallback behavior specified

#### From D4.3 (Spec)

- [x] This document: complete Path C spec section
- [x] Reviewed against existing spec style (sections 1-11)
- [x] Performance targets defined with measurement method

#### From D4.4 (Prototype)

- [ ] Throwaway: `GRAPH_COMPUTE_ALL` handler runs on triton 2-GPU
- [ ] Output matches per-device compute bit-exact
- [ ] Findings documented for implement phase

#### From D4.5 (Implement)

- [ ] `RPC_CMD_GRAPH_COMPUTE_ALL` handled in `ggml-rpc.cpp` server
- [ ] Server-side scheduler creates multi-device splits
- [ ] Client capability detection + fallback works
- [ ] Old client + new server: no regression
- [ ] New client + old server: fallback works

#### From D4.6 (Test)

- [ ] RTTs reduced: 5 -> 3 on triton 2-GPU
- [ ] Server GPU duty cycle: measurable improvement
- [ ] G non-regression: >= C1 baseline (42.8 t/s)
- [ ] Correctness: token sequence matches per-device compute
- [ ] Multi-turn: 8149 tokens without KV corruption

---

*Section 12 added 2026-07-11 — Path C Server-Side Scheduling. ADR-0004 Option B+ accepted.*

---

## 13. Deeper Pipelining — n_stages > 2 with Per-Backend Sub-Stages

This section defines the extension of GPipe Mode A from 2 stages to `n_stages > 2`.
The design splits the Stage 0 "compute" phase into per-backend sub-stages, enabling
straggler isolation and concurrent cross-token compute across heterogeneous backends.

ADR-0003: **Accepted (Option C: Hybrid)** — Topology-aware static default with adaptive
opt-in. Full decision at `docs/adr/0003-adaptive-pipeline-depth.md`.

D5.1 split timing analysis at `docs/wayfinder/D5.1-split-timing-analysis.md`.

---

### 13.1 Problem Statement

The 2-stage GPipe pipeline cannot overlap within-token RPC compute across backends.
As documented in D5.1:

- Split 1 (RPC0 3060 Ti) takes 10.0 ms (63.8% of Stage 0)
- Split 2 (ROCm0 7900 XTX) takes 5.7 ms (36.2% of Stage 0)
- Because they are serial, cycle time = 10.0 + 5.7 = 15.7 ms
- With per-backend sub-stages, cycle time = max(10.0, 5.7) = 10.0 ms (+57%)

The same pattern applies to the production 5-GPU cluster, where RPC1 (3060) at
4.0 ms dominates the pipeline and blocks faster backends (2.1-3.5 ms).

---

### 13.2 Architecture: Per-Backend Sub-Stages

#### Pipeline Structure

```
Current (2-stage):
  Stage 0: embed + Split 0 (CPU) + Split 1 (RPC) + Split 2 (ROCm) + Split 3..N
  Stage 1: gather + KV write + sample

Deeper (n_stages > 2):
  Stage 0: embed (CPU)
  Stage 1: backend[0] compute (first RPC or local backend)
  Stage 2: backend[1] compute (second backend)
  ...
  Stage N-1: backend[N-1] compute + gather + KV write + sample (last backend)
```

Each backend computes its assigned layer range independently. Faster backends
can begin processing token T+1 while slower backends finish token T.

#### Overlap Diagram (dual-GPU, n_stages=3)

```
Token T:   [emb(T)] [RPC0(T).............] [ROCm+gather(T)............]
Token T+1:         [emb(T+1)][RPC0(T+1).............][ROCm+gather(T+1)............]
```

With per-backend isolation, RPC0(T+1) runs concurrently with ROCm+gather(T),
reducing effective cycle time from 15.7 ms to 10.0 ms.

---

### 13.3 Stage Assignment Model

#### Static (topology-aware default)

```
n_stages = min(n_backends + 1, GGML_SCHED_GPIPE_DEPTH, LLAMA_GPIPE_MAX_STAGES)
```

| n_backends | n_stages | Sub-stages |
|:----------:|:--------:|-----------|
| 1 (local only) | 2 | embed+compute, gather (current behavior) |
| 2 (dual-GPU RPC) | 3 | embed, RPC0, ROCm+gather |
| 3 | 4 | embed, RPC0, RPC1, ROCm+gather |
| 4+ (5-GPU prod) | 5+ | embed + per-backend ... + last+gather |

#### Adaptive (opt-in via `GGML_SCHED_GPIPE_ADAPTIVE=1`)

After 5 warm-up decodes, measure per-backend timing from telemetry.
If `max_backend_time / min_backend_time < 1.3x`: collapse to 2-stage (homogeneous).
If ratio >= 1.3x: assign each backend its own stage.
Straggler (3x+ mean): reduce its layer share.

#### Fallback

If `n_stages < 2` or `n_stages > LLAMA_GPIPE_MAX_STAGES`, fall back to
`n_stages = n_backends + 1` with a log warning.

---

### 13.4 API Contracts

#### Event Signaling (per sub-stage)

```cpp
// Event protocol for n-stage pipeline:
//
// Sub-stage 0 (embed):
//   compute_embed() -> ggml_sched_gpipe_record(sched, 0)
//
// Sub-stage i (backend i-1 compute, for i in 1..n_stages-2):
//   ggml_sched_gpipe_wait(sched, i-1)  // wait for previous stage
//   -> compute_backend_layers(backend[i-1])
//   -> ggml_sched_gpipe_record(sched, i)
//
// Sub-stage n_stages-1 (last backend + gather + KV write):
//   ggml_sched_gpipe_wait(sched, n_stages-2)
//   -> compute_backend_layers(backend[n_stages-2])
//   -> gather + KV write
//   -> ggml_sched_gpipe_record(sched, n_stages-1)  // kv_ready for T+1
```

#### New Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GGML_SCHED_GPIPE_DEPTH` | 0 (auto) | Override n_stages; 0 = topology-aware default (n_backends+1) |
| `GGML_SCHED_GPIPE_ADAPTIVE` | 0 | Enable timing-based adaptive depth refinement |

#### Modified State Machine

```cpp
// llama_decode_gpipe_impl() extended for n_stages > 2:
switch (ctx->gpipe.cur_stage) {
    case 0:
        // embed
        compute_embed();
        ggml_sched_gpipe_record(sched, 0);
        ctx->gpipe.cur_stage = 1;
        break;
    case 1 ... LLAMA_GPIPE_MAX_STAGES-2:
        // per-backend compute (backends 0..n_backends-2)
        ggml_sched_gpipe_wait(sched, ctx->gpipe.cur_stage - 1);
        compute_backend(ctx->gpipe.cur_stage - 1);
        ggml_sched_gpipe_record(sched, ctx->gpipe.cur_stage);
        ctx->gpipe.cur_stage++;
        break;
    case N-1:
        // last backend + gather + KV write
        ggml_sched_gpipe_wait(sched, ctx->gpipe.cur_stage - 1);
        compute_backend_last();
        gather_outputs();
        kv_write();
        ggml_sched_gpipe_record(sched, ctx->gpipe.cur_stage);
        ctx->gpipe.cur_stage = 0;  // wrap around
        break;
}
```

#### Backend-to-Stage Mapping

```cpp
// In ggml_backend_sched, the GPipe stage assignment is:
//
// For each split_id in the scheduler's split list:
//   - Split with backend_id == 0 and phase "embed": maps to GPipe stage 0
//   - Split with backend_id == i (i > 0): maps to GPipe stage i+1
//   - Split with backend_id == n_backends-1: maps to GPipe stage n_backends
//     (last stage includes gather + KV write)
//
// The mapping is determined at sched_reserve() time and stored in:
//   int split_to_gpipe_stage[GGML_SCHED_MAX_SPLITS];
```

#### Cross-Token Event Protocol

```
Token T sub-stage k records gpipe_event[k]
Token T+1 sub-stage 0 records gpipe_event[0]
Token T+1 sub-stage k waits on gpipe_event[k-1] (previous sub-stage of T+1)
Token T+1 sub-stage k also waits on gpipe_event[k+1] (next sub-stage of T, if k+1 exists)
```

This ensures each sub-stage only starts when (a) the previous sub-stage of the
same token is done, AND (b) the next sub-stage of the previous token is done
(so the previous token has vacated the pipeline slot).

---

### 13.5 Integration with Existing GPipe Infrastructure

| Component | Change |
|-----------|--------|
| `llama_gpipe_state` | `n_stages` now configured; `cur_stage` ranges `0..n_stages-1` |
| `llama_decode_gpipe_impl()` | `switch` extended from 2 cases to N cases |
| `ggml_sched_gpipe_init()` | Called with `n_stages` from topology-aware default |
| `ggml_sched_gpipe_record/wait()` | No change — already supports arbitrary stage IDs up to `GGML_SCHED_MAX_STAGES=8` |
| `ggml_backend_sched` | No struct changes needed; `gpipe_events[8]` already allocated |
| `llama_context` | No new members; `gpipe.n_stages` used for stage count |
| RPC protocol | No changes; per-split `GRAPH_COMPUTE` continues to work |

---

### 13.6 Performance Targets

| Metric | 2-stage (current) | 3-stage (target) | 5-stage (stretch, cluster) |
|--------|:----------------:|:----------------:|:--------------------------:|
| Cycle time (romulus dual-GPU) | 15.7 ms | 10.0 ms (-36%) | N/A |
| Per-token latency (romulus) | 15.7 ms | 10.0 ms | N/A |
| Cycle time (production 5-GPU) | ~12.8 ms est. | ~8.8 ms (-31%) | ~4.0 ms (-69%) |
| global_3bk_pct | <1% | TBD | >=25% (target) |
| overlap_pct | 0.1-0.2% | TBD | >=5% (target) |

---

### 13.7 Acceptance Criteria

#### From D5.5 (Implementation)

- [ ] Stage 0 split into embed + per-backend sub-stages
- [ ] Event signaling correct: each sub-stage records/waits on correct event
- [ ] Straggler isolation: fast backends not blocked by straggler
- [ ] Dual-GPU (romulus): 3 sub-stages active (embed + RPC0 + ROCm+gather)
- [ ] Single-GPU: falls back to 2-stage (current behavior)
- [ ] `GGML_SCHED_GPIPE=1` enables n_stages > 2 automatically

#### From D5.6 (Adaptive Depth)

- [ ] `GGML_SCHED_GPIPE_DEPTH` overrides n_stages
- [ ] `GGML_SCHED_GPIPE_ADAPTIVE` enables timing-based adaptation
- [ ] Fallback to static if adaptive fails
- [ ] `GGML_SCHED_GPIPE=0` disables all GPipe (no change)

#### From D5.7 (Test)

- [ ] Romulus dual-GPU: per-token latency improves vs 2-stage
- [ ] Correctness: logits match 2-stage GPipe ON output
- [ ] No regression when GPipe OFF

---

### 13.8 References

- `docs/wayfinder/D5.1-split-timing-analysis.md` — per-backend timing data
- `docs/adr/0003-adaptive-pipeline-depth.md` — depth decision (Option C)
- `docs/wayfinder/D0.5-implementation-seam.md` — implementation seam (section 7)
- `docs/hot-paths-analysis.md` — tensor deployment map
- `docs/wayfinder/D4.1-romulus-baseline-analysis.md` — D4 baseline

---

*Section 13 added 2026-07-11 — Deeper Pipelining n_stages > 2. ADR-0003 Option C accepted.*