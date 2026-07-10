# Path D Specification — GPipe Client Scheduler

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-10  
**Status:** Specification Phase  
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

**Goal:** Prove server-side scheduling on triton's co-localized dual-GPU before deeper client-side pipelining.

| Step | What | Output |
|------|------|--------|
| Research | C1 baseline: per-device RPC splits, RTT count, server GPU util | `D4.1-triton-baseline-analysis.md` |
| Design | ADR-004: Server-side scheduling model | `docs/adr/0004-server-side-scheduling.md` |
| Spec | Path C spec section | `docs/path-d-spec.md` (this file) |
| Prototype | Throwaway: test `GRAPH_COMPUTE_ALL` concept | Findings for implement |
| Implement | C2: server-side sched with `GRAPH_COMPUTE_ALL` | Code in `ggml-rpc.cpp` |
| Test | Verify server GPU duty improvement | Metrics in TRACKING.md |

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

### 10.3 Phase D6 — Mode B Microbatch / Multi-Seq

**Goal:** Support multiple sequences at different pipeline positions (server multi-slot).

| Step | What | Output |
|------|------|--------|
| Research | Multi-slot requirements, KV cache interaction | `D6.1-multi-seq-requirements.md` |
| Design | ADR-005: Multi-seq GPipe scheduling | `docs/adr/0005-multi-seq-gpipe.md` |
| Spec | Multi-seq spec section | `docs/path-d-spec.md` (this file) |
| Prototype | Throwaway: multi-seq token tracking | Findings for implement |
| Implement | Extend state machine for multi-seq | Code in `llama-context.cpp` |
| Implement | Server multi-slot dispatch | Code in `ggml-rpc.cpp` |
| Test | Verify concurrent multi-seq decode | Metrics in TRACKING.md |

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
- `docs/rpc-multi-backend-pipeline-plus/DESIGN-path-d-layer-pipeline.md`

---

*Specification extended with Beyond Mode A (D4-R3) — 2026-07-10*