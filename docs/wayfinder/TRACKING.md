# TRACKING — Path D Full Workflow

**Branch:** Path-D-Gpipeline-Assembly-Line
**Date:** 2026-07-10
**Parent:** `docs/rpc-multi-backend-pipeline-plus/TRACKING.md`
**Status:** IN PROGRESS — RPC event bug blocking D2.2/D3.2

---

## Phase Status

| Phase | Status | Completion Date |
|-------|--------|-----------------|
| Investigation (D0.1-D0.5) | COMPLETE | 2026-07-10 |
| Specification | COMPLETE | 2026-07-10 |
| Work Breakdown | COMPLETE | 2026-07-10 |
| Implementation (D1.1-D1.7) | COMPLETE | 2026-07-10 |
| Testing (D2.1-D2.3) | PARTIAL | 2026-07-10 |
| Production Hardening (D3.1-D3.3) | IN PROGRESS | - |
| Path C Stepping Stone (D4.1-D4.6) | PENDING | - |
| Deeper Pipelining (D5.1-D5.7) | PENDING | - |
| Mode B Microbatch (D6.1-D6.7) | PENDING | - |
| Advanced Optimization (R3.1-R3.5) | PENDING | - |

## Blockers

| Blocker | Affects | Status |
|---------|---------|--------|
| RPC event drain bug | D2.2, D3.2, D4-R3 | ACTIVE — investigating |

---

## D2 Findings

**Build fix:** Tests required `-DGGML_RPC=ON` (was OFF by default) and
`target_link_libraries(test-gpipe-* PRIVATE ggml-rpc)` in `tests/CMakeLists.txt`.

**Pre-existing RPC event bug:** `drain_pending_event_response` fails after ~247ms on
second token decode. Crash occurs with both `GGML_SCHED_GPIPE=1` and `GGML_SCHED_GPIPE=0`.
Root cause: `ggml-rpc.cpp` event handling blocks on `wait_compute_idle()` in the
server's event record handler.

**Fixes attempted:**
1. Removed `wait_compute_idle()` from server event handler — reduced failure time
   but didn't fix crash
2. Changed to blocking `send_rpc_cmd` for event record — crash moved to different
   location (line 2117)

**Performance metrics:** Cannot be collected until RPC event bug is fixed.

---

## Implementation Ticket Status

### D1 — Mode A Implementation (complete)

| Ticket | Status | Notes |
|--------|--------|-------|
| D1.1 | ✅ complete | `llama_gpipe_state` struct added |
| D1.2 | ✅ complete | `llama_gpipe_enabled()` helper implemented |
| D1.3 | ✅ complete | `GGML_SCHED_GPIPE` env var handling added |
| D1.4 | ✅ complete | `llama_decode_gpipe_impl()` skeleton implemented |
| D1.5 | ✅ complete | `ggml_sched_gpipe_init()` implemented |
| D1.6 | ✅ complete | `ggml_sched_gpipe_wait()` implemented |
| D1.7 | ✅ complete | Stage state machine dispatch logic complete |

### D2 — Testing (partial)

| Ticket | Status | Notes |
|--------|--------|-------|
| D2.1 | ✅ complete | All 8 GPipe unit tests pass (state, enabled, env, init, wait, stage, stage-full, decode-skel) |
| D2.2 | ⚠️ blocked | Model loads, first token decodes, event drain crashes on second token — pre-existing RPC bug |
| D2.3 | ⚠️ confirmed | GPipe OFF also crashes — confirms crash is NOT a GPipe regression |

### D2 Findings

**Build fix:** Tests required `-DGGML_RPC=ON` (was OFF by default) and `target_link_libraries(test-gpipe-* PRIVATE ggml-rpc)` in `tests/CMakeLists.txt`.

**Pre-existing RPC event bug:** `drain_pending_event_response` fails after ~247ms on second token decode. Crash occurs with both `GGML_SCHED_GPIPE=1` and `GGML_SCHED_GPIPE=0`. Root cause: `ggml-rpc.cpp` event handling race condition, not GPipe-specific.

**Performance metrics:** Cannot be collected until RPC event bug is fixed. The 2-stage GPipe pipeline's event record/wait logic is exercised but the crash prevents multi-token measurement.

### D3 — Production Hardening (pending)

| Ticket | Status | Notes |
|--------|--------|-------|
| D3.1 | ⏳ pending | Runbook delta in PIPELINE.md |
| D3.2 | ⏳ pending | Profiler acceptance |
| D3.3 | ⏳ pending | README.md + docs/ update |

### D4 — Path C Stepping Stone (pending)

| Ticket | Status | Notes |
|--------|--------|-------|
| D4.1 | ⏳ pending | C1 baseline: triton dual-GPU analysis |
| D4.2 | ⏳ pending | ADR-004: Server-side scheduling model |
| D4.3 | ⏳ pending | Spec section for Path C |
| D4.4 | ⏳ pending | Prototype: GRAPH_COMPUTE_ALL concept |
| D4.5 | ⏳ pending | C2: server-side sched implementation |
| D4.6 | ⏳ pending | Test: server GPU duty improvement |

### D5 — Deeper Pipelining (pending)

| Ticket | Status | Notes |
|--------|--------|-------|
| D5.1 | ⏳ pending | Split timing analysis |
| D5.2 | ⏳ pending | ADR-003: Adaptive pipeline depth |
| D5.3 | ⏳ pending | Spec section for deeper pipelining |
| D5.4 | ⏳ pending | Prototype: per-backend sub-stages |
| D5.5 | ⏳ pending | Implement per-backend sub-stages |
| D5.6 | ⏳ pending | Dynamic stage assignment |
| D5.7 | ⏳ pending | Test: `global_3bk_pct >= 25%` |

### D6 — Mode B Microbatch (pending)

| Ticket | Status | Notes |
|--------|--------|-------|
| D6.1 | ⏳ pending | Multi-seq requirements analysis |
| D6.2 | ⏳ pending | ADR-005: Multi-seq GPipe scheduling |
| D6.3 | ⏳ pending | Spec section for multi-seq |
| D6.4 | ⏳ pending | Prototype: multi-seq token tracking |
| D6.5 | ⏳ pending | Extend state machine for multi-seq |
| D6.6 | ⏳ pending | Server multi-slot dispatch |
| D6.7 | ⏳ pending | Test: concurrent multi-seq decode |

### R3 — Advanced Optimization (pending)

| Ticket | Status | Notes |
|--------|--------|-------|
| R3.1 | ⏳ pending | Adaptive depth analysis |
| R3.2 | ⏳ pending | ADR-006: Finalize adaptive depth |
| R3.3 | ⏳ pending | Deprecation warnings for B+11/B+14/B+7f |
| R3.4 | ⏳ pending | Adaptive depth refinement |
| R3.5 | ⏳ pending | Test: no regression |

---

## Implementation Commits

| Commit | Message | Files Changed |
|--------|---------|-------------|
| `c593c2dce` | Add GPipe infrastructure and scaffolding tests | 11 files |
| `3a3c89f98` | Implement D1.7 stage state machine dispatch logic | 4 files |

---

## Safety

Pre-flight checklist: `bash scripts/safety-check.sh`  
Run before every docker build, server start, or benchmark.  
Aborts if VRAM/RAM/disk/running-instances indicate OOM risk.

---

## Next Actions

1. **D2 Testing** — Build and run GPipe test suite
2. **D3 Production Hardening** — After tests pass
3. **D4 Path C Stepping Stone** — De-risks scheduling on triton
4. **D5 Deeper Pipelining** — Main throughput lever (n_stages > 2)
5. **D6 Mode B Microbatch** — Multi-seq support
6. **R3 Advanced Optimization** — Adaptive depth + deprecation cleanup

---

## Test Execution

```bash
# Build tests
cmake --build build-rocm-docker --target test-gpipe-state test-gpipe-enabled test-gpipe-env test-gpipe-init test-gpipe-wait test-gpipe-stage-full

# Run tests
cd build-rocm-docker
ctest -R test-gpipe --output-on-failure
```

---

*Tracking file updated — extension plan created (D2-R3 phases added)*
