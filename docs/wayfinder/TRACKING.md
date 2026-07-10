# TRACKING — Path D Full Workflow

**Branch:** Path-D-Gpipeline-Assembly-Line  
**Date:** 2026-07-10  
**Parent:** `docs/rpc-multi-backend-pipeline-plus/TRACKING.md`

---

## Phase Status

| Phase | Status | Completion Date |
|-------|--------|-----------------|
| Investigation (D0.1-D0.5) | ✅ complete | 2026-07-10 |
| Specification | ✅ complete | 2026-07-10 |
| Work Breakdown | ✅ complete | 2026-07-10 |
| Implementation (D1.1-D1.7) | ✅ complete | 2026-07-10 |
| Testing (D2.1-D2.3) | ⏳ pending | - |
| Production Hardening (D3.1-D3.3) | ⏳ pending | - |
| Path C Stepping Stone (D4.1-D4.6) | ⏳ pending | - |
| Deeper Pipelining (D5.1-D5.7) | ⏳ pending | - |
| Mode B Microbatch (D6.1-D6.7) | ⏳ pending | - |
| Advanced Optimization (R3.1-R3.5) | ⏳ pending | - |

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

### D2 — Testing (pending)

| Ticket | Status | Notes |
|--------|--------|-------|
| D2.1 | ⏳ pending | Correctness: logits hash, KV fill, MoE routing |
| D2.2 | ⏳ pending | Performance: `global_3bk_pct`, `overlap_pct`, G targets |
| D2.3 | ⏳ pending | Regression: Path-B+ baseline unchanged |

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
