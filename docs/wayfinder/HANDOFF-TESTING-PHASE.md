# Handoff: Testing Phase (D2.1-D2.3)

**Date:** 2026-07-10  
**From:** Implementation Phase  
**To:** Testing Phase  
**Branch:** Path-D-Gpipeline-Assembly-Line

---

## Current State

### Completed Implementation

All D1.1-D1.7 tickets are complete. See commits:

| Commit | Message |
|--------|---------|
| `c593c2dce` | Add GPipe infrastructure and scaffolding tests |
| `3a3c89f98` | Implement D1.7 stage state machine dispatch logic |

### Key Files

| File | Purpose |
|------|---------|
| `src/llama-context.h` | `llama_gpipe_state` struct definition |
| `src/llama-context.cpp` | `llama_decode_gpipe_impl()` with stage state machine |
| `ggml/src/ggml-backend.cpp` | `ggml_sched_gpipe_init()`, `wait()`, `record()` |
| `tests/test-gpipe-*.cpp` | 8 test files for GPipe functions |

---

## Testing Requirements

### D2.1 — Correctness Tests

**Goal:** Verify GPipe produces correct outputs matching classic decode.

**Test Files:**
- `tests/test-gpipe-state.cpp` — struct exists and initializes correctly
- `tests/test-gpipe-enabled.cpp` — env var handling
- `tests/test-gpipe-env.cpp` — environment variable parsing
- `tests/test-gpipe-init.cpp` — `ggml_sched_gpipe_init()` functionality
- `tests/test-gpipe-wait.cpp` — `ggml_sched_gpipe_wait()` functionality
- `tests/test-gpipe-stage-full.cpp` — stage transitions and event pairing

**Acceptance Criteria:**
- [ ] All tests pass with `ctest -R test-gpipe`
- [ ] Logits hash smoke: T+1 logits match classic decode
- [ ] Multi-turn KV fill: 8149 tokens without corruption
- [ ] No crashes on invalid stage_id handling

### D2.2 — Performance Tests

**Goal:** Verify GPipe achieves overlap targets.

**Metrics:**
- `global_3bk_pct >= 25%` (target)
- `overlap_pct >= 5%` (target)
- G (A1) >= 180 t/s on romulus-local

**Test Script:**
```bash
# Build
cmake --build build-rocm-docker --target llama-server

# Run with profiling
export GGML_SCHED_GPIPE=1
export GGML_PIPELINE_PLUS=1
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE=1

bash scripts/romulus-local-up.sh \
  --model /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  --ts 50,50 \
  --ctx 8192
```

### D2.3 — Regression Tests

**Goal:** Verify no regressions on Path-B+ baseline.

**Test:**
- Run `b6-2gpu-f-romulus-local` baseline comparison
- Verify Path-B+ benchmarks unchanged when GPipe OFF

---

## Build Instructions

```bash
# From repo root
cmake -B build-rocm-docker -S . \
  -DGGML_BACKEND_HIP=ON \
  -DGGML_HIP_DYNAMIC_LINK=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-rocm-docker --target llama-server -j$(nproc)
cmake --build build-rocm-docker --target test-gpipe-* -j$(nproc)
```

---

## Test Execution Commands

```bash
# Run all GPipe tests
cd build-rocm-docker
ctest -R test-gpipe --output-on-failure

# Run specific test
./test-gpipe-stage-full
./test-gpipe-state
./test-gpipe-enabled
```

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/path-d-spec.md` | Specification for what to test |
| `docs/wayfinder/D0.5-implementation-seam.md` | Implementation details |
| `docs/adr/0002-gpipe-kv-ordering.md` | KV ordering decisions |
| `docs/wayfinder/TRACKING.md` | Current status |

---

## Next After Testing

If tests pass:
1. Update `README.md` with GPipe feature
2. Update `docs/` with GPipe documentation
3. Create `docs/path-d-complete-report.md`

If tests fail:
1. Create bug tickets
2. Hand back to `/implement` for fixes

---

## Session Handoff

In new session:

```
/handoff "Testing Phase: Run GPipe tests and verify performance"
→ Read this file
→ Build with cmake
→ Run ctest -R test-gpipe
→ Run romulus-local benchmark
→ Update TRACKING.md with results
```

---

## Beyond Testing — Next Phases (D4-R3)

After D2 testing completes, the following phases extend Mode A into the full GPipe architecture. Each follows the workflow loop: research → design → spec → prototype → implement → test → review (max 3 loops).

| Phase | Goal | Agent Plan |
|-------|------|------------|
| **D3** Production Hardening | Make Mode A deployable | Tickets D3.1-D3.3 |
| **D4** Path C Stepping Stone | Server-side sched on romulus dual-GPU (7900 XTX + 3060 Ti) | `D4-PATH-C-AGENT-PLAN.md` |
| **D5** Deeper Pipelining | n_stages > 2, per-backend sub-stages | `D5-DEEPER-PIPELINE-AGENT-PLAN.md` |
| **D6** Mode B Microbatch | Multi-seq pipeline sharing | `D6-MODE-B-AGENT-PLAN.md` |
| **R3** Advanced Optimization | Adaptive depth + deprecation cleanup | `R3-ADVANCED-OPT-AGENT-PLAN.md` |

### Execution Rules

- **No human intervention** during execution — continue through all failures
- **Milestone commit** after each phase (self-describing, pushed to GitHub)
- **Intermediate commits** after every ticket
- **Safety check** (`bash scripts/safety-check.sh`) before every resource-intensive operation
- **3-loop limit** per phase: if not converged, mark FAILED SPIKE and continue
- **Review only after full plan completion**

### Safety Reminders

- One llama-server instance at a time
- Never run llama-cli (OOM risk)
- Use `pathb-rpc-vram-preflight.sh` for VRAM pre-calculation
- Models in `/mnt/models` or `~/models`
- Clean up docker after test builds: `docker system prune -f`
- Do not kill romulus (hosts this session)

---

*Handoff prepared for Testing Phase continuation — 2026-07-10*