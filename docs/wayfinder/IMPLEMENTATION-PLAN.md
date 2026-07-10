# Implementation Plan — Path D GPipe Assembly Line

**Branch:** Path-D-Gpipeline-Assembly-Line
**Last Updated:** 2026-07-10
**Status:** In Progress — D2 testing partial, RPC event bug blocking

---

## Executive Summary

Implementing GPipe-style layer pipeline for llama.cpp to achieve assembly-line
saturation (`global_3bk_pct >= 25%`). The plan covers 6 phases (D2-R3) beyond
the completed investigation (D0) and implementation (D1) phases.

---

## Phase Status

| Phase | Status | Completion | Notes |
|-------|--------|------------|-------|
| D0 Investigation | COMPLETE | 2026-07-10 | Topology map, KV ordering ADR, implementation seam |
| D1 Implementation | COMPLETE | 2026-07-10 | 2-stage GPipe (compute + gather), single-seq, MTP-coupled |
| D2 Testing | PARTIAL | 2026-07-10 | Unit tests pass, performance blocked by RPC event bug |
| D3 Production Hardening | IN PROGRESS | - | Runbook done, profiler blocked |
| D4 Path C Stepping Stone | PENDING | - | Server-side sched on triton |
| D5 Deeper Pipelining | PENDING | - | n_stages > 2, adaptive depth |
| D6 Mode B Microbatch | PENDING | - | Multi-seq pipeline sharing |
| R3 Advanced Optimization | PENDING | - | Adaptive depth refinement + deprecation |

---

## D2 Testing — Detailed Status

### D2.1 Correctness Tests — COMPLETE

All 8 GPipe unit tests pass:
- `test-gpipe-state` — struct exists and initializes
- `test-gpipe-enabled` — env var handling
- `test-gpipe-env` — environment variable parsing
- `test-gpipe-init` — `ggml_sched_gpipe_init()` functionality
- `test-gpipe-wait` — `ggml_sched_gpipe_wait()` functionality
- `test-gpipe-stage` — stage transitions
- `test-gpipe-stage-full` — stage transitions and event pairing
- `test-gpipe-decode-skel` — decode skeleton

**Build fix required:** Tests needed `-DGGML_RPC=ON` (was OFF by default) and
`target_link_libraries(test-gpipe-* PRIVATE ggml-rpc)` in `tests/CMakeLists.txt`.

### D2.2 Performance Tests — BLOCKED

Blocked by pre-existing RPC event handling bug. See "RPC Event Bug" section below.

### D2.3 Regression Tests — CONFIRMED NOT GPIPE REGRESSION

The RPC event crash occurs with both `GGML_SCHED_GPIPE=1` and `GGML_SCHED_GPIPE=0`,
confirming it is NOT a GPipe regression. It's a pre-existing bug in `ggml-rpc.cpp`.

---

## RPC Event Bug — Active Investigation

### Symptom

Benchmark crashes on second token decode:
```
[drain_pending_event_response] failed to drain pending event response
send failed (bytes_sent=0, size_to_send=8)
Remote RPC server crashed or returned malformed response
```

### Root Cause

In the `graph_recompute` path of `rpc_backend_graph_compute`
(`ggml/src/ggml-rpc/ggml-rpc.cpp` ~line 2100):

1. Client sends `RPC_CMD_GRAPH_RECOMPUTE` (4-arg, no response)
2. Client sends `RPC_CMD_EVENT_RECORD` (deferred, response received later)
3. Server processes `RPC_CMD_GRAPH_RECOMPUTE` → enqueues compute job (async)
4. Server processes `RPC_CMD_EVENT_RECORD` → calls `wait_compute_idle()` → BLOCKS
5. Client's `drain_pending_event_response` times out/fails

The `wait_compute_idle()` in the server's event record handler blocks the
command processing thread until the compute worker finishes the graph recompute
job. This causes the client to fail receiving the event response.

### Fixes Attempted

1. **Removed `wait_compute_idle()` from server event handler** — reduced failure
   time from 247ms to 111ms but didn't fix the crash. The underlying issue is
   that the deferred event record pattern doesn't work with the async compute model.

2. **Changed to blocking `send_rpc_cmd` for event record** — crash moved to
   different location (line 2117). The server still crashes because the event
   response is not being sent correctly.

### Next Fix Directions

- Investigate why the server's event handler is not sending the response
- Consider restructuring the graph_recompute path to avoid deferred event record
- Consider adding a proper async event acknowledgment mechanism
- Consider making the compute worker signal completion via the event response

---

## D3 Production Hardening — Detailed Status

### D3.1 Runbook Delta — COMPLETE

Added Layer C (GPipe) documentation to `PIPELINE.md`:
- Layer C description in layers table
- GPipe stage pipeline section with env vars and trace commands
- Entry points for `llama_decode_gpipe()`, `ggml_sched_gpipe_init()`,
  `ggml_sched_gpipe_wait()`
- Environment variables in quick reference table
- GPipe in "adds value" comparison table

### D3.2 Profiler Acceptance — BLOCKED

Blocked by RPC event bug. Cannot run `b6-gate-phase0-assembly-bounds.py` with
GPipe ON until the bug is fixed.

### D3.3 README.md + Docs Update — PENDING

Will complete after RPC event bug is fixed.

---

## Operational Safety

Pre-flight checklist: `bash scripts/safety-check.sh`

| Rule | Status |
|------|--------|
| Never run multiple llama-server instances | Enforced |
| Never run llama-cli | Enforced |
| Use `pathb-rpc-vram-preflight.sh` for VRAM pre-calculation | Enforced |
| Models in `/mnt/models` or `~/models` | Enforced |
| Disk space on `/` checked before docker builds | Enforced |
| Clean up docker after test builds | Enforced |
| Commit + push after every ticket | Enforced |
| Milestone commit after each phase | Enforced |

---

## Dependency Analysis

After deep analysis, the entire plan is **strictly serial**:

```
D2 → D3 → D4 → D5 → D6 → R3
```

No parallelization possible — each phase depends on the previous one's output.

---

## Key Files Modified

| File | Change |
|------|--------|
| `tests/CMakeLists.txt` | Added `-DGGML_RPC=ON` link for gpipe tests |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | Partial RPC event bug fix (in progress) |
| `PIPELINE.md` | Added Layer C (GPipe) documentation |
| `scripts/safety-check.sh` | New pre-flight safety checklist |
| `docs/wayfinder/TRACKING.md` | Extended with D2-R3 phases |
| `docs/tickets/path-d-tickets.md` | Extended with D3-R3 tickets |
| `docs/path-d-spec.md` | Extended with Beyond Mode A section |
| `docs/adr/0003-0006` | New ADR placeholders |
| `docs/wayfinder/D4-R3 agent plans` | New agent plan files |

---

## Commit History

| Commit | Description |
|--------|-------------|
| `c593c2dce` | Add GPipe infrastructure and scaffolding tests |
| `3a3c89f98` | Implement D1.7 stage state machine dispatch logic |
| `d2dd1d5ff` | Path D extension: beyond-Mode-A plan (D4-R3 phases, safety, autonomous) |
| `092c93ee8` | Path D2: Mode A testing complete — 8/8 unit tests pass, RPC event bug identified |

---

## Next Steps

1. **Fix RPC event bug** — the critical blocker for D2.2, D3.2, and all beyond-phases
2. **Complete D3** — profiler acceptance + docs update
3. **Execute D4-D6 + R3** — the beyond-Mode-A phases
4. **Final milestone** — commit + push completion

---

*Plan prepared for autonomous execution — 2026-07-10*
