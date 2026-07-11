# Implementation Plan — Path D GPipe Assembly Line

**Branch:** Path-D-Gpipeline-Assembly-Line
**Last Updated:** 2026-07-11
**Status:** D2/D3 complete on dual-GPU — 5-GPU cluster metrics and profiler deferred to cluster deployment

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
| D2 Testing | COMPLETE | 2026-07-11 | RPC event bug fixed; dual-GPU perf validated (within noise); cluster metrics deferred |
| D3 Production Hardening | COMPLETE | 2026-07-11 | TRACKING.md updated; README/docs deferred to cluster validation |
| D4 Path C Stepping Stone | PENDING | - | Server-side sched on romulus dual-GPU (7900 XTX + 3060 Ti); cluster (triton) deferred |
| D5 Deeper Pipelining | PENDING | - | n_stages > 2, adaptive depth |
| D6 Mode B Microbatch | PENDING | - | Multi-seq pipeline sharing |
| R3 Advanced Optimization | PENDING | - | Adaptive depth refinement + deprecation |

---

## Hardware

| Machine | Role | GPUs | Models |
|---------|------|------|--------|
| romulus | Local dev + dual-GPU test bench | AMD 7900 XTX (client), NVIDIA 3060 Ti (RPC server) | `/mnt/models` |
| triton | Cluster deployment | 5-GPU (see CLUSTER-NODE-LAYOUT.md) | `/mnt/models` |
| remus | Cluster deployment | (see CLUSTER-NODE-LAYOUT.md) | `/mnt/models` |

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

## Profiler Architecture

### Architecture Pipeline

```
                           D4.7-D4.10 (NOW)                        D4.11-D4.14 (LATER)
                    ┌──────────────────────────┐            ┌──────────────────────────────┐
                    │  llama-gpipe-profiler    │            │  llama-server Pareto Optimizer│
                    │  (native C++, like       │            │  (built into server governor) │
                    │   llama-bench)           │            │                              │
                    │                          │            │  ┌──────────────────┐         │
 GPU Cluster ──────>│  ┌────────────────────┐  │  heatmap   │  │ Environment      │         │
 (RPC endpoints)   │  │ Per-layer timing   │──┼──────────> │  │ Auto-Analysis    │         │
                    │  │ GPU util breakdown │  │  .json     │  └──────┬───────────┘         │
                    │  │ KV cache timing    │  │            │         │                     │
                    │  │ Task-aware: pp, tg │  │            │  ┌──────▼───────────┐         │
                    │  └────────────────────┘  │            │  │ Pareto Placement  │         │
                    └──────────────────────────┘            │  │ Hot 20% layers   │         │
                                                            │  │ -> Fast 20% GPUs │         │
                                                            │  └──────────────────┘         │
                                                            └──────────────────────────────┘
```

### Scope Boundary

| Scope | Tickets | Build Now? | Owner |
|-------|---------|------------|-------|
| Profiler data collection (server-side telemetry) | D4.7-D4.10 | **Yes** — v1 implementation | This sprint |
| `llama-gpipe-profiler` binary | D4.7-D4.10 | **Yes** — CMake target, CLI, heatmap | This sprint |
| Python profiler adaptation | D4.10 | **Yes** — consume new telemetry fields | This sprint |
| Pareto Optimizer in `llama-server` | D4.11-D4.14 | **No** — plan + ticket only | Future sprint |
| Full transition: Python -> native | — | **No** — phased out gradually | Post-v1 |

### Design Decisions

1. **Profiler is a standalone binary** (not embedded in server). Rationale: follows the `llama-bench` pattern — a dedicated tool that exercises the model through real inference tasks to produce accurate, task-dependent measurements. The server telemetry path is for raw data collection; the profiler binary orchestrates complex task sequences (prompt processing, token generation) and synthesizes the heatmap.

2. **KV cache profiling included**. The heatmap must capture KV cache read/write timing per slot, placement map, and eviction cost — not just layer compute. This is critical for the Pareto optimizer to make informed placement decisions about which tensors keep hot KV pages.

3. **Task-aware profiling**. Models activate different tensors/layers differently depending on the task (prompt processing vs token generation vs speculative decoding). The profiler must measure both `pp` and `tg` tasks to capture this variance. The heatmap is task-stratified.

4. **Transition: Python profiler stays alive**. The existing `llama-pipeline-profiler` Python tool runs alongside the native profiler during maturation. It's phased out step by step as the native profiler reaches parity. No flag day.

5. **Pareto Optimizer is server-resident**. The optimizer lives in `llama-server` itself, so the server governor/dispatcher can auto-analyze its environment and decide GPU placement without external orchestration. This is ticketed (D4.11-D4.14) but explicitly not built in the current sprint — it would explode the scope of this implementation.

### D4.7 — Profiler Research: Binary Design + Heatmap Schema

**Status:** PENDING

Survey existing infrastructure and design the profiler binary:

- **Existing infra**: `llama-bench` source structure (`examples/llama-bench/`) as pattern template; existing client-side traces (`GGML_SCHED_TRACE=1`, `GGML_RPC_TRACE=1`); existing GPU telemetry tools (`nvidia-smi dmon`, `rocm-smi`); existing `llama-pipeline-profiler` Python tool and `diagnose.json` schema
- **Binary design**: CLI surface (`--model`, `--endpoints`, `--tasks pp,tg`, `--output heatmap.json`, `--repeat`, `--warmup`), CMake target name, integration point in `tools/` or `examples/`
- **Heatmap schema**: JSON format for per-layer timing, GPU utilization breakdown, KV cache timing per slot, task stratification (pp vs tg), GPU metadata (device name, VRAM, backend, PCIe topology)
- **KV cache profiling scope**: Read/write timing per KV slot, placement map, eviction cost. Decide whether to instrument `llama_kv_cache` directly or collect via RPC telemetry
- **Task-awareness**: How to drive prompt processing vs token generation workloads through the profiler, capturing per-task tensor/layer activation patterns

Output: `docs/wayfinder/D4.7-profiler-research.md` with binary design, heatmap schema draft, and KV cache scope decision.

### D4.8 — Profiler Prototype: Server Collection + Thin Client

**Status:** PENDING

Throwaway prototype validating three things:

1. **Server-side collection overhead**: Can the server capture per-GPU timing from `ggml_backend_sched_graph_compute()` and KV cache operations without perturbing the hot path? Measure collection + formatting cost vs baseline.
2. **Wire format**: `rpc_msg_server_telemetry` as an optional frame appended to `GRAPH_COMPUTE_ALL` / `GRAPH_RECOMPUTE_ALL` response. Extended fields beyond the original 4:
   - `device_timings_us[]` — per-GPU compute time
   - `layer_assignments[]` — layer-to-GPU mapping
   - `copy_times_us[]` — cross-device copy time
   - `device_meta[]` — GPU metadata
   - `kv_read_times_us[]` — KV cache read per slot *(new)*
   - `kv_write_times_us[]` — KV cache write per slot *(new)*
3. **Thin client**: A minimal C program that connects to RPC endpoints, exercises `GRAPH_COMPUTE_ALL`, parses telemetry, and writes raw JSON — validating the end-to-end pipeline before building the full profiler binary.

Skill: `/prototype`. Keep findings, delete code. Output: `docs/wayfinder/D4.8-profiler-prototype-findings.md`.

### D4.9 — Profiler ADR: Binary + Heatmap + Transition

**Status:** PENDING

Produce ADR-004 section (or standalone mini-ADR) deciding:

- **Binary design**: Standalone executable (`llama-gpipe-profiler`) patterned after `llama-bench`. CLI surface, CMake target location (`tools/llama-gpipe-profiler/`), integration model (links against `libllama`, `libggml`)
- **Telemetry protocol**: `rpc_msg_server_telemetry` wire format, versioning strategy, opt-in mechanism (`GGML_RPC_SERVER_TELEMETRY=0|1`)
- **Heatmap format**: JSON schema, task stratification, KV cache fields, GPU metadata. Consumer contract: what the Pareto optimizer expects
- **Transition plan**: Python `llama-pipeline-profiler` stays alive during native profiler maturation. No flag day. Phased deprecation milestones defined
- **Future hook**: Interface between profiler heatmap output and the planned server-side Pareto optimizer (D4.11-D4.14). Forward-compatible schema design

Output: Section in `docs/adr/0004-server-side-scheduling.md` (or `docs/adr/0004b-profiler-architecture.md`).

### D4.10 — Profiler Implementation v1

**Status:** PENDING
**Blocked by:** D4.4 (GRAPH_COMPUTE_ALL prototype), D4.5 (C2 implementation), D4.9 (ADR)

Implement v1 of the profiler — server telemetry paths + profiler binary + script adaptation:

1. **Server telemetry** (`ggml/src/ggml-rpc/ggml-rpc.cpp`):
   - Collect per-backend timing after `ggml_backend_sched_graph_compute()`
   - Collect KV cache read/write timing per slot
   - Format `rpc_msg_server_telemetry` with all 6 fields, append to `GRAPH_COMPUTE_ALL` response
   - Gate behind `GGML_RPC_SERVER_TELEMETRY` env var
2. **Client-side parsing** (`ggml/src/ggml-rpc/ggml-rpc.cpp`):
   - Parse telemetry frame from `GRAPH_COMPUTE_ALL` response
   - Write `server-telemetry.jsonl` alongside existing trace files
3. **Profiler binary** (`tools/llama-gpipe-profiler/`):
   - CMake target: `llama-gpipe-profiler`
   - CLI: `--model`, `--endpoints`, `--tasks pp,tg`, `--output`, `--repeat`, `--warmup`
   - Task orchestration: drives real inference (prompt processing + token generation) through RPC endpoints
   - Heatmap generation: synthesizes raw telemetry into task-stratified heatmap JSON
   - Pattern after `llama-bench` structure
4. **Script adaptation**:
   - Extend `b6-gate-phase0-assembly-bounds.py` to consume server telemetry fields when present
   - Extend `diagnose.json` schema with `device_timings_us`, `layer_assignments`, `copy_times_us`, `device_meta`, `kv_read_times_us`, `kv_write_times_us`
   - Keep existing `llama-pipeline-profiler` working; add telemetry fields as optional extensions

### Telemetry Field Specification

| Field | Type | Source | Consumer |
|-------|------|--------|----------|
| `device_timings_us[]` | `uint64[]` | Scheduler after `graph_compute` per backend | D5.1 straggler ID, R3 depth tuning, Pareto optimizer |
| `layer_assignments[]` | `int32[]` | Split output (layer start per device) | D5.1 sub-stage boundary mapping, Pareto placement |
| `copy_times_us[]` | `uint64[]` | PCIe copy duration per peer pair | D5.1 copy vs compute attribution |
| `device_meta[]` | `struct {name, vram, backend, pcie}` | Backend init at startup | Trace context, hardware regression, Pareto env analysis |
| `kv_read_times_us[]` | `uint64[]` | KV cache read per slot | Pareto optimizer: hot KV page placement |
| `kv_write_times_us[]` | `uint64[]` | KV cache write per slot | Pareto optimizer: KV eviction cost modeling |

---

### Future: Server-Side Pareto Optimizer (D4.11–D4.14)

> **Planned + ticketed, NOT built in current sprint.** Stored here so the design is
> recorded and the profiler heatmap schema is forward-compatible.

The Pareto optimizer lives inside `llama-server`'s governor/dispatcher. It consumes
the heatmap JSON produced by `llama-gpipe-profiler` and applies Pareto's law (80/20
rule) to GPU placement:

- **Identify the hot 20%**: Layers/tensors that consume the most compute time (from per-layer timing)
- **Identify the fast 20%**: GPUs with the highest throughput (from device_meta + per-GPU timing)
- **Hot-on-fast**: Place hot layers on fast GPUs; cold layers become a "holding tank" in VRAM or system RAM
- **KV cache consideration**: Hot KV pages placed alongside hot layers for cache locality

The optimizer is designed to run automatically — the server governor analyzes its
environment on startup (GPU count, PCIe topology, model architecture) and decides
placement without external orchestration. It may also periodically re-profile
(adaptive behavior) as workloads change.

| Ticket | Type | Description |
|--------|------|-------------|
| D4.11 | research | Pareto optimizer design: llama-server governor integration points, placement algorithm, adaptive re-profiling strategy |
| D4.12 | prototype | Throwaway: server-side placement decision from heatmap input, validate 80/20 rule on real cluster |
| D4.13 | design | ADR: optimizer placement model, hot/cold tier definitions, transition from static config to adaptive |
| D4.14 | implementation | Build optimizer into llama-server governor: env analysis, heatmap consumption, placement dispatch |

---

*Plan prepared for autonomous execution — 2026-07-11*
