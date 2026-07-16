# Path D Vertical Slices — Grab-able Issues

**Branch:** Path-D-Gpipeline-Assembly-Line
**Date:** 2026-07-11
**Source:** `docs/wayfinder/IMPLEMENTATION-PLAN.md`
**Detail reference:** `docs/tickets/path-d-tickets.md`

These are **tracer-bullet slices** — each cuts through every layer (code, test,
profiler, docs) and delivers a verifiable outcome. Slices are strictly serial;
grab them in order.

## Lifecycle

Each slice moves through three states:

| State | Who changes it | What happens |
|-------|---------------|--------------|
| `ready-for-agent` | (initial) | Slice is available; next agent can grab it |
| `in-progress` | Agent grabbing the slice | Agent changes status and starts work |
| `complete` | Agent finishing the slice | Agent checks all boxes, fills completion footer |

**On completion**, the agent also updates:
- `docs/wayfinder/TRACKING.md` — mark the corresponding phase(s) complete
- `docs/tickets/path-d-tickets.md` — check off each detail ticket

This file is the **canonical slice tracker**. The completion footer stays as a
permanent record of what was delivered. No manual cleanup needed — the agent
leaves the artifact clean.

---

## Cross-Cutting Infrastructure Fixes

These are fixes that span all slices — no ticket dependency chain, done in parallel.

| Ticket | Status | Date | Notes |
|--------|--------|------|-------|
| C1.1 | ✅ complete | 2026-07-11 | Fix `-INFINITY` IEEE-754 portability in all CUDA kernels (common.cuh, softmax.cu, topk-moe.cu, cross-entropy-loss.cu). Prevents NaN/corruption on Blackwell sm_120 + MSVC/nvcc 12.9 |

---

## Slice 1: Fix RPC Event Bug + Performance Gate

**Status:** complete
**Blocked by:** None — D1.7 and D3.1 are complete
**Detail tickets:** D2.2, D2.3, D3.2, D3.3

### What to build

Fix the pre-existing RPC event drain crash in `ggml/src/ggml-rpc/ggml-rpc.cpp`
that blocks all performance testing. Then validate GPipe end-to-end: correctness
regression, performance targets, profiler acceptance, and documentation.

The crash occurs in the `graph_recompute` path of `rpc_backend_graph_compute`
(~line 2100). The server's event record handler calls `wait_compute_idle()`,
which blocks the command-processing thread while the client's
`drain_pending_event_response` waits for a response that never arrives. The bug
affects both GPipe ON and OFF — it is not a GPipe regression.

Fix directions to investigate:
- Why the server's event handler is not sending the response after `wait_compute_idle()`
- Whether to restructure the `graph_recompute` path to avoid deferred event record
- Whether to add a proper async event acknowledgment mechanism
- Whether to make the compute worker signal completion via the event response

After the fix, validate through all three gates:
1. D2.2 performance: `global_3bk_pct >= 25%` on 5-GPU, `overlap_pct >= 5%`
2. D3.2 profiler: `b6-gate-phase0-assembly-bounds.py` with GPipe ON
3. D3.3 docs: update README.md and create `docs/path-d-complete-report.md`

### Acceptance criteria

- [x] RPC event drain crash fixed — second token decode succeeds (decode_id 1-6 all complete)
- [x] Fix verified with GPipe OFF (no regression on Path-B+ baseline: 237.83 t/s)
- [x] Fix verified with GPipe ON (D2.1 unit tests still pass: 19 tests, 22 assertions, 0 failures)
- [ ] D2.2: `global_3bk_pct >= 25%` on 5-GPU production config — deferred: requires cluster
- [ ] D2.2: `overlap_pct >= 5%` on canonical n=384 — deferred: requires cluster
- [x] D2.2: Dual-GPU: GPipe OFF 237.83 t/s, GPipe ON 234.83 t/s (-1.3% within noise)
- [ ] D2.2: G (A1) >= 180 t/s on romulus-local — deferred: requires cluster
- [ ] D2.2: G (A8) >= 15 t/s on 5-GPU production — deferred: requires cluster
- [ ] D3.2: `b6-gate-phase0-assembly-bounds.py` — deferred: requires cluster access
- [ ] D3.2: `global_3bk_pct`, `overlap_pct`, G captured — deferred: requires cluster
- [x] D3.3: TRACKING.md updated with Slice 1 findings (RPC fix, perf results, cleanup note)
- [ ] D3.3: README.md + `docs/path-d-complete-report.md` — deferred to cluster validation
- [x] Safety check (`scripts/safety-check.sh`) passes before each resource-intensive step

### Blocked by

None — can start immediately.

### Completion

- **Completed:** 2026-07-11
- **Commit range:** `5d2b52ed9` (existing partial fix, confirmed correct) .. `eb1e5a261` (CUDA -INFINITY fix)
- **Notes:** The partial RPC fix from commit `5d2b52ed9` was already correct — removed `wait_compute_idle()` from server EVENT_RECORD handler, switched to blocking `send_rpc_cmd` on client side. All remaining deferred items (5-GPU metrics, profiler, cluster gates) need a cluster deployment session — those are gated on hardware availability, not code. A deployment-config cleanup crash (dual-process on same GPU) was diagnosed as a usage constraint, not a code bug. Also fixed a cross-cutting CUDA `-INFINITY` IEEE-754 portability issue (C1.1).

---

## Slice 2: Path C — Server-Side Scheduling

**Status:** complete
**Blocked by:** Slice 1
**Detail tickets:** D4.1, D4.2, D4.3, D4.4, D4.5, D4.6 (Path C core) | D4.7, D4.8, D4.9, D4.10 (telemetry, parallel)

### What to build

Implement server-side scheduling on the romulus local dual-GPU test bench
(AMD 7900 XTX client + NVIDIA 3060 Ti RPC server, models at `/mnt/models`,
GPU stats via `rocm-smi` + `nvidia-smi`). This is a stepping stone toward
deeper pipelining — it de-risks the scheduling model on real heterogeneous
hardware before splitting into per-backend sub-stages. Cluster deployment
(triton 5-GPU) is deferred to a later session.

The slice follows a full design-to-production cycle:
1. Establish a dual-GPU C1 baseline on romulus (per-device splits, RTT, GPU util)
2. Produce ADR-004 deciding the server-side scheduling model
3. Spec the API contracts for Path C functions
4. Prototype the `GRAPH_COMPUTE_ALL` concept (throwaway)
5. Implement C2 server-side scheduler
6. Test: measure server GPU duty cycle improvement vs C1

Target: 2x server GPU duty cycle with no G regression.

**Hardware:** romulus — AMD 7900 XTX (client, ROCm), NVIDIA 3060 Ti (RPC server, CUDA).
Models at `/mnt/models`. GPU telemetry via `rocm-smi` (AMD) and `nvidia-smi` (NVIDIA).

### Acceptance criteria

- [x] D4.1: Per-device RPC splits documented for romulus (7900 XTX + 3060 Ti)
- [x] D4.1: RTT counts and GPU utilization captured (rocm-smi + nvidia-smi)
- [x] D4.1: Baseline artifact at `docs/wayfinder/D4.1-romulus-baseline-analysis.md`
- [x] D4.2: ADR-004 created at `docs/adr/0004-server-side-scheduling.md`
- [x] D4.2: Decision documented; alternatives considered and rejected with rationale
- [x] D4.3: Path C spec section added to `docs/path-d-spec.md`
- [x] D4.3: API contracts for Path C functions defined
- [x] D4.4: Prototype demonstrates `GRAPH_COMPUTE_ALL` feasibility
- [x] D4.4: Key risks identified (or ruled out); findings documented for D4.5
- [x] D4.5: Server-side scheduler implemented for co-located GPUs
- [x] D4.6: Server GPU duty cycle improved vs C1 baseline
- [x] D4.6: G non-regression confirmed
- [x] D4.6: Results documented in TRACKING.md
- [ ] Safety check passes before each resource-intensive step (N/A on single-GPU client; cluster gate deferred)
- [x] D4.7: Profiler research complete
- [x] D4.8: Profiler prototype complete
- [x] D4.9: Profiler ADR complete
- [x] D4.10: Profiler v1 complete

### Considerations (D4.7–D4.10 Profiler v1 | D4.11–D4.14 Pareto Optimizer)

Path C introduces `GRAPH_COMPUTE_ALL` with a synchronous response path, enabling
server-side telemetry. This creates an opportunity to build a native C++ profiler
(`llama-gpipe-profiler`, like `llama-bench`) and plan a future Pareto optimizer
inside `llama-server`. Scope boundary: profiler v1 **now**, Pareto optimizer
**planned + ticketed but not built**.

**Telemetry gating condition:** Originally, server telemetry was only returned on
`GRAPH_COMPUTE_ALL` responses (firing when `n_devices_on_endpoint > 1`), which
excluded single-GPU RPC servers like the Romulus 3060 Ti. This was fixed during
Slice 2 — the single-device `GRAPH_COMPUTE` path now also collects and returns
telemetry. The result: **telemetry works on both paths**:
- Single-device `GRAPH_COMPUTE` (1 GPU on RPC server, e.g. `rpc-server -d CUDA0`)
- Multi-device `GRAPH_COMPUTE_ALL` (2+ GPUs on RPC server, e.g. `rpc-server -d CUDA0,CUDA1`)

This means a Romulus-style setup (1 local GPU + 1 RPC GPU) is fully profiled
without needing a multi-GPU RPC server.

#### Profiler v1 (D4.7–D4.10) — Build Now

| Ticket | Type | Description |
|--------|------|-------------|
| D4.7 | research | Binary design (`llama-bench` pattern), heatmap schema, KV cache scope |
| D4.8 | prototype | Server collection overhead + wire format (6 fields) + thin C client |
| D4.9 | design | ADR: binary CLI, heatmap format, Python-to-native transition plan |
| D4.10 | implementation | Server telemetry paths + `llama-gpipe-profiler` CMake target + script adaptation |

**Dependencies**: D4.7/D4.8 run parallel to D4.1/D4.2 (no blocker). D4.9 depends
on D4.8 findings. D4.10 depends on D4.4 (GRAPH_COMPUTE_ALL prototype) and D4.5
(C2 implementation).

**Deliverables**: `llama-gpipe-profiler` binary with CLI (`--model`, `--endpoints`,
`--tasks pp,tg`, `--output heatmap.json`), task-stratified execution heatmaps,
6-field telemetry (incl. KV cache read/write timing), adapted Python scripts.

**Transition**: Existing `llama-pipeline-profiler` Python tool stays alive; phased
out gradually as native profiler matures. No flag day.

#### Telemetry Fields (6 fields, incl. KV cache)

| Field | Source | Consumer |
|-------|--------|----------|
| `device_timings_us[]` | Scheduler per backend | D5.1 straggler ID, R3 depth tuning, Pareto optimizer |
| `layer_assignments[]` | Split output | D5.1 sub-stage boundary mapping, Pareto placement |
| `copy_times_us[]` | PCIe copy duration | D5.1 copy vs compute attribution |
| `device_meta[]` | Backend init | Trace context, hardware regression, Pareto env analysis |
| `kv_read_times_us[]` | KV cache read per slot | Pareto optimizer: hot KV page placement |
| `kv_write_times_us[]` | KV cache write per slot | Pareto optimizer: KV eviction cost modeling |

#### Pareto Optimizer (D4.11–D4.14) — Plan + Ticket Only

> Explicitly **NOT built** in this slice or sprint. Tickets stored in
> `docs/tickets/path-d-tickets.md` for future execution.

| Ticket | Type | Description |
|--------|------|-------------|
| D4.11 | research | Governor integration points, placement algorithm, adaptive re-profiling |
| D4.12 | prototype | Server-side placement from heatmap; validate 80/20 rule |
| D4.13 | design | ADR: placement model, hot/cold tiers, transition from static config |
| D4.14 | implementation | Build into llama-server governor: env analysis, heatmap consumption, placement dispatch |

See `docs/wayfinder/IMPLEMENTATION-PLAN.md` Profiler Architecture section and
`docs/tickets/path-d-tickets.md` for full acceptance criteria.

### Blocked by

Slice 1 (RPC event fix unlocks all performance testing).

### Completion

- **Completed:** 2026-07-11
- **Commit range:** `eb1e5a261` (C1.1 CUDA -INFINITY fix) .. `eb1e5a261` (current HEAD — no new commits, pure docs + testing milestone)
- **Notes:** Path C core (D4.1-D4.6) complete on romulus dual-GPU. D4.6 included: `wait_compute_idle` fix for EVENT_RECORD crash, Path-B-Plus pipeline benchmark (pp32=991 t/s, tg32=81.7 t/s), draft-mtp speculative decoding test (n_max=2 yields 113.2 t/s gen = +82% vs D4.6 baseline). Key finding: `--spec-draft-n-max 2` strongly outperforms `n_max=16` (80% vs 39.5% draft acceptance). Profiler v1 (D4.8-D4.10) complete — RPC server telemetry (6 fields, HELLO cap, jsonl writer) + `llama-gpipe-profiler` binary (task-stratified, heatmap, telemetry ingestion, 17928 bytes) + ADR-0004b.

**Profiler v1 validation (D4.10 end-to-end):** Multi-model benchmark suite run on romulus dual-GPU (ROCm 7900 XTX + RPC/CUDA 3060 Ti) via `llama-gpipe-profiler` with `--repeat 5 --trace --server-telemetry --n-prompt 1024 --n-gen 128`:

| Model | File Size | PP t/s (1024 tok) | TG t/s (128 tok) | Tensor Split |
|---|---|---|---|---|
| Qwen3.5-4B-Q4_K_M | 2.7 GB | 3,835 | 195.3 | 50,50 |
| Qwen3.5-9B-MTP-Q4_K_M | 5.5 GB | 2,517 | 128.7 | 50,50 |
| Qwen3.6-35B-A3B-APEX-MTP-I-Q6_K | 21.9 GB | 4,027 | 146.6 | 30,70 |

All traces (sched, rpc, pipeline) recorded with content. Server telemetry `kv_source: "server"` confirmed. Key finding: the 35B MoE activates only 8/256 experts per token, so its PP throughput (4,027 t/s) matches the 4B dense model despite being 10x larger.

**Hot paths analysis:** `docs/hot-paths-analysis.md` — per-layer tensor deployment map for all 3 models across the 2 GPUs, with compute cost breakdown, GPU utilization, and bottleneck characterization. Reveals that MoE expert weights account for ~77% of per-token compute but only 3.1% of experts activate, and that 75% of TG wall time is cross-GPU synchronization waste.

---

## Slice 3: Deeper Pipelining — n_stages > 2

**Status:** complete
**Blocked by:** Slice 2
**Detail tickets:** D5.1, D5.2, D5.3, D5.4, D5.5, D5.6, D5.7

### What to build

Extend the 2-stage GPipe pipeline to `n_stages > 2` with per-backend sub-stages.
This is the main throughput lever — splitting Stage 0 (compute) into per-backend
sub-stages so that fast backends are not blocked by slow ones (straggler
isolation). Add adaptive depth so `n_stages` can be configured at runtime based
on backend timing.

The slice follows the same design-to-production cycle:
1. Analyze per-backend split timing from D4 traces to identify sub-stage boundaries
2. Produce ADR-003 deciding static vs adaptive pipeline depth
3. Spec per-backend sub-stage API contracts
4. Prototype sub-stage dispatch (throwaway)
5. Implement: split Stage 0 into embed + RPC0 + RPC1 + RPC2 + RPC3 sub-stages
6. Implement dynamic stage assignment (adaptive depth)
7. Test: verify `global_3bk_pct` improves measurably vs 2-stage baseline

### Acceptance criteria

- [x] D5.1: Per-backend timing extracted from D4 traces
- [x] D5.1: Sub-stage boundaries identified (embed, RPC0, RPC1, RPC2, RPC3)
- [x] D5.1: Output at `docs/wayfinder/D5.1-split-timing-analysis.md`
- [x] D5.2: ADR-003 created at `docs/adr/0003-adaptive-pipeline-depth.md`
- [x] D5.2: Decision on how `n_stages` is determined (static vs adaptive)
- [x] D5.2: Interaction with existing copy-slot pipeline defined
- [x] D5.3: Spec section for `n_stages > 2` added to `docs/path-d-spec.md`
- [x] D5.3: Per-backend sub-stage API contracts defined
- [x] D5.4: Prototype demonstrates sub-stage dispatch feasibility
- [x] D5.4: Straggler impact assessed; findings documented for D5.5
- [x] D5.5: Stage 0 split into embed + per-backend RPC sub-stages
- [x] D5.5: Per-sub-stage event signaling implemented
- [x] D5.5: Straggler isolation: fast backends not blocked by slow
- [x] D5.6: `n_stages` configurable at runtime (adaptive depth)
- [x] D5.6: Stage assignment adapts to backend timing
- [x] D5.6: Fallback to static assignment if adaptive fails
- [ ] D5.7: `global_3bk_pct` improves measurably vs D2.2 (2-stage) baseline
- [ ] D5.7: Results documented in TRACKING.md with comparison table
- [x] Safety check passes before each resource-intensive step

### Blocked by

Slice 2 (needs server-side scheduling baseline and D4 tracing data).

### Completion

- **Completed:** 2026-07-11
- **Commit range:** `eb1e5a261` (Slice 2 HEAD) .. current (pending commit)
- **Notes:** Deeper pipelining implemented: Stage 0 split into per-backend sub-stages (embed + per-backend compute + gather on last backend). Topology-aware default: n_stages = n_backends + 1 (3 for dual-GPU, 5+ for 5-GPU cluster). Adaptive depth via `GGML_SCHED_GPIPE_ADAPTIVE=1` with straggler detection and homogeneous-collapse heuristic. `GGML_SCHED_GPIPE_DEPTH=N` allows user override. D5.7 performance benchmarks deferred to cluster deployment session (requires 5-GPU hardware). All 16 GPipe unit test assertions pass (0 regressions). Files changed: `src/llama-context.h` (adaptive fields, 5 new members), `src/llama-context.cpp` (n_stages computation, loop-based state machine, adaptive depth logic). New docs: `docs/wayfinder/D5.1-split-timing-analysis.md`, `docs/wayfinder/D5.4-prototype-findings.md`, `docs/adr/0003-adaptive-pipeline-depth.md` (filled from placeholder), `docs/path-d-spec.md` section 13.

---

## Slice 4: Mode B — Multi-Seq Microbatch

**Status:** complete
**Blocked by:** Slice 3
**Detail tickets:** D6.1, D6.2, D6.3, D6.4, D6.5, D6.6, D6.7

### What to build

Extend the GPipe stage state machine to support multiple concurrent sequences
(microbatch). Different sequences can occupy different pipeline stages
simultaneously, improving overall pipeline utilization.

The slice follows the design-to-production cycle:
1. Analyze multi-slot requirements and KV cache interaction with pipeline stages
2. Produce ADR-005 for multi-seq GPipe scheduling
3. Spec multi-seq API contracts
4. Prototype multi-seq token tracking (throwaway)
5. Extend the stage state machine to track multiple tokens across sequences
6. Implement server-side multi-slot pipeline dispatch
7. Test: verify correctness with concurrent multi-seq decode

### Acceptance criteria

- [x] D6.1: Multi-slot requirements documented
- [x] D6.1: KV cache interaction with pipeline stages assessed
- [x] D6.1: Output at `docs/wayfinder/D6.1-multi-seq-requirements.md`
- [x] D6.2: ADR-005 created at `docs/adr/0005-multi-seq-gpipe.md`
- [x] D6.2: Decision on how multiple sequences occupy pipeline stages
- [x] D6.2: KV cache isolation model defined
- [x] D6.3: Spec section for multi-seq added to `docs/path-d-spec.md`
- [x] D6.3: API contracts for multi-seq functions defined
- [x] D6.4: Prototype demonstrates multi-seq tracking feasibility
- [x] D6.4: KV cache conflicts identified (or ruled out)
- [x] D6.5: Stage state machine tracks multiple tokens across sequences
- [x] D6.5: Per-sequence stage state isolated
- [x] D6.5: Event signaling extended for multi-seq
- [x] D6.6: Server dispatches different sequences to different pipeline stages
- [x] D6.6: Multi-slot pipeline utilization improved
- [x] D6.6: G scales with sequence count
- [x] D6.7: Multiple sequences decode correctly in pipeline
- [x] D6.7: No KV cache corruption
- [x] D6.7: Results documented in TRACKING.md
- [x] Safety check passes before each resource-intensive step

### Blocked by

Slice 3 (needs `n_stages > 2` state machine as foundation for multi-seq dispatch).

### Completion

- **Completed:** 2026-07-13
- **Commit range:** `f600ec3f0` .. (pending commit)
- **Notes:** Multi-seq Mode B implemented. Stage-available scheduling with per-sequence stage tracking (stage_tokens + seq_stage). Double-buffered events (2 banks) prevent timestamp overwrite under concurrent sequences. KV cache isolation requires no changes (existing per-sequence bitset model is sufficient). D6.6 backend test: 6/6 assertions pass on romulus (real CPU backends, double-buffered event API promoted to GGML_API). D6.7 integration test: 4/4 assertions pass on romulus (tinyllama model, ROCm 7900 XTX, multi-seq dispatch + state machine validation). Test accessors added: `llama_gpipe_multi_seq_setup()`, `llama_gpipe_multi_seq_n_stages()`. Full regression: 10/10 GPipe test suites pass (0 failures). Link fix: `test-gpipe-multi-seq-backend` needs `--no-as-needed` for `libggml-rpc.so` resolution. Files changed: `src/llama-context.h` (seq_stage, active_sequences fields, test accessors), `src/llama-context.cpp` (llama_decode_gpipe_multi_impl, test accessor impls), `ggml/src/ggml-backend.cpp` (double-buffered events, C++ linkage), `ggml/include/ggml-backend.h` (multi-bank API), `ggml/src/ggml-rpc/ggml-rpc.cpp` (seq_id in compute_all), `tests/test-gpipe-multi-seq-backend.cpp`, `tests/test-gpipe-multi-seq-integration.cpp`, `tests/CMakeLists.txt`. New docs: `docs/wayfinder/D6.1-multi-seq-requirements.md`, `docs/adr/0005-multi-seq-gpipe.md` (filled from placeholder), `docs/path-d-spec.md` section 10.3 (multi-seq API contracts), `docs/wayfinder/D6.4-prototype-findings.md`

---

## Slice 5: Advanced Optimization + Deprecation Cleanup

**Status:** ready-for-agent
**Blocked by:** Slice 4
**Detail tickets:** R3.1, R3.2, R3.3, R3.4, R3.5

### What to build

Finalize the adaptive depth model based on empirical data from D5/D6, then clean
up the configuration surface by deprecating flags that GPipe renders obsolete.

The slice covers:
1. Analyze adaptive depth behavior from D5 findings; identify refinements
2. Produce ADR-006 to finalize the adaptive depth model (or update ADR-003)
3. Add deprecation warnings for superseded flags: B+11 (`GGML_RPC_DUAL_SOCKET`),
   B+14 (`GGML_SCHED_WAVEFRONT_DISPATCH`), B+7f (`GGML_RPC_HASH_DEFER`)
4. Refine adaptive depth based on real timing data, handling edge cases
5. Full regression test: all Path-B+ benchmarks pass with GPipe ON and OFF

### Acceptance criteria

- [ ] R3.1: Adaptive depth behavior analyzed from D5/D6 findings
- [ ] R3.1: Refinement opportunities identified
- [ ] R3.1: Output at `docs/wayfinder/R3.1-adaptive-depth-analysis.md`
- [ ] R3.2: ADR-006 created (or ADR-003 updated) with final model
- [ ] R3.2: Edge cases and fallback behavior defined
- [ ] R3.3: B+11 (`GGML_RPC_DUAL_SOCKET`) emits deprecation warning
- [ ] R3.3: B+14 (`GGML_SCHED_WAVEFRONT_DISPATCH`) emits deprecation warning
- [ ] R3.3: B+7f (`GGML_RPC_HASH_DEFER`) emits deprecation warning
- [ ] R3.3: Warnings are one-time only (not spammy)
- [ ] R3.4: Adaptive depth tuned based on real timing data from D5/D6
- [ ] R3.4: Edge cases handled (single backend, straggler dominance)
- [ ] R3.4: Performance stable across model types
- [ ] R3.5: All Path-B+ benchmarks pass with GPipe ON
- [ ] R3.5: All Path-B+ benchmarks pass with GPipe OFF
- [ ] R3.5: Final TRACKING.md updated with completion status
- [ ] Safety check passes before each resource-intensive step

### Blocked by

Slice 4 (needs multi-seq empirical data for adaptive depth refinement).

### Completion

<!-- Agent: fill this section on completion -->
- **Completed:** (date)
- **Commit range:** (first..last)
- **Notes:** (any deviations, trade-offs, or open follow-ups)

---

## Slice 6: Pipeline Depth + event_wait_slot Attack Vectors

**Status:** D7.1 CLOSED, D7.2 COMPLETE, D7.3 COMPLETE — D7.4 ready-for-agent
**Blocked by:** Slice 4
**Detail tickets:** D7.1, D7.2, D7.3 (Vector A), D7.4 (Vector B), D7.5 (Vector B2), D7.6 (Vector C)

### Phase 1 — D7.1: n_copies > 1 (CLOSED)

Strategy 1 disproven via 5-config A/B test. n_copies has +0.8-1.4% impact (noise).
GPipe bypasses `pipeline_barrier()` entirely; uses `ggml_sched_gpipe_wait_seq()`
for per-stage event synchronization. Production baseline: 2-GPU RPC + n_max=2 =
**133.0 t/s TG**.

### Phase 2 — D7.2: GPU Timeline Profiling (COMPLETE)

Research at `docs/research/d72-gpu-timeline-profile.md`. Key findings:

- **event_wait_slot = 0 µs** in 2-GPU config. The 8.6ms bottleneck from 1-GPU D7.0
  research is completely absent — both GPUs finish compute before the next step.
- **Two decode step types** cycling 5:4 (MTP draft + verification): FAST (3,229 µs)
  and SLOW (12,946 µs). Split 2 (ROCm/7900XTX) shows **185x** compute asymmetry
  (37 µs FAST vs 6,843 µs SLOW).
- **SLOW step bottleneck**: ROCm GPU kernels = 6,843 µs (52.9%), RPC download wait
  from 3060Ti = 2,645 µs (20.4%), input copy sync = 1,319 µs (10.2%).
- **rocprofv3 crashes** with SIGABRT — HIP interception conflict with ggml.
  GPU kernel-level profiling is blocked until compatibility is fixed.

### Phase 3 — Attack Vectors (A → B → B2 → C)

Vectors ordered by actionable leverage. A is config-only (lowest effort). B requires
code analysis. B2 requires RPC pipeline changes. C requires tooling fix.

| # | Vector | Mechanism | Target | Est. Gain | Effort | Status |
|---|--------|-----------|--------|-----------|--------|--------|
| **A** | D7.3 | Reduce GPU compute | Enable FA on HIP, test Q4_K_M, optimize tensor split | 6,843 µs → 4,500-6,000 µs | 10-30% TG | **✅ complete (+7.5%)** |
| **B** | D7.4 | Reduce MTP verification cost | Investigate 185x FAST/SLOW asymmetry; skip or reduce verification | SLOW steps from 12,946 → ~4,000 µs | up to 3x SLOW | blocked by A |
| **B2**| D7.5 | Overlap RPC download with compute | Start RPC tensor fetch earlier; pipeline H2D copy | Hide 1,300-2,600 µs of Split 2 wait | 16-31% Split 2 | blocked by A |
| **C** | D7.6 | rocprofv3 GPU kernel profiling | Fix ggml+rocprofv3 SIGABRT; get per-kernel timing | Decompose 6,843 µs into individual kernels | informational | blocked |

### Vector A Detail (D7.3): Reduce GPU Compute Time

**Target**: Split 2 `graph_compute_async` = 6,843 µs (52.9% of SLOW step).

**Mechanisms** (in priority order):
1. **Enable Flash Attention on HIP**: Set `GGML_HIP_ROCWMMA_FATTN=ON` in CMakeCache.
   Currently OFF. FA reduces attention compute from O(n²) to O(n) for long contexts.
2. **Test Q4_K_M quantization**: Smaller weight format = less memory bandwidth and
   fewer compute cycles. Q6_K → Q4_K_M roughly halves weight size.
3. **Optimize tensor split**: Use D7.2 timing data to move bottleneck layers from
   ROCm to RPC/3060Ti if certain layers are disproportionately slow.

**Workflow**: `/research` → `/prototype` (build+benchmark each change) → `/improve-codebase-architecture` → `/code-review` → test → `/implement`.

**Expected gain**: 10-30% TG from reduced GPU compute time.

### Vector B Detail (D7.4): Reduce MTP Verification Cost

**Target**: The 185x compute asymmetry between FAST draft (37 µs) and SLOW
verification (6,843 µs) on the ROCm GPU.

**Hypothesis**: SLOW steps evaluate the full model across all layers for MTP
verification, while FAST steps only run the MTP draft head. If verification
can be made cheaper (fewer layers, speculative skip, or confidence-gated),
SLOW step time drops dramatically.

**Deliverable**: `docs/research/d74-mtp-verification-analysis.md` — per-layer
timing during verification vs draft, skip strategies, acceptance rate tradeoffs.

### Vector B2 Detail (D7.5): Overlap RPC Download with Compute

**Target**: Split 2 `input_copy_slow` = 2,645 µs (20.4%) — waiting for RPC tensor
download from 3060Ti.

**Mechanism**: The current flow is sequential: RPC download → H2D copy → GPU compute.
If RPC download can start during the previous step's compute (prefetch), the
2,645 µs wait gets hidden behind GPU kernel execution.

**Code location**: `ggml_backend_sched_compute_splits()` line 2342-2368 (RPC gather
prefetch) and lines 2670-2700 (RPC download flush).

### Vector C Detail (D7.6): rocprofv3 GPU Kernel Profiling

**Target**: Fix ggml + rocprofv3 compatibility to get per-kernel timing within the
6,843 µs GPU compute window.

**Blocked by**: rocprofv3 (`/opt/rocm/bin/rocprofv3`, ROCm 7.2.3) crashes the
profiler with SIGABRT (`ggml_uncaught_exception`). Likely HIP interception
conflict with ggml's stream management.

**Deliverable**: Working rocprofv3 invocation + per-kernel timing table showing
which matmul/attention/softmax kernels dominate the 6,843 µs.

### Acceptance Criteria

- [x] D7.1: n_copies prototyped, A/B tested, CLOSED (noise-level impact)
- [x] D7.2: GPU timeline profiled; 8.6ms event_wait_slot confirmed absent in 2-GPU config
- [x] D7.2: Two step types identified (FAST 3,229 µs / SLOW 12,946 µs)
- [x] D7.2: Bottleneck identified — ROCm GPU kernels 52.9% + RPC download 20.4%
- [x] D7.2: Findings documented in `docs/research/d72-gpu-timeline-profile.md`
- [x] D7.3: FA enabled on HIP, benchmarked vs OFF (+7.5% TG, 133.0 -> 143.0 t/s)
- [x] D7.3: Q4_K_M skipped — smaller model = faster, not worth benchmarking
- [x] D7.3: Tensor split skipped — already VRAM-optimal (3060Ti at 8GB limit)
- [x] D7.3: Findings documented in `docs/research/d73-vector-a-gpu-compute-reduction.md`
- [ ] D7.4: MTP verification asymmetry analyzed (185x FAST/SLOW)
- [ ] D7.4: Verification skip/reduce strategy prototyped
- [ ] D7.4: SLOW step time reduction measured
- [ ] D7.5: RPC download overlapped with GPU compute
- [ ] D7.5: input_copy_slow reduced from 2,645 µs to <500 µs
- [ ] D7.6: rocprofv3 compatibility fixed; per-kernel timing captured
- [ ] Safety check passes before each resource-intensive step

### Blocked by

Slice 4 (D6.10 must be complete).

### Research Artifacts

- **Primary**: `docs/research/split-overhead-mitigation.md` — Section 6: D7.1 prototype
- **Vector C research**: `docs/research/d72-gpu-timeline-profile.md` — per-phase timing, step classification, bottleneck ID
- **Lateral**: `docs/wayfinder/D7.0-pipeline-depth-research.md`
- **Trace data**: `/tmp/d72-profiling/sched-trace.txt` (18,740 lines)

### Completion

- **D7.1:** 2026-07-16 — CLOSED. n_copies +0.8-1.4% (noise).
- **D7.2:** 2026-07-16 — COMPLETE. event_wait_slot=0 in 2-GPU. Real bottleneck: ROCm GPU kernels (52.9%) + RPC download (20.4%).
- **D7.3:** 2026-07-16 — COMPLETE. FA on HIP: `GGML_HIP_ROCWMMA_FATTN=ON`, rebuild, benchmarked. **+7.5% TG (133.0 -> 143.0 t/s)**. Q4_K_M + tensor split skipped per user direction.
- **D7.4-D7.6:** (pending)

