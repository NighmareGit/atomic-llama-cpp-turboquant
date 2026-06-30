# MISSION — rpc-multi-backend-pipeline-plus

## Problem Statement

Even after Path B event support and pipeline parallelism (`Plus` mode), llama-server multi-backend orchestration (local CUDA + RPC workers) remains dominated by a **serial RPC critical path**.

This manifests as:

- Burst-then-idle GPU utilization patterns (high instantaneous power/util during compute segments, near-zero between RPC round-trips)
- 7–22+ blocking RPC RTTs per token still present
- Sub-linear scaling when adding devices (RX6600 third hop actively harmful for 35–36B A3B MoE)
- Low average `nvidia-smi` utilization despite high VRAM occupancy
- **`overlap_pct` stuck at 0.1–0.7%** after B+1–B+6 and partial B+7 (B+6 gate FAIL, 2026-06-29)

The current `Path-B-Event-Support-Pipeline-Plus` state delivers excellent incremental gains (legacy 39.3 t/s → 48.9 t/s on optimal 2-device), but the fundamental orchestration model is still client-serial until copy-slot pipelining stays alive across RPC RTTs.

## Mission

**Deliver a production-grade, well-instrumented multi-backend RPC orchestration layer** that:

1. Makes the best possible use of `Path-B-Event-Support-Pipeline-Plus` (events + explicit pipeline stages / graph splits).
2. Clearly quantifies and mitigates the remaining serial sync blockers **within Path-B+ bounds** (no Path C server aggregation until explicitly bridged).
3. **Passes the B+6 overlap gate** — `overlap_pct >= 5%` (M3) on measured topologies — by untangling client→RPC I/O and scheduler copy-wait without parallelizing splits within a token.
4. Provides topology-aware defaults, profiling harnesses, and decision frameworks so operators can choose optimal 2-device vs 3-device vs 4-GPU cluster configurations without guesswork.
5. Creates a clean bridge to `Path-C-Distributed-Orchestration` only after Path-B+ mitigation ladder is exhausted (deferred as long as possible for upstream compatibility).

## Success Criteria (measurable)

| Criterion | Target | Current (2026-06-30) | Notes |
|-----------|--------|----------------------|-------|
| Optimal 36B NL MoE throughput (2-device) | ≥48 t/s sustained | **48.9 t/s** (`trace-f-2gpu-plus`) | **Achieved** |
| 3-device vs 2-device penalty | <10% regression | +32% uplift by dropping 6600 | **Achieved** — do not use 3-device for this model class |
| **B+6 M3 overlap gate** | **`overlap_pct >= 5%`** | **0.1–0.7%** (best G2 n=128 only) | **Primary active gap** |
| B+6 M1 interim | `overlap_pct >= 1%` @ n=384 | 0.7% @ n=128 only | FAIL |
| GPU power duty cycle @ ≥20% TDP during gen | >15% | 3.8% (best run) | Orchestration stall signature |
| Per-split / per-RPC timing visibility | Full histogram in scheduler trace | Partial (`llama-pipeline-profiler`, sched trace) | Phase 1.1 |
| Documentation & runbook completeness | All common topologies covered | This doc root + `rpc-patch/` | **In progress** |
| Path-C readiness | Stable 4-GPU baseline + handoff points | `trace-g-4gpu-primary` @ ~43 t/s | Ready; **do not start until B+ ladder exhausted** |

## Non-Goals (for this phase)

- Implementing continuous batching or speculative decoding
- **Path C server-side aggregation** (unified multi-GPU rpc-server, parallel split loop) — explicitly deferred
- Rewriting the core llama.cpp scheduler beyond `Plus`-flagged barrier/copy-wait narrowing
- Supporting VL / embedding / FIM workloads (focus remains chat / completion GGUFs ≥9B)

## Guiding Principles

- **Measure first, optimize second** — every claim backed by profiler `diagnose.json`, `nvidia-smi` CSV, power duty cycle, and artifact bundles.
- **Topology is king** — the right number of devices + correct tensor split beats "more devices".
- **Honest documentation** — explicitly call out what Path B events + Pipeline-Plus bought us and what they did not.
- **Incremental & shippable** — small, testable bisects in `ggml-rpc.cpp` / `ggml-backend.cpp`; one fix per profiler re-run.
- **Path C is last resort** — preserve fork/mother-repo compatibility until M3 fails after B+7–B+13 ladder.

## Relationship to Existing Work

- Builds on [docs/cuda-windows-5070ti/PROFILING.md](../cuda-windows-5070ti/PROFILING.md) and `trace-f-*-plus` / `profile-*-no6600` runs.
- Complements [rpc-patch/](../../rpc-patch/) protocol and Linux harness work.
- B+6 gate execution: [rpc-patch/docs/b6-gate/](../../rpc-patch/docs/b6-gate/).
- Sync-site audit (B+7+): [rpc-patch/docs/pathb-sync-site-audit.md](../../rpc-patch/docs/pathb-sync-site-audit.md).

**Owner:** NighmareGit  
**Review cadence:** After every major profile/profiler run or topology change.