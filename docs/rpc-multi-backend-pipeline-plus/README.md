# rpc-multi-backend-pipeline-plus

**Documentation root for Path-B-Event-Support-Pipeline-Plus enhanced multi-backend RPC orchestration.**

This is the canonical home for analysis, planning, tracking, and artifacts related to hybrid (local CUDA + remote RPC) llama-server deployments using the current `Path-B-Event-Support-Pipeline-Plus` state.

**Recommended production topology (as of 2026-06-29):** `trace-f-2gpu-plus` — 5070 Ti + remus 5060 Ti (`ts=50,50`, `Plus=1`) delivering **48.9 t/s** on Qwen3.6-35B-A3B-UD-IQ4_NL_XL.

## Quick links

- [MISSION.md](MISSION.md) — Goals, success criteria, and problem statement
- [PLAN.md](PLAN.md) — Phased execution plan (Path-B+ → overlap gate → Path-C bridge)
- [TRACKING.md](TRACKING.md) — Current status, open items, decisions
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — B+8..B+13, C-full schema, Phase 1.2 work queue
- [FUTURE-EXPANSIONS.md](FUTURE-EXPANSIONS.md) — deferred work (C-full sample API, post-B+15 perf hunt)
- [ANALYSIS/Path-B-Plus-MultiBackend-RPC-Orchestration-Audit.md](ANALYSIS/Path-B-Plus-MultiBackend-RPC-Orchestration-Audit.md) — Core technical audit of serial workflow and sync blockers
- [BENCHMARKS/](BENCHMARKS/) — Reference runs, matrix summaries, profile artifacts
- Related mission docs (same branch):
  - [rpc-patch/docs/rpc-path-b-plus-overview.md](../../rpc-patch/docs/rpc-path-b-plus-overview.md)
  - [rpc-patch/docs/b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md) — B+6 overlap gate living checklist
  - [rpc-patch/docs/pathb-sync-site-audit.md](../../rpc-patch/docs/pathb-sync-site-audit.md) — B+7+ candidate table

## Scope

- Multi-backend = local CUDA (Windows 5070 Ti or Linux) + 1..N RPC workers (remus 5060 Ti, RX6600, romulus cluster, etc.)
- Focus: Event-driven RPC (Path B) + pipeline parallelism (`Plus` mode, `graph splits`, `sched_reserve`)
- Primary concern: Serial RPC critical path and sync blockers that cause burst-idle GPU behavior
- Active gate: **B+6** — `overlap_pct >= 5%` (M3) without Path C server aggregation
- Out of scope (for now): Full Path-C server-side aggregation, continuous batching, speculative decoding

## How to contribute / update

1. Run new profile/matrix with `-Profile` flag or `llama-pipeline-profiler` presets (`b6-*`)
2. Add artifact bundle under [BENCHMARKS/](BENCHMARKS/) or reference `benches/path-b-plus/`
3. Update [TRACKING.md](TRACKING.md) with new measurements and decisions
4. Mirror gate milestones in [rpc-patch/docs/b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md)
5. Evolve [PLAN.md](PLAN.md) as mitigation experiments land

---

**Status:** Active development (`Path-B-Event-Support-Pipeline-Plus` branch)  
**Last updated:** 2026-07-01  
**Owner:** NighmareGit (with Grok analysis support)