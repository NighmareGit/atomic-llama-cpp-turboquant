# rpc-multi-backend-pipeline-plus

**Documentation root for Path-B-Event-Support-Pipeline-Plus enhanced multi-backend RPC orchestration.**

This is the canonical home for analysis, planning, tracking, and artifacts related to hybrid (local CUDA + remote RPC) llama-server deployments using the current `Path-B-Event-Support-Pipeline-Plus` state.

**Recommended production topology (as of 2026-06-29):** `trace-f-2gpu-plus` — 5070 Ti + remus 5060 Ti (`ts=50,50`, `Plus=1`) delivering **48.9 t/s** on Qwen3.6-35B-A3B-UD-IQ4_NL_XL.

## Quick links

- [MISSION.md](MISSION.md) — Goals, success criteria, and problem statement
- [PLAN.md](PLAN.md) — Phased execution plan (Path-B+ → Phase D layer pipeline)
- [DESIGN-path-d-layer-pipeline.md](DESIGN-path-d-layer-pipeline.md) — **Phase D0** GPipe + Path C path forward (grill-backed)
- [TRACKING.md](TRACKING.md) — Current status, open items, decisions
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — B+8..B+13, C-full schema, Phase 1.2 work queue
- [RPC-PROTOCOL.md](RPC-PROTOCOL.md) — wire format, version history, proto 4.4 CHANNEL_BIND
- [FEATURE-b11-dual-socket-rpc.md](FEATURE-b11-dual-socket-rpc.md) — B+11 cmd/rsp split, bisect, ops
- [DESIGN-b13-input-wait-audit.md](DESIGN-b13-input-wait-audit.md) — B+13 split-2 gather wait breakdown
- [FUTURE-EXPANSIONS.md](FUTURE-EXPANSIONS.md) — deferred work (C-full sample API, Phase D backlog)
- Handover: [HANDOVER-SESSION-2026-07-01-phase-d.md](../../rpc-patch/patch/HANDOVER-SESSION-2026-07-01-phase-d.md) — **continue here** for path decision
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
- **Path-B+ gate:** B+6 M3 **closed FAIL** — structural ceiling; branch frozen for production deploy
- **Active mission:** **Phase D0** — layer pipeline / assembly-line saturation ([DESIGN-path-d-layer-pipeline.md](DESIGN-path-d-layer-pipeline.md))
- Out of scope on Path-B+ branch: GPipe sched, Path C implementation (pursued on proposed `Path-D-Layer-Pipeline` branch)

## How to contribute / update

1. Run new profile/matrix with `-Profile` flag or `llama-pipeline-profiler` presets (`b6-*`)
2. Add artifact bundle under [BENCHMARKS/](BENCHMARKS/) or reference `benches/path-b-plus/`
3. Update [TRACKING.md](TRACKING.md) with new measurements and decisions
4. Mirror gate milestones in [rpc-patch/docs/b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md)
5. Evolve [PLAN.md](PLAN.md) as mitigation experiments land

---

**Status:** Path-B+ **deploy-ready** (frozen); Phase D0 **active** (design)  
**Protocol:** RPC v4.4.2 (B+11 dual-socket shipped, default OFF)  
**Last updated:** 2026-07-01 (V6 Path D fork)
**Owner:** NighmareGit (with Grok analysis support)