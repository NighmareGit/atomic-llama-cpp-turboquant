# Path C: Distributed Orchestration & Async RPC Worker Work

**Status:** Planning / Architecture Phase  
**Parent Branch:** `Path-B-Event-Support`  
**Current Focus:** Designing distributed work distribution across RPC workers with metadata-driven workflow.

## Goal

Enable more efficient utilization of multiple RPC workers (and local GPUs) by moving away from strictly serialized, server-orchestrated dispatch toward an asynchronous, metadata-driven workflow model.

This is an evolutionary step beyond Path B (client ↔ single worker pipeline parallelism).

## High-Level Vision

- Phase 1: Server remains central orchestrator but performs **asynchronous forwarding** using rich metadata.
- Later phases: Gradually give workers more autonomy (controlled forwarding → full P2P routing).
- Foundation: Strong metadata + routing layer that supports all phases.

## Current Approach

We are following a structured architecture process:
1. Global overview of topics
2. Rigorous decision-making per topic (grill-me style)
3. Documented decisions in this folder
4. Phased implementation plans

## Folder Contents

| File                        | Purpose                                      | Status      |
|----------------------------|----------------------------------------------|-------------|
| `README.md`                | This file                                    | Done        |
| `OVERVIEW.md`              | Global view + all major topics + recommendations | In Progress |
| `TOPICS_CHECKLIST.md`      | Living checklist of decisions                | Planned     |
| `ARCHITECTURE.md`          | Core data models and design                  | Planned     |
| `PHASE_1_PLAN.md`          | Concrete implementation plan for Phase 1     | Planned     |
| `PHASES_ROADMAP.md`        | Evolution path across phases                 | Planned     |
| `DECISIONS.md`             | Locked decisions from architecture discussions | Planned     |

## How to Contribute / Work on This

1. Branch from `Path-B-Event-Support`
2. Work inside `rpc-patch/distributed-orchestration/`
3. Update documents as decisions are made
4. Only move to implementation once architecture is sufficiently clear

## Related Work

- Path A: Protocol batching & pipelining
- Path B: Event-based pipeline parallelism (client ↔ worker)
- Path C: Distributed orchestration across multiple workers

---

*This is skunkworks / experimental work. Security and production hardening are intentionally deprioritized until the core idea is validated.*