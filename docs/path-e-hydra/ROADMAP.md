# Path-E: Hydra — Draft Roadmap (v0.1)

> **Status:** DRAFT. Milestones are **dependency-honest**: Hydra is the destination tier; each
> Path-D milestone below is a prerequisite that also pays for itself on the current architecture.

## Phase map (from the wayfinder re-run, `.scratch/plans/rpc-architecture-final.md`)

| Phase | Work | Path-D payoff | Hydra unlock |
|-------|------|---------------|--------------|
| P1 | **NW1** (topology-stable graph uids) + **A2′** (shared compute engine) | +42–97% / correctness | fast + correct base to stream against |
| P2 | **NW3** (async GRAPH_COMPUTE) → **NW4** (GET_TENSOR batch, same-host) | +5–15%, N-tax trim | network headroom to hide streaming |
| P3 | **A7** (RPC-to-RPC chain) | O(1) head round-trips | the line the streams travel |
| P4 | **A5** (MTP-on-RPC via BUG-002a double-buffered state) | +37% tokens/step | MTP drafts = router-head fuel |
| P5 | **A3** (connection-independent server) | N-stable multi-client | Hydra's control plane |
| **P6** | **Path-E: Hydra** (streaming expert offload, multi-head) | >aggregate-VRAM models run | — |

## P6 milestones (initial sketch)

1. **M0 — Tiering:** VRAM/RAM/SSD weight tiers; placement policy (frequency-of-use heuristic first).
2. **M1 — Streaming protocol:** RPC staged-expert-load command; prefetch window; overlap with compute.
3. **M2 — Router heads:** MTP-draft-fed prediction on fast backends; hit-rate gate (≥93% target from prior art).
4. **M3 — Hydra dispatch:** multi-head candidates, dedup, staged execution.
5. **M4 — Join:** ordered reassembly (scoreboard); rollback on misprediction (A5 mechanics).
6. **M5 — Models:** Mixtral 72 GB (unblocked), MiniMax 172B (expert offload), Qwen3.5-122B retest.

## Kill criteria (per milestone)

- M1: streaming window > compute window at target hit-rate → economics fail → kill.
- M2: router hit-rate < 90% on real workloads → prediction doesn't pay → kill.
- M4: join overhead > streaming savings → kill.

## Open questions (see VISION.md)

Split-mode design, streaming protocol, multi-head dispatch, join semantics.
