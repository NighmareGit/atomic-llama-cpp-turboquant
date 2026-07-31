# Path-E: Hydra — Draft Vision (v0.1)

> **Status:** DRAFT — idea collection branch. No implementation. Forked off the canonical ancestor
> (`feature/turboquant-kv-cache` @ `066cc29b4`) so it syncs cleanly when `good-prototype` re-merges.
> **Owner:** hunter + Grok (TOPIC-001 research loop, 2026-07-31)
> **Canonical idea ledger:** workspace `.scratch/research/LEDGER.md` entry 24 (A8)

---

## One-liner

**Streaming speculative expert offload**: run >aggregate-VRAM MoE models across the fleet by keeping only
*predicted-likely* experts in VRAM, streaming the rest from node RAM/SSD hidden behind network/compute
overlap, with multiple speculative routing heads (fed by MTP drafts) deciding what to stage next — a
hydra of compute heads instead of a starfish.

## Why this phase exists

The fleet has 5 GPUs ≈ 80 GB aggregate VRAM. Models on the shelf that exceed it:

| Model | Size | Requirement |
|-------|------|-------------|
| MiniMax-M2.7-REAP-172B | 92 GiB | expert offload required |
| Mixtral-8x22B | 72 GiB | blocked (needs ≥12 GiB per slot) |
| Qwen3.5-122B-A10B | 70 GiB | 5-GPU MoE (retest pending BUG-001a/002a) |

Path-D (layer-split assembly line) peaks at the minimum GPU count that fits a model. Hydra's premise:
for models that fit *nowhere* in aggregate VRAM, the only lever is **weight movement** — and weight
movement can be hidden behind the network/compute pipeline if we know *what to move next*.

## The architecture (components)

1. **Offload buffers:** node system RAM + local SSD as the weight tier; only likely-needed experts in VRAM.
2. **Speculative routing (revives A6):** MTP draft hidden states feed the routers (cheap linear gates) →
   predict expert selection for the upcoming verify batch. A6 was killed for layer-split ("nothing to
   hide at RPC"); Hydra *creates the transfer to hide* → prediction now has economic value.
3. **Network-hidden streaming:** stream predicted experts from RAM/SSD while the network/compute does
   this token's work (A7 pipelining applied to *weights*, not just activations).
4. **CPU yanking:** easy/small experts run on capable node CPUs when it relieves the pipeline head.
5. **Hydra heads:** multiple routing heads deployed on the fastest RPC backends → parallel speculation
   (several candidate expert-sets staged concurrently).
6. **Reassembly:** parallel compute chains must rejoin in order — the hard join problem (reorder-buffer /
   scoreboard discipline; A5's rollback mechanics made physical).

## Lineage

```
Path-B  (event support)      → foundational RPC events
Path-B+ (pipeline-plus)      → pipeline flags, deferred events
Path-C  (distributed orch.)  → multi-node orchestration
Path-D  (gpipeline assembly) → layer-split assembly line (current)
Path-E  (HYDRA)              → streaming speculative expert offload   ← you are here (draft)
```

## Dependency map (what Path-D must deliver first)

| Prerequisite | Vector | Why |
|--------------|--------|-----|
| Stall elimination | NW1 (uid stability) + NW3 (async) | network/compute must be fast enough to hide streaming |
| Correctness | A2′ (shared compute engine) | chain of buggy handoffs worse than starfish |
| Topology | A7 (RPC-to-RPC chain) | the line Hydra's streams flow along |
| Speculation | A5 (MTP-on-RPC, fix BUG-002a) | MTP drafts are the router heads' fuel |

## Catches (recorded, not solved)

1. **New split mode:** Hydra supersedes layer-split → expert-parallel + streaming (Phase-5-tier change).
2. **Bandwidth economics:** SSD ~1–3 GB/s vs PCIe ~30 GB/s — "network hides streaming" holds only if
   selection hit-rate keeps the streaming window ≤ compute time (speculation economics at the weight level).
3. **Hydra join:** N parallel chains reassemble in order — the scoreboard/reorder problem.
4. **A6 revival is conditional:** prediction only pays where a transfer exists — Hydra supplies it, but the
   hit-rate gate (93–97% prior art) still applies per-token.

## Open questions

- Split-mode design: expert shards vs whole experts; placement policy (VRAM/RAM/SSD/CPU tiers).
- Streaming protocol: RPC command for staged expert loads; prefetch window sizing.
- Multi-head dispatch: which routers on which backends; how candidates are deduped/merged.
- Join semantics: barriers vs scoreboard; rollback on misprediction (A5 mechanics).

## References

- Idea ledger: workspace `.scratch/research/LEDGER.md` entry 24 (A8)
- Prior art: `docs/path-e-hydra/research/a6-router-prediction.md` (captured 2026-07-31)
- Roadmap: `docs/path-e-hydra/ROADMAP.md`
- Parent-lineage context: `TURBOQUANT_UPSTREAM_MERGE.md`, `docs/wayfinder/`
