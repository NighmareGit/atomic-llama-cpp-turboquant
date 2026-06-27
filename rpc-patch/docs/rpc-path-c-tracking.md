# Path C: Multi-GPU RPC Aggregation -- Tracking

**Branch:** Path-B-Event-Support-Pipeline-Plus (Phase 6 kickoff)  
**Plan:** [rpc-path-c-plan.md](rpc-path-c-plan.md) | **B+1 tracking:** [rpc-path-b-plus-tracking.md](rpc-path-b-plus-tracking.md)

## Status: PHASE 6 KICKOFF (C1 baseline + feasibility)

### Build: N/A | Test: N/A | Benchmark: baseline cataloged from Phase 5 traces

### Current Phase: C1 (baseline + feasibility)
### Phase Status: IN PROGRESS (6a/6b); C2 not started

---

## Implementation Log

| Date | Phase | Action |
|------|-------|--------|
| 2026-06-27 | C1 / 6a | Client-split baseline cataloged: `trace-f-3gpu-plus` (G=42.8, 4 splits) vs `trace-f-2gpu-plus` (G=48.9, 3 splits) |
| 2026-06-27 | C1 / 6b | Remus feasibility: `:50051` CUDA-only 1-GPU container, `:50052` separate ROCm container; Path C needs C2 aggregation for CUDA+ROCm |

---

## C1 Baseline (from Path-B Plus benches)

| Run | Topology | Client RPC endpoints | Splits | G (t/s) | Role |
|-----|----------|---------------------|--------|---------|------|
| trace-f-3gpu-plus | 5070 + 5060 + 6600 | :50051 + :50052 | 4 | 42.8 | **Path C "before"** (client serializes remus hops) |
| trace-f-2gpu-plus | 5070 + 5060 | :50051 only | 3 | 48.9 | Ops workaround (drop 6600; not Path C) |

Path C target on Config F: one remus endpoint, client sees fewer splits, 6600 VRAM usable without +10 ms/tok hop tax.

---

## Feasibility (remus Config F)

| Component | Today | Path C requirement |
|-----------|-------|-------------------|
| pathb-rpc-remus (:50051) | 1x NVIDIA, `rpc-server -d CUDA0` | Internal multi-GPU sched |
| rx6600-rpc (:50052) | Separate ROCm container | Merge into one logical backend |
| `rpc-server -d CUDA0,CUDA1` | Works for 2 CUDA GPUs same process | Does not cover CUDA + ROCm mix |

**Conclusion:** Remus 5060+6600 needs **C2 server internal scheduler** presenting one logical device. Multi-device flag alone is insufficient.

**Near-term spike host:** Config A/B (2 NVIDIA on one rpc-server) for 6c before remus hetero merge.

---

## Issues Log

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|
| - | - | - | - | - |

---

## Benchmark Results

| Run | Settings | Baseline tok/s | Modified tok/s | Delta |
|-----|----------|----------------|----------------|-------|
| trace-f-3gpu-plus | 3-device client split | 42.8 | - | C1 "before" |
| trace-f-2gpu-plus | 2-device ops | 48.9 | - | Not Path C (topology trim) |

---

## Success Criteria

- [ ] **RTTs per token reduced**: from 7-22 to ~3-5 (one SET_TENSOR batch, one GRAPH_COMPUTE_ALL, one GET_TENSOR)
- [ ] **Correct results**: output matches per-device compute bit-exact (compare token sequence)
- [ ] **Throughput improvement**: measurable gain in cross-GPU RPC scenarios
- [ ] **Backward compatible**: old clients work with new server and vice versa

---

## Next steps

1. **6c:** Spike `rpc-server -d CUDA0,CUDA1` on 2-NVIDIA host; bench vs 2-endpoint client split.
2. **6d / C2:** Server-side `ggml_backend_sched` for graph_compute across server GPUs.
3. **C3:** Single logical backend RPC cap + client fallback.