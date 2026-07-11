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
| 2026-07-11 | D4.5/D4.6 | Romulus GRAPH_COMPUTE_ALL implementation and test. Fixed `wait_compute_idle()` removal bug in EVENT_RECORD handler that caused CUDA context corruption. Single-device test confirms no regression (pp32 ~800 t/s, tg32 ~62 t/s). Path-B-Plus + draft-mtp test: pp32=991 t/s (+21%), tg32=81.7 t/s (+32%). draft-mtp n_max=2 gives 113.2 t/s gen throughput (+41.5% vs no-spec, +82% vs D4.6 baseline). n_max=2 strongly preferred over n_max=16 (80% vs 39.5% acceptance). |

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
| 2026-07-11 | RPC_CMD_EVENT_RECORD handler removed `wait_compute_idle()` | D4.5 optimization removed CUDA sync, causing `cudaFuncGetAttributes` to fail with ILLEGAL_ADDRESS on next command | Re-added `server.wait_compute_idle()` to handler | Fixed |

---

## Benchmark Results

| Run | Settings | Baseline tok/s | Modified tok/s | Delta |
|-----|----------|----------------|----------------|-------|
| trace-f-3gpu-plus | 3-device client split | 42.8 | - | C1 "before" |
| trace-f-2gpu-plus | 2-device ops | 48.9 | - | Not Path C (topology trim) |
| 2026-07-11 D4.6 (romulus) | D4.5 client + fixed server, 9B MTP, ts=45 | 816 pp32 / 62 tg32 (baseline) | 786 pp32 / 62 tg32 (fixed) | No regression; single-device test |
| 2026-07-11 D4.6 (romulus) | Path-B-Plus + RPC multi-device, 9B MTP, no spec | 991 pp32 / 81.7 tg32 (bench) | 80.0 tg128 (server) | Path-B-Plus: +21% pp, +32% tg vs D4.6 baseline |
| 2026-07-11 D4.6 (romulus) | Path-B-Plus + RPC multi-device + draft-mtp, 9B MTP | 96.9 tg128 (server) | -- | draft-mtp n_max=16: +21% gen throughput over no-spec; 39.5% draft acceptance |
| 2026-07-11 D4.6 (romulus) | Path-B-Plus + RPC multi-device + draft-mtp (n_max=2), 9B MTP | 113.2 tg128 (server) | -- | n_max=2 strongly recommended: 80% draft acceptance, +41.5% over no-spec, +16.8% over n_max=16 |

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