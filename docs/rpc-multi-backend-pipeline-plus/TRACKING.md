# TRACKING — rpc-multi-backend-pipeline-plus

**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Date:** 2026-07-01  
**Status:** Phase 1 (Instrumentation) + Phase 1b (B+6 gate) — in progress (triton n=128/384 runs completed; B+9 bisect started; instrumentation enhancements begun)

Mirror gate checklist: [rpc-patch/docs/b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md)

## Key Decisions Log

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-06-27 | Drop RX6600 from 35–36B A3B MoE workloads | 3-device F = 37–42.8 t/s vs 2-device = 48.9 t/s (+32%) | Deprecate 3-device Config F for these models |
| 2026-06-27 | Declare `trace-f-2gpu-plus` production default | Highest measured throughput + clean 2-device topology | Runbooks and matrix defaults |
| 2026-06-27 | 4-GPU cluster (no 6600) is stable | `trace-g-4gpu-primary` @ ~43 t/s | Path-C bridge testbed |
| 2026-06-29 | B+6 gate verdict D3+D1 | Drain-bound canonical 4-GPU; triton fixes drain not overlap | Next code: B+7a′ + B+8–B+10 |
| 2026-06-29 | Create `rpc-multi-backend-pipeline-plus/` doc root | Structured mission/plan/tracking + formalized audit | This document set |
| 2026-06-30 | **M3 is hard mission complete criterion** | User decision: B+6 gate must pass or structural ceiling documented | Mitigation ladder before Path C |
| 2026-06-30 | **Path C deferred** | Mother-repo / fork compatibility | Stay in `ggml-rpc.cpp` + `ggml-backend.cpp` until ladder exhausted |
| 2026-06-30 | **Implement order B+8→B+9→B+10→B+7a′** | Profiler: overlap is pipelining depth; drain fixes alone insufficient | See [PLAN.md](PLAN.md) Phase 2 |

## Current Champion Runs

| Label | Topology | Plus | G (t/s) | overlap_pct | Notes | Status |
|-------|----------|------|---------|-------------|-------|--------|
| `trace-f-2gpu-plus` | 5070 + 5060 `ts=50,50` | 1 | **48.9** | 0.6% | **Production default** 36B NL MoE | Shipped |
| `b6-2gpu-f-triton-guard-n128` | romulus + triton 3090 (guard) | 1 | 116.0 | 0.9% | n=128 guard run; stall improved vs baseline | Gate data (MIXED) |
| `b6-2gpu-f-triton-n384` | romulus + triton 3090 | 1 | 128.8 | 0.2% | canonical n=384 Plus=1; straggler dominant | Gate data (STRAGGLER_DOMINANT) |
| `b6-2gpu-f-triton` | romulus + triton 3090 | 1 | 186.6 | 0.3% | Fast worker; overlap still FAIL | Gate ref |
| `b6-2gpu-f` | romulus + remus 5060 | 1 | 75.6 | 0.2% | Post-B7 partial | Gate FAIL |
| `b6-4gpu-g` | 4-GPU JUPITER canonical n=384 | 1 | 77.2 | 0.1% | drain 50.4s | Gate FAIL |
| `b6-4gpu-g-triton` | 4-GPU triton swap n=384 | 1 | 63.1 | 0.1% | drain 5.9s | Gate FAIL |
| `trace-g-4gpu-primary` | romulus 4-GPU (no 6600) | 1 | ~43.0 | 0.1% | Stable cluster | Path-C baseline |

## Open Items / Blockers

### Primary — B+6 overlap gate (M3)

1. **Copy-slot pipelining collapsed** — `stall_ratio` 0.92–0.95; `input_wait_copy_ms` >> `graph_compute_async_ms`
2. **EVENT_RECORD recv on hot path** — `drain_flush_ms` tracks EVENT count (b6-2gpu-f: 4514ms / 385 tok)
3. **4-GPU drain amplification** — canonical 50.4s vs triton 5.9s (B+7a′ multi-socket)
4. **Overlap metric ceiling** — 0.1–0.7% after B+1–B+6; needs B+8–B+10 before declaring structural limit

**Actions:** See [PLAN.md](PLAN.md) Phase 2 table; update [pathb-sync-site-audit.md](../../rpc-patch/docs/pathb-sync-site-audit.md) after each bisect.

### Secondary — Instrumentation (Phase 1.1)

- Per-split / per-RPC RTT breakdown in operator-facing logs (extend existing trace parsers)
- `BENCHMARKS/2026-07-comparison-matrix.md` after refresh

### Ops / hygiene

- RX6600 guard in Config F matrix scripts
- 500 ms + 45 s flush default for all bench scripts
- Link doc root from all entry READMEs (**done 2026-06-30**)

## Recently Completed

- n=128 triton guard + n=384 canonical Plus=1 runs on triton (2026-07-01); data collected (0.9% and 0.2% overlap)
- B+9 (DEFER=0) bisect on n=384 triton launched
- RPC send combined-buffer improvement + early-cmd drain skip + post-hello pending reset (helped runs complete)
- Per-split / basic RPC RTT histogram starter in pathb-rpc-trace-parse.sh (plan 1.1)
- `rpc-multi-backend-pipeline-plus/` doc root incorporated into repo (2026-06-30)
- Reconciled with B+6 gate, b6-gate/TRACKING, grilling lateral ladder (B+8–B+13)
- Formalized [Path-B-Plus-MultiBackend-RPC-Orchestration-Audit.md](ANALYSIS/Path-B-Plus-MultiBackend-RPC-Orchestration-Audit.md)

## Next 7 Days

- [x] B+8–B+10, B+7a′, B+13 code landed (2026-06-30, **untested** — see [IMPLEMENTATION.md](IMPLEMENTATION.md)); 2-GPU triton n=128/384 data collected
- [x] n=384 triton Plus=1 + initial B+9 bisect launched (2026-07-01)
- [ ] Complete B+ bisects (B+8/B+9/B+10) + n=384 re-bench on triton (Q5 contract); compare DEFER=0 vs Plus=1
- [ ] B+7a′ validate on `b6-4gpu-g` n=384 (after 2-GPU pass/partial)
- [ ] Phase 1.1: full per-split timing + RPC RTT histogram in scheduler trace / profiler output + parsers (starter added)
- [ ] Fix `--validate-rpc` hang (still using --skip; handshake debug in progress)
- [ ] Topology guard in `pathb-vram-calc.ps1` / Config F scripts
- [ ] Publish `BENCHMARKS/2026-07-comparison-matrix.md` after instrumentation refresh
- [ ] Triton ops (full git sync, start/stop scripts); rebuilds; push session work + handover

## Metrics Dashboard

| Metric | Value | Gate |
|--------|-------|------|
| Best 36B NL MoE (2-device) | **48.9 t/s** | PASS (production) |
| B+6 overlap (best) | **0.7%** (G2 n=128) | **FAIL** (M3 needs ≥5%) |
| Best power duty @20% TDP | **3.8%** | FAIL (target >15%) |
| 4-GPU stable baseline | **~43 t/s** | PASS (cluster) |
| b6-4gpu canonical drain | **50.4s** / n=384 | FAIL (target <10s) |

## References to Raw Data

- `docs/cuda-windows-5070ti/benchmarks/trace-f-2gpu-plus/`
- `benches/path-b-plus/b6-2gpu-f/` (romulus profiler artifacts)
- `benches/path-b-plus/regression.jsonl`
- `rpc-patch/docs/pathb-sync-site-audit.md`
- `rpc-patch/patch/HANDOVER-SESSION-2026-06-29.md`

---

**Update this file after every profile/profiler run or topology decision.**  
**Last edit:** 2026-06-30 — handoff package incorporated + B+6 gate reconciliation.