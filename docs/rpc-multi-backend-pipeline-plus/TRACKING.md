# TRACKING — rpc-multi-backend-pipeline-plus

**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Date:** 2026-07-01  
**Status:** Phase 1b — 2-GPU triton ladder **exhausted** (partial verdict below); 4-GPU gate run in progress on romulus.

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
| 2026-07-01 | **B+9 OFF null on n=384 triton** | DEFER=0 vs canonical: overlap 0.2% both, stall ~0.87 | Next: B+8 OFF bisect; see [ADR-0001](../adr/0001-b6-ladder-execution-post-b9-null.md) |
| 2026-07-01 | **Topology-agnostic gate client** | Any synced node may run profiler; romulus primary | [CONTEXT.md](CONTEXT.md) |

## Current Champion Runs

| Label | Topology | Plus | G (t/s) | overlap_pct | Notes | Status |
|-------|----------|------|---------|-------------|-------|--------|
| `trace-f-2gpu-plus` | 5070 + 5060 `ts=50,50` | 1 | **48.9** | 0.6% | **Production default** 36B NL MoE | Shipped |
| `b6-2gpu-f-triton-guard-n128` | romulus + triton 3090 (guard) | 1 | 116.0 | 0.9% | n=128 guard run; stall improved vs baseline | Gate data (MIXED) |
| `b6-2gpu-f-triton-n384-remus-docker` | remus docker + triton 3090 | 1 | 128.8 | 0.2% | archived from legacy n384; B+9 NULL baseline | Gate data |
| `b6-2gpu-f-triton-n384-romulus-native` | romulus 7900XTX + triton 3090 | 1 | 200.8 | 0.2% | canonical romulus-native @ deacf5e65; stall 0.858 | Gate data (STRAGGLER_DOMINANT) |
| `b6-2gpu-f-triton` | romulus + triton 3090 | 1 | 186.6 | 0.3% | Fast worker; overlap still FAIL | Gate ref |
| `b6-2gpu-f` | romulus + remus 5060 | 1 | 75.6 | 0.2% | Post-B7 partial | Gate FAIL |
| `b6-4gpu-g` | 4-GPU JUPITER canonical n=384 (2026-06) | 1 | 77.2 | 0.1% | drain 50.4s | Gate FAIL |
| `b6-4gpu-g-n384-romulus-native` | romulus 4-GPU + JUPITER n=384 | 1 | 69.5 | 0.2% | drain 6.4s; stall 0.96 | Gate FAIL |
| `b6-4gpu-g-triton` | 4-GPU triton swap n=384 (2026-06) | 1 | 63.1 | 0.1% | drain 5.9s | Gate FAIL |
| `b6-4gpu-g-triton-n384-romulus-native` | romulus 4-GPU + triton 3090 n=384 | 1 | 73.2 | 0.1% | drain 4.6s; stall 0.94 | Gate FAIL |
| `trace-g-4gpu-primary` | romulus 4-GPU (no 6600) | 1 | ~43.0 | 0.1% | Stable cluster | Path-C baseline |

## Partial 2-GPU verdict (2026-07-01)

**Topology:** romulus 7900XTX client + triton 3090 `:50054`, n=384, `ts=50,50`, Plus=1, SHA `deacf5e65`.

| Bisect | `overlap_pct` | `stall_ratio` | vs canonical Δoverlap | Verdict |
|--------|---------------|---------------|----------------------|---------|
| canonical (`romulus-native`) | 0.2% | 0.858 | — | baseline |
| B+9 OFF (`no-defer`, docker baseline) | 0.2% | 0.870 | 0.0 | NULL |
| B+8 OFF (`no-partial`) | 0.2% | 0.904 | 0.0 | NULL |
| B+10 OFF (`no-async-copy`) | 0.2% | 0.906 | 0.0 | NULL |

**Conclusion:** B+8, B+9, B+10 do **not** move `overlap_pct` on 2-GPU triton n=384. Stall worsens slightly when each flag is OFF (Δstall +0.046 to +0.048) — mitigations help stability marginally, not pipelining depth. **M1 FAIL** (best overlap 0.9% on guard-n128 only; n=384 canonical stuck at 0.2%). Straggler remains backend 1 (triton RPC) @ ~5.6-8.2 ms/tok depending on client.

**4-GPU summary (2026-07-01):** JUPITER canonical 0.2%/6.4s drain; B+7a′ OFF 0.1%; triton swap 0.1%/4.6s drain, G=73.2. All M1 FAIL.

**Next:** Structural ceiling doc per PLAN stop rule; optional B+11–B+13 only with explicit scope.

## Open Items / Blockers

### Primary — B+6 overlap gate (M3)

1. **Copy-slot pipelining collapsed** — `stall_ratio` 0.92–0.95; `input_wait_copy_ms` >> `graph_compute_async_ms`
2. **EVENT_RECORD recv on hot path** — `drain_flush_ms` tracks EVENT count (b6-2gpu-f: 4514ms / 385 tok)
3. **4-GPU drain amplification** — canonical 50.4s vs triton 5.9s (B+7a′ multi-socket)
4. **2-GPU overlap ceiling** — 0.2% at n=384 after B+8–B+10 OFF bisects (2026-07-01); 4-GPU ladder pending

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
- B+9 (DEFER=0) bisect on n=384 triton — **null result** (overlap 0.2% vs canonical 0.2%)
- Grill-with-docs session: execution plan locked ([ADR-0001](../adr/0001-b6-ladder-execution-post-b9-null.md), [CONTEXT.md](CONTEXT.md))
- Tokens read from romulus ~/tokens (gitea 8ca5... PAT etc); .git-credentials + insteadOf + local token mirrors updated on remus+triton; git ls-remote OK for origin/gitea (2026-07-01)
- RPC send combined-buffer improvement + early-cmd drain skip + post-hello pending reset (helped runs complete)
- Per-split / basic RPC RTT histogram starter in pathb-rpc-trace-parse.sh (plan 1.1)
- `rpc-multi-backend-pipeline-plus/` doc root incorporated into repo (2026-06-30)
- Reconciled with B+6 gate, b6-gate/TRACKING, grilling lateral ladder (B+8–B+13)
- Formalized [Path-B-Plus-MultiBackend-RPC-Orchestration-Audit.md](ANALYSIS/Path-B-Plus-MultiBackend-RPC-Orchestration-Audit.md)

## Next 7 Days (grill-locked 2026-07-01)

- [x] B+8–B+10, B+7a′, B+13 code landed; 2-GPU triton n=128/384 collected
- [x] B+9 OFF bisect on n=384 triton — null
- [x] PR 2 + PR 6: `b6-gate-bisect-run.sh`, env audit in cluster, docker `GGML_*` forward, `--skip-rpc-validate` auto (2026-07-01)
- [x] Phase 0: archive `b6-2gpu-f-triton-n384-remus-docker`; romulus reset `deacf5e65`; canonical-romulus re-bench (2026-07-01)
- [ ] Triton git sync (repo git fs-boundary quirk); rpc-server :50054 was already up
- [x] **B+8 OFF** bisect on romulus: overlap 0.2% (NULL vs canonical), stall 0.904 vs 0.858 (borderline MITIGATION_HELPS) — 2026-07-01
- [x] **B+10 OFF** bisect on romulus: overlap 0.2% (NULL), stall 0.906 vs 0.858 — 2026-07-01
- [x] **2-GPU partial verdict** recorded (2026-07-01)
- [x] **`b6-4gpu-g-n384-romulus-native`** on romulus — overlap 0.2%, drain 6.4s, M1 FAIL (2026-07-01)
- [x] B+7a′ OFF (`no-flush`) on 4-GPU: overlap 0.1% (Δ-0.1), drain 5.3s vs 6.4s — NULL/worse overlap (2026-07-01)
- [ ] Phase 1.1 parallel: re-parse canonical/no-defer traces for full RTT hist + per-split RPC
- [ ] **PR 8:** validate-rpc investigation matrix (client x endpoint x server state); not remus-specific — fix root cause, re-enable for 4-GPU after proven
- [x] `b6-4gpu-g-triton-n384-romulus-native` — overlap 0.1%, G=73.2, drain 4.6s (2026-07-01)
- [ ] **Structural ceiling doc** (PLAN stop rule: B+8–B+10 + B+7a′ exhausted, M1 FAIL on 2-GPU + 4-GPU)
- [ ] Phase 1.1 re-parse; PR 3 bisect comparator; PR 8 validate-rpc
- [ ] Triton ops (git fs-boundary, start/stop scripts); commit/push session work

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
**Last edit:** 2026-07-01 — grill session locked execution plan (ADR-0001).