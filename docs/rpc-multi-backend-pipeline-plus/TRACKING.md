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
| 2026-07-01 | **Structural ceiling + Path C out of scope** | B+8–B+10 + B+7a' exhausted; M1 FAIL 2-GPU + 4-GPU | TRACKING ceiling section; no Path C on this branch |

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

**Next:** Phase 1.1 complete; B+11–B+13 parked (Path C out of scope for path-b-plus).

## Structural ceiling (2026-07-01 — PLAN stop rule)

**Verdict:** M3 (`overlap_pct >= 5%`) is **not reachable** within Path-B+ scope on measured 2-GPU and 4-GPU topologies after the mitigation ladder (B+8, B+9, B+10, B+7a') is exhausted.

**Scope boundary:** Path C (server-side scheduling / distributed orchestration) is **out of scope** for `rpc-multi-backend-pipeline-plus`. Ideas from Path C may inform future work elsewhere; **no Path C implementation** on this branch.

### Evidence summary

| Signal | 2-GPU romulus+triton n=384 | Interpretation |
|--------|---------------------------|----------------|
| `overlap_pct` | 0.2% (canonical); 0.9% best (guard-n128 only) | M1 FAIL; M3 FAIL |
| `stall_ratio` | 0.86–0.91 | Copy-slot pipelining collapsed |
| `input_wait_copy_ms` | 1976 vs `graph_compute_async_ms` 323 | Wait dominates compute (~6x) |
| `straggler` | backend 1 (triton RPC) @ 5.6–8.2 ms/tok | Serial RPC stage bounds assembly line |
| `EVENT_RECORD` | 385 events / 1752 ms drain | Drain tracks event count, not overlap lever |
| Bisect B+8/B+9/B+10 OFF | NULL overlap (Δ=0); stall worsens when OFF | Mitigations help marginally; not pipelining depth |

| Signal | 4-GPU romulus n=384 | Interpretation |
|--------|----------------------|----------------|
| `overlap_pct` | 0.1–0.2% | M1 FAIL |
| `drain_flush_ms` | 4.6–6.4s (triton swap helps vs JUPITER 50s) | Drain fix != overlap fix |
| B+7a' OFF (`no-flush`) | overlap 0.1% (worse) | NULL / harmful to overlap |

### Phase 1.1 trace proof (re-parse, no new bench)

Romulus-native canonical (`78e8f3c45` ladder):

- `rpc_rtt_p50_ms=0.35` `p95=3.68` `p99=71.55` — tail latency spikes on RPC hot path
- `per_split_rpc_rtt`: split=2 backend=1 (triton) accounts for bulk RPC RTT in gen window
- `assembly_overlap_count=990` / `pairs=444675` -> **0.2%** — backends rarely overlap in time
- `trace-parse-extended.json` written per dir; run: `bash scripts/b6-gate-phase11-reparse.sh`

### Audit blocker mapping (pathb-sync-site-audit)

| Blocker | Trace evidence | Ladder result |
|---------|----------------|---------------|
| 7a serial split dispatch | `stall_ratio` 0.86+, low `overlap_pct` | Unchanged by B+8–B+10 |
| 7b RPC RTT on hot path | `rpc_rtt_ms` 1370ms / 385 tok; straggler backend 1 | Triton swap cuts drain, not overlap |
| 7c input_wait_copy stall | 1976ms >> 323ms compute | B+10 OFF worsens stall |
| 7d EVENT_RECORD drain | `drain_flush_ms` = EVENT_RECORD sum | B+9 OFF null |
| 7e assembly-line gap | 0.2% overlap at n=384 | Guard-n128 0.9% is n=128 only |

### What remains in path-b-plus (no Path C)

1. ~~**PR 8** validate-rpc~~ (**done 2026-07-01** — matrix all PASS; RPC handshake fix; `b6-gate-validate-rpc-matrix.sh`)
2. **Production guard** — `trace-f-2gpu-plus` @ 48.9 t/s; RX6600 excluded for 35B+ A3B MoE
3. **B+11–B+13** — active (grill 2026-07-01: aim M3; implementation bug not Path C). **Staged analysis D:** 1.2C done → 1.2 A+B next
4. **Comparison matrix** — `BENCHMARKS/2026-07-comparison-matrix.md` when 1.1 data is folded in

## Open Items / Blockers

### Primary — B+6 overlap gate (M3) — **STRUCTURAL CEILING**

Ladder exhausted; see **Structural ceiling** section above. No further bisect waves without explicit new scope.

**Remaining hygiene:** PR 8 validate-rpc; regression guard for 48.9 t/s production champion.

### Secondary — Instrumentation (Phase 1.1)

- ~~Per-split / per-RPC RTT in trace parsers~~ (**done 2026-07-01** — `pathb-rpc-trace-parse.sh`, `b6-gate-phase11-reparse.sh`)
- ~~`BENCHMARKS/2026-07-comparison-matrix.md`~~ (**done 2026-07-01**)

### Ops / hygiene

- ~~RX6600 guard in Config F matrix scripts~~ (**done 2026-07-01** — `pathb-config-f-matrix.ps1`, `rpc-server-bench.ps1`)
- ~~Triton Linux ops scripts~~ (**done 2026-07-01** — `b6-gate-triton-{git-sync,start-rpc,stop-rpc,sync-rebuild,remote}.sh`)
- ~~Triton git repair~~ (**done 2026-07-01** — missing `.git/objects/`; force checkout to `78e8f3c45`)
- 500 ms telemetry default in profiler (`pipeline-gpu-telemetry`); 45 s flush in `rpc-server-bench.ps1` — verify cluster scripts inherit
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
- [x] Triton git sync + repair (missing objects/); SHA `78e8f3c45`; rpc-server :50054 up (2026-07-01)
- [x] **B+8 OFF** bisect on romulus: overlap 0.2% (NULL vs canonical), stall 0.904 vs 0.858 (borderline MITIGATION_HELPS) — 2026-07-01
- [x] **B+10 OFF** bisect on romulus: overlap 0.2% (NULL), stall 0.906 vs 0.858 — 2026-07-01
- [x] **2-GPU partial verdict** recorded (2026-07-01)
- [x] **`b6-4gpu-g-n384-romulus-native`** on romulus — overlap 0.2%, drain 6.4s, M1 FAIL (2026-07-01)
- [x] B+7a′ OFF (`no-flush`) on 4-GPU: overlap 0.1% (Δ-0.1), drain 5.3s vs 6.4s — NULL/worse overlap (2026-07-01)
- [x] Phase 1.1 re-parse: RTT p50/p95/p99, per-split RPC RTT, `trace-parse-extended.json` (2026-07-01)
- [x] PR 3 `b6-gate-bisect-compare.sh` — bisect verdict TSV (2026-07-01)
- [x] **PR 8:** validate-rpc matrix all PASS; RPC-only preflight helper (2026-07-01)
- [x] **Comparison matrix** `BENCHMARKS/2026-07-comparison-matrix.md` (2026-07-01)
- [x] `b6-4gpu-g-triton-n384-romulus-native` — overlap 0.1%, G=73.2, drain 4.6s (2026-07-01)
- [x] **Structural ceiling doc** — PLAN stop rule met (2026-07-01)
- [x] Triton ops scripts + commit/push session work (2026-07-01)

## Metrics Dashboard

| Metric | Value | Gate |
|--------|-------|------|
| Best 36B NL MoE (2-device) | **48.9 t/s** | PASS (production) |
| B+6 overlap (best) | **0.9%** (triton guard-n128) | **FAIL** (M3 needs ≥5%) |
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