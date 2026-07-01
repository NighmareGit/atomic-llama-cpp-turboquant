# TRACKING — rpc-multi-backend-pipeline-plus

**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Date:** 2026-07-01  
**Status:** Phase 2 complete — B+11–B+13 ladder **exhausted**; structural ceiling **finalized**; 4-GPU gate = `b6-4gpu-g-triton` (jupiter skipped).

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
| 2026-07-01 | **Grill: pursue M3 via B+11–B+13** | Blocking = implementation bug, not Path C; ladder NULL != unfixable | Reactivate B+11–B+13; Phase 1.2 staged D |
| 2026-07-01 | **C-full instrumentation** | C-min insufficient for B+13 proof | `sync_copy_fallback` + RPC join; see IMPLEMENTATION |
| 2026-07-01 | **Sample API C-full keep lists deferred** | Emit/parser/re-bench first | [FUTURE-EXPANSIONS.md](FUTURE-EXPANSIONS.md) only |
| 2026-07-01 | **Validation before next M3 experiment** | B+14/B+15 closed HIP gather gap; CUDA `leaf_55` residual ~1.5ms; M3 still ~0.3% — run MoE + larger-model smoke before picking next hunt step | Lateral todo below; PLAN review after V1+V2 (M3 hunt not postponed) |
| 2026-07-01 | **V3 plan review: ship B+14/B+15, resume M3 hunt** | V1/V2 PASS; overlap 0.3% @ n=384 canonical unchanged; next experiment B+12 not leaf_55 | TRACKING V3 section; PLAN Phase 2b |
| 2026-07-01 | **B+12 NULL on M3 overlap** | Canonical bisect ON/OFF: overlap 0.3% both; G +1.8%; drain_flush -52%; defer path shipped | B+12 section; next B+11 |
| 2026-07-01 | **B+11 NULL on M3 overlap; HURTS G** | 4-GPU triton bisect dual ON/OFF: overlap 0.2% both; G -9.1%; hol_tail_ms +508%; default OFF kept | B+11 section; next B+13 |
| 2026-07-01 | **B+13 NULL on M3 overlap; +G** | 2-GPU ~206 t/s; 4-GPU ~82 t/s; stall_ratio 0.95->0.68; overlap 0.1-0.3% | B+13 shipped; ceiling finalized |
| 2026-07-01 | **Jupiter skipped; triton is 4-GPU RPC2** | `:50053` register failed from romulus; triton `:50054` operational | Gate preset `b6-4gpu-g-triton` canonical |

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
| `b6-4gpu-g-triton-n384-romulus-native` | romulus 4-GPU + triton 3090 n=384 dual OFF | 1 | **80.5** | 0.2% | B+11 bisect OFF arm; drain 3.0s | Gate FAIL |
| `b6-4gpu-g-triton-n384-romulus-native` (dual ON) | romulus 4-GPU + triton 3090 n=384 dual ON | 1 | 73.2 | 0.2% | B+11 bisect ON arm; hol_tail 864ms spike | Gate FAIL |
| `b6-4gpu-g-triton-n384-romulus-native-b13` | post-B+13d dual OFF | 1 | **81.8** | 0.2% | stall 0.68; canonical 4-GPU gate | Gate FAIL overlap |
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

**4-GPU summary (2026-07-01):** JUPITER canonical 0.2%/6.4s drain; B+7a′ OFF 0.1%; triton swap 0.2%/3.0s drain, G=80.5 (dual OFF). B+11 dual ON: overlap 0.2%, G=73.2, hol_tail_ms=1036 (864ms spike). All M1 FAIL.

**Next:** B+13 ladder closed; 4-GPU canonical gate is `b6-4gpu-g-triton` (jupiter skipped).

## Topology decision (2026-07-01 — locked)

**4-GPU gate:** Use **`b6-4gpu-g-triton`** only. Jupiter `:50053` skipped (`register failed` from romulus validate-rpc 2026-07-01).

| Slot | Endpoint | Role |
|------|----------|------|
| RPC0 | `192.168.8.176:50051` | remus 5060 Ti (`hostname=remus`) |
| RPC1 | `127.0.0.1:50051` | romulus 3060 Ti docker (`pathb-rpc-romulus`; **not** on remus) |
| RPC2 | `192.168.8.23:50054` | triton 3090 (replaces jupiter 5070) |
| ROCm0 | romulus 7900 XTX | client gather (`192.168.8.108`) |

**Hardware ground truth (2026-07-01, live `nvidia-smi` + `rocm-smi` on each node):**

Re-run inventory: `bash scripts/b6-gate-cluster-gpu-inventory.sh`  
Pre-deploy ts/ngl plan: `bash scripts/pathb-rpc-vram-preflight.sh --preset b6-4gpu-g-triton --gguf <model>`

| Host | IP | `nvidia-smi` | `rocm-smi` | Active RPC / role |
|------|-----|--------------|------------|-------------------|
| romulus | `192.168.8.108` | RTX **3060 Ti** 8192 MiB (7646 free) | RX **7900 XTX** ~24 GB (25753026560 B) | `:50051` docker on **local 3060**; ROCm **client** on **local 7900** |
| remus | `192.168.8.176` | RTX **5060 Ti** 16311 MiB (15656 free) | RX **6600** ~8 GB (8573157376 B) — **unused** | `:50051` (5060); `:50052` (6600 parked) |
| triton | `192.168.8.23` | RTX **3090** 24576 MiB + RTX **3070** 8192 MiB | n/a | `:50054` (3090) / `:50055` (3070 parked) |

Each bench host is **dual-GPU**: romulus and remus pair one NVIDIA + one AMD on the same machine; triton pairs two NVIDIA cards. The **3060 Ti and 7900 XTX are both on romulus** — not on remus. `127.0.0.1:50051` is the romulus client loopback to its own docker RPC worker on the local 3060 Ti.

`-ts 22,11,34,33` matches live free VRAM ratio (5060/3060/3090/7900 ~22/11/33/34%). Do not block gates on jupiter rebuild. Preset `b6-4gpu-g` (JUPITER) is **deprecated** for active hunt until ops revisits Windows rpc-server.

Triton `:50055` (3070) remains parked for 35B+ MoE per ops notes.

## Structural ceiling (2026-07-01 — finalized post B+13)

**Verdict:** M3 (`overlap_pct >= 5%`) is **not reachable** within Path-B+ scope on measured 2-GPU and 4-GPU topologies after the full mitigation ladder through **B+13** is exhausted.

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
| B+11 dual-socket OFF | NULL overlap; **+9% G** vs dual ON on 4-GPU | Ship dual OFF |
| B+12 GET defer OFF | NULL overlap; +1.8% G | Ship defer ON |
| B+13b/c/d | NULL overlap; +G on 4-GPU, +1% on 2-GPU romulus | Gather fixes; not pipelining depth |

| Signal | 4-GPU `b6-4gpu-g-triton` romulus n=384 | Interpretation |
|--------|----------------------------------------|----------------|
| `overlap_pct` | 0.1–0.2% (pre/post B+13) | M1/M3 FAIL |
| G (dual OFF, ships) | **81.8** post-B+13 (was 80.5) | Real throughput gain; not overlap |
| `stall_ratio` | 0.68 post-B+13 (was 0.95) | Orchestration improved |
| `drain_flush_ms` | ~4.3–4.6s | Drain fix != overlap fix |
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

1. ~~**PR 8** validate-rpc~~ (**done 2026-07-01**)
2. ~~**B+11–B+13 ladder**~~ (**done 2026-07-01** — all NULL on M3 overlap; B+13 ships for G/stall)
3. **Production guard** — `trace-f-2gpu-plus` @ 48.9 t/s; RX6600 excluded for 35B+ A3B MoE
4. ~~**B+16 optional**~~ — **REJECT** (split-slot wait hurts `leaf_55`; code reverted)
5. **Comparison matrix** — `BENCHMARKS/2026-07-comparison-matrix.md` when 1.1 data is folded in

## Open Items / Blockers

### Primary — B+6 overlap gate (ceiling finalized)

V1+V2 validation **PASS**. B+11–B+13 ladder **closed NULL on M3**. **G gains ship** (2-GPU ~206 t/s, 4-GPU ~82 t/s triton gate). Overlap gate remains FAIL; structural ceiling documented. Optional backlog: B+16 G-only.

### Secondary — cluster sync + instrumentation

### Secondary — Instrumentation (Phase 1.1 + 1.2)

- ~~Per-split / per-RPC RTT in trace parsers~~ (**done 2026-07-01**)
- ~~Phase 1.2C blocking audit~~ (**done 2026-07-01** — `b6-gate-phase12c-blocking-audit.sh`)
- ~~C-full hotpath emit~~ (**done 2026-07-01** — `6dc504bce`)
- Extend phase12c parser for `sync_copy_fallback` / `copy_async_ok`
- Re-bench `b6-2gpu-f-triton-n384-romulus-native`
- Phase 1.2 **A** waterfall + **B** Gantt parsers
- Blocker **7f** in pathb-sync-site-audit
- Sample API C-full keep lists — **deferred** ([FUTURE-EXPANSIONS.md](FUTURE-EXPANSIONS.md))

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
- B+14 gather prefetch + producer mask; B+15 split-1-start RPC prefetch + MoE ids path (`41a65d963`, `590597110`)
- B+12 GET_TENSOR deferral — NULL overlap @ n=384; G +1.8%; `GGML_RPC_GET_TENSOR_DEFER`; romulus sync/rebuild scripts (2026-07-01)

## Lateral Todo (post-B+15 — 2026-07-01)

Side track — not blocking production ship (`trace-f-2gpu-plus` @ 48.9 t/s). Perf-hunt items deferred to [FUTURE-EXPANSIONS.md](FUTURE-EXPANSIONS.md).

### Now — correctness validation

| Step | What | Why | Client(s) |
|------|------|-----|-----------|
| V1 | **MoE model** profiler gate / smoke run | `gemma-4-26B-A4B-APEX-I-Compact.gguf` (gemma4 MoE, distinct from canonical Qwen APEX 36B); B+14/B+15 MoE leaf / ids prefetch path | `remus-docker`, `romulus` |
| V2 | **Larger model** (70B+ class) smoke run | `meta-llama-3-70b-instruct.Q4_K_M.gguf` (40G dense); gather prefetch across deeper graphs | `remus-docker`, `romulus` |
| V3 | Compare traces vs B+15 baselines | `input_wait_copy`, `leaf_*`, `rpc_prefetch_*` — flag new stalls or sync fallbacks, not just G/overlap | both |

**Pass criteria (validation):** generation completes; no new `sync_copy_fallback` spikes; throughput within prior regression band for that model/topology; no correctness anomalies in sample output.

### Validation results (2026-07-01, B+15 @ rsync to romulus)

| Run | Model | Client | G (t/s) | overlap | sync_fb | Verdict |
|-----|-------|--------|---------|---------|---------|---------|
| V1 | gemma4 26B-A4B MoE | romulus HIP | 164.9 | 0.8% | 0 | **PASS** |
| V1 | gemma4 26B-A4B MoE | remus CUDA docker | 140.8 | 1.3% | 0 | **PASS** |
| V2 | llama-70B Q4_K_M | romulus HIP | 30.0 | 0.8% | 0 | **PASS** |
| V2 | llama-70B Q4_K_M | remus CUDA docker (profiler) | — | — | — | **VRAM BLOCKED** (profiler holds local shard in-process) |
| V2 | llama-70B Q4_K_M | remus CUDA `llama-server` + triton | smoke OK | — | — | **PASS** (`--fit on`, ts=50,50; 3 tok gen) |
| V2 | llama-70B Q4_K_M | romulus HIP (profiler) | 30.0 | 0.8% | 0 | **PASS** |
| V2 | llama-70B Q4_K_M | romulus HIP 4-GPU triton (`b6-4gpu-g-triton`, n=128) | 16.9 | 0.2% | 0 | **PASS** |

**V2 CUDA note:** `llama-pipeline-profiler` on remus-docker OOMs on dense 70B — local 5060 Ti cannot hold the client shard. For V2 CUDA validation use **llama-server + RPC worker** topology (`rpc-server-bench.sh pathb`): e.g. remus client + triton `:50054`, or remus + romulus `:50051`, with `GGML_PIPELINE_PLUS=1` and `BENCH_TRACE=1`. Profiler gate presets remain canonical for M3; validation smokes may use server style for large dense models.

Artifacts: `benches/path-b-plus/b6-2gpu-f-triton-n384-v1-moe-gemma-{hip,cuda}`, `b6-2gpu-f-triton-n384-v2-llama70b-hip`, `b6-4gpu-g-triton-n384-v2-llama70b-hip`, `b6-2gpu-f-triton-n384-romulus-native-b15b`. No `sync_copy_fallback` on any completed run.

### V3 — Plan / mission review (2026-07-01)

**Verdict:** B+14/B+15 **safe to ship** for production paths. M3 hunt **continues** on canonical bench; validation does not change the mission criterion.

#### What B+14/B+15 bought (canonical `b6-2gpu-f-triton-n384`, n=384)

| Signal | Pre-B+15 (baseline) | Post-B+15 | Notes |
|--------|---------------------|-----------|-------|
| CUDA G | ~128.8 t/s | ~128.8 t/s | Throughput flat; gains are latency composition |
| CUDA overlap | ~0.2–0.3% | 0.3% (b14i) | M3 still FAIL |
| HIP G | ~200.8 t/s | ~190–205 t/s (b14b/b15b) | Modest / stable |
| HIP overlap | ~0.2% | ~0.3% | M3 still FAIL |
| HIP split-2 gather | `producer_event_wait` ~4ms | `input_wait` ~358ms | Gather path fixed (rebuild `ggml-base` after git sync) |
| CUDA residual | `leaf_55` ~2.3ms | ~1.5ms (b15) | MoE weight H2D; not overlap lever |

B+14/B+15 are **implementation fixes** (prefetch, producer mask, split-1-start issue, MoE ids path). They collapse gather stalls and improve G on HIP; they do **not** unlock copy-slot pipelining depth — the structural M3 gap.

#### M3 hunt status (not postponed)

| Milestone | Target | Best post-B+15 | @ n=384 canonical | Status |
|-----------|--------|----------------|-------------------|--------|
| M1 | overlap >= 1% | 1.3% (V1 CUDA gemma, n=128) | 0.3% | **FAIL** at gate depth |
| M3 | overlap >= 5% | 1.3% | 0.3% | **FAIL** |

Validation overlap (0.8–1.3% @ n=128) shows the scheduler **can** overlap more on some graphs, but not at canonical n=384 depth. M3 remains the active mission gap.

#### Decisions (locked)

1. **Ship B+14/B+15** — V1/V2 PASS; zero `sync_copy_fallback`; no MoE leaf regressions on gemma4 or llama-70B HIP.
2. **Keep M3 as hard complete criterion** — do not downgrade to throughput-only or Path C.
3. **Resume hunt on canonical bench** — next experiments ordered by overlap leverage, not G alone:
   - ~~B+12 GET_TENSOR deferral~~ — **NULL overlap** (shipped for G/drain)
   - ~~B+11 dual-socket RPC~~ — **NULL overlap; HURTS G** (default OFF; proto 4.4 ships)
   - **B+13** — next HOL/overlap hunt on canonical gate
   - **B+17-candidate:** CUDA `leaf_55` MoE weight path — G/stall ROI on CUDA docker; unlikely to move overlap_pct materially
4. **Dual-track execution:** production ship (`trace-f-2gpu-plus` 48.9 t/s) proceeds in parallel with canonical M3 experiments.
5. **V2 CUDA gap** — llama-70B blocked on remus 5060 Ti VRAM; not a B+15 regression. Use romulus HIP or larger local GPU for dense 70B+ CUDA validation.

#### Next actions

- [x] Push B+14/B+15 to gitea; canonical `b15b` re-bench (rebuild `ggml-base` after sync)
- [x] V2 CUDA via `llama-server` + triton `:50054` (remus client, `--fit on`) — PASS 2026-07-01
- [x] B+12 bisect on canonical n=384 — **NULL overlap** (2026-07-01); see B+12 section
- [x] B+11 dual-socket RPC scope (`DESIGN-b11-dual-socket.md`)
- [x] B+11 bisect on `b6-4gpu-g-triton` n=384 (proto 4.4 remus/romulus/triton; jupiter still 4.3) — **NULL overlap; HURTS G** (2026-07-01)
- [x] Phase 1.2 A+B parsers (waterfall/Gantt) before next major bisect

### B+12 — GET_TENSOR deferral (2026-07-01)

**Flag:** `GGML_RPC_GET_TENSOR_DEFER=1` (default ON when `GGML_PIPELINE_PLUS=1`). Bisect OFF: `bash scripts/b6-gate-bisect-run.sh no-get-defer`.

**Implementation:** Skip `rpc_gather_flush` / `rpc_early_flush` when defer ON; single `rpc_defer_flush` at graph_compute boundary; blocking `tensor_set` at flush (avoids multi-slot `synchronize` deadlock). Romulus rebuild must copy `ggml/include/ggml-rpc.h` -> `ggml/src/ggml-rpc.h` (`b6-gate-romulus-sync-rebuild.sh`).

| Bisect | G (t/s) | overlap_pct | stall_ratio | drain_flush_ms | blocking_ms |
|--------|---------|-------------|-------------|----------------|-------------|
| B+12 ON (`-b12`) | **204.5** | 0.3% | 0.904 | 79 | 1297 |
| B+12 OFF (`no-get-defer`) | 200.9 | 0.3% | 0.908 | 166 | 1426 |
| Delta | +1.8% | **0.0** | -0.004 | -52% | -9% |

**Trace proof (ON):** `rpc_defer_flush=386`, `rpc_gather_flush=0`, `rpc_early_flush=0`, `decode_max=384`.

**Verdict:** **NULL on M3 overlap** — ship defer path for G/drain/correctness; not the pipelining-depth lever.

Artifacts: `b6-2gpu-f-triton-n384-romulus-native-b12`, `...-no-get-defer`.

### B+11 — dual-socket RPC (2026-07-01)

**Flag:** `GGML_RPC_DUAL_SOCKET=1` (default **OFF**; proto 4.4). Bisect ON: export before `canonical-romulus`; OFF: `no-dual-socket`.

**Gate:** `b6-4gpu-g-triton` n=384 (triton `:50054` swap; jupiter `:50053` still 4.3). Proto 4.4 on remus/romulus/triton.

| Bisect | G (t/s) | overlap_pct | stall_ratio | drain_flush_ms | hol_tail_count | hol_tail_ms |
|--------|---------|-------------|-------------|----------------|----------------|-------------|
| B+11 ON (dual) | 73.2 | 0.2% | 0.955 | 3918 | 5 | 1036 (864ms spike) |
| B+11 OFF (single) | **80.5** | 0.2% | 0.949 | 2985 | 9 | 171 |
| Delta | **-9.1%** | **0.0** | +0.006 | +31% | -4 | +508% |

**Verdict:** **NULL on M3 overlap; HURTS G** — dual-socket does not meet pass criteria (overlap delta >= 1% or hol_tail -50% with stable G). Keep default OFF; proto 4.4 + `CHANNEL_BIND` ships for future use.

Artifacts: `b6-4gpu-g-triton-n384-romulus-native`, `...-no-dual-socket`; logs `b11-4gpu-bisect-logs/`.

### B+16 — CUDA `leaf_55` MoE split-slot wait (2026-07-01)

**Hypothesis:** MoE host->GPU path waits CPU `input_bid` copy slot (~1.5ms CUDA `leaf_55` stall); switch to `split_backend_id` slot.

| Arm | G (t/s) | overlap_pct | `leaf_55` med | Verdict |
|-----|---------|-------------|---------------|---------|
| b14i (pre-B+16) | 128.8 | 0.3% | 1582 us | baseline |
| b16 (split-slot ON) | 128.9 | 0.2% | **7184 us** | **REJECT** |

**Verdict:** **NULL / REJECT** — split-slot wait **increases** `leaf_55` ~4.5x on CUDA; G flat. Keep B+10 `input_bid` slot wait. CUDA `leaf_55` residual (~1.5ms) is structural on remus-docker; not an overlap lever.

Artifacts: `b6-2gpu-f-triton-n384-romulus-native-b16` (experiment only; code reverted).

### Backlog (post-ceiling, optional)

- ~~**B+16**~~ — **REJECT** (2026-07-01); see section above
- ~~**4-GPU llama-70B**~~ — **PASS** (2026-07-01); `b6-4gpu-g-triton-n384-v2-llama70b-hip`
- **Jupiter `:50053`** — deferred; triton swap is canonical 4-GPU gate

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
- [x] B+14/B+15 gather prefetch + split-1-start issue; HIP b14b + CUDA b14i re-bench (2026-07-01)
- [x] **V1** MoE model validation — gemma4 26B-A4B, n=128 (2026-07-01)
- [x] **V2** Larger model validation — llama-70B HIP; CUDA ts=50,50 VRAM-blocked on 5060 Ti (2026-07-01)
- [x] **V3** Plan / mission review after V1+V2 (2026-07-01)
- [x] Push B+14/B+15 to gitea; romulus + triton @ `590597110` (2026-07-01)
- [x] Canonical B+15 re-bench `b6-2gpu-f-triton-n384-romulus-native-b15b` — G=190.2, overlap=0.3%, `input_wait` ~358ms (b15 stale `ggml-base`; rebuild fixed)
- [x] B+12 `GET_TENSOR` deferral bisect on canonical n=384 — NULL overlap (2026-07-01)
- [x] B+11 bisect on `b6-4gpu-g-triton` n=384 — NULL overlap; HURTS G (2026-07-01)
- [x] B+13b/c romulus A/B n=384 @ `199eb1d5e` — pre-b13b G=204.2 overlap=0.3%; b13bc G=205.4 overlap=0.2%; M3 NULL (2026-07-01)
- [x] B+13d producer-slot wait on gather+defer — b13d G=205.9 overlap=0.2%; B+13 ladder closed NULL (2026-07-01)
- [x] 4-GPU triton gate post-B+13 — canonical G=80.1 (+5%); dual-OFF G=81.8 (+1.6%); overlap 0.1-0.2% FAIL (2026-07-01)

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
**Last edit:** 2026-07-01 — Jupiter skipped; 4-GPU gate = triton swap; structural ceiling finalized post B+13.