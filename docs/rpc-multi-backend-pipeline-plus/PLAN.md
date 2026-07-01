# PLAN — rpc-multi-backend-pipeline-plus

Navigation: [TRACKING.md](TRACKING.md) | [MISSION.md](MISSION.md) | [rpc-patch/docs/b6-gate/PLAN.md](../../rpc-patch/docs/b6-gate/PLAN.md)

## Phase 0 — Foundation (Complete)

- Path-B-Event-Support merged
- Path-B-Event-Support-Pipeline-Plus branch active
- Windows 5070 Ti native build + portable artifacts
- Config E (2-device) and Config F (3-device) matrix runs complete through 80B
- Stable 4-GPU Config G cluster baseline (`trace-g-4gpu-primary` series)
- Core profiling harness with `-Profile`, 500 ms telemetry, power duty cycle, and artifact bundles
- B+1–B+6 shipped; B+7-7a + B7-1b partial; `llama-pipeline-profiler` + B+6 gate presets

**Deliverables:** `benchmarks/config-*/`, `profile-*/`, `benches/path-b-plus/`, `PROFILING.md`, `MULTI-NODE.md`

## Phase 1 — Instrumentation & Visibility (Current — 2026-06-30)

**Goal:** Make the serial RPC critical path visible; establish bisect evidence for mitigation ladder.

### 1.1 Scheduler / RPC trace enhancements

- Per-graph-split timing + RPC RTT histogram when `Plus=1` (extend existing `GGML_SCHED_TRACE` / `GGML_RPC_TRACE`)
- Expose via `-Profile`, `pathb-profile-parse.ps1`, `pathb-rpc-trace-parse.sh`, `llama-pipeline-profiler`
- Log `cudaEvent` / RPC event wait durations separately from compute
- **Profiler gate presets:** `b6-2gpu-f`, `b6-4gpu-g`, `b6-4gpu-g-triton` (n=384 canonical)

**Success metric:** Operator sees per-split wait vs compute; `diagnose.json` maps to audit blocker IDs (7a–7e).

### 1.2 Topology decision engine

- Extend `pathb-vram-calc.ps1` to emit hard recommendations/warnings
- Add `config-e-recommended` vs `config-f-recommended` presets in matrix scripts
- Enforce "no RX6600 for 35B+ A3B MoE" guard in Config F scripts

### 1.3 Comparison matrix refresh

- Re-run champion + 4-GPU canonical with full profiler telemetry
- Deliverable: `BENCHMARKS/2026-07-comparison-matrix.md` (after 1.1)

## Phase 1b — B+6 Overlap Gate (Active — parallel with 1.1)

**Goal:** Pass M3 (`overlap_pct >= 5%`) without Path C. See [b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md).

**2026-06-29 verdict:** D3 drain-bound + D1 worker-class partial. M1/M3 FAIL.

| Milestone | Target | Best measured | Status |
|-----------|--------|---------------|--------|
| M1 | overlap ≥ 1% | 0.9% (triton guard-n128) | FAIL |
| M3 | overlap ≥ 5% | 0.3% (2-GPU triton n=384 post-B+15) | FAIL |

**2026-07-01 post-B+15:** B+14/B+15 fixed gather stalls (HIP `input_wait` ~355us vs ~4ms pre-fix). Overlap at canonical n=384 still 0.3%. Validation on gemma4/llama-70B hit 0.8–1.3% @ n=128 — graph-dependent, not gate depth.

**2026-07-01 post-B+11:** B+12 defer **NULL** on overlap (+1.8% G, shipped). B+11 dual-socket **NULL** on overlap and **-9.1% G** when `GGML_RPC_DUAL_SOCKET=1` (default OFF; proto 4.4 ships). **Next experiment: B+13** — prove/fix silent `sync_copy_fallback` on the local→RPC upload path.

## Phase 2 — Mitigation Experiments (Path-B+ only, no Path C)

Ordered by leverage on `overlap_pct` (profiler-led). One bisect per re-bench.

| ID | Target | Code area | Bench gate | Status (2026-07-01) |
|----|--------|-----------|------------|---------------------|
| **B+8** | Partial `pipeline_barrier` — wait dependency frontier only, not all backends | `ggml-backend.cpp` | `b6-2gpu-f` n=384 | NULL overlap |
| **B+9** | Defer `EVENT_RECORD` recv to barrier boundary | `ggml-rpc.cpp` | `b6-2gpu-f` n=384 | NULL overlap |
| **B+10** | MoE `input_wait_copy` de-sync (APEX path ~1682) | `ggml-backend.cpp` | `b6-2gpu-f` n=384 | NULL overlap |
| **B+7a′** | 4-RPC multi-socket flush (canonical 50s vs 5s drain) | `ggml-rpc.cpp` | `b6-4gpu-g` n=384 | NULL overlap; drain helped |
| **B+12** | `GET_TENSOR` deferral (Path A2-style) | `ggml-rpc.cpp` | `b6-2gpu-f-triton` n=384 | NULL overlap; **+1.8% G** (shipped) |
| **B+11** | Dual-socket RPC (cmd + response per endpoint, proto 4.4) | `ggml-rpc.cpp` | `b6-4gpu-g-triton` n=384 | NULL overlap; **-9.1% G** when ON; default OFF |
| **B+13** | Verify `cpy_tensor_async` local→RPC not sync-fallback | `ggml-rpc.cpp` + sched | `b6-2gpu-f-triton` n=384 | **ACTIVE** |
| **B+16** | CUDA `leaf_55` MoE weight path | CUDA backend | remus docker | G lever only; weak overlap ROI |

**Recommended implement order:** B+8 → B+9 → B+10 → B+7a′ (4-GPU drain prerequisite if canonical is gate topology).

**Post-B+15 hunt order (2026-07-01):** B+14/B+15 shipped → B+12 **NULL** (defer shipped) → B+11 **NULL** (dual OFF) → **B+13** (sync_copy_fallback) → B+16 optional G-only.

**Stop rule:** If M1 not reached after B+8–B+10 + B+7a′ on 2-GPU and 4-GPU, document structural ceiling in TRACKING. **Triggered 2026-07-01** — see TRACKING structural ceiling section. Ceiling coexists with continued B+13 hunt until M3 PASS or explicit mission revision.

### 2.1 B+11 result (closed)

Gate `b6-4gpu-g-triton` n=384, romulus client, proto 4.4 on remus/romulus/triton.

| Arm | `GGML_RPC_DUAL_SOCKET` | G (t/s) | overlap_pct | Notes |
|-----|------------------------|---------|-------------|-------|
| bisect ON (`canonical-romulus` in bg script) | 1 | 73.2 | 0.2% | hol_tail 864ms spike; **slower** |
| bisect OFF (`no-dual-socket`) | 0 | **80.5** | 0.2% | production default |

**Do not read ON as baseline.** Dual-socket was the experiment; OFF is what ships. Details: [FEATURE-b11-dual-socket-rpc.md](FEATURE-b11-dual-socket-rpc.md), [TRACKING.md](TRACKING.md) B+11 section.

### 2.2 B+13 active (next)

**Initial hypothesis:** `sync_copy_fallback` when `cpy_tensor_async` fails blocks the upload path.

**Audit result (2026-07-01, canonical `b6-2gpu-f-triton-n384-romulus-native` C-full trace):**

| Phase | ms (gen) | Notes |
|-------|----------|-------|
| `sync_copy_fallback` | **0** | `copy_async_ok` = 2310 — async path succeeds |
| `input_wait_copy` | 2806 | split 2 (RPC hop) = 2751 ms |
| `event_sync_slot` | 2705 | backend 1 only — pipeline slot wait |
| `input_copy_slow` | 2760 | `l_out-16` RPC→CUDA0 = 2717 ms (382 rows, reject=`unknown`) |
| `rpc_flush_downloads` | 216 | GET_TENSOR defer flush |
| `graph_compute_async` | 307 | compute not the bottleneck |

**Revised hypothesis:** Overlap is blocked by **split-2 input drain on the RPC→client path** — correlated `event_sync_slot` + `input_copy_slow` for `l_out-16` (MoE output RPC→CUDA0), not by `sync_copy_fallback`. B+13 work shifts to: why `input_copy_slow` fires with `reject=unknown` despite `copy_async_ok`, and whether slot waits can overlap with RPC download.

**Code areas:** `ggml_backend_sched_compute_splits` (`input_copy_slow`, `event_sync_slot`), `ggml_backend_rpc_try_download_tensor`, `rpc_flush_downloads` / B+12 defer boundary.

**Pass criteria (M3):**

- `overlap_pct` delta >= 1% vs canonical on `b6-2gpu-f-triton-n384-romulus-native`, or
- `input_copy_slow` + `event_sync_slot` ms down >= 50% on split 2 with stable G

**Execution steps:**

1. [x] C-full audit — `scripts/b6-gate-phase12c-blocking-audit.sh` (see `telemetry/blocking-audit-c.json`)
2. [x] Code audit — [DESIGN-b13-input-wait-audit.md](DESIGN-b13-input-wait-audit.md); trace `reject` labels fixed (B+13a)
3. [ ] B+13b: issue split-2 deferred GETs before `wait_copy_slot` on gather splits
4. [ ] B+13c: tighten B+15 prefetch overlap (split 1 end sync vs start issue)
5. Re-gate `b6-2gpu-f-triton-n384-romulus-native`; optional 4-GPU if 2-GPU overlap moves

**Trace env:** `GGML_SCHED_TRACE=1`, `GGML_RPC_TRACE=1` (C-full phases in [IMPLEMENTATION.md](IMPLEMENTATION.md)).

**Path C boundary:** Path C (server-side scheduling / distributed orchestration) is **out of scope** for `rpc-multi-backend-pipeline-plus`. Reference Path C docs for ideas only; do not implement Path C on this branch.

### 2.2 Straggler / ops (throughput, not overlap gate)

- Triton `:50054` as RPC2 eval vs G4 `-ts` `30,14,16,40`
- Formal RX6600 exclusion from 35B+ MoE scripts

### 2.3 Telemetry & operator UX

- `b6-gate-diagnose-runs.sh` after each bisect
- Append `benches/path-b-plus/regression.jsonl`
- Regression guard for 48.9 t/s 2-device baseline

## Phase 3 — Path-C Bridge (**out of scope**)

**Not part of path-b-plus.** Structural ceiling is documented; M3 remains FAIL. Path C may be pursued in a separate effort.

- This mission may **read** [rpc-patch/docs/rpc-path-c-plan.md](../../rpc-patch/docs/rpc-path-c-plan.md) for inspiration
- **No Path C implementation** on `Path-B-Event-Support-Pipeline-Plus`
- Production cluster baseline (`trace-g-4gpu-primary` ~43 t/s) remains a throughput reference only

## Phase 4 — Production Hardening

- Default topology matrix (model class → config)
- Runbooks for Windows client + Linux RPC workers
- Performance SLOs (t/s, tail latency, GPU duty cycle, overlap gate)
- Documentation sign-off

## Dependencies & Risks

| Risk | Mitigation | Phase |
|------|------------|-------|
| M3 unreachable without Path C | Honest ceiling doc; don't block 48.9 t/s production ship | 2 |
| Core scheduler changes need upstream coordination | Isolate behind `GGML_PIPELINE_PLUS` | 1b–2 |
| Instrumentation overhead | Off by default; profiler-only | 1.1 |
| Path C breaks mother-repo compatibility | Defer until Phase 3 entry criteria met | 3 |

## Command Examples

```powershell
# Production champion (Phase 0)
.\scripts\cuda-windows-5070ti\rpc-server-bench.ps1 `
  -Label trace-f-2gpu-plus -Config config-f `
  -RpcEndpoint 192.168.8.176:50051 `
  -TensorSplit 50,50 -Profile -GenTokens 256 -Runs 3 -EnsurePathbRpc
```

```bash
# B+6 gate (romulus)
bash scripts/b6-gate-run-remote.sh b6-2gpu-f
bash scripts/b6-gate-diagnose-runs.sh b6-2gpu-f b6-4gpu-g

# B+13 proof (canonical 2-GPU triton n=384)
bash scripts/b6-gate-bisect-run.sh canonical-romulus
bash scripts/b6-gate-phase12a-waterfall.sh b6-2gpu-f-triton-n384-romulus-native
bash scripts/b6-gate-phase12c-blocking-audit.sh b6-2gpu-f-triton-n384-romulus-native
```

## Review Gates

- End of Phase 1: Instrumentation + first comparison matrix published
- End of Phase 2: M3 PASS **or** structural ceiling documented with trace proof
- Phase 3 start: Explicit approval only

**Next action (2026-07-01):** B+11 closed — NULL overlap, -9.1% G when dual ON; default OFF. **Proceed B+13:** C-full audit of `sync_copy_fallback` on `b6-2gpu-f-triton-n384-romulus-native`, fix async upload path, re-gate n=384. Details: [TRACKING.md](TRACKING.md), [IMPLEMENTATION.md](IMPLEMENTATION.md).