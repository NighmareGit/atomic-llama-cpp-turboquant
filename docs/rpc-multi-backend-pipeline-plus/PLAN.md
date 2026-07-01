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

## Phase 1 — Instrumentation & Visibility (Complete — 2026-07-01)

**Goal:** Make the serial RPC critical path visible; establish bisect evidence for mitigation ladder.

### 1.1 Scheduler / RPC trace enhancements

- Per-graph-split timing + RPC RTT histogram when `Plus=1` (extend existing `GGML_SCHED_TRACE` / `GGML_RPC_TRACE`)
- Expose via `-Profile`, `pathb-profile-parse.ps1`, `pathb-rpc-trace-parse.sh`, `llama-pipeline-profiler`
- Log `cudaEvent` / RPC event wait durations separately from compute
- **Profiler gate presets:** `b6-2gpu-f-triton`, `b6-4gpu-g-triton` (n=384 canonical). `b6-4gpu-g` (jupiter) **deferred** — use triton `:50054` as RPC2.

**Success metric:** Operator sees per-split wait vs compute; `diagnose.json` maps to audit blocker IDs (7a–7e).

### 1.2 Topology decision engine (partial — 2026-07-01)

- [x] `pathb-rpc-vram-preflight.py` live probe + `--ts-mode equal` + planning reserves (`a9fbf3a5d`)
- [x] Presets: `b6-5gpu-g-prod`, `b6-4gpu-g-triton`, gate integration via `PATHB_VRAM_PREFLIGHT=1`
- [ ] Extend `pathb-vram-calc.ps1` Windows-side warnings (optional)
- [x] RX6600 exclusion documented in TRACKING topology decisions
- [ ] P2 `--probe-fit` / P3 calibration — deferred ([MISSION.md](MISSION.md))

### 1.3 Comparison matrix refresh

- Re-run champion + 4-GPU canonical with full profiler telemetry
- Deliverable: `BENCHMARKS/2026-07-comparison-matrix.md` (after 1.1)

## Phase 1b — B+6 Overlap Gate (Closed — structural ceiling 2026-07-01)

**Goal:** Pass M3 (`overlap_pct >= 5%`) without Path C. See [b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md).

**2026-06-29 verdict:** D3 drain-bound + D1 worker-class partial. M1/M3 FAIL.

| Milestone | Target | Best measured | Status |
|-----------|--------|---------------|--------|
| M1 | overlap ≥ 1% | 0.9% (triton guard-n128) | FAIL |
| M3 | overlap ≥ 5% | 0.3% (2-GPU triton n=384 post-B+15) | FAIL |

**2026-07-01 post-B+15:** B+14/B+15 fixed gather stalls (HIP `input_wait` ~355us vs ~4ms pre-fix). Overlap at canonical n=384 still 0.3%. Validation on gemma4/llama-70B hit 0.8–1.3% @ n=128 — graph-dependent, not gate depth.

**2026-07-01 post-B+13:** Full ladder B+8–B+13 + B+11/B+12 **NULL** on M3. Structural ceiling documented. Hunt pauses unless new hypothesis.

**2026-07-01 post-B+14 wavefront:** W1+W2 factorial on 5-GPU prod — `global_3bk` < 1%; default OFF. Not an overlap lever.

## Phase 2 — Mitigation Experiments (Path-B+ only, no Path C) — **Ladder exhausted**

Ordered by leverage on `overlap_pct` (profiler-led). One bisect per re-bench.

| ID | Target | Code area | Bench gate | Status (2026-07-01) |
|----|--------|-----------|------------|---------------------|
| **B+8** | Partial `pipeline_barrier` — wait dependency frontier only, not all backends | `ggml-backend.cpp` | `b6-2gpu-f` n=384 | NULL overlap |
| **B+9** | Defer `EVENT_RECORD` recv to barrier boundary | `ggml-rpc.cpp` | `b6-2gpu-f` n=384 | NULL overlap |
| **B+10** | MoE `input_wait_copy` de-sync (APEX path ~1682) | `ggml-backend.cpp` | `b6-2gpu-f` n=384 | NULL overlap |
| **B+7a′** | 4-RPC multi-socket flush (canonical 50s vs 5s drain) | `ggml-rpc.cpp` | `b6-4gpu-g` n=384 | NULL overlap; drain helped |
| **B+12** | `GET_TENSOR` deferral (Path A2-style) | `ggml-rpc.cpp` | `b6-2gpu-f-triton` n=384 | NULL overlap; **+1.8% G** (shipped) |
| **B+11** | Dual-socket RPC (cmd + response per endpoint, proto 4.4) | `ggml-rpc.cpp` | `b6-4gpu-g-triton` n=384 | NULL overlap; **-9.1% G** when ON; default OFF |
| **B+13** | Verify `cpy_tensor_async` local→RPC not sync-fallback | `ggml-rpc.cpp` + sched | `b6-2gpu-f-triton` n=384 | NULL overlap; **+G** (shipped) |
| **B+14** | Wavefront assembly line W1+W2 | `ggml-backend.cpp` | `b6-5gpu-g-prod` n=64/384 | NULL overlap; default OFF |
| **B+15** | L4 layer spread + VRAM planning | preflight scripts | `b6-5gpu-g-prod` 70B/72B | **PASS** deploy; not overlap |
| **B+16** | CUDA `leaf_55` MoE weight path | CUDA backend | remus docker | **REJECT** |

**Recommended implement order:** B+8 → B+9 → B+10 → B+7a′ (4-GPU drain prerequisite if canonical is gate topology).

**Ladder order (final):** B+8–B+10 → B+7a′ → B+12 → B+11 → B+13 → B+14 wavefront → B+16. All **NULL** on M3 except G/stall improvements.

**Stop rule:** **Triggered 2026-07-01** — structural ceiling in TRACKING. No further overlap bisects without explicit new hypothesis and V5 review.

### 2.5 — 5-GPU production deploy (2026-07-01)

**Goal:** Run 70B+ dense on all-Linux 5-GPU without 3060 OOM.

| Step | Status |
|------|--------|
| Planning reserves in preflight | **shipped** |
| L4 equal-safe TS for A8/A13 | **PASS** (4/4 loads) |
| Wavefront default OFF on prod | **shipped** |
| Optional n=384 L4 confirmation | backlog |

```bash
# Pre-deploy (romulus or dev host with SSH to cluster)
bash scripts/pathb-rpc-vram-preflight.sh --preset b6-5gpu-g-prod \
  --gguf /mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf \
  --ts-mode equal --phase load

# Bench spike (optional)
bash scripts/b6-gate-b15-l4-layer-spread-spike.sh
```

### 2.1 B+11 result (closed)

Gate `b6-4gpu-g-triton` n=384, romulus client, proto 4.4 on remus/romulus/triton.

| Arm | `GGML_RPC_DUAL_SOCKET` | G (t/s) | overlap_pct | Notes |
|-----|------------------------|---------|-------------|-------|
| bisect ON (`canonical-romulus` in bg script) | 1 | 73.2 | 0.2% | hol_tail 864ms spike; **slower** |
| bisect OFF (`no-dual-socket`) | 0 | **80.5** | 0.2% | production default |

**Do not read ON as baseline.** Dual-socket was the experiment; OFF is what ships. Details: [FEATURE-b11-dual-socket-rpc.md](FEATURE-b11-dual-socket-rpc.md), [TRACKING.md](TRACKING.md) B+11 section.

### 2.2 B+13 result (closed)

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
3. [x] B+13b: issue split-2 deferred GETs before `wait_copy_slot` on gather splits (`rpc_gather_prefetch_early`)
4. [x] B+13c: skip RPC `event_synchronize` at split-1 end when defer on (`rpc_prefetch_end`)
5. [x] Re-gate romulus n=384 — pre-b13b / b13b / b13bc A/B (`199eb1d5e`); overlap 0.3% all arms; G ~204 t/s
6. [x] B+13d: gather+defer uses producer `event_wait` instead of full copy-slot wait (`event_wait_producer_slot`)

**B+13 closed (NULL on M3 overlap).** Structural ceiling finalized (TRACKING). Backlog: B+16 G-only (optional).

### 2.3 B+13 4-GPU gate (2026-07-01, `d5c5f2fb0`, triton swap)

| Arm | G (t/s) | overlap | stall_ratio | vs pre-B+13 |
|-----|---------|---------|-------------|-------------|
| canonical (dual ON) | 80.1 | 0.1% | 0.70 | +5% G (76.3) |
| dual OFF (ships) | **81.8** | 0.2% | 0.68 | +1.6% G (80.5) |

M3 still FAIL. B+13 cuts orchestration stall on 4-GPU; G gain is real but overlap gate unchanged. Production 4-GPU: keep **dual OFF** (B+11).

### 2.4 4-GPU topology (locked — jupiter skipped)

Triton provides the third RPC hop (3090 `:50054`) in place of jupiter 5070 `:50053`. Romulus validate-rpc reports jupiter `register failed`; do not gate on jupiter ops.

```bash
B6_GATE_PRESET=b6-4gpu-g-triton bash scripts/b6-gate-bisect-run.sh canonical-romulus
B6_GATE_PRESET=b6-4gpu-g-triton bash scripts/b6-gate-bisect-run.sh no-dual-socket  # production ships
```

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

## Phase 1c — Assembly line production (Active — 2026-07-01)

**Goal:** Working 5-GPU assembly line for deploy — all GPUs loaded, stable G, operator runbook. **Not** M3 `overlap_pct` (closed).

Overlap hunt on this branch is **done**. See [DESIGN-b14-parallel-assembly-line.md](DESIGN-b14-parallel-assembly-line.md) section 10.

| Step | Work | Gate |
|------|------|------|
| 1 | Deploy runbook + equal-safe preflight | 70B+ load PASS |
| 2 | L1 `GGML_RPC_HASH_DEFER` spike @ n=384 | G >= 0%; no regression |
| 3 | L4 n=384 confirm (A8/A13) | optional |
| 4 | Path C bridge criteria doc | when to escalate W2 |

**Next action (2026-07-01, V5):** Branch **deploy-ready**. Continue Phase 1c: L1 spike, then optional n=384 L4. Details: [TRACKING.md](TRACKING.md) V4; assembly line: DESIGN-b14 s10.