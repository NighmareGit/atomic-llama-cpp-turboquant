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
| M3 | overlap ≥ 5% | 0.2% (2-GPU triton n=384) | FAIL |

**Profiler evidence (b6-2gpu-f):** `stall_ratio=0.95`, `input_wait_copy_ms=4821` vs `graph_compute_async_ms=250`, `EVENT_RECORD` = `drain_flush_ms`. Triton cut drain/straggler but overlap moved 0.2%→0.3% only → **pipelining depth**, not throughput, is the gate lever.

## Phase 2 — Mitigation Experiments (Path-B+ only, no Path C)

Ordered by leverage on `overlap_pct` (profiler-led). One bisect per re-bench.

| ID | Target | Code area | Bench gate |
|----|--------|-----------|------------|
| **B+8** | Partial `pipeline_barrier` — wait dependency frontier only, not all backends | `ggml-backend.cpp` | `b6-2gpu-f` n=384 |
| **B+9** | Defer `EVENT_RECORD` recv to barrier boundary | `ggml-rpc.cpp` | `b6-2gpu-f` n=384 |
| **B+10** | MoE `input_wait_copy` de-sync (APEX path ~1682) | `ggml-backend.cpp` | `b6-2gpu-f` n=384 |
| **B+7a′** | 4-RPC multi-socket flush (canonical 50s vs 5s drain) | `ggml-rpc.cpp` | `b6-4gpu-g` n=384 |
| **B+11** | Dual-socket RPC (cmd + response per endpoint, proto 4.4?) | `ggml-rpc.cpp` | `b6-4gpu-g` |
| **B+12** | `GET_TENSOR` deferral (Path A2-style) | `ggml-rpc.cpp` | either topology |
| **B+13** | Verify `cpy_tensor_async` CUDA→RPC not sync-fallback | `ggml-rpc.cpp` + sched | trace hotpath |

**Recommended implement order:** B+8 → B+9 → B+10 → B+7a′ (4-GPU drain prerequisite if canonical is gate topology).

**Stop rule:** If M1 not reached after B+8–B+10 + B+7a′ on 2-GPU and 4-GPU, document structural ceiling in TRACKING. **Triggered 2026-07-01** — see TRACKING structural ceiling section.

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
```

## Review Gates

- End of Phase 1: Instrumentation + first comparison matrix published
- End of Phase 2: M3 PASS **or** structural ceiling documented with trace proof
- Phase 3 start: Explicit approval only

**Next action (2026-07-01):** Structural ceiling closed. Remaining: PR 8 validate-rpc hygiene, comparison matrix refresh. B+11–B+13 parked (not Path C).