# B+6 overlap gate plan (Path-B-Plus)

Navigation: [TRACKING.md](TRACKING.md) | [rpc-path-b-plus-overview.md](../rpc-path-b-plus-overview.md) | [pathb-sync-site-audit.md](../pathb-sync-site-audit.md)

**Branch:** Path-B-Event-Support-Pipeline-Plus  
**Goal:** Pass B+6 (`overlap_pct >= 5%`) on 2-GPU F without Path C.  
**Tracking:** Update [TRACKING.md](TRACKING.md) after **every** completed step (status + evidence).

Profiler tooling (R1-R5) is **done**; this mission uses `llama-pipeline-profiler` as a read-only measurement harness only.

---

## Baseline (2026-06-28)

| Label | overlap_pct | stall_ratio | straggler | Source |
|-------|-------------|-------------|-----------|--------|
| romulus 4-GPU v2 | 0.2% | 0.96 | 5060 @ 9.2 ms/tok | `benches/path-b-plus/profiler-4gpu-primary-romulus-trace-v2` |
| trace-f-2gpu-plus | 0.6% | 0.68 | remus 5060 | regression seed / offline diagnose |

S5 PASS, B+6 FAIL on all cited production topologies.

---

## Strategy

### 1. Profiler as measurement only

Run `llama-pipeline-profiler` / `romulus-profiler-production-trace.sh` for `diagnose.json` and jsonl traces. Append [`benches/path-b-plus/regression.jsonl`](../../../benches/path-b-plus/regression.jsonl). No further profiler tool development in this mission.

### 2. Core A/B: remus 5060 vs triton 3090 (same client)

Hold **romulus 7900 client** constant; swap only the RPC worker. Triton is faster overall -- isolates **5060 straggler** vs **client-side drain**.

| Label | `--rpc` | Purpose |
|-------|---------|---------|
| `b6-2gpu-f` | `192.168.8.176:50051` | remus 5060 baseline |
| `b6-2gpu-f-triton` | `192.168.8.23:50054` | triton 3090 worker |

| Spike outcome | Verdict | Next |
|---------------|---------|------|
| triton: lower straggler, higher overlap | `STRAGGLER_DOMINANT` | `-ts` tuning; triton as dev RPC default |
| triton: similar drain/overlap | `DRAIN_DOMINANT` | B+7 `ggml-rpc.cpp` first |
| both move | `MIXED` | parallel tracks |

### 3. B+6 milestones

Also mirrored in [rpc-path-b-plus-overview.md](../rpc-path-b-plus-overview.md).

| ID | overlap_pct | stall_ratio | Status |
|----|-------------|-------------|--------|
| Baseline | 0.6% (2-GPU) | 0.68 | CURRENT |
| Spike ref | record delta | record delta | PENDING |
| M1 | >= 1.0% | < 0.80 | PENDING |
| M2 | >= 2.5% | < 0.60 | PENDING |
| M3 PASS | >= 5.0% | < 0.50 | PENDING |

Stop rule: M1 not reached after two B+7 fixes on 2-GPU F -> document structural ceiling; no Path C.

### 4. Deferred

- 5-GPU cluster (triton as 5th hop)
- 3070 as `:50055` (8 GB too small for 35B MoE RPC share)
- Path C unified multi-GPU rpc-server

---

## Phases (map to TRACKING rows)

### Phase A -- Windows CUDA build merge + triton spike prep

- Shared `docs/cuda-windows/BUILD.md` + `scripts/cuda-windows/build.ps1` (`-Profile all|triton`)
- Lateral `docs/cuda-windows-triton/`, `scripts/cuda-windows-triton/pathb-rpc-server.ps1` (`:50054`)
- Sync triton `C:\projects\...\Path-B-Event-Support-Pipeline-Plus`, rebuild portable, R5 smoke

### Phase 0 -- Profiler matrix (ordered)

1. `b6-2gpu-f` (remus)
2. `b6-2gpu-f-triton` (same romulus client)
3. `b6-4gpu-g`
4. `b6-2gpu-f-plus0`

n=384, [`benches/path-b-plus/prompts/profiler-reasoning-long.txt`](../../../benches/path-b-plus/prompts/profiler-reasoning-long.txt), `GGML_PIPELINE_PLUS=1`.

### Phase 1 -- Stall ledger + A/B verdict

Update [pathb-sync-site-audit.md](../pathb-sync-site-audit.md) with trace-proven blockers.

### Phase 2 -- B+7 RPC fixes

One fix per bisect in `ggml-rpc.cpp`; priority from Phase 1 verdict. Re-run triton spike after each fix.

### Phase 3 -- `-ts` sweep

If `STRAGGLER_DOMINANT`: 4-GPU G and 2-GPU F grid. **DONE 2026-06-29:** 5-row grid n=128 + G2/G4 confirm n=384; best grid G2 0.7% (M1 not reached @ n=384).

### Phase 4 -- Gate close

M3 on 2-GPU F; sync overview, GATES, PIPELINE SS9 if needed.

### Phase 5-6 -- Diagnosis + mission routing

After profiler matrix and ts sweep: run `b6-gate-diagnose-runs.sh`, refresh [pathb-sync-site-audit.md](../pathb-sync-site-audit.md), record verdict in [TRACKING.md](TRACKING.md). **2026-06-29 verdict:** D3 drain-bound (canonical 50s vs triton 5.9s) + D1 worker swap partial; next B+7 4-socket bisect.

---

## Path C stop line

No `rpc-server -d CUDA0,CUDA1` aggregation, no parallel split loop in `ggml_backend_sched_compute_splits`.

## Key paths

- Measurement: `tools/llama-pipeline-profiler/`, `scripts/romulus-profiler-production-trace.sh`
- Regression: `benches/path-b-plus/regression.jsonl`
- RPC hot path: `ggml/src/ggml-rpc/ggml-rpc.cpp`
- Gates reference: `tools/llama-pipeline-profiler/GATES.md`