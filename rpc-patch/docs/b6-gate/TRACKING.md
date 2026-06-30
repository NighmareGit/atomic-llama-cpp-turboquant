# B+6 gate tracking (Path-B-Plus)

Plan: [PLAN.md](PLAN.md)

**Overall:** IN PROGRESS (4-GPU triton A/B + ts sweep done; B+6 still FAIL; M1 not reached)  
**Branch:** Path-B-Event-Support-Pipeline-Plus  
**Last updated:** 2026-06-30  
**Mission root:** [docs/rpc-multi-backend-pipeline-plus/](../../../docs/rpc-multi-backend-pipeline-plus/)

> Update this file after **each** completed plan step: set Status, Evidence (path or label), and bump **Last updated**. Mirror milestones in [rpc-path-b-plus-overview.md](../rpc-path-b-plus-overview.md).

---

## Milestones

| ID | overlap_pct | stall_ratio | Status | Evidence |
|----|-------------|-------------|--------|----------|
| Baseline (2-GPU remus) | 0.6% | 0.68 | CURRENT | `trace-f-2gpu-plus` / regression seed |
| Baseline (4-GPU romulus) | 0.2% | 0.96 | CURRENT | `profiler-4gpu-primary-romulus-trace-v2` |
| Spike ref (remus vs triton) | 0.1% -> 0.3% | 0.95 -> 0.92 | DONE | `b6-2gpu-f-triton` |
| M1 | >= 1.0% | < 0.80 | PENDING | - |
| M2 | >= 2.5% | < 0.60 | PENDING | - |
| M3 (B+6 PASS) | >= 5.0% | < 0.50 | PENDING | - |

## A/B verdict

| Field | Value |
|-------|-------|
| Verdict | **MIXED** (4-GPU: drain topology-bound; overlap ceiling ~0.2-0.7%) |
| 2-GPU remus `straggler_ms_per_token` | 12.40 (backend1 5060, post-B7) |
| 2-GPU triton `straggler_ms_per_token` | 6.95 (backend1 3090) |
| 2-GPU remus `drain_flush_ms` | 16993 (post-B7) |
| 2-GPU triton `drain_flush_ms` | 2293 |
| **4-GPU canonical** `b6-4gpu-g` n=384 ts=25,12,25,38 | overlap 0.1%, drain **50400**, straggler backend3 5070 @ 11.6 ms/tok, G=77.2 |
| **4-GPU triton** `b6-4gpu-g-triton` n=384 ts=22,11,34,33 | overlap 0.1%, drain **5924**, straggler backend3 3090 @ 9.0 ms/tok, G=63.1 |
| ts sweep best n=128 | G2 legacy `36,24,24,16` overlap **0.7%**, drain 2312, straggler backend1 5060 |
| ts confirm best n=384 | G4 `30,14,16,40` overlap **0.2%**, drain 4837, G=75.8 |

---

## Phase A -- Build merge + triton spike prep

| Step | Status | Evidence |
|------|--------|----------|
| A1 Shared `docs/cuda-windows/BUILD.md` | DONE | `docs/cuda-windows/BUILD.md` |
| A2 Shared `scripts/cuda-windows/build.ps1` (-Profile) | DONE | `scripts/cuda-windows/build.ps1` |
| A3 5070ti build forwarder | DONE | `scripts/cuda-windows-5070ti/build.ps1` |
| A4 `docs/cuda-windows-triton/` host stub | DONE | `docs/cuda-windows-triton/README.md` |
| A5 `scripts/cuda-windows-triton/pathb-rpc-server.ps1` :50054 | DONE | `scripts/cuda-windows-triton/pathb-rpc-server.ps1` |
| A6 Audit triton `C:\projects\...` tree | DONE | SSH OK; 3090+3070; `C:\backup\lcuda` legacy |
| A7 Sync triton tree to branch + rebuild portable | DONE | git `948c6a534`; JUPITER build -> `C:\backup\pathb-portable` + repo `build-cuda-b-bin/portable` |
| A8 Triton :50054 RPC smoke | DONE | `PathB-Triton-RPC-50054` schtask; HELLO proto 4.3 peer_copy=yes |
| A9 R5 preflight romulus -> 192.168.8.23:50054 | DONE | validate-rpc OK |
| A10 Git commit/push (user-approved) | DONE | `948c6a534` on gitea |

---

## Phase 0 -- Profiler matrix (measurement runs)

| Label | Status | G | overlap_pct | stall_ratio | gate_b6 | Artifact |
|-------|--------|---|-------------|-------------|---------|----------|
| `b6-2gpu-f` (remus) | DONE | 75.56 | 0.2% | 0.949 | FAIL | `benches/path-b-plus/b6-2gpu-f/` |
| `b6-2gpu-f-triton-guard-n128` | DONE | 116.0 | 0.9% | 0.6357 | FAIL | guard run (MIXED); stall improved |
| `b6-2gpu-f-triton-n384` | DONE | 128.8 | 0.2% | 0.867 | FAIL | canonical (STRAGGLER_DOMINANT) |
| `b6-2gpu-f-triton` | DONE | 186.61 | 0.3% | 0.919 | FAIL | `benches/path-b-plus/b6-2gpu-f-triton/` |
| `b6-4gpu-g` | DONE | 77.2 wall / 37.0 diag | 0.1% | 0.929 | FAIL | `benches/path-b-plus/b6-4gpu-g/` n=384, ts=25,12,25,38, JUPITER OK |
| `b6-4gpu-g-triton` | DONE | 63.1 | 0.1% | 0.958 | FAIL | `benches/path-b-plus/b6-4gpu-g-triton/` n=384, ts=22,11,34,33 |
| `b6-2gpu-f-plus0` | DONE | 72.15 | 0.2% | 0.955 | FAIL | romulus `b6-2gpu-f-plus0/` |
| regression.jsonl append | DONE | - | - | - | - | triton + ts sweep rows on romulus |

---

## Phase 1 -- Stall ledger

| Step | Status | Evidence |
|------|--------|----------|
| P1-1 Stall ledger from profiler artifacts | DONE | `b6-2gpu-f` diagnose + trace-summary |
| P1-2 A/B verdict (STRAGGLER/DRAIN/MIXED) | DONE | MIXED preliminary (triton pending) |
| P1-3 `pathb-sync-site-audit.md` updated | DONE | B+7 candidates table |

---

## Phase 2 -- B+7 RPC fixes

| Fix | Status | overlap_pct delta | blocking_rpc delta | Evidence |
|-----|--------|-------------------|--------------------|----------|
| B7-7a socket-scoped GET flush | DONE | 0.2% -> 0.1% (noise) | unchanged (1 RPC sock) | `ggml-rpc.cpp` `flush_pending_get_tensor_for_socket` |
| B7-1b GET_ALLOC_SIZE cache | DONE | 0.1% (no move) | 2416 -> 808 | `ggml-rpc.cpp` `tls_alloc_size_cache` |
| Post-fix triton re-spike | DONE | 0.1% -> 0.3% | drain 16993 -> 2293 | `b6-2gpu-f-triton` |

**Phase 2 verdict:** Two B+7 fixes shipped; **M1 not reached** on 2-GPU F (`overlap_pct=0.1%`). Straggler-bound (`backend1` 12.4 ms/tok). Socket-scoped flush likely needs **4-GPU** to show drain benefit. Stop-rule note: structural ceiling on single-RPC 2-GPU unless triton A/B or `-ts` moves straggler.

---

## Phase 3 -- ts sweep

| Step | Status | Evidence |
|------|--------|----------|
| P3-1 4-GPU G `-ts` grid (5 rows n=128) | DONE | `benches/path-b-plus/b6-4gpu-ts-sweep/` G0-G4 |
| P3-2 Confirm top 2 n=384 | DONE | G2-confirm 0.2%, G4-confirm 0.2% (ranked G2, G4 from grid) |
| P3-3 Best grid overlap | DONE | G2 `36,24,24,16` **0.7%** @ n=128 (M1 not reached @ n=384) |

## Phase 5-6 -- Diagnosis + mission routing

| Step | Status | Evidence |
|------|--------|----------|
| P5-1 Diagnosis matrix | DONE | `benches/path-b-plus/b6-diagnosis-matrix.tsv` on romulus |
| P5-2 Drain post-mortem | DONE | canonical ts=25,12,25,38 drain 50s vs triton 5.9s vs G4-confirm 4.8s |
| P6-1 Mission verdict | DONE | **D3 + D1** (see below) |

**Phase 6 verdict (primary D3 + secondary D1):**

- **D3 DRAIN-bound on JUPITER canonical split:** `b6-4gpu-g` ts=25,12,25,38 n=384 drain **50.4s** vs same-topology ts rows at n=128 drain **1.5-2.3s** vs G4-confirm n=384 drain **4.8s**. Drain is highly `-ts` and token-count sensitive on 3-RPC JUPITER topology; B+7 multi-socket flush bisect is the next code track.
- **D1 Worker-class (partial):** triton 4-GPU swap cuts drain **50.4s -> 5.9s** with same overlap (0.1%); straggler improves 11.6 -> 9.0 ms/tok. Overlap does not move -> not sufficient alone for M1.
- **D5 Stop-rule note:** No row reaches M1 (1%) at n=384 confirm; best grid peek G2 **0.7%** @ n=128 only.

**Next scheduled mission items:** B+8 partial `pipeline_barrier` → B+9 EVENT defer → B+10 MoE copy-wait ([PLAN.md](../../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) Phase 2); then B+7a′ 4-GPU drain. Ops eval of triton `:50054` parallel only.

---

## Phase 4 -- Gate close

| Step | Status | Evidence |
|------|--------|----------|
| P4-1 M3 PASS on 2-GPU F | PENDING | - |
| P4-2 `rpc-path-b-plus-overview.md` milestones | PENDING | - |
| P4-3 `GATES.md` / PIPELINE SS9 if metrics shifted | PENDING | - |

---

## Deferred

| Item | Status | Notes |
|------|--------|-------|
| 5-GPU triton as 5th hop | DEFERRED | After M1 or scale need |
| 3070 `:50055` | DEFERRED | 8 GB VRAM |
| Path C | OUT OF SCOPE | - |

---

## Implementation log

| Date | Step | Action | Artifact |
|------|------|--------|----------|
| 2026-06-28 | - | Plan + tracking created under `rpc-patch/docs/b6-gate/` | PLAN.md, this file |
| 2026-06-28 | A1-A5 | Shared cuda-windows build + triton collateral | docs/cuda-windows/, scripts/ |
| 2026-06-28 | A6 | Triton unreachable (nc timeout :22 :50054) | remus :50051 OK, romulus :22 OK |
| 2026-06-28 | - | Added `scripts/b6-gate-profiler-romulus.sh` | Phase 0 presets |
| 2026-06-28 | P0 | `b6-2gpu-f` romulus: G=75.56, overlap=0.2%, stall=0.95, gate_b6 FAIL | `benches/path-b-plus/b6-2gpu-f/` |
| 2026-06-28 | P0 | `b6-2gpu-f-plus0`: G=72.15, overlap=0.2% (Plus does not move overlap) | romulus |
| 2026-06-28 | P0 | `b6-4gpu-g` blocked: romulus->192.168.8.21:50053 timeout | use trace-v2 seed |
| 2026-06-28 | P1 | Stall ledger + MIXED verdict in audit | `pathb-sync-site-audit.md` B+7 table |
| 2026-06-29 | P2 | B7-7a socket-scoped flush; romulus rebuild `3efe5b3d8-dirty` | `ggml-rpc.cpp` |
| 2026-06-29 | P2 | B7-1b GET_ALLOC_SIZE shape cache | blocking_rpc 808, overlap 0.1% FAIL |
| 2026-06-29 | P0 | Post-B7 `b6-2gpu-f` romulus | `benches/path-b-plus/b6-2gpu-f-b7-combined/` |
| 2026-06-29 | A6 | Triton sshd installed+running; LAN :22 still closed (firewall) | SMB `\\192.168.8.23\C$` works |
| 2026-06-29 | A6 | Staged `b6-gate-triton-sshd-firewall.ps1` + Startup hook on triton | needs one local run or re-login |
| 2026-06-29 | A7-A9 | Triton SSH + pathb-portable :50054 + schtask | proto 4.3 peer_copy=yes |
| 2026-06-29 | P0 | `b6-2gpu-f-triton` spike | G=186.6, overlap=0.3%, straggler=6.95ms/tok |
| 2026-06-29 | A10 | Commit/push B6 collateral to gitea | `948c6a534` |
| 2026-06-29 | A7 | Triton git sync + JUPITER rebuild deploy | `948c6a534`, `C:\backup\pathb-portable` |
| 2026-06-29 | JUPITER | abort() root cause: portable built `86-real` only; `GGML_ASSERT` on failed graph_compute | rebuild `ggml-cuda.dll` with `120a-real`; `b6-gate-jupiter-rebuild-rpc.cmd` |
| 2026-06-29 | P3 | `-ts` rework: RPC-first VRAM split 4-GPU `25,12,25,38`, 5-GPU `22,11,22,11,34` | `b6-gate-profiler-romulus.sh`, validate-rpc OK all endpoints |
| 2026-06-29 | P0 | `b6-4gpu-g` smoke n=16 with JUPITER :50053 alive | G=42, overlap=2%, straggler backend3 (5070) 57ms/tok |
| 2026-06-29 | P0 | `b6-4gpu-g` full n=384 canonical 4-GPU + JUPITER | G=77.2, overlap=0.1%, drain=50.4s, straggler backend3 11.6ms/tok, gate_b6 FAIL |
| 2026-06-29 | P0 | `b6-4gpu-g-triton` n=384 A/B | G=63.1, overlap=0.1%, drain=5.9s, straggler backend3 3090 9.0ms/tok |
| 2026-06-29 | P3 | ts grid G0-G4 n=128 + G2/G4 confirm n=384 | best grid G2 0.7%; confirm G4 0.2% drain=4.8s |
| 2026-06-29 | P6 | Mission routing D3+D1 | diagnosis matrix; M1 not reached |
| 2026-06-30 | Docs | `rpc-multi-backend-pipeline-plus/` mission root incorporated | PLAN Phase 2 B+8–B+13 ladder; M3 hard criterion |