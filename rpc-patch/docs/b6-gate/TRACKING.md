# B+6 gate tracking (Path-B-Plus)

Plan: [PLAN.md](PLAN.md)

**Overall:** IN PROGRESS (triton spike done; B+6 still FAIL; M1 not reached)  
**Branch:** Path-B-Event-Support-Pipeline-Plus  
**Last updated:** 2026-06-29

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
| Verdict | MIXED (straggler drives G; overlap ceiling ~0.3%) |
| remus `straggler_ms_per_token` | 12.40 (backend1 5060, post-B7) |
| triton `straggler_ms_per_token` | 6.95 (backend1 3090) |
| remus `drain_flush_ms` | 16993 (post-B7) |
| triton `drain_flush_ms` | 2293 |
| remus `overlap_pct` / G | 0.1% / ~85 t/s |
| triton `overlap_pct` / G | 0.3% / 187 t/s |

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
| A7 Sync triton tree to branch + rebuild portable | DONE | JUPITER `pathb-portable` scp -> `C:\backup\pathb-portable` (proto 4.3) |
| A8 Triton :50054 RPC smoke | DONE | `PathB-Triton-RPC-50054` schtask; HELLO proto 4.3 peer_copy=yes |
| A9 R5 preflight romulus -> 192.168.8.23:50054 | DONE | validate-rpc OK |
| A10 Git commit/push (user-approved) | IN PROGRESS | staging B6 collateral |

---

## Phase 0 -- Profiler matrix (measurement runs)

| Label | Status | G | overlap_pct | stall_ratio | gate_b6 | Artifact |
|-------|--------|---|-------------|-------------|---------|----------|
| `b6-2gpu-f` (remus) | DONE | 75.56 | 0.2% | 0.949 | FAIL | `benches/path-b-plus/b6-2gpu-f/` |
| `b6-2gpu-f-triton` | DONE | 186.61 | 0.3% | 0.919 | FAIL | `benches/path-b-plus/b6-2gpu-f-triton/` |
| `b6-4gpu-g` | BLOCKED | 53.2* | 0.2%* | 0.96* | FAIL* | romulus->:50053 timeout; use `profiler-4gpu-primary-romulus-trace-v2` |
| `b6-2gpu-f-plus0` | DONE | 72.15 | 0.2% | 0.955 | FAIL | romulus `b6-2gpu-f-plus0/` |
| regression.jsonl append | PENDING | - | - | - | - | - |

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
| P3-1 4-GPU G `-ts` grid (if STRAGGLER/MIXED) | PENDING | - |
| P3-2 Best `-ts` row in regression | PENDING | - |

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