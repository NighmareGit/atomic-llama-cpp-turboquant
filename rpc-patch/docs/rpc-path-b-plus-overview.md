# Path-B Plus: Project Overview

Entry point for the **Path-B-Event-Support-Pipeline-Plus** branch. Path-B Plus finishes the assembly-line pipeline that Path B started for multi-RPC llama.cpp clusters.

**Primary repo (Windows):** `D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant` -- edit, commit, and push here (not Grok worktrees).

**Status (2026-07-01):** B+11 dual-socket shipped (proto 4.4, default OFF); B+12 NULL; M3 hunt -> B+13. Mission root: [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/).
**Protocol:** RPC v4.4.2 (`CHANNEL_BIND` dual-socket; `COPY_TENSOR_PEER`; default single-socket HELLO)  
**Frozen base:** `Path-B-Event-Support` @ `26a9353`

---

## What this branch is

Path B (v4.2.2) enabled event-based pipeline parallelism (`sched copies = 4`) across local CUDA and remote RPC backends. On multi-hop Windows/remus topologies, throughput plateaued because the scheduler still forced full synchronization on every graph reuse and RPC drains emptied overlap windows.

Path-B Plus unblocks that pipeline and adds per-hop copy improvements. The branch is a **private fork extension** -- not intended for upstream llama.cpp PR submission as-is.

| Layer | Scope |
|-------|-------|
| **B+1 (Tier 0)** | Scheduler barrier, narrow sampling sync, scoped RPC drain, trace/overlap metrics |
| **Tier 1** | Async COPY, deferred COPY drain, same-host peer COPY (proto 4.3) |
| **Phase 5** | Production topology (2-device F) + validation spikes |
| **Deferred** | Path C unified remus rpc-server; 5-endpoint 72B+ cluster (S0) |

---

## Problem and fix (one paragraph)

Multi-RPC inference runs graph splits **serially** per token. Path B rotated pipeline copy slots only on first allocation; graph reuse reused the same buffers and triggered `sched_synchronize` on every token. RPC `send_rpc_cmd` drained pending events on fire-and-forget sends, collapsing overlap. Path-B Plus fixes copy rotation (P0), narrows post-decode sync to sampling backends (P1), scopes drain to blocking RPC ops (P2), and instruments overlap for validation (P3).

Full root-cause analysis: [RPC-BUG-HUNT.md](../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md). Implementation plan: [rpc-path-b-plus-plan.md](rpc-path-b-plus-plan.md).

---

## Measured results

Primary bench model: `Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf`, ctx=4096, q4_0 KV, ngl=99, Windows RTX 5070 Ti + remus RTX 5060 Ti.

| Topology | Plus | G (t/s) | vs pre-Plus | Notes |
|----------|------|---------|-------------|-------|
| 3-device F (`ts=30,12,58`, + RX6600) | 0 | 38.0 | baseline | bug-hunt pre-Plus |
| 3-device F | 0 | 39.3 | +3% | legacy; broken copy rotation |
| 3-device F | 1 | **42.8** | +13% | best 3gpu; still below 45+ target |
| 2-device F (`ts=50,50`, :50051 only) | 1 | **48.9** | +29% vs 3gpu pre | **production default** |

Correctness: S4 smoke PASS on gemma-4-E4B (44.2 t/s) and 36B NL (48.9 t/s). Artifacts under `docs/cuda-windows-5070ti/benchmarks/`.

---

## Production deployment (Config F, 36B NL MoE)

| Setting | Value |
|---------|-------|
| Branch / build | `Path-B-Event-Support-Pipeline-Plus`, client build 9964+ |
| Topology | 5070 Ti + remus 5060 Ti (**2-device**) |
| `--rpc` | `192.168.8.176:50051` (do not attach `:50052` unless VRAM requires it) |
| `-ts` | `50,50` |
| Env | `GGML_PIPELINE_PLUS=1` |

Verify after deploy:

1. Server log: `pipeline parallelism enabled`, `sched copies = 4`
2. remus HELLO: `proto 4.4` (4.3.2+ acceptable with dual OFF)
3. Trace: `assembly_overlap_count > 0`; G ~49 t/s

Ops detail: [rpc-path-b-plus-handover.md](rpc-path-b-plus-handover.md). Windows multi-node: [MULTI-NODE.md](../../docs/cuda-windows-5070ti/MULTI-NODE.md).

```bash
# remus (from repo root via WSL)
./rpc-patch/scripts/pathb-remus-rpc.sh rebuild
```

```powershell
# Windows validation bench
.\scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-2gpu-plus
.\scripts\cuda-windows-5070ti\pathb-rpc-trace-parse.ps1 -TraceDir docs\cuda-windows-5070ti\benchmarks\trace-f-2gpu-plus\telemetry
```

---

## Code map

| Component | File(s) | Change |
|-----------|---------|--------|
| P0 pipeline barrier | `ggml/src/ggml-backend.cpp`, `src/llama-context.cpp` | Rotate `cur_copy` on graph reuse; event-wait per slot |
| P1 sampling sync | `src/llama-context.cpp` | `synchronize_sampling()` when `GGML_PIPELINE_PLUS=1` |
| P2 scoped drain | `ggml/src/ggml-rpc/ggml-rpc.cpp` | Drain only on blocking `send_rpc_cmd` |
| P3 trace | `ggml-backend.cpp`, `pathb-rpc-trace-parse.ps1` | `copy` field; assembly-line overlap metric |
| B+2 async COPY | `ggml-rpc.cpp` | `cpy_tensor_async`, deferred COPY drain |
| B+3 peer COPY | `ggml-rpc.cpp`, rpc-server | `COPY_TENSOR_PEER`, HELLO minor=3 |
| B+11 dual-socket | `ggml-rpc.cpp`, `transport.h` | `RPC_CMD_CHANNEL_BIND`, optional rsp channel (default OFF) |
| Deploy sync | `rpc-patch/scripts/pathb-remus-deploy-sync.sh` | Prevent stale remus `GIT_BRANCH` / proto drift |

Env vars:

| Var | Default | Effect |
|-----|---------|--------|
| `GGML_PIPELINE_PLUS` | `1` when pipeline active | P0/P1 narrow sync; `0` = legacy full sync |
| `GGML_SCHED_TRACE` | `0` | Per-split jsonl trace |
| `GGML_RPC_TRACE` | `0` | RPC client jsonl trace |
| `GGML_RPC_DUAL_SOCKET` | `0` | B+11 cmd/rsp split (proto 4.4; bisect NULL) |

---

## Branch lineage

```text
Path-B-Event-Support (26a9353)     Path B shipped, proto 4.2.2
        |
        v
Path-B-Event-Support-Pipeline-Plus  B+1 + Tier 1 + Phase 5
        |
        +-- Path A/B baseline docs: rpc-path-b-plan.md, rpc-path-b-tracking.md
        +-- Path C (planned): patch/rpc-path-c-plan.md
```

Predecessor behavior and Config A-D matrix: [rpc-path-b-plan.md](rpc-path-b-plan.md), [patch/HANDOVER.md](../patch/HANDOVER.md).

---

## Phase and spike status

| Phase | Description | Status |
|-------|-------------|--------|
| B+0 | Planning from bug-hunt | DONE |
| B+1 | P0-P3 pipeline unblock | DONE |
| Tier 1 | B+2 async COPY, B+3 peer COPY | SHIPPED (topology-limited on 3gpu F) |
| Phase 2 | Trace parser + HELLO/copy_issue fields | DONE |
| Deploy | remus sync hardening, proto 4.3.2 | DONE |
| Phase 5a | 2-device F production default | DONE |
| Phase 5b | Spikes S4/S5/S1/S3 | DONE |
| S0 | 5-endpoint 72B+ baseline | DEFERRED |
| Phase 6 / Path C | C1 baseline + feasibility; C2 server sched | **IN PROGRESS** |

Spike details: [rpc-path-b-plus-spikes.md](rpc-path-b-plus-spikes.md). Live status and issue log: [rpc-path-b-plus-tracking.md](rpc-path-b-plus-tracking.md).

---

## B+6 overlap gate milestones (active mission)

Plan and step checklist: [b6-gate/PLAN.md](b6-gate/PLAN.md), [b6-gate/TRACKING.md](b6-gate/TRACKING.md) (update tracking after each step).

| ID | overlap_pct | stall_ratio | Topology | Status |
|----|-------------|-------------|----------|--------|
| Baseline | 0.6% / 0.2% | 0.68 / 0.96 | 2-GPU F remus / 4-GPU G | **CURRENT** |
| Post-B7 remus (2-GPU F) | 0.1% | 0.95 | romulus + remus 5060 | FAIL (2026-06-29) |
| Spike triton (2-GPU F) | 0.3% | 0.92 | romulus + triton 3090 :50054 | FAIL; G=187 t/s |
| 4-GPU canonical `b6-4gpu-g` n=384 | 0.1% | 0.93 | JUPITER :50053, ts=25,12,25,38 | FAIL; drain 50s |
| 4-GPU triton `b6-4gpu-g-triton` n=384 | 0.1% | 0.96 | triton :50054, ts=22,11,34,33 | FAIL; drain 5.9s |
| ts sweep best (G2 n=128) | 0.7% | - | legacy ts=36,24,24,16 | FAIL @ n=384 confirm |
| Spike ref | remus vs triton delta recorded | - | 2-GPU + 4-GPU A/B | DONE |
| M1 | >= 1.0% | < 0.80 | 2-GPU F / 4-GPU G | **PENDING** |
| M2 | >= 2.5% | < 0.60 | 2-GPU F | PENDING |
| M3 (PASS) | **>= 5.0%** | < 0.50 | 2-GPU F | PENDING |

Core diagnostic: remus 5060 (`:50051`) vs triton 3090 (`192.168.8.23:50054`) with client held constant.

---

## Document index

| Document | Purpose |
|----------|---------|
| **This file** | Project overview and navigation |
| [rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/) | **Mission doc root** — MISSION, PLAN, TRACKING, orchestration audit |
| [RPC-PROTOCOL.md](../../docs/rpc-multi-backend-pipeline-plus/RPC-PROTOCOL.md) | Wire format, version table, proto 4.4 |
| [FEATURE-b11-dual-socket-rpc.md](../../docs/rpc-multi-backend-pipeline-plus/FEATURE-b11-dual-socket-rpc.md) | B+11 feature, bisect, rebuild checklist |
| [b6-gate/PLAN.md](b6-gate/PLAN.md) | B+6 gate mission plan (Path-B-Plus, no Path C) |
| [b6-gate/TRACKING.md](b6-gate/TRACKING.md) | B+6 step checklist -- living state |
| [rpc-path-b-plus-plan.md](rpc-path-b-plus-plan.md) | Technical plan, tiers, success metrics |
| [rpc-path-b-plus-tracking.md](rpc-path-b-plus-tracking.md) | Implementation log, benchmarks, issues, checklists |
| [rpc-path-b-plus-spikes.md](rpc-path-b-plus-spikes.md) | Validation spikes S0-S5, commands |
| [rpc-path-b-plus-handover.md](rpc-path-b-plus-handover.md) | Production ops, verify, fallback |
| [HANDOVER-SESSION-2026-06-29.md](../patch/HANDOVER-SESSION-2026-06-29.md) | **Latest session end** -- resume B+6 here |
| [RPC-BUG-HUNT.md](../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md) | Trace-led root cause (pre-Plus) |
| [benchmarks/README.md](../../docs/cuda-windows-5070ti/benchmarks/README.md) | Bench artifact index |
| [rpc-path-c-plan.md](rpc-path-c-plan.md) | Path C / Phase 6 implementation plan |
| [rpc-path-c-tracking.md](rpc-path-c-tracking.md) | Path C status and C1 baseline |
| [README.md](../README.md) | rpc-patch folder, hardware configs, scripts |

---

## Known limits

1. **3-device F 45+ t/s:** Not achieved. Serial split sum (~27 ms/tok) dominates; RX6600 third hop adds ~16 ms/tok. Use 2-device F for throughput.
2. **B+3 peer COPY on 3gpu F:** Copies are CUDA<->RPC (`SET_TENSOR_HASH`), not RPC<->RPC. Peer copy benefits Path C / multi-RPC-same-host layouts.
3. **Overlap % gate:** B+6 targets **>5%** after rebuild (baseline 0.5% on pre-B+6 binary).
4. **72B+ cluster:** Phase 9 S0-lite via `pathb-72b-cluster-matrix.sh` on Config G.

---

## What's next

1. **Mission plan** -- [rpc-multi-backend-pipeline-plus/PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md): B+8→B+9→B+10 bisects, then B+7a′ 4-GPU drain.
2. **B+6 gate** -- [b6-gate/TRACKING.md](b6-gate/TRACKING.md): living checklist; M3 (`overlap_pct >= 5%`) is hard complete criterion.
3. **Phase 1.1** -- per-split / RPC timing visibility (parallel with bisects).
4. **Path C** -- deferred until mitigation ladder exhausted ([rpc-multi-backend-pipeline-plus/PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) Phase 3 entry criteria).