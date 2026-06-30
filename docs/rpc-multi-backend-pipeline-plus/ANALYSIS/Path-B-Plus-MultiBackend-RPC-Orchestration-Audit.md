# Path-B-Plus Multi-Backend RPC Orchestration Audit

**Date:** 2026-06-29 (audit); **2026-06-30** (B+6 gate + profiler appendix)  
**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Context:** Windows 5070 Ti client + remus RPC workers (Config E/F) and romulus 4-GPU cluster (Config G)  
**Mission doc root:** [../README.md](../README.md)

---

## Executive Summary

Path-B-Event-Support-Pipeline-Plus delivers **real, measurable throughput gains** in hybrid multi-backend inference, but the **B+6 overlap gate (M3: `overlap_pct >= 5%`) remains FAIL** as of 2026-06-29. The dominant limiter is **client-side orchestration** that collapses copy-slot pipelining — not PCIe, NIC, or raw GPU compute.

**What Path B events + Pipeline-Plus bought us:**

- Explicit pipeline parallelism with `graph splits` and `sched_reserve`
- `Plus=1` uplift: legacy 39.3 t/s → 42.8 t/s (3-device) / **48.9 t/s** (2-device)
- Production default: `trace-f-2gpu-plus` (`ts=50,50`)
- B+4–B+6 RPC hot-path fixes (hash cache, async server queue, GRAPH entry drain removal)

**What it did not solve:**

- **M3 overlap gate** — best 0.7% (ts grid n=128); 0.1–0.3% at n=384 confirm
- 7–22+ blocking RPC RTTs per token (partially reduced; `blocking_rpc_count` still ~800–2400 on 4-GPU)
- Burst-then-idle GPU signature (`stall_ratio` 0.92–0.95, power duty 3.8% @20% TDP)
- RX6600 third hop harmful for 35–36B A3B MoE

**Verdict:** Production throughput mission **achieved** on 2-device F. **Overlap / pipelining mission active** — untangle client→RPC I/O and scheduler copy-wait within Path-B+ bounds before Path C.

---

## Branch Progression Context

| Branch | Key Addition | Measured Impact (36B NL MoE) |
|--------|--------------|------------------------------|
| `Path-B-Event-Support` | RPC events v4.2.2 | Foundation |
| `Path-B-Event-Support-Pipeline-Plus` | Plus + B+1–B+6 + proto 4.3 | **48.9 t/s** 2-device; overlap ~0.1–0.7% |
| `Path-C-Distributed-Orchestration` | Server aggregation | **Deferred** — compatibility constraint |

---

## Bottleneck Classification

### 1. Serial RPC Critical Path (PRIMARY)

**Evidence:** `pipeline parallelism enabled`, `sched copies = 4`, power/util burst-idle, profiler `ORCHESTRATION_STALL` + `LOW_DUTY_CYCLE`.

**Why events + Plus helped throughput but not overlap:**

- Path B improves intra-segment sync; Plus rotates copy slots (P0).
- Per-token split loop remains **serial** ([RPC-WAIT-MAP.md](../../cuda-windows-5070ti/RPC-WAIT-MAP.md)).
- Overlap metric counts **cross-backend split temporal overlap** across the sched timeline — requires copy slots to stay in flight while RPC RTTs complete.

### 2. RX6600 Third-Hop Straggler (SECONDARY — actionable)

3-device F: 37–42.8 t/s vs 2-device **48.9 t/s** (+32% by dropping 6600). **Do not use Config F 3-device for 35–36B A3B MoE** unless VRAM forces it.

### 3. Drain Amplification on 4-GPU (B+7a — active)

Canonical `b6-4gpu-g` n=384: drain **50.4s** vs triton swap **5.9s** at same overlap (0.1%). Topology/worker-class affects drain class; overlap unchanged → pipelining depth is separate from drain.

### 4. Measurement (MITIGATED)

`-Profile` runs (500 ms `nvidia-smi` + 45 s flush) are authoritative. Legacy 2 s polling ruled out.

### 5. PCIe / NIC / RAM (RULED OUT for GEN)

Low pages/sec, ~1.6 Gbps NIC peak during generation — orchestration-bound.

---

## Profiler Appendix (2026-06-29 B+6 gate)

Source: `benches/path-b-plus/b6-2gpu-f/telemetry/` (romulus + remus 5060, APEX MoE, n=384).

| Signal | Value | Implication |
|--------|-------|-------------|
| `stall_ratio` | 0.95 | GPUs idle 95% of token window |
| `input_wait_copy_ms` | 4821 | Scheduler copy-wait dominates |
| `graph_compute_async_ms` | 250 | Async launch is small fraction |
| `backend1 split_total` | 5057ms / 385 tok | RPC worker ~99% of split time |
| `EVENT_RECORD` total | 4514ms (= `drain_flush_ms`) | Client recv blocks pipelining |
| `overlap_pct` | 0.2% | 1028 / 444675 cross-backend split pairs |
| Triton A/B | drain 4.5→2.3s, G 75→187 | Overlap 0.2→0.3% only |

**Interpretation:** Fixing drain/straggler improves **G** but not **overlap_pct** sufficiently. M3 requires **copy-slot pipelining depth** — B+8 (`pipeline_barrier`), B+9 (EVENT defer), B+10 (MoE copy-wait).

Full candidate table: [rpc-patch/docs/pathb-sync-site-audit.md](../../rpc-patch/docs/pathb-sync-site-audit.md).

---

## Path-B+ Mitigation Ladder (no Path C)

| ID | Blocker | Direction | Status |
|----|---------|-----------|--------|
| B+7-7a | Socket-scoped GET flush | `ggml-rpc.cpp` | SHIPPED (2-GPU); 4-GPU still drain-bound |
| B+7-1b | GET_ALLOC_SIZE cache | `ggml-rpc.cpp` | SHIPPED |
| **B+8** | Full-backend `pipeline_barrier` wait | Partial frontier wait | **NEXT** |
| **B+9** | EVENT recv on hot path | Defer to barrier | PENDING |
| **B+10** | MoE `input_wait_copy` sync | De-sync expert copy path | PENDING |
| **B+7a′** | 4-socket drain | Canonical 50s vs 5s | PENDING |
| B+11–B+13 | Dual-socket, GET defer, cpy async verify | `ggml-rpc.cpp` | PENDING |

**Path C boundary (do not cross yet):** parallel split loop in `ggml_backend_sched_compute_splits`; unified multi-GPU rpc-server aggregation.

---

## Recommendations (Ranked)

1. **Enforce 2-device preference** for 35–36B A3B MoE (scripts + `MULTI-NODE.md`).
2. **Execute B+8→B+9→B+10** with `b6-2gpu-f` n=384 gate after each bisect.
3. **B+7a′** on 4-GPU canonical if 4-GPU remains gate topology.
4. **Phase 1.1 instrumentation** — per-split / RPC wait histogram in trace output.
5. **Path-C bridge** only after M3 pass or trace-proven structural ceiling + explicit approval.

---

## Open Engineering Questions

- Can `pipeline_barrier` wait a dependency subset without breaking MoE correctness?
- Is 5% overlap achievable without parallel splits (Path C), given serial split sum ~13 ms/tok on RPC?
- How much of `input_wait_copy` is MoE expert path vs CUDA→RPC sync fallback?
- Dual-socket RPC (proto 4.4) — compatibility cost vs HOL blocking removal?

---

## Conclusion

Path-B-Event-Support-Pipeline-Plus is a **successful incremental throughput step** (`trace-f-2gpu-plus` @ 48.9 t/s) with a **stable 4-GPU baseline**. The **active mission** is B+6 M3 overlap — profiler data shows pipelining collapse, not hardware limits.

Next wins: **visibility (Phase 1.1) + B+8–B+10 bisects** within `ggml-backend.cpp` / `ggml-rpc.cpp`. Path C remains last resort.

**Ready for Phase 1.1 + B+8 implementation.**

---

*Formalizes 2026-06-29 analysis; updated 2026-06-30 with B+6 gate profiler appendix and mitigation ladder.*