# Path B Sync-Site Audit (living doc)

Trace-proven blocking sites for multi-RPC pipeline stalls. Static map: [`RPC-WAIT-MAP.md`](../../docs/cuda-windows-5070ti/RPC-WAIT-MAP.md). Parse tools: `rpc-patch/scripts/pathb-rpc-trace-parse.sh`, `pathb-hotpath-summary.sh`.

**Last updated:** 2026-07-01  
**Mission plan:** [docs/rpc-multi-backend-pipeline-plus/PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) Phase 2
**Evidence bench:** `b6-2gpu-f`, `b6-4gpu-g`, `b6-4gpu-g-triton`, `b6-4gpu-ts-sweep` (romulus 4-GPU, 2026-06-29), `trace-f-2gpu-plus` (Windows), `profiler-4gpu-primary-romulus-trace-v2`

---

## Six proven blockers (trace-led)

These six sites were proven on `GGML_RPC_TRACE=1` / `GGML_SCHED_TRACE=1` runs and explain the Path B plateau before Plus.

| # | Blocker | Symptom in trace | Fix tier | Status | Code refs |
|---|---------|------------------|----------|--------|-----------|
| 1 | **Graph reuse skips copy rotation** — full `sched_synchronize` every token | `copy3` skew (508 vs ~130 on legacy); G regression vs Plus | B+1 **P0** | **SHIPPED** | `ggml/src/ggml-backend.cpp` `ggml_backend_sched_pipeline_barrier` (~1972); `src/llama-context.cpp` (~1365) |
| 2 | **`llama_get_logits_*` syncs all backends** after async decode | Extra host-side drain between tokens; sampling path blocked | B+1 **P1** | **SHIPPED** | `src/llama-context.cpp` `synchronize_sampling()` (~728); call sites ~3729+ |
| 3 | **`send_rpc_cmd` drain on fire-and-forget sends** | High `drain_flush_ms`; collapsed RPC overlap window | B+1 **P2** | **SHIPPED** | `ggml/src/ggml-rpc/ggml-rpc.cpp` scoped drain (~456, ~618-619) |
| 4 | **`tls.ev` stale `response_pending`** after scoped drain | Plus regression G=28 vs 38; EVENT recv corruption risk | B+1b **P2.1** | **SHIPPED** | `ggml/src/ggml-rpc/ggml-rpc.cpp` `rpc_finish_event_response` / event TLS (~413-431, ~1117-1120) |
| 5 | **`cpy_tensor_async` NULL on RPC** — scheduler forced sync COPY | Large `COPY_TENSOR` avg_us (~280k) on pre-B+2 3gpu traces | B+2 | **SHIPPED** (client) | `ggml/src/ggml-rpc/ggml-rpc.cpp` `ggml_backend_rpc_cpy_tensor_async` (~976); `ggml-backend.cpp` copy path (~1698) |
| 6 | **Cross-port COPY GET+SET relay** — no peer path on wire | `SET_TENSOR_HASH` dominates RPC ms on CUDA↔RPC; `COPY_TENSOR` on 3gpu F | B+3 | **SHIPPED** proto 4.3.2; **topology-limited** on 3gpu F (CUDA↔RPC hash, not RPC↔RPC) | `ggml/src/ggml-rpc/ggml-rpc.cpp` `RPC_CMD_COPY_TENSOR_PEER` (~934-948), `SET_TENSOR_HASH` (~832); server `rpc_serve_client` |

**Tier 0/1 exit:** 2-device F `trace-f-2gpu-plus` — G=48.9, `assembly_overlap_count=267`, `drain_flush_ms` 2523→1734 vs legacy, `SET_TENSOR_HASH` hash path with `COPY_TENSOR=0`.

**4-GPU cluster exit:** `trace-g-4gpu-primary-trace` — G~40, `split_total` 20.7 ms/tok, 5060 straggler 9.6 ms/tok, `assembly_overlap_count=1075`, slot init OK (vs RX6600 4-GPU hang at `initializing slots`).

---

## Remaining hot path (B+4 .. B+6)

Post–B+3 stalls visible in `pathb-hotpath-summary.sh` output. Not regressions; next tuning targets.

| ID | Work | Target symptom | Status | Code / doc refs |
|----|------|----------------|--------|-----------------|
| **B+4** | `SET_TENSOR_HASH` client cache — skip redundant hash RTTs after first hit | 168 calls ~43s on 2gpu-plus load+gen window | **SHIPPED** | `ggml-rpc.cpp` `tls_hash_present` + `rpc_hash_cache_key` in `ggml_backend_rpc_buffer_set_tensor` |
| **B+5** | Server async compute queue — recv next cmd while GPU computes | Server serial cmd loop blocked on sync `graph_compute` | **SHIPPED** | `ggml-rpc.cpp` `rpc_server::enqueue_graph_*`, `wait_compute_idle` on `RPC_CMD_EVENT_RECORD` |
| **B+6** | Client assembly-line unlock — no proactive EVENT drain before GRAPH | Collapsed overlap window at `graph_compute` entry | **SHIPPED** | `ggml-rpc.cpp` removed `drain_pending_event_response` from `ggml_backend_rpc_graph_compute`; re-bench `overlap_pct` gate >5% |

---

## B+7 candidates (trace-proven 2026-06-28, `b6-2gpu-f`)

Profiler label `b6-2gpu-f` (romulus 7900 + remus 5060, `ts=50,50`, q8_0 APEX, n=384). Verdict: **MIXED** — Plus improves G (75.6 vs 72.1 t/s) but **not** `overlap_pct` (0.2% both).

| # | Blocker | Evidence (`diagnose.json`) | B+7 direction |
|---|---------|------------------------------|---------------|
| 7a | Central drain on blocking `send_rpc_cmd` | 2-GPU: `drain_flush_ms=4514`; **4-GPU canonical** `b6-4gpu-g` n=384 ts=25,12,25,38 **drain=50400**; triton swap **5924**; G4-confirm **4837** | **SHIPPED** B7-7a on 2-GPU; **4-GPU canonical split still drain-bound** -- next bisect: per-socket flush across 3 RPC HELLO peers |
| 7b | RPC straggler (topology-dependent) | 2-GPU remus backend1 5060 @ 12.4 ms/tok; **4-GPU JUPITER** backend3 5070 @ 11.6; **triton** backend3 3090 @ 9.0; ts grid n=128 straggler **backend1 5060** on G0-G4 | triton swap helps ms/tok but not overlap; `-ts` shifts straggler identity |
| 7c | Serial `input_wait_copy` | `stall_ratio=0.92-0.96` on 4-GPU runs; `input_wait_copy_ms` >> `graph_compute_async_ms` | Token pipe starved; overlaps 7a+7b |
| 7d | Plus does not raise overlap | All 4-GPU gate rows 0.1-0.7% overlap | B+7 drain/straggler, not P0/P1 barrier |
| 7e | GET_ALLOC_SIZE RPC storm | 4-GPU `blocking_rpc_count=2392-2404` (post B7-1b) | Cache shipped; blocking count still high on 4-GPU -- investigate GET_TENSOR / SET_TENSOR_HASH share |

Mission tracking: [b6-gate/TRACKING.md](b6-gate/TRACKING.md).

### B+8–B+13 ladder (2026-06-30, profiler-led, Path-B+ only)

| ID | Blocker | Direction | Status | Flag |
|----|---------|-----------|--------|------|
| B+8 | Full-backend `pipeline_barrier` quiesce | F2 partial frontier wait | **SHIPPED** (untested) | `GGML_PIPELINE_BARRIER_PARTIAL` |
| B+9 | EVENT recv on blocking path | Defer to barrier | **SHIPPED** (untested) | `GGML_RPC_EVENT_DEFER_BARRIER` |
| B+10 | MoE `input_wait_copy` sync (~1682) | Copy-slot event wait | **SHIPPED** (untested) | `GGML_SCHED_MOE_ASYNC_COPY` |
| B+7a′ | 4-GPU drain 50s vs 5s | Multi-socket RPC flush | **SHIPPED** (untested) | `GGML_RPC_MULTI_SOCKET_FLUSH` |
| B+13 | `cpy_tensor_async` sync fallback | Dst-then-src async try | **SHIPPED** (untested) | (with Plus) |
| B+11 | Single TCP HOL blocking | Dual-socket proto 4.4 | PENDING | — |
| B+12 | GET_TENSOR blocking storm | Full Path A2 deferral | **NULL overlap** (shipped) | `GGML_RPC_GET_TENSOR_DEFER` |

### B+7f — hot-path observability (Phase 1.2C-full, 2026-07-01)

| # | Blocker | Evidence | Fix tier | Status |
|---|---------|----------|----------|--------|
| 7f | C-full trace join | Pre-C-full: `LOCAL_SYNC_FALLBACK_LIKELY` with `copy_issue=0`; `input_wait_copy_ms` >> wire COPY | C-full emit: `sync_copy_fallback`, RPC `(decode_id,split,backend)` | **SHIPPED** emit; parsers + B+13 fix **active** |

Bench: `b6-2gpu-f` n=384 first (G1). See [IMPLEMENTATION.md](../../docs/rpc-multi-backend-pipeline-plus/IMPLEMENTATION.md).

---

## Trace checklist (per run)

```bash
./rpc-patch/scripts/pathb-rpc-trace-parse.sh docs/cuda-windows-5070ti/benchmarks/trace-f-2gpu-plus/telemetry
./rpc-patch/scripts/pathb-hotpath-summary.sh docs/cuda-windows-5070ti/benchmarks/trace-f-2gpu-plus/telemetry
```

| Check | Pass (2gpu production) | Fail signal |
|-------|------------------------|-------------|
| S5 overlap | `assembly_overlap_count > 0` | 0 → Plus off or broken rotation |
| S1 COPY budget | `COPY_TENSOR=0`, hash path active | High `COPY_TENSOR` avg_us |
| S3 drain | `drain_flush_ms` down vs legacy 3gpu | Rising with scoped-drain regression |
| Copy rotation | `copy0..3` roughly even (~99/99/96/96) | One slot >> others |
| HELLO | `minor=3 peer_copy=true` (post remus 4.3.2) | `minor=2` → redeploy rpc-server |

---

## Update log

| Date | Change |
|------|--------|
| 2026-06-27 | Initial audit: 6 proven blockers + B+4/B+5/B+6; linked bash parse/summary scripts |
| 2026-06-27 | B+4/B+5/B+6 shipped in `ggml-rpc.cpp`; Linux trace tooling + Config G cluster scripts |
| 2026-06-29 | 4-GPU triton A/B + ts sweep: drain 50s vs 5s topology split; overlap ceiling 0.7% @ n=128; M1 not reached @ n=384 |
| 2026-06-30 | B+8–B+13 ladder added; linked to `docs/rpc-multi-backend-pipeline-plus/` mission plan |

When a blocker ships or a trace disproves a site, update the table and bump **Last updated**.