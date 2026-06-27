# Path B Sync-Site Audit (living doc)

Trace-proven blocking sites for multi-RPC pipeline stalls. Static map: [`RPC-WAIT-MAP.md`](../../docs/cuda-windows-5070ti/RPC-WAIT-MAP.md). Parse tools: `rpc-patch/scripts/pathb-rpc-trace-parse.sh`, `pathb-hotpath-summary.sh`.

**Last updated:** 2026-06-27  
**Evidence bench:** `trace-f-2gpu-plus` (production 2-device F), `trace-f-3gpu-plus` (3-device comparison)

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

---

## Remaining hot path (B+4 .. B+6)

Post–B+3 stalls visible in `pathb-hotpath-summary.sh` output. Not regressions; next tuning targets.

| ID | Work | Target symptom | Status | Code / doc refs |
|----|------|----------------|--------|-----------------|
| **B+4** | `SET_TENSOR_HASH` client cache — skip redundant hash RTTs after first hit | 168 calls ~43s on 2gpu-plus load+gen window | **SHIPPED** | `ggml-rpc.cpp` `tls_hash_present` + `rpc_hash_cache_key` in `ggml_backend_rpc_buffer_set_tensor` |
| **B+5** | Server async compute queue — recv next cmd while GPU computes | Server serial cmd loop blocked on sync `graph_compute` | **SHIPPED** | `ggml-rpc.cpp` `rpc_server::enqueue_graph_*`, `wait_compute_idle` on `RPC_CMD_EVENT_RECORD` |
| **B+6** | Client assembly-line unlock — no proactive EVENT drain before GRAPH | Collapsed overlap window at `graph_compute` entry | **SHIPPED** | `ggml-rpc.cpp` removed `drain_pending_event_response` from `ggml_backend_rpc_graph_compute`; re-bench `overlap_pct` gate >5% |

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

When a blocker ships or a trace disproves a site, update the table and bump **Last updated**.