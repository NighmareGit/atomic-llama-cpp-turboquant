# Implementation — B+8..B+13 mitigation flags + C-full hotpath

**Status:** B+8–B+10 ladder exhausted (NULL overlap); **B+11–B+13 active** (grill 2026-07-01). C-full trace shipped `6dc504bce`; re-bench + parsers next.  
**Bench gate:** `b6-2gpu-f-triton-n384-romulus-native` per [TRACKING.md](TRACKING.md).

## Environment flags (default on when `GGML_PIPELINE_PLUS=1`)

| Flag | Fix | `=0` rollback |
|------|-----|---------------|
| `GGML_PIPELINE_BARRIER_PARTIAL` | B+8 F2 partial `pipeline_barrier` | Full all-backend event wait (legacy) |
| `GGML_RPC_EVENT_DEFER_BARRIER` | B+9 defer EVENT recv to barrier | Drain EVENT on every blocking RPC |
| `GGML_RPC_MULTI_SOCKET_FLUSH` | B+7a′ flush all RPC sockets at synchronize | Per-socket flush only |
| `GGML_SCHED_MOE_ASYNC_COPY` | B+10 MoE copy-slot event wait | Full `ggml_backend_synchronize` on MoE path |

B+13 (dual-side `cpy_tensor_async` try) is always on in scheduler copy path when Plus is enabled.

## Code map

| ID | File | Symbol / area |
|----|------|----------------|
| B+8 | `ggml/src/ggml-backend.cpp` | `barrier_copy_src_mask`, `ggml_backend_sched_update_barrier_src_mask`, `ggml_backend_sched_pipeline_barrier` |
| B+9 | `ggml/src/ggml-rpc/ggml-rpc.cpp` | `rpc_event_defer_barrier`, `send_rpc_cmd` blocking path |
| B+7a′ | `ggml/src/ggml-rpc/ggml-rpc.cpp` | `rpc_drain_all_endpoints_pending`, `ggml_backend_rpc_drain_all_endpoints` |
| B+10 | `ggml/src/ggml-backend.cpp` | MoE `MUL_MAT_ID` weight path ~1682 |
| B+13 | `ggml/src/ggml-backend.cpp` | `input_wait_copy` async try dst then src |

## Bisect procedure (G1 — 2-GPU triton n=384)

```bash
# After sync + rebuild (romulus client, triton :50054)
bash scripts/b6-gate-bisect-run.sh no-partial          # B+8 OFF
bash scripts/b6-gate-bisect-run.sh no-async-copy       # B+10 OFF
bash scripts/b6-gate-bisect-run.sh canonical-romulus   # romulus-native baseline

# Fallback (remus docker CUDA client)
B6_GATE_CLIENT=remus-docker bash scripts/b6-gate-bisect-run.sh no-partial

bash scripts/b6-gate-diagnose-runs.sh b6-2gpu-f-triton-n384-*
```

Compare `diagnose.json` to baseline in `benches/path-b-plus/b6-2gpu-f/`.

**PASS → B+9 already stacked;** evaluate Q5 contract. **PARTIAL → B+8b** tighten mask. **FAIL →** set `GGML_PIPELINE_BARRIER_PARTIAL=0` and revert commit.

## Rollback one-liner (production)

```bash
export GGML_PIPELINE_BARRIER_PARTIAL=0
export GGML_RPC_EVENT_DEFER_BARRIER=0
export GGML_RPC_MULTI_SOCKET_FLUSH=0
export GGML_SCHED_MOE_ASYNC_COPY=0
```

Or `GGML_PIPELINE_PLUS=0` for full legacy Path B behavior.

---

## C-full hotpath instrumentation

Phase 1.2C on **pre-C-full** traces showed `copy_issue=0`, `COPY_TENSOR RPC=0 ms`, yet `input_wait_copy_ms` dominated — local B+13 sync fallback was invisible. Grill Q3/Q4: **C-full** (not C-min).

### Code (shipped)

| Layer | Mechanism | Location |
|-------|-----------|----------|
| Sched context | Thread-local `(split_id, backend_id)` per split | `ggml_hotpath_trace_set_sched_ctx()` |
| B+13 phases | `sync_copy_fallback` (us); `copy_async_ok` (marker) | `ggml_backend_sched_compute_splits` ~1865–1887 |
| RPC join | `decode_id`, `split`, `backend` on all rpc_trace rows | `rpc_trace_emit_hotpath_fields()` |

### Trace env

```bash
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE=1
export GGML_PIPELINE_TRACE=1
```

### Schema v1 (sched-trace.jsonl)

| phase | Meaning |
|-------|---------|
| `input_wait_copy` | Full input-drain window |
| `sync_copy_fallback` | Local sync + `tensor_copy` when async copy fails |
| `copy_async_ok` | Async copy succeeded (`elapsed_us=0`) |
| `graph_compute_async` | Split graph dispatch |
| `event_record` | Per-split event record |
| `split_total` | Wall time for split |

RPC rows add optional `decode_id`, `split`, `backend`. Join key for parsers: `(decode_id, split, backend)`.

### Phase 1.2 execution (staged D — active now)

| Step | Work | Status |
|------|------|--------|
| **C** | Phase 1.2C blocking audit on legacy traces | **done** — `scripts/b6-gate-phase12c-blocking-audit.sh` |
| **C-full** | Emit + schema above | **done** — `6dc504bce` |
| **C audit** | Extend phase12c for `sync_copy_fallback` / `copy_async_ok` | **next** |
| **Re-bench** | `b6-2gpu-f-triton-n384-romulus-native` with C-full build | **next** |
| **A** | Per-token blocking waterfall parser | **next** |
| **B** | Assembly-line Gantt (`decode_id` x split x cmd) | **next** |
| **7f** | Register hot-path observability in pathb-sync-site-audit | **next** |

### B+11–B+13 implement order (post-C audit)

1. **B+13** — prove/fix `cpy_tensor_async`; stop silent `sync_copy_fallback`
2. **B+12** — `GET_TENSOR` deferral
3. **B+11** — dual-socket RPC (if A+B shows HOL)

### Deferred (sample API only)

C-full keep lists in `llama-pipeline-trace-sample.sh` — see [FUTURE-EXPANSIONS.md](FUTURE-EXPANSIONS.md).