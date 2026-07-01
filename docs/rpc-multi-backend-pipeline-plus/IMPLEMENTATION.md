# Implementation — B+8..B+13 mitigation flags

**Status:** Code landed 2026-06-30 (untested — cluster bench deferred).  
**Bench gate:** `b6-2gpu-f` n=384 per [PLAN.md](PLAN.md) Q5 contract.

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