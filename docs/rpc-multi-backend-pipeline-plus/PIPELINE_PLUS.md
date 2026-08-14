# GGML_PIPELINE_PLUS — Behavior and Usage

## Overview

`GGML_PIPELINE_PLUS` is the master enable flag for **Path-B Plus** (P0-P3), a set of
scheduler optimizations that tighten synchronization in multi-GPU pipeline-parallel
inference. Its purpose is **cross-token tail overlap**: the RPC head of token *N+1* can
launch while token *N*'s local GPU tail is still running, using copy-slot rotation,
per-backend events, and narrow synchronization.

This is a **downstream-only** extension to upstream llama.cpp's `pipeline_parallel` mode.
It is not present in upstream llama.cpp.

---

## Enable / Disable

Set via environment variable:

```bash
# Enable (default)
export GGML_PIPELINE_PLUS=1

# Disable — reverts to full sync per token (pre-Plus Path B behavior)
export GGML_PIPELINE_PLUS=0
```

**Default: `1`** in all three independent consumers:

| Component | File | Function | Default |
|-----------|------|----------|---------|
| Scheduler | `ggml/src/ggml-backend.cpp:54` | `ggml_sched_pipeline_plus_enabled()` | `1` |
| RPC layer | `ggml/src/ggml-rpc/ggml-rpc.cpp:297` | `rpc_pipeline_plus_enabled()` | `1` |
| Context | `src/llama-context.cpp:34` | `llama_pipeline_plus_enabled()` | `1` |

The default is overridden to `0` when either of these environment variables is set:

| Override | Effect |
|----------|--------|
| `GGML_PIPELINE_SCHED_LEGACY=1` | Forces Plus OFF in scheduler + RPC layer |
| `GGML_PIPELINE_MULTI_BACKEND_SEQ=1` | Forces Plus OFF (TSC repair bundle) |

---

## What It Changes (P0-P3)

### P0 — Pipeline Barrier (graph reuse)

When `pipeline_parallel` is active and `GGML_PIPELINE_PLUS=1`, graph reuse uses
`ggml_backend_sched_pipeline_barrier()` instead of `ggml_backend_sched_synchronize()`:

| Mode | Synchronization | Cost |
|------|----------------|------|
| Plus=1 | `pipeline_barrier()` — per-slot `event_wait` | O(n_backends) |
| Plus=0 | `sched_synchronize()` — full backend sync | O(n_backends x pending ops) |

`pipeline_barrier()` waits only on backends whose copy slot is being reused
(`barrier_copy_src_mask`), using `ggml_backend_event_synchronize` per slot.

**Narrowed further by sub-flags:**
- `GGML_PIPELINE_BARRIER_PARTIAL=1` (B+8): narrows wait mask further
- `GGML_PIPELINE_BARRIER_PARTIAL_STRICT=1` (B+8b): intersects with `pending_mask`

### P1 — Narrow Sampling Synchronization

When `GGML_PIPELINE_PLUS=1` and `pipeline_parallel` is true, `synchronize_sampling()`
syncs only the backend holding logits/sampling tensors, instead of all backends:

```cpp
// Plus=1: narrow sync
sync_tensor(t_sampled_logits);
sync_tensor(get_logits());

// Plus=0: full backend sync
synchronize();
```

This avoids synchronizing the RPC backend when it may still be computing on token *N*'s
tail, while sampling for token *N*-2 proceeds on the host.

Override: `GGML_PIPELINE_P1_FULL_SYNC=1` reverts to full sync for bisection.

### P2 — Scoped RPC Drain

In the RPC layer, `send_rpc_cmd` drains events only on blocking RPC operations rather
than on every command. See `ggml-rpc.cpp` for details.

### P3 — Trace / Overlap Metric

Enables `assembly_overlap_count` in the hotpath summary, consumed by
`pathb-rpc-trace-parse` for overlap analysis.

---

## Interaction With Downstream Flags

When `GGML_PIPELINE_PLUS=1`, the following flags inherit `1` as their default unless
explicitly set by the user. When `GGML_PIPELINE_PLUS=0`, these flags have no effect
regardless of their value.

| Flag | Default | Purpose |
|------|---------|---------|
| `GGML_PIPELINE_BARRIER_PARTIAL` | `1` | Narrow barrier wait to backends whose slot is reused |
| `GGML_PIPELINE_BARRIER_PARTIAL_STRICT` | `1` | Intersect F2 src mask with per-slot pending |
| `GGML_SCHED_MOE_ASYNC_COPY` | `1` | Async MoE weight copies via event_wait |
| `GGML_RPC_EVENT_DEFER_BARRIER` | `1` | Defer EVENT recv to pipeline barrier |
| `GGML_RPC_GET_TENSOR_DEFER` | `1` | Batch RPC GET downloads to graph boundary |
| `GGML_RPC_MULTI_SOCKET_FLUSH` | `1` | Multi-socket flush optimization |

**Effective value logic:**
```
effective = (user_env || plus_default) && plus_enabled
```

This means setting a flag explicitly ON when `GGML_PIPELINE_PLUS=0` is a no-op.

---

## Three RPC Features Gated Behind Plus

These features require Plus=1 AND their own env var:

| Feature | Extra Guard | Line |
|---------|------------|------|
| Event defer barrier | `GGML_RPC_EVENT_DEFER_BARRIER != 0` AND >= 1 server | `ggml-rpc.cpp:2134` |
| GET_TENSOR defer | `GGML_RPC_GET_TENSOR_DEFER != 0` | `ggml-rpc.cpp:2141` |
| Hash defer | `GGML_RPC_HASH_DEFER != 0` | `ggml-rpc.cpp:2145` |

---

## Interaction With Other Major Flags

### `GGML_SCHED_GPIPE` (Path D)

Independent flag layered on top of Plus:

```
GGML_PIPELINE_PLUS=1, GGML_SCHED_GPIPE=0  ->  Path-B+ (default production)
GGML_PIPELINE_PLUS=1, GGML_SCHED_GPIPE=1  ->  Path-D GPipe on top of B+
GGML_PIPELINE_PLUS=0                       ->  Legacy Path B full sync
```

### `GGML_RPC_DUAL_SOCKET`

**Deprecated.** Defaults to `0` regardless of Plus. Slated for removal in Phase R3.

### `GGML_SCHED_WAVEFRONT_DISPATCH`

**Deprecated.** Defaults to `0`. Slated for removal in Phase R3.

### `GGML_RPC_HASH_DEFER`

**Deprecated.** Defaults to `0`. Slated for removal in Phase R3.

---

## Structural Ceiling (Why Path D Exists)

The B+ mitigation ladder is exhausted. Despite all nine mitigation flags being enabled,
the overlap metrics fall short of targets:

| Metric | Achieved | Target |
|--------|----------|--------|
| `overlap_pct` | 0.1-1.3% | >= 5% |
| `global_3bk_pct` | < 1% | >= 25% |

The scheduler dispatch loop is fully serial per split — it cannot overlap RPC-to-RPC
splits within a single token. `GGML_PIPELINE_PLUS` enables cross-token tail overlap
(RPC head of T+1 with ROCm tail of T) but cannot break the within-token serial dispatch
ceiling. This is why Path D (`GGML_SCHED_GPIPE`) exists.

---

## Blessed Configuration (4-GPU reference)

```bash
export GGML_PIPELINE_PLUS=1
export GGML_PIPELINE_BARRIER_PARTIAL=1
export GGML_PIPELINE_BARRIER_PARTIAL_STRICT=1
export GGML_RPC_EVENT_DEFER_BARRIER=1
export GGML_RPC_GET_TENSOR_DEFER=1
export GGML_RPC_MULTI_SOCKET_FLUSH=1
export GGML_SCHED_MOE_ASYNC_COPY=1
```

Rollback to legacy:
```bash
export GGML_PIPELINE_PLUS=0
```

---

## Source Code References

| File | Lines | Purpose |
|------|-------|---------|
| `ggml/src/ggml-backend.cpp` | 50-155 | Flag gate definitions, pipeline_barrier() |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 293-370, 2128-2145 | RPC flag gates, defer logic |
| `src/llama-context.cpp` | 34-45, 955-994, 1590-1630 | llama-level Plus checks |
| `scripts/gpu-host-local-up.sh` | 142 | Production launcher default |
| `tools/llama-pipeline-profiler/llama-pipeline-profiler.cpp` | 174, 372+ | Profiler integration |

See also:
- `CONFIGURATION.md` — Blessed config and deprecation policy
- `IMPLEMENTATION.md` — Full flag table with status
- `CONTEXT.md` — Domain vocabulary
- `CURRENT-STATE-pipeline-flow.md` — Pipeline flow details and structural ceiling
