# Path-B Plus: Multi-RPC Pipeline Extension

Fork-local successor to shipped Path B. Completes the assembly-line Path B started: unblock pipeline sync, then per-hop copy fixes.

**Predecessor:** [rpc-path-b-plan.md](rpc-path-b-plan.md) | **Baseline:** [../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md](../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md)

## Goal

**25-30+ t/s** on 72B+ multi-worker topologies (head + 3 workers, 5+ RPC backends) without Path C unified rpc-server.

## Root cause (why Path B plateaued)

Path B enabled `pipeline_parallel` and `n_copies=4`, but:

1. **Graph reuse never rotated copy slots** (`is_alloc` stays true; `alloc_graph` skipped) -> same buffers every token -> forced full `sched_synchronize`.
2. **C API `llama_get_logits_*` syncs all backends** after async decode.
3. **`send_rpc_cmd` drain on every fire-and-forget send** emptied RPC overlap windows.
4. **COPY cross-port on same host** falls back to client GET+SET relay.

## Tier 0 -- Pipeline unblock (B+1)

| ID | Fix | File(s) |
|----|-----|---------|
| P0 | `ggml_backend_sched_pipeline_barrier` -- rotate copy slot + event wait only | `ggml-backend.cpp`, `llama-context.cpp` |
| P1 | `synchronize_sampling()` -- sync logits/sampling backends only | `llama-context.cpp` |
| P2 | Scoped drain -- response-read `send_rpc_cmd` only | `ggml-rpc.cpp` |
| P3 | Trace: `copy` field + assembly-line overlap metric | `ggml-backend.cpp`, `pathb-rpc-trace-parse.ps1` |

Env: `GGML_PIPELINE_PLUS=1` (default on when `pipeline_parallel` active). Set `0` to revert to legacy full sync.

## Tier 1 -- Per-hop (after Tier 0 bench)

| Phase | Work |
|-------|------|
| B+3 | Same-host peer COPY |
| B+2 | `cpy_tensor_async` + HELLO caps |
| B+4/B+5 | Drain tuning, server prefetch |

## Tier 2 -- Topology

- Pipeline segment ordering (`--rpc` list matches layer flow)
- `pathb-hotpath-summary.ps1` (trace + layer map)

## Success metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| trace-f-3gpu G | 38 t/s | 45+ after B+1 |
| split ms/token | 26.8 | <20 |
| sched overlap | none | S5 metric >0 |
| 72B+ cluster | TBD | 25-30+ t/s |

## Doc map

- [rpc-path-b-plus-tracking.md](rpc-path-b-plus-tracking.md) -- status
- [rpc-path-b-plus-spikes.md](rpc-path-b-plus-spikes.md) -- validation spikes
- [rpc-path-b-plus-handover.md](rpc-path-b-plus-handover.md) -- ops when shipped