# Path-B Plus: Multi-RPC Pipeline Extension

Fork-local successor to shipped Path B. Completes the assembly-line Path B started: unblock pipeline sync, fix proven blockers, unlock cross-token overlap.

**Predecessor:** [rpc-path-b-plan.md](rpc-path-b-plan.md) | **Baseline:** [../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md](../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md)

## Goal

**25-30+ t/s** on 72B+ multi-worker topologies without Path C unified rpc-server.

## Proven root cause

Source-code lateral forced-sync + serial server loop — not hardware. See [RPC-WAIT-MAP.md](../../docs/cuda-windows-5070ti/RPC-WAIT-MAP.md), [pathb-sync-site-audit.md](pathb-sync-site-audit.md).

## Tier 0 -- Pipeline unblock (B+1) COMPLETE

| ID | Fix | Status |
|----|-----|--------|
| P0 | `ggml_backend_sched_pipeline_barrier` | SHIPPED |
| P1 | `synchronize_sampling()` | SHIPPED |
| P2 | Scoped drain | SHIPPED |
| P3 | Trace copy/overlap metric | SHIPPED |

## Tier 1 -- Per-hop COMPLETE

| Phase | Work | Status |
|-------|------|--------|
| B+2 | `cpy_tensor_async` + deferred COPY drain | SHIPPED |
| B+3 | COPY_TENSOR_PEER proto 4.3 | SHIPPED |
| B+4 | SET_TENSOR_HASH client cache | SHIPPED |
| B+5 | Server async compute queue | SHIPPED |
| B+6 | Client assembly-line (no GRAPH entry drain) | SHIPPED |

## Phase 5 COMPLETE

2-device F production default (`ts=50,50`, G ~49 t/s).

## Phase 6 CLOSED

Path C C2+ declined. Feasibility 6a/6b done.

## Phase 7 -- Tier 2 trace tooling (Romulus)

| Deliverable | Status |
|-------------|--------|
| `pathb-rpc-trace-parse.sh` | SHIPPED |
| `pathb-hotpath-summary.sh` | SHIPPED |
| `pathb-sync-site-audit.md` | SHIPPED |
| `BENCH_TRACE=1` in `rpc-server-bench.sh` | SHIPPED |

## Phase 8 -- Config G cluster standup

| Item | Status |
|------|--------|
| `Atomic-Llama-Romulus-PathB` deploy | SHIPPED |
| `pathb-romulus-rpc.sh` / `pathb-cluster-up.sh` | SHIPPED |
| `pathb-rpc-server.ps1` (Windows :50053) | SHIPPED |
| `config-g` in `pathb-72b-vram-calc.py` | SHIPPED |

## Phase 9 -- S0-lite benches

Run after rebuild with B+4..B+6:

```bash
WIN_RPC_IP=<lan_ip> ./rpc-patch/scripts/pathb-72b-cluster-matrix.sh qwen72b coder-next-q4
```

Pass: MoE **>=25 t/s**, `overlap_pct` **>5%**, positive worker scaling vs 2-device.

## Doc map

- [rpc-path-b-plus-overview.md](rpc-path-b-plus-overview.md)
- [rpc-path-b-plus-tracking.md](rpc-path-b-plus-tracking.md)
- [pathb-sync-site-audit.md](pathb-sync-site-audit.md)
- [rpc-multi-node-remus.md](rpc-multi-node-remus.md)