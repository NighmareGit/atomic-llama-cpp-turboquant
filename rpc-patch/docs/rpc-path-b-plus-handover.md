# Path-B Plus Handover

**Status:** B+1 production-ready on 2-device Config F (2026-06-27).

## What changed

Path-B Plus completes Path B assembly-line scheduling for multi-RPC clusters.

### B+1 (shipped)

- **P0:** `ggml_backend_sched_pipeline_barrier` -- rotates `cur_copy` on graph reuse; event-wait per slot instead of full sched sync.
- **P1:** Sampling/logits C API uses narrow backend sync when `GGML_PIPELINE_PLUS=1`.
- **P2:** RPC fire-and-forget sends no longer drain pending events/GETs; blocking ops still drain.
- **P3:** Sched trace includes `copy` field; parser reports assembly-line overlap.

### Tier 1 (shipped, topology-limited benefit)

- **B+2:** `cpy_tensor_async` + deferred COPY drain (client).
- **B+3:** `COPY_TENSOR_PEER` proto v4.3 (client+server). Peer copy applies to RPC<->RPC hops; 2-device F uses CUDA<->RPC hash path.

## Production default (36B NL MoE, Config F)

| Setting | Value |
|---------|-------|
| Topology | 5070 Ti + remus 5060 Ti (**2-device**, no RX6600) |
| `--rpc` | `192.168.8.176:50051` |
| `-ts` | `50,50` |
| `-ngl` | `99` |
| `--fit` | `off` |
| KV | `-ctk q4_0 -ctv q4_0` |
| Env | `GGML_PIPELINE_PLUS=1` |

Expected throughput: **~49 t/s** (`trace-f-2gpu-plus` measured 48.9).

Do **not** attach `:50052` for this model unless VRAM requires it; 3-device adds ~10 ms/tok serial hop (~38-43 t/s).

## Env vars

| Var | Default | Effect |
|-----|---------|--------|
| `GGML_PIPELINE_PLUS` | `1` | Enable P0/P1 narrow sync. `0` = legacy Path B behavior |
| `GGML_SCHED_TRACE` | `0` | Per-split trace jsonl |
| `GGML_RPC_TRACE` | `0` | RPC client trace jsonl |

## Verify

1. Server log: `pipeline parallelism enabled`, `sched copies = 4`
2. `trace-summary.txt`: `assembly_overlap_count > 0` during GEN
3. G(t/s) vs baseline: **48.9 t/s** on 2-device F (was 38.0 on 3-device pre-Plus)
4. S4 smoke: 4B + 36B both PASS (no crash, coherent generation)

Quick check:

```powershell
.\scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-2gpu-plus
.\scripts\cuda-windows-5070ti\pathb-rpc-trace-parse.ps1 -TraceDir docs\cuda-windows-5070ti\benchmarks\trace-f-2gpu-plus\telemetry
```

## Remus deploy

```bash
./rpc-patch/scripts/pathb-remus-rpc.sh rebuild
```

Confirm `[hello] version: 4.3.2` and client trace `"minor":3,"peer_copy":true`.

## Fallback

`GGML_PIPELINE_PLUS=0` or rebuild without B+1 commits restores full `sched_synchronize` on graph reuse.

## What's next

- **Path C:** single remus rpc-server spanning 5060+6600 (removes client-side split between remus GPUs).
- **5-endpoint spike (S0):** deferred until 72B+ cluster bench.