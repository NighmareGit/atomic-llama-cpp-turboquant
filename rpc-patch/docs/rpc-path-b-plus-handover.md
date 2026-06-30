# Path-B Plus Handover

**Status:** B+1 production-ready on 2-device F; B+4..B+6 shipped; **B+6 gate IN PROGRESS** (M3 not reached, 2026-06-30).

**Mission doc root:** [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/) (MISSION, PLAN, TRACKING, audit).

**Resume here:** [HANDOVER-SESSION-2026-06-29.md](../patch/HANDOVER-SESSION-2026-06-29.md) (cluster session) + [rpc-multi-backend-pipeline-plus/TRACKING.md](../../docs/rpc-multi-backend-pipeline-plus/TRACKING.md) (mission state).

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

### B+6 gate (active, 2026-06-29)

- Profiler presets: `b6-2gpu-f`, `b6-2gpu-f-triton`, `b6-4gpu-g`, `b6-4gpu-g-triton`
- ts sweep: `scripts/b6-gate-ts-sweep-4gpu.sh`
- Diagnosis: `scripts/b6-gate-diagnose-runs.sh`, [b6-gate/TRACKING.md](b6-gate/TRACKING.md)

**4-GPU canonical gate (n=384):** overlap 0.1%, drain 50.4s, straggler 5070 @ 11.6 ms/tok.  
**4-GPU triton swap:** overlap 0.1%, drain 5.9s. **Next:** B+7 multi-socket drain bisect.

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

Expected throughput: **~49 t/s** (`trace-f-2gpu-plus` measured 48.9). Triton 3090 spike: **~187 t/s** (`b6-2gpu-f-triton`).

## 4-GPU Config G (romulus client)

| Setting | Value |
|---------|-------|
| `-rpc` | `192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053` |
| `-ts` | `25,12,25,38` (RPC-first: 5060, 3060, 5070, 7900) |
| JUPITER ops | `scripts/b6-gate-jupiter-start-rpc-task.ps1` |
| Profiler | `bash scripts/b6-gate-run-remote.sh b6-4gpu-g` |

See [docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md](../../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md).

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
5. B+6: `llama-pipeline-profiler --validate-rpc` all endpoints proto 4.3 `peer_copy=yes`

## Remus deploy

```bash
./rpc-patch/scripts/pathb-remus-rpc.sh rebuild
```

Confirm `[hello] version: 4.3.2` and client trace `"minor":3,"peer_copy":true`.

## Fallback

`GGML_PIPELINE_PLUS=0` or rebuild without B+1 commits restores full `sched_synchronize` on graph reuse.

## What's next

See [docs/rpc-multi-backend-pipeline-plus/PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) Phase 2 (mitigation ladder).

- **B+8:** Partial `pipeline_barrier` wait (`ggml-backend.cpp`) — profiler-led, first bisect.
- **B+9:** EVENT defer-to-barrier (`ggml-rpc.cpp`).
- **B+10:** MoE `input_wait_copy` de-sync (`ggml-backend.cpp`).
- **B+7a′:** 4-RPC socket drain on canonical 4-GPU (`ggml-rpc.cpp`).
- **Ops:** triton `:50054` as RPC2 eval; G4 `-ts` `30,14,16,40` (throughput, not overlap gate).
- **Path C:** deferred until B+8–B+13 ladder exhausted.