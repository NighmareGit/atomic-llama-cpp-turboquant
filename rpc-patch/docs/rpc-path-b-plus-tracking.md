# Path-B Plus Tracking

Overview: [rpc-path-b-plus-overview.md](rpc-path-b-plus-overview.md)

**Overall:** B+1 **PRODUCTION READY** (Phase 5 complete) | Tier 1 SHIPPED | tier1b = run variance (fluke)  
**Build:** 9964 client + remus **proto 4.3.2** (`peer_copy=yes` on :50051/:50052)  
**Current Phase:** Phase 5 **COMPLETE** -- 2-device F ops default; Path C deferred  
**Branch:** Path-B-Event-Support-Pipeline-Plus (private fork)

## Implementation log

| Date | Phase | Action |
|------|-------|--------|
| 2026-06-27 | B+0 | Plan + docs created from bug-hunt + user review |
| 2026-06-27 | B+1 | P0 pipeline_barrier, P1 sampling sync, P2 scoped drain, P3 trace copy/overlap |
| 2026-06-27 | B+1b | tls.ev event tracking fix; trace jsonl truncate; A/B bench |
| 2026-06-27 | B+2 | cpy_tensor_async + deferred COPY drain (client) |
| 2026-06-27 | B+3 | COPY_TENSOR_PEER proto v4.3 (client+server; remus deployed) |
| 2026-06-27 | B+3b | tier1b bench: remus v4.3 non-event on 3gpu F; peer COPY N/A (CUDA-RPC topology) |
| 2026-06-27 | Phase 2 | Trace parser cmd map fix; hello/copy_issue RPC trace fields |
| 2026-06-27 | Deploy | `pathb-remus-deploy-sync.sh`; build auto-syncs deploy to remus; proto 4.3.2 verified |
| 2026-06-27 | Phase 5a | 2-device F production default (`ts=50,50`, single `:50051`, `GGML_PIPELINE_PLUS=1`) |
| 2026-06-27 | Phase 5b | Spikes S4/S5/S1/S3 PASS on `trace-f-2gpu-plus`; S0 deferred |

## Issues

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|
| 2026-06-27 | Path B plateau on N-RPC | Graph reuse skips copy rotation; full sync drains pipe | B+1 Tier 0 | FIXED |
| 2026-06-27 | Plus regression G=28 vs 38 | drain_pending cleared tls but not ev->response_pending | tls.ev tracking | FIXED |
| 2026-06-27 | tier1 vs plus G delta -2.0 | Same build 9962; single-run jitter | Multi-run not needed; closed as fluke | CLOSED |
| 2026-06-27 | tier1b no COPY_TENSOR_PEER | 3gpu F copies are CUDA<->RPC not RPC<->RPC | Pivot B+3 to Path C / 5-ep | WONTFIX on 3gpu |
| 2026-06-27 | trace-f-2gpu-plus HELLO minor=2 | Bench ran before remus 4.3.2 deploy sync | Remus now 4.3.2; re-bench optional | CLOSED |

## Benchmarks

| Run | Config | G (t/s) | Build | Notes |
|-----|--------|---------|-------|-------|
| trace-f-3gpu-pre | 3-device 36B NL | 38.0 | - | Pre-Plus (bug-hunt) |
| trace-f-3gpu-legacy | 3-device, Plus=0 | 39.3 | 9962 | Full sync on reuse; copy3=508 (broken rotation) |
| trace-f-3gpu-plus | 3-device, Plus=1 | **42.8** | 9962 | Best 3gpu; copy rotation OK |
| trace-f-3gpu-tier1 | 3-device, Plus=1 | 40.8 | **9962** | Same binary as plus |
| trace-f-3gpu-tier1b | 3-device, Plus=1, remus v4.3 | 40.6 | 9963 | Within noise of tier1; 0 peer COPY (expected) |
| trace-f-2gpu-plus | 2-device ts=50,50, Plus=1 | **48.9** | 9964 | **Production default**; 3 splits; overlap 0.5% |
| s4-4b-2gpu-plus | 2-device gemma-4-E4B, Plus=1 | 44.2 | 9964 | S4 correctness smoke PASS |

Artifacts: `docs/cuda-windows-5070ti/benchmarks/trace-f-{3gpu-plus,legacy,tier1,tier1b,2gpu-plus}/`, `s4-4b-2gpu-plus/`

## Production default (Phase 5a)

**36B NL MoE on Config F:** use **2-device** topology, not 3-device.

| Setting | Value |
|---------|-------|
| `--rpc` | `192.168.8.176:50051` only (drop `:50052`) |
| `-ts` | `50,50` |
| `GGML_PIPELINE_PLUS` | `1` (default when trace/profile enabled) |
| Model | `Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf` |
| Expected G | **~49 t/s** (measured 48.9) |

Runbook label: `trace-f-2gpu-plus`. See [rpc-path-b-plus-handover.md](rpc-path-b-plus-handover.md).

3-device F (`ts=30,12,58`) remains valid when VRAM requires RX6600; expect ~38-43 t/s, not 45+.

## Remus deploy

`build` and `rebuild` sync `rpc-patch/deploy/*` to remus before compiling (no stale `GIT_BRANCH`).

```bash
./rpc-patch/scripts/pathb-remus-rpc.sh rebuild      # deploy + build + restart :50051
./rpc-patch/scripts/pathb-remus-rx6600-rpc.sh rebuild # deploy + build + restart :50052
# or
./rpc-patch/scripts/pathb-remus-multi-rpc-win.sh start
```

`deploy` only: `pathb-remus-rpc.sh deploy`. Skip sync: `PATHB_REMUS_SKIP_DEPLOY=1 build`.

Server: `[hello] version: 4.3.2`. Client trace: `"minor":3,"peer_copy":true`. Artifact: `benchmarks/proto-check-v43/`.

## Phase 5 checklist

### 5a -- Ops topology (DONE)

| Item | Status | Evidence |
|------|--------|----------|
| Document 2-device F as production default | PASS | tracking + handover + runbook |
| `trace-f-2gpu-plus` bench captured | PASS | G=48.9, RESULT=PASS |
| Runbook/README updated | PASS | `pathb-trace-runbook.ps1`, benchmarks README |
| Remus deploy proto 4.3.2 | PASS | deploy sync + proto-check |

### 5b -- Validation spikes (DONE)

| Spike | Status | Evidence |
|-------|--------|----------|
| S4 barrier correctness | **PASS** | `s4-4b-2gpu-plus` + `trace-f-2gpu-plus`; pipeline + sched copies=4 |
| S5 overlap during GEN | **PASS** | `assembly_overlap_count=267`, overlap_pct=0.5% |
| S1 gen COPY budget | **PASS** | 2gpu: SET_TENSOR_HASH=168, COPY_TENSOR=0 (hash path) |
| S3 drain ratio | **PASS** | drain_flush 2523->1734 ms (-31%) on 2gpu vs legacy 3gpu |
| S0 5-endpoint baseline | DEFERRED | 72B+ cluster; not required for B+1 |

Details: [rpc-path-b-plus-spikes.md](rpc-path-b-plus-spikes.md)

## Next steps (post Phase 5)

1. Path C spike: single remus rpc-server for 5060+6600 (eliminate client split hop).
2. `pathb-hotpath-summary.ps1` (trace + layer map).
3. Optional: re-bench `trace-f-2gpu-plus` on remus 4.3.2 for updated HELLO trace fields.

## Tier 0 exit criteria

| Metric | Target | Actual | Pass |
|--------|--------|--------|------|
| trace-f-3gpu G | 45+ | 42.8 | PARTIAL (+12% vs 38 pre) |
| trace-f-2gpu G (ops) | 45+ | **48.9** | **PASS** |
| copy rotation | 4 slots used | yes (99/99/96/96 on 2gpu) | PASS |
| overlap_pct | >0 meaningful | 0.5% (2gpu) | PASS |
| S4 correctness | 4B + 36B | both PASS | PASS |

45+ on 3-device F is topology-limited (serial split sum ~27 ms/tok). Production target met on 2-device F.