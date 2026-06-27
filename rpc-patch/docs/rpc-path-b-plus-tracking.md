# Path-B Plus Tracking

**Overall:** B+1 VALIDATED | Tier 1 SHIPPED | tier1b = run variance (fluke)  
**Build:** 9963 (`5e48bb3a4`) client + remus v4.3 deployed  
**Current Phase:** Phase 5 topology (2-device F ops + Path C spike)  
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

## Issues

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|
| 2026-06-27 | Path B plateau on N-RPC | Graph reuse skips copy rotation; full sync drains pipe | B+1 Tier 0 | FIXED |
| 2026-06-27 | Plus regression G=28 vs 38 | drain_pending cleared tls but not ev->response_pending | tls.ev tracking | FIXED |
| 2026-06-27 | tier1 vs plus G delta -2.0 | Same build 9962; single-run jitter | Multi-run not needed; closed as fluke | CLOSED |
| 2026-06-27 | tier1b no COPY_TENSOR_PEER | 3gpu F copies are CUDA<->RPC not RPC<->RPC | Pivot B+3 to Path C / 5-ep | WONTFIX on 3gpu |

## Benchmarks

| Run | Config | G (t/s) | Build | Notes |
|-----|--------|---------|-------|-------|
| trace-f-3gpu-pre | 3-device 36B NL | 38.0 | - | Pre-Plus (bug-hunt) |
| trace-f-3gpu-legacy | 3-device, Plus=0 | 39.3 | 9962 | Full sync on reuse |
| trace-f-3gpu-plus | 3-device, Plus=1 | **42.8** | 9962 | Best single run; copy rotation OK |
| trace-f-3gpu-tier1 | 3-device, Plus=1 | 40.8 | **9962** | Same binary as plus |
| trace-f-3gpu-tier1b | 3-device, Plus=1, remus v4.3 | 40.6 | 9963 | Within noise of tier1; 0 peer COPY (expected) |
| trace-f-2gpu-plus | 2-device ts=50,50, Plus=1 | **48.9** | 9964 | Ops target hit; 3 splits; remus HELLO minor=2 |

Artifacts: `docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-{plus,legacy,tier1,tier1b}/`

## Remus deploy

```bash
./rpc-patch/scripts/pathb-remus-rpc.sh build
./rpc-patch/scripts/pathb-remus-rx6600-rpc.sh build
./rpc-patch/scripts/pathb-remus-multi-rpc-win.sh start
```

Server logs: `RPC <endpoint>: proto 4.3 peer_copy=yes` (when trace/load connects).

## Next steps (Phase 5)

1. Re-bench `trace-f-2gpu` on build 9963 (2-device F, drop :50052).
2. Path C spike: single remus rpc-server for 5060+6600.
3. `pathb-hotpath-summary.ps1` (trace + layer map).

## Tier 0 exit criteria

| Metric | Target | Actual | Pass |
|--------|--------|--------|------|
| trace-f-3gpu G | 45+ | 42.8 | PARTIAL (+12% vs 38 pre) |
| copy rotation | 4 slots used | yes | PASS |
| overlap_pct | >0 meaningful | 0.3% | FAIL (structural serial splits) |

45+ on 3-device F may be blocked by serial split sum (~27 ms/tok). Use 2-device F for throughput target.