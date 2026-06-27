# Path-B Plus Tracking

**Overall:** B+1 VALIDATED | Tier 1 CLIENT SHIPPED (server deploy pending)  
**Build:** PASS (2026-06-27)  
**Current Phase:** Remus rpc-server v4.3 deploy + B+3 bench  
**Branch:** Path-B-Event-Support-Pipeline-Plus (private fork)

## Implementation log

| Date | Phase | Action |
|------|-------|--------|
| 2026-06-27 | B+0 | Plan + docs created from bug-hunt + user review |
| 2026-06-27 | B+1 | P0 pipeline_barrier, P1 sampling sync, P2 scoped drain, P3 trace copy/overlap |
| 2026-06-27 | B+1b | tls.ev event tracking fix; trace jsonl truncate; A/B bench |
| 2026-06-27 | B+2 | cpy_tensor_async + deferred COPY drain (client) |
| 2026-06-27 | B+3 | COPY_TENSOR_PEER proto v4.3 (client+server code; remus image pending) |

## Issues

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|
| 2026-06-27 | Path B plateau on N-RPC | Graph reuse skips copy rotation; full sync drains pipe | B+1 Tier 0 | FIXED |
| 2026-06-27 | Plus regression G=28 vs 38 | drain_pending cleared tls but not ev->response_pending; stale flags broke pipeline_barrier | tls.ev tracking + rpc_finish_event_response | FIXED |

## Benchmarks

| Run | Config | Baseline G | Plus G | Delta | Notes |
|-----|--------|------------|--------|-------|-------|
| trace-f-3gpu-pre | 3-device 36B NL | 38.0 | - | - | Pre-Plus (bug-hunt) |
| trace-f-3gpu-legacy | 3-device, Plus=0 | - | 39.3 | - | Full sync on reuse; copy3 stuck (508/520 splits) |
| trace-f-3gpu-plus | 3-device, Plus=1 | 39.3 | **42.8** | **+3.5** | Copy rotation 132/132/128/128; overlap_pct=0.3% |
| trace-f-3gpu-tier1 | 3-device, Plus=1 + B+2 client | 42.8 | 40.8 | -2.0 | 72 deferred COPY; peer COPY blocked (remus v4.2) |

Artifacts: `docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-{plus,legacy,tier1}/`

## Remus deploy (required for B+3)

```bash
# After pushing branch to git remote used by remus build.sh:
./rpc-patch/scripts/pathb-remus-rpc.sh build
./rpc-patch/scripts/pathb-remus-rx6600-rpc.sh build
./rpc-patch/scripts/pathb-remus-multi-rpc-win.sh start
```

Server must report `RPC_PROTO_MINOR_VERSION=3` in logs for `COPY_TENSOR_PEER`.

## Tier 0 exit criteria

| Metric | Target | Actual | Pass |
|--------|--------|--------|------|
| trace-f-3gpu G | 45+ | 42.8 | PARTIAL (+12% vs 38 pre) |
| copy rotation | 4 slots used | yes | PASS |
| overlap_pct | >0 meaningful | 0.3% | FAIL (Tier 1) |