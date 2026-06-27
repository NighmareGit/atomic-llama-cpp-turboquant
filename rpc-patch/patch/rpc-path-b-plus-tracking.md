# Path-B Plus Bench Log

Mirror of [../docs/rpc-path-b-plus-tracking.md](../docs/rpc-path-b-plus-tracking.md) for run artifacts.

**Status:** B+1 production ready. B+4..B+6 shipped (rebuild required). Phase 7-8 tooling/cluster shipped. Phase 9 S0-lite pending rebuild+bench.

## Runs

| Label | Dir | G | overlap | Notes |
|-------|-----|---|---------|-------|
| trace-f-3gpu-pre | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu | 38.0 | 0.1% | Pre-Plus baseline |
| trace-f-3gpu-legacy | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-legacy | 39.3 | 0.1% | Plus=0, copy3=508 |
| trace-f-3gpu-plus | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-plus | 42.8 | 0.3% | Plus=1, 4 splits |
| trace-f-3gpu-tier1 | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-tier1 | 40.8 | 0.2% | B+2 client; build 9962 |
| trace-f-3gpu-tier1b | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-tier1b | 40.6 | 0.3% | remus v4.3; 0 peer COPY |
| trace-f-2gpu-plus | docs/cuda-windows-5070ti/benchmarks/trace-f-2gpu-plus | **48.9** | 0.5% | **Production default** |
| s4-4b-2gpu-plus | docs/cuda-windows-5070ti/benchmarks/s4-4b-2gpu-plus | 44.2 | - | S4 smoke PASS |
| proto-check-v43 | docs/cuda-windows-5070ti/benchmarks/proto-check-v43 | - | - | HELLO minor=3 peer_copy |

## Deploy / probe artifacts

| Label | Dir | Notes |
|-------|-----|-------|
| proto-check-v43 | benchmarks/proto-check-v43 | remus 4.3.2 negotiation |
| probe-notes | benchmarks/telemetry/probe-notes.txt | PCIe dmon methodology |

Canonical tracking, Phase 5 checklist, and Phase 6 plan: [../docs/rpc-path-b-plus-tracking.md](../docs/rpc-path-b-plus-tracking.md).