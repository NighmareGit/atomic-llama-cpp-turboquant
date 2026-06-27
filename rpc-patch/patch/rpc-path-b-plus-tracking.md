# Path-B Plus Bench Log

Mirror of [../docs/rpc-path-b-plus-tracking.md](../docs/rpc-path-b-plus-tracking.md) for run artifacts.

## Runs

| Label | Dir | G | overlap | Notes |
|-------|-----|---|---------|-------|
| trace-f-3gpu-pre | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu | 38.0 | 0 | Pre-Plus baseline |
| trace-f-3gpu-legacy | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-legacy | 39.3 | 0.1% | Plus=0, copy3=508 |
| trace-f-3gpu-plus | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-plus | 42.8 | 0.3% | Plus=1, balanced copies |
| trace-f-3gpu-tier1 | docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-tier1 | 40.8 | 0.2% | B+2 client; remus still v4.2 |