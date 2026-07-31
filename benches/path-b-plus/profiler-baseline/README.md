# Profiler baselines (P0-1 offline seed)

Cluster traced rerun (`trace-g-4gpu-primary-trace` with `decode_id` + `pipeline-trace.jsonl`)
is pending romulus power-on. Until then, regression rows are seeded from:

| Label | Source | Notes |
|-------|--------|-------|
| trace-f-2gpu-plus | `docs/cuda-windows-5070ti/benchmarks/trace-f-2gpu-plus/telemetry` | diagnose.json 2026-06-28 |
| trace-f-3gpu-plus | `docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-plus/telemetry` | import script |
| trace-f-3gpu-legacy | `docs/cuda-windows-5070ti/benchmarks/trace-f-3gpu-legacy/telemetry` | Plus=0 |
| trace-g-4gpu-primary-trace | hotpath summary | G~37, overlap 0.1%; full jsonl on romulus |

```bash
./scripts/llama-pipeline-import-baseline.sh
./scripts/llama-pipeline-profiler-cluster.sh profiler-4gpu-primary   # when cluster online
```

See [../../docs/llama-pipeline-profiler/TRACKING.md](../../docs/llama-pipeline-profiler/TRACKING.md).