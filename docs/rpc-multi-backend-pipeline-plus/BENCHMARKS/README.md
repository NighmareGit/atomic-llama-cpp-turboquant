# BENCHMARKS — rpc-multi-backend-pipeline-plus

Curated reference runs and matrix summaries for multi-backend RPC orchestration work.

## Primary Reference Locations (canonical tree)

| Path | Contents |
|------|----------|
| `docs/cuda-windows-5070ti/benchmarks/` | Windows `result.meta`, `bench.result`, `server.log`, GPU monitor |
| `benches/path-b-plus/` | Profiler gate artifacts (`b6-*`, `regression.jsonl`) |
| `rpc-patch/patch/bench-results/cluster-4gpu-primary/` | Romulus 4-GPU stable series |
| `docs/cuda-windows-5070ti/PROFILING.md` | 2026-06-27 bottleneck classification |

## Key Runs Catalog

| Label | Topology | Plus | G (t/s) | overlap_pct | Artifact | Notes |
|-------|----------|------|---------|-------------|----------|-------|
| `trace-f-2gpu-plus` | 5070+5060 `ts=50,50` | 1 | 48.9 | 0.6% | `benchmarks/trace-f-2gpu-plus/` | **Production default** |
| `profile-f-36b-nl-no6600` | Same | 1 | 48.9 | — | `benchmarks/profile-f-36b-nl-no6600/` | Best power telemetry |
| `trace-f-3gpu-plus` | 3-device F | 1 | 42.8 | 0.2% | `benchmarks/trace-f-3gpu-plus/` | Topology-limited |
| `b6-2gpu-f` | romulus + remus | 1 | 75.6 | 0.2% | `benches/path-b-plus/b6-2gpu-f/` | B+6 gate |
| `b6-2gpu-f-triton` | romulus + triton 3090 | 1 | 186.6 | 0.3% | `benches/path-b-plus/b6-2gpu-f-triton/` | Straggler A/B |
| `b6-4gpu-g` | 4-GPU JUPITER canonical | 1 | 77.2 | 0.1% | romulus only (not in git) | drain 50.4s |
| `trace-g-4gpu-primary` | romulus 4-GPU | 1 | ~43.0 | 0.1% | `cluster-4gpu-primary/` | Path-C baseline |

## How to Add New Runs

1. Run with `rpc-server-bench.ps1 -Profile` or `b6-gate-run-remote.sh <preset>`
2. Reference artifact path in [TRACKING.md](../TRACKING.md)
3. Append row to `benches/path-b-plus/regression.jsonl` via profiler import
4. Update audit if bottleneck classification changes

## Planned Matrices

- `2026-07-comparison-matrix.md` — 2-device vs 3-device vs 4-GPU with per-split timing (after Phase 1.1)

---

Raw data stays in canonical trees above. This folder holds summaries and cross-references only.