# Path-B Plus benchmark artifacts

Published results for **pipeline layer** benchmarks on branch
`Path-B-Event-Support-Pipeline-Plus`. This tree is separate from upstream
`benches/dgx-spark/` and `benches/mac-m2-ultra/` style `llama-bench` dumps.

Methodology and run contracts: [BENCHMARKING.md](../../BENCHMARKING.md).
Architecture: [PIPELINE.md](../../PIPELINE.md).
Mission / gate plan: [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/).
Native profiler: [docs/llama-pipeline-profiler/OVERVIEW.md](../../docs/llama-pipeline-profiler/OVERVIEW.md).

## Layout

| Directory | Tier | Question |
|-----------|------|----------|
| [depth-2/](depth-2/) | T0 | Does `LLAMA_PIPELINE_DEPTH2` help speculative overlap? |
| [plus-ab/](plus-ab/) | T1 | Does `GGML_PIPELINE_PLUS` unblock split overlap? |
| [regression.jsonl](regression.jsonl) | T1 | Append-only profiler gate history |
| [profiler-baseline/](profiler-baseline/) | P0-1 | Offline seeded baselines until cluster rerun |
| [composition/](composition/) | T3 | Full-stack cells (spec + turbo + plus + topology) |

Each run should ship:

- `summary.md` - host metadata + result table
- `env.txt` - env vars and git sha at run time
- Optional: symlink or path reference to full telemetry under `rpc-patch/patch/bench-results/` or `docs/cuda-windows-5070ti/benchmarks/`

## Seeded results (imported from existing artifacts)

### T1 -- Path-B Plus A/B (multi-GPU)

See [plus-ab/summary.md](plus-ab/summary.md). Source artifacts:

- [rpc-patch/docs/rpc-path-b-plus-spikes.md](../../rpc-patch/docs/rpc-path-b-plus-spikes.md)
- [docs/cuda-windows-5070ti/benchmarks/README.md](../../docs/cuda-windows-5070ti/benchmarks/README.md)
- [rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md](../../rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md)

### T0 -- depth-2 A/B (single-node)

No published rows yet. Run:

```bash
# MTP server must already be listening (e.g. scripts/run-gemma4-mtp-server.sh)
HOST=127.0.0.1 PORT=8080 N_PREDICT=128 ./scripts/bench-pipeline-depth2.sh
```

Gate target: >= +15% median tps vs `LLAMA_PIPELINE_DEPTH2=0` on f16-mtp short prompt
(3 runs). See [docs/development/pipeline-depth-2-pure-overlap.md](../../docs/development/pipeline-depth-2-pure-overlap.md).

## Contributing a row

1. Run the appropriate stub from [BENCHMARKING.md](../../BENCHMARKING.md) quick reference.
2. Copy or reference artifacts into the matching subdirectory.
3. Add a row to the summary table in this README or the tier `summary.md`.
4. Include host metadata (machine, backend, build sha, model, ctx, KV, topology).