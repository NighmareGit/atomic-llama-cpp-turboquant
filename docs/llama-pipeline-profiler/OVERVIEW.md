# llama-pipeline-profiler

Multi-backend pipeline diagnosis for llama.cpp: scheduler split stalls, RPC hop
budget, pipeline saturation, and hybrid GPU telemetry. The tool is a cross-platform
C++ binary (llama-bench parity) with in-process diagnose, sampling, and regression.

## Doc map

| Document | Purpose |
|----------|---------|
| [PLAN.md](PLAN.md) | Implementation plan, R1-R5 refactor, success criteria |
| [TRACKING.md](TRACKING.md) | Living log, checklist, gate results (update per atomic step) |
| [../../tools/llama-pipeline-profiler/README.md](../../tools/llama-pipeline-profiler/README.md) | Tool syntax, examples, output formats |
| [../../tools/llama-pipeline-profiler/TELEMETRY.md](../../tools/llama-pipeline-profiler/TELEMETRY.md) | Artifact and `diagnose.json` schemas |
| [../../tools/llama-pipeline-profiler/GATES.md](../../tools/llama-pipeline-profiler/GATES.md) | S1/S3/S4/S5, B+6, smell detectors |

## Related project docs

| Document | Scope |
|----------|-------|
| [../../BENCHMARKING.md](../../BENCHMARKING.md) | Global bench taxonomy T0-T3, run contract |
| [../../PIPELINE.md](../../PIPELINE.md) | Layer A depth-2 + Layer B Path-B Plus architecture |
| [../../benches/path-b-plus/README.md](../../benches/path-b-plus/README.md) | Published matrix artifacts |

## Tool roles

| Tool | When to use |
|------|-------------|
| **llama-pipeline-profiler** | Native-client pipeline trace, Plus A/B, spike gates, R5 RPC preflight |
| **llama-gpipe-profiler** | Task-stratified native profiler with server telemetry, heatmap synthesis. Use for new gpipe/heatmap work. See [tools/llama-gpipe-profiler/README.md](../../tools/llama-gpipe-profiler/README.md). |
| **llama-bench** | Generic throughput matrices (no pipeline internals) |
| **rpc-server-bench.sh** | HTTP production gate (`BENCH_HTTP=1` fallback in T1 stub) |
| **bench-pipeline-depth2.sh** | T0 depth-2 A/B via live `llama-server` |

## Quick start

```bash
# Build
cmake --build build --target llama-pipeline-profiler

# R5 preflight (no model)
BENCH_RPC_ENDPOINT=host:50051,host2:50052 BENCH_TS=50,50 \
  ./scripts/llama-pipeline-r5-validate.sh

# 2-GPU trace + in-process diagnose
./build/bin/llama-pipeline-profiler -m MODEL.gguf \
  -rpc host:50051 -ts 50,50 -n 128 --mode trace --out-dir ./out

# Offline diagnose on saved telemetry
./build/bin/llama-pipeline-profiler --diagnose-only ./out/telemetry
```

Cluster (romulus): `scripts/llama-pipeline-profiler-cluster.sh` (runs R5 preflight first)

Regression history: `benches/path-b-plus/regression.jsonl`