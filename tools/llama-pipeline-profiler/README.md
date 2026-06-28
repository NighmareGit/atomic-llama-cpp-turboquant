# llama.cpp/tools/llama-pipeline-profiler

Cross-platform pipeline profiler for llama.cpp (llama-bench parity). Measures
throughput, captures scheduler/RPC trace telemetry, and runs in-process diagnose,
regression append, and trace sampling without bash/Python on the default path.

Project docs: [docs/llama-pipeline-profiler/OVERVIEW.md](../../docs/llama-pipeline-profiler/OVERVIEW.md)

## Table of contents

1. [Syntax](#syntax)
2. [Examples](#examples)
3. [Output layout](#output-layout)
4. [Offline and preflight modes](#offline-and-preflight-modes)
5. [Optional scripts](#optional-scripts)

## Syntax

```
usage: llama-pipeline-profiler -m <model.gguf> [options]

options:
  -h, --help
  -m, --model <path>              GGUF model (required for live runs)
  --mode <name>                   throughput | trace | profile | ab-plus | spike-check | trace-observer
  --out-dir <path>                artifact root (default: ./profiler-out)
  --trace-dir <path>              sched/rpc jsonl dir (default: <out-dir>/telemetry)
  -o, --output <jsonl|none>       stdout format (default: jsonl)
  -rpc, --rpc <endpoints>         comma-separated RPC servers (required for multi-GPU)
  -ts, --tensor-split <ts>        e.g. 50,50 or 36,24,24,16
  -ngl, --n-gpu-layers <n>        default 99
  -sm, --split-mode <mode>        none | layer | row | tensor (default: layer)
  -ctk, --cache-type-k <t>        default q4_0
  -ctv, --cache-type-v <t>        default q4_0
  -r, --repetitions <n>           repeat each cell n times (trace on last rep only)
  -n, --n-gen <n>                  generation tokens (default: 128)
  -b, --batch-size <n>            default 512
  -ub, --ubatch-size <n>           default 512
  -t, --threads <n>               default: auto
  --sync-per-token                sync after each token (legacy llama-bench compare)
  --trace                         enable GGML sched/rpc/pipeline trace
  --with-gpu-telemetry            local nvidia-smi CSV sampler (C++, during gen)
  --overlap-target <pct>          B+6 gate threshold (default: 5)
  --trace-sample <n>              downsample trace jsonl (profile mode default: 10)
  --regression-file <path>        append diagnose row after trace runs
  --diagnose-only <dir>           offline diagnose on existing telemetry
  --validate-rpc                  probe -rpc endpoints and exit (R5 preflight)
  --skip-rpc-validate             skip automatic RPC probe before model load
  --no-warmup                     skip 1-token warmup
```

Environment (trace modes set these when `--trace` / `--trace-dir` is active):

- `GGML_SCHED_TRACE=1`, `GGML_SCHED_TRACE_FILE`
- `GGML_RPC_TRACE=1`, `GGML_RPC_TRACE_FILE`
- `GGML_PIPELINE_TRACE=1`, `GGML_PIPELINE_TRACE_FILE`
- `GGML_PIPELINE_PLUS` (per run cell)

Stdout default is one jsonl object per repetition. Artifacts land under `--out-dir`
when set.

## Examples

### 2-GPU trace (explicit RPC layout)

```bash
./build/bin/llama-pipeline-profiler -m /path/to/model.gguf \
  -rpc 192.168.8.176:50051 -ts 50,50 \
  -n 128 --mode trace --out-dir ./out
```

Diagnose runs automatically when `--trace` is on. Offline re-run:

```bash
./build/bin/llama-pipeline-profiler --diagnose-only ./out/telemetry
```

### Plus A/B (T1)

```bash
BENCH_RPC_ENDPOINT=192.168.8.176:50051 BENCH_TS=50,50 \
  ./scripts/bench-pipeline-plus-ab.sh my-cell
```

Or direct:

```bash
./build/bin/llama-pipeline-profiler -m /path/to/model.gguf \
  -rpc 192.168.8.176:50051 -ts 50,50 \
  -n 128 --mode ab-plus --trace --out-dir ./plus-ab-run
```

### Trace observer (throughput vs trace overhead)

```bash
./build/bin/llama-pipeline-profiler -m /path/to/model.gguf \
  -rpc 192.168.8.176:50051 -ts 50,50 \
  -n 128 -r 3 --mode trace-observer --out-dir ./observer-run
```

### 4-GPU cluster client

```bash
BENCH_RPC_ENDPOINT="192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053" \
BENCH_TS="36,24,24,16" \
  ./scripts/llama-pipeline-profiler-cluster.sh profiler-4gpu-primary
```

Preflight only (R5):

```bash
BENCH_RPC_ENDPOINT="192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053" \
BENCH_TS="36,24,24,16" \
  ./scripts/llama-pipeline-r5-validate.sh
```

### Repetitions (llama-bench style)

```bash
./build/bin/llama-pipeline-profiler -m /path/to/model.gguf \
  -rpc 192.168.8.176:50051 -ts 50,50 \
  -n 128 -r 5 --mode throughput -o jsonl
```

## Output layout

```text
<out-dir>/
  summary.md          # mean G over repetitions
  env.txt
  result.jsonl        # one row per repetition
  telemetry/          # when --trace
    sched-trace.jsonl
    rpc-trace.jsonl
    pipeline-trace.jsonl
    trace-summary.txt
    diagnose.json
    diagnose-summary.txt
    gpu/              # with --with-gpu-telemetry or --mode profile
      schema.json
      nvidia-local.csv
```

`env.txt` always includes `client_kind=native`.

## Offline and preflight modes

| Flag | Model required | Purpose |
|------|----------------|---------|
| `--diagnose-only <dir>` | no | Parse existing telemetry -> `diagnose.json` |
| `--validate-rpc` | no | HELLO + device memory probe for each `-rpc` endpoint |

Live runs with `-rpc` run `--validate-rpc` automatically unless `--skip-rpc-validate`.

## Optional scripts

Cluster and matrix wrappers are conveniences only; the binary is self-contained.

| Script | Role |
|--------|------|
| `scripts/llama-pipeline-r5-validate.sh` | R5 RPC preflight |
| `scripts/llama-pipeline-profiler-cluster.sh` | SSH/local cluster runner + preflight |
| `scripts/bench-pipeline-plus-ab.sh` | T1 Plus A/B matrix |
| `scripts/gpu-telemetry-collect.sh` | Remote ROCm/remus SSH sampler (optional) |
| `scripts/llama-pipeline-import-baseline.sh` | Seed regression from offline telemetry |

Legacy bash diagnose/regression scripts remain for reference; the binary uses C++.

See [TELEMETRY.md](TELEMETRY.md) and [GATES.md](GATES.md).