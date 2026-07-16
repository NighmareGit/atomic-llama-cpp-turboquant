# rocprofv3 GPU Kernel Profiling Guide

**Date:** 2026-07-16
**Purpose:** Practical reference for using rocprofv3 to profile ggml GPU kernels
**ROCm Version:** 7.2.3 (gfx1100 / Radeon RX 7900 XTX)

## 1. Overview

rocprofv3 is AMD's GPU profiling tool that captures per-kernel timing via HSA/ROCtracer interception. For ggml-based inference, it provides kernel-level breakdowns (MatMul, attention, SSM, MoE routing, quantization, etc.) that the built-in `GGML_SCHED_TRACE` cannot — the built-in tracer only sees "graph_compute_async" as an opaque block.

## 2. Prerequisites

### 2.1 Disable CUDA Graphs

**Required.** CUDA graphs batch multiple kernel launches into a single opaque call, hiding individual kernels from the profiler:

```bash
export GGML_CUDA_GRAPHS=0
```

Without this, rocprofv3 sees a single "graph launch" entry instead of individual MatMul, attention, and normalization kernels.

### 2.2 RPC Server

The RPC server must be running before the profiler starts:

```bash
rpc-server -H 0.0.0.0 -p 50051 -d CUDA0
```

## 3. Known Issues

### 3.1 `--hip-trace` Crash (SIGABRT)

**Do NOT use `--hip-trace`.** HIP API interception conflicts with ggml's stream management, causing `ggml_uncaught_exception()` SIGABRT:

```bash
# CRASHES:
rocprofv3 --hip-trace --kernel-trace --stats --summary -- ./llama-gpipe-profiler ...

# WORKS:
rocprofv3 --kernel-trace --stats --summary -- ./llama-gpipe-profiler ...
```

The `--kernel-trace` flag alone is sufficient for per-kernel timing. HIP API-level tracing (memcpy, malloc, etc.) is not needed.

### 3.2 Profiling Overhead (~9.5x slowdown)

rocprofv3 adds substantial overhead by intercepting every kernel launch:

| Metric | Unprofiled | Profiled | Factor |
|--------|-----------|----------|--------|
| TG TPS (Qwen35B-MTP Q6_K) | 143 t/s | 15.1 t/s | 9.5x |
| Per-step GPU time | ~1,200 ms | ~1,156 ms (inflated) | N/A |

**Relative kernel percentages are reliable.** Absolute timings are inflated by the interception overhead. Compare percentages, not absolute µs.

### 3.3 n_gen Impact

Because of the overhead, use fewer generation tokens for profiling. 64 tokens is fine for a full breakdown; 32 tokens gives proportional results at half the runtime.

## 4. Single-Model Profiling

### 4.1 Basic Command

```bash
rocprofv3 --kernel-trace --stats --summary -- \
  ./build/bin/llama-gpipe-profiler \
  -m /path/to/model.gguf \
  -rpc 127.0.0.1:50051 \
  --n-gpu-layers 99 \
  --gpipe-stages 3 \
  --tasks tg \
  --n-gen 64 \
  --repeat 1 \
  --no-warmup \
  -o /tmp/profile/heatmap.json \
  --out-dir /tmp/profile
```

For MTP models, add:
```bash
  --spec-type draft-mtp --spec-draft-n-max 2
```

### 4.2 Output Files

rocprofv3 writes to a timestamped subdirectory under `--out-dir`:

```
/tmp/profile/
  results.stats.csv           # Per-kernel statistics
  results.json                # Full trace (large)
  pmc_perf.csv                # Performance counters (if --pmc)
```

The key file is `results.stats.csv` with columns:

| Column | Description |
|--------|-------------|
| Name | Kernel function name |
| Calls | Number of invocations |
| TotalDurationNs | Total GPU time (nanoseconds) |
| AverageNs | Average per-call time |
| Percentage | % of total GPU kernel time |
| MinNs, MaxNs, StdDev | Timing distribution |

### 4.3 With env vars

Multiple env vars can be combined with `rocprofv3`:

```bash
env GGML_CUDA_GRAPHS=0 GGML_SCHED_TRACE=1 \
  rocprofv3 --kernel-trace --stats --summary -- \
  ./build/bin/llama-gpipe-profiler ...
```

Note: `GGML_SCHED_TRACE` adds host-side scheduling trace lines to stderr, which interleave with rocprofv3's output but don't affect kernel profiling.

## 5. Batch Profiling

### 5.1 Script Pattern

Use indexed arrays (not associative, which have non-deterministic iteration order in bash):

```bash
#!/bin/bash
set -euo pipefail

OUTDIR=/tmp/batch-profile
mkdir -p "$OUTDIR"

RUNS=(
  "qwen36-q6k|/mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf|--spec-type draft-mtp --spec-draft-n-max 2"
  "gemma4-12b|/mnt/models/gemma-4-12b-it-Q4_K_M.gguf|"
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r name model_path extra_flags <<< "$entry"
  extra_flags="${extra_flags:-}"
  rundir="$OUTDIR/$name"
  mkdir -p "$rundir"

  env GGML_CUDA_GRAPHS=0 timeout 300 rocprofv3 \
    --kernel-trace --stats --summary \
    -d "$rundir" -- \
    ./build/bin/llama-gpipe-profiler \
    -m "$model_path" \
    -rpc 127.0.0.1:50051 \
    --n-gpu-layers 99 \
    --gpipe-stages 3 \
    --tasks tg --n-gen 64 --repeat 1 --no-warmup \
    -o "$rundir/heatmap.json" \
    --out-dir "$rundir" \
    ${extra_flags:-} \
    2>&1 | tee "$rundir/run.log"

  echo "Done: $name"
done
```

### 5.2 Critical: env before timeout

`timeout` treats the first argument as the command. Always use `env`:

```bash
# WRONG -- timeout treats GGML_CUDA_GRAPHS=0 as a command:
timeout 300 GGML_CUDA_GRAPHS=0 rocprofv3 ...

# CORRECT:
env GGML_CUDA_GRAPHS=0 timeout 300 rocprofv3 ...
```

### 5.3 Critical: Default Tensor Split

Do NOT use `--tensor-split` with RPC. The default VRAM-proportional split respects the RPC GPU's VRAM limit. Explicit splits can force too many weights onto the RPC GPU, causing OOM:

| Explicit Split | RPC VRAM Requested | Result |
|---------------|--------------------|--------|
| `--tensor-split 30,70` | 7.77 GB (Qwen35B) | OOM on 8 GB 3060Ti |
| Default (none) | ~6 GB | Works |

If a model still fails with default split (e.g., 18 GB dense model on 8 GB RPC GPU), it exceeds the dual-GPU capacity and cannot be profiled in this configuration.

## 6. Parsing Results

### 6.1 Extract Top Kernels

```bash
# Top 15 kernels by GPU time:
grep -v '^$' results.stats.csv | sort -t',' -k4 -nr | head -16
```

### 6.2 Categorize Kernels

Map kernel names to categories for architectural analysis:

| Pattern | Category |
|---------|----------|
| `*matmul*` / `*mm*` / `*gemm*` | MatMul |
| `*attn*` / `flash_attn*` | Attention |
| `*topk_moe*` / `*moe*` | MoE Routing |
| `*ssm*` / `*delta*` | SSM |
| `*quantize*` / `*dequant*` | Quantization |
| `*norm*` / `rms*` | Normalization |
| `*rope*` | RoPE |
| `*copy*` / `*memcpy*` / `*set_rows*` / `*get_rows*` | Data Movement |
| `*add*` / `*mul*` / `*silu*` / `*gelu*` / `*swiglu*` | Element-wise |

## 7. Common Gotchas

| Gotcha | Symptom | Fix |
|--------|---------|-----|
| CUDA graphs enabled | Single "graph launch" kernel, no breakdown | `GGML_CUDA_GRAPHS=0` |
| `--hip-trace` crash | SIGABRT in ggml_uncaught_exception | Remove `--hip-trace`, use `--kernel-trace` only |
| timeout before env | `timeout: failed to run command 'GGML_CUDA_GRAPHS=0'` | `env GGML_CUDA_GRAPHS=0 timeout ...` |
| Wrong RPC tensor split | `alloc_tensor_range: failed to allocate RPC0 buffer` | Remove `--tensor-split`, use default |
| `pkill -f` kills wrapper | Bash script self-terminates | Use `kill $(pgrep ...)` or let processes exit naturally |
| Model too large | All layers on ROCm, RPC scratch allocation fails | Model exceeds dual-GPU capacity; skip |

## 8. Related Docs

- `docs/research/d76-rocprofv3-kernel-profile.md` — Single-model Qwen3.6-MTP Q6_K kernel breakdown
- `docs/research/d76b-multi-model-kernel-comparison.md` — 6-model comparison across Qwen and Gemma-4
- `docs/research/d72-gpu-timeline-profile.md` — Host-side timeline profiling with GGML_SCHED_TRACE
- `/tmp/d76-multi-profile/` — Raw profiling artifacts and run logs

---

*Guide maintained for Path-D GPU pipeline optimization. Update when rocprofv3 flags or ggml compatibility changes.*
