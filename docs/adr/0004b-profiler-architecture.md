# ADR-0004b: Profiler Binary Architecture (`llama-gpipe-profiler`)

## Status

**ACCEPTED** — 2026-07-11

## Context

D4.7 research (docs/wayfinder/D4.7-profiler-research.md) established the
design space for a task-stratified, multi-GPU profiler. D4.8 prototype
(Agent A, parallel) validates the server-telemetry wire format end-to-end
via the `ggml-rpc.cpp` client. This ADR finalizes the binary architecture,
CLI surface, heatmap schema, and telemetry ingestion contract for the
production `llama-gpipe-profiler` binary (D4.10).

The existing `llama-pipeline-profiler` (tools/llama-pipeline-profiler/)
provides proven RPC registration, trace-env setup, and diagnose machinery.
The new binary borrows those patterns rather than reimplementing them.

### Problem

The current profiler stack is split: `llama-pipeline-profiler` handles
multi-GPU trace capture and diagnose, while Python scripts
(`b6-gate-phase0-assembly-bounds.py`) consume raw jsonl. There is no
single native binary that:
1. Drives both prompt-processing (pp) and token-generation (tg) tasks.
2. Produces a task-stratified heatmap JSON consumed by the Pareto optimizer.
3. Ingests server-side telemetry (per-layer timing, KV cache timing) when
   available, degrading gracefully when not.

### Inputs

| Source | Finding |
|--------|---------|
| D4.7 research | Binary pattern: `llama-bench`; KV: server telemetry (Option B); heatmap: task-stratified |
| D4.8 prototype | `rpc_msg_server_telemetry` 6-field frame validated end-to-end |
| D4.10 acceptance criteria | CMake target, CLI, task orchestration, heatmap synthesis, script adaptation |
| llama-pipeline-profiler | `register_rpc_servers`, `setup_trace_env`, `run_session` patterns to reuse |

---

## Decision

### Binary Architecture

Standalone `llama-gpipe-profiler` in `tools/llama-gpipe-profiler/`,
mirroring the `llama-bench` file layout:

```
tools/llama-gpipe-profiler/
  CMakeLists.txt          # llama-gpipe-profiler-impl (lib) + llama-gpipe-profiler (exe)
  main.cpp                # thin entry: llama_gpipe_profiler(argc, argv)
  llama-gpipe-profiler.cpp # all logic: CLI, model load, RPC, tasks, heatmap
  README.md               # usage
```

**CMake targets:**

| Target | Type | Sources | Links |
|--------|------|---------|-------|
| `llama-gpipe-profiler-impl` | library | llama-gpipe-profiler.cpp | llama-common, llama, [ggml-rpc] |
| `llama-gpipe-profiler` | executable | main.cpp | llama-gpipe-profiler-impl |

**Linkage:** identical to `llama-pipeline-profiler` — links `llama-common`
and `llama` unconditionally; `ggml-rpc` only when `GGML_RPC` is set, with
`GGML_BACKEND_DL` guard (dependency-only, no link when dynamic loading).

**Reuse strategy:** the new binary does NOT refactor shared code into a
common library at this stage. It copies the small, stable helpers from
`llama-pipeline-profiler` (`register_rpc_servers`, `setup_trace_env`,
`parse_tensor_split`, `parse_cache_type`, `run_prompt`, `run_gen`) into
`llama-gpipe-profiler.cpp`. Rationale: the two binaries have different
evolution velocities (pipeline-profiler is tied to B+6 gate scripts;
gpipe-profiler targets Pareto optimization). A shared library would couple
their release cycles. Revisit in D5 if divergence becomes painful.

### CLI Surface

Finalized from D4.7 section 4.1:

```
  -m, --model PATH           GGUF model path (required)
  -rpc, --rpc HOST:PORT      RPC endpoint (comma-separated for multiple)
  -ts, --tensor-split LIST   Per-device weight split (comma-separated floats)
  --tasks LIST               Tasks to profile: pp, tg (comma-separated, default: pp,tg)
  -p, --n-prompt N           Prompt tokens for pp task (default: 512)
  -f, --file PATH            Prefill from prompt text file (overrides -p)
  -n, --n-gen N              Generation tokens for tg task (default: 128)
  -r, --repeat N             Repetitions per task (default: 5)
  --warmup, --no-warmup      Warmup run before profiling (default: enabled)
  -o, --output PATH          Heatmap output path (default: heatmap.json)
  --out-dir PATH             Artifact directory (default: ./profiler-out)
  --trace                    Enable sched/rpc/pipeline traces
  --server-telemetry         Request server telemetry (sets GGML_RPC_SERVER_TELEMETRY=1)
  -ngl, --n-gpu-layers N     GPU layers (default: 99)
  -sm, --split-mode MODE     Split strategy: none, layer, row, tensor (default: layer)
  -ctk, --cache-type-k TYPE  KV cache K type (default: q4_0)
  -ctv, --cache-type-v TYPE  KV cache V type (default: q4_0)
  -b, --batch-size N         Batch size (default: 512)
  -ub, --ubatch-size N       Micro-batch size (default: 512)
  -t, --threads N            Thread count (default: auto)
  --ctx-size N               Context size (default: 4096)
  --overlap-target N         B+6 gate threshold percent (default: 5)
  -h, --help                 Usage
```

**Design rationale:**
- `--tasks pp,tg` drives task stratification. Each task runs as an
  independent cell with its own trace capture and timing.
- `--server-telemetry` sets `GGML_RPC_SERVER_TELEMETRY=1` before model load.
  Without it, only client-side traces are available.
- `--trace` enables `GGML_SCHED_TRACE`, `GGML_RPC_TRACE`,
  `GGML_PIPELINE_TRACE` and writes jsonl under `--out-dir/telemetry/`.
- Repetitions: trace capture on last rep only (matches pipeline-profiler
  behavior -- trace on final rep avoids first-rep cache effects).
- Warmup runs are excluded from timing and trace capture.

### Heatmap JSON Schema

Primary output (default `heatmap.json`). Schema version 1, additive-only
evolution. Unknown keys ignored by consumers.

```json
{
  "schema_version": 1,
  "generated_at": "2026-07-11T12:00:00Z",
  "git_sha": "abc1234",
  "label": "qwen36-35b-triton-2gpu",
  "model": {
    "path": "/models/Qwen3.6-35B-A3B.gguf",
    "n_layers": 60,
    "param_count_b": 36.0,
    "ctx_size": 4096
  },
  "tasks": {
    "pp": {
      "n_prompt_tokens": 512,
      "n_batch": 512,
      "wall_ms": 1250.3,
      "tps": 409.6,
      "layers": [ { "idx": 0, "ms": 18.2, "gpu_id": 0 } ],
      "kv": { "read_ms": 1.1, "write_ms": 0.8, "evict_count": 0 }
    },
    "tg": {
      "n_gen_tokens": 128,
      "wall_ms": 3850.0,
      "tps": 33.2,
      "layers": [ { "idx": 0, "ms": 58.1, "gpu_id": 0 } ],
      "kv": { "read_ms": 0.05, "write_ms": 0.03, "evict_count": 2 }
    }
  },
  "gpu_metadata": [
    {
      "id": 0,
      "name": "NVIDIA RTX 3090",
      "backend": "CUDA",
      "vram_total_mib": 24576,
      "pci_bus_id": "0000:01:00.0",
      "pci_link_gen": 4,
      "pci_link_width": 16
    }
  ],
  "summary": {
    "straggler_gpu": 1,
    "overlap_pct": 0.2,
    "gate_s5": "PASS",
    "gate_b6": "FAIL"
  }
}
```

**Per-layer entry fields:**

| Field | Type | Description |
|-------|------|-------------|
| `idx` | int | Layer index (0-based) |
| `ms` | float | Compute time (ms) for this layer |
| `gpu_id` | int | GPU that executed this layer |

**KV cache timing fields (optional, server-telemetry only):**

| Field | Type | Description |
|-------|------|-------------|
| `read_ms` | float | KV cache read time (ms) |
| `write_ms` | float | KV cache write time (ms) |
| `evict_count` | int | Number of KV slot evictions |

When server telemetry is unavailable, the `kv` object is omitted from the
task entry and `summary.kv_source` is set to `"estimated"`.

### Telemetry Ingestion

The profiler reads `server-telemetry.jsonl` produced by the `ggml-rpc`
client library (Agent A's parallel work). Each line is JSON:

**Gating condition:** Telemetry is returned on both the `GRAPH_COMPUTE` (single-device)
and `GRAPH_COMPUTE_ALL` (multi-device) response paths. The single-device path
covers any RPC server with 1+ GPUs (e.g., `rpc-server -d CUDA0`), while the
multi-device path activates when `n_devices_on_endpoint > 1` (e.g.,
`rpc-server -d CUDA0,CUDA1`). This means a setup with a local GPU and a single-GPU
RPC server (like Romulus: 7900 XTX + 3060 Ti `-d CUDA0`) is fully profiled.
When telemetry is unavailable, the profiler degrades gracefully (see "Ingestion
rules" below).

```json
{
  "event": "server_telemetry",
  "ts_us": 123456,
  "trace_id": 1,
  "device_timings_us": [1234, 5678],
  "layer_assignments": [0, 32],
  "copy_times_us": [100, 200],
  "device_meta": [
    { "name": "NVIDIA RTX 3090", "vram_mib": 24560, "backend": "ROCm", "pcie_gen": 4, "pcie_width": 16 }
  ],
  "kv_read_times_us": [45, 30],
  "kv_write_times_us": [80, 12]
}
```

**Ingestion rules:**
1. If `--server-telemetry` is passed, set `GGML_RPC_SERVER_TELEMETRY=1`
   before model load.
2. After the run, read `server-telemetry.jsonl` from `--out-dir/telemetry/`.
3. Parse each `server_telemetry` entry; aggregate per-layer timing across
   repetitions (p50 for the heatmap `layers[].ms`).
4. If the file is missing or empty, fall back to client-side traces only.
   Emit a stderr warning: `"server telemetry unavailable; heatmap kv fields omitted"`.
5. Raw telemetry is stored under `--out-dir/telemetry/` when `--trace` is on.

**No compile-time dependency** on Agent A's struct changes. The profiler
parses the JSONL format only. If the wire format changes, only the JSON
parsing logic in `llama-gpipe-profiler.cpp` needs updating.

### Script Adaptation

- `b6-gate-phase0-assembly-bounds.py`: accept optional `--server-telemetry`
  path argument. When provided, read the 6 telemetry fields and incorporate
  into the overlap/straggler analysis. When absent, behavior is unchanged.
- `diagnose.json`: extend schema with optional fields:
  `device_timings_us`, `layer_assignments`, `copy_times_us`, `device_meta`,
  `kv_read_times_us`, `kv_write_times_us`. Existing consumers ignore unknown
  keys.
- `llama-pipeline-profiler`: stays working unchanged. New fields are purely
  additive.

---

## Consequences

| Positive | Negative |
|----------|----------|
| Single native binary drives pp + tg tasks with task-stratified output | Duplicates ~200 lines of helpers from llama-pipeline-profiler |
| Heatmap schema is forward-compatible (additive-only) | No shared library yet -- divergence risk over time |
| Graceful degradation when server telemetry unavailable | Telemetry ingestion is file-based (not in-process) |
| No compile-time coupling to Agent A's struct changes | JSONL parse adds a small post-processing step |
| Reuses proven RPC registration and trace-env patterns | |

---

## What Changes

| Component | Change |
|-----------|--------|
| `tools/llama-gpipe-profiler/` | New directory: CMakeLists.txt, main.cpp, llama-gpipe-profiler.cpp, README.md |
| `tools/CMakeLists.txt` | `add_subdirectory(llama-gpipe-profiler)` |
| `docs/adr/0004b-profiler-architecture.md` | This ADR |
| `scripts/b6-gate-phase0-assembly-bounds.py` | Optional `--server-telemetry` input |
| `tools/llama-pipeline-profiler/pipeline-diagnose.cpp` | Optional telemetry fields in diagnose.json |

## What Does NOT Change

- `llama-pipeline-profiler` binary (stays working; new fields optional)
- `llama-bench` (unaffected)
- `ggml-rpc.cpp` server (Agent A's domain; this binary only consumes its output)
- Server-side telemetry gating (controlled by Agent A's `GGML_RPC_SERVER_TELEMETRY`)

---

## Implementation Sketch

```
main:
  parse args -> config
  llama_backend_init()
  ggml_backend_load_all()
  load model (llama_model_load_from_file + llama_init_from_model)
  register RPC endpoints (if --rpc)
  setup trace env (if --trace)
  for each task in config.tasks:
    repeat config.repeat times:
      if last repeat: enable trace capture
      run task:
        pp: batched llama_decode of n_prompt tokens, time wall
        tg: serial llama_decode of n_gen tokens, time wall
      if last repeat: traces land in out-dir/telemetry/
  parse server-telemetry.jsonl (if --server-telemetry)
  synthesize heatmap:
    per-layer ms from server telemetry device_timings/layer_assignments
    pp/tg wall time from run loop
    gpu_metadata from server telemetry device_meta
  write heatmap.json
  llama_backend_free()
```

---

*Decision finalized 2026-07-11. Feeds D4.10 implementation.*
