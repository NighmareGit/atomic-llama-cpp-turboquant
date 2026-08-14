# llama-gpipe-profiler

Task-stratified, multi-GPU profiler for llama.cpp pipeline (gpipe) inference.
Drives prompt-processing (pp) and token-generation (tg) tasks across local
and RPC endpoints, captures both client-side traces and optional server
telemetry, and synthesizes a task-stratified heatmap JSON consumed by the
Pareto optimizer (D4.11+).

Patterned after `tools/llama-bench` and `tools/llama-pipeline-profiler`.

## Relationship to llama-pipeline-profiler

| Tool | Role | When to use |
|------|------|-------------|
| **llama-gpipe-profiler** | Task-stratified native profiler with server telemetry, heatmap synthesis, and graceful degradation | New profiling work; server-side timing needed; Pareto optimizer input |
| **llama-pipeline-profiler** | Native pipeline trace, diagnose, Plus A/B, spike gates, R5 RPC preflight | Existing regression gates; pipeline internals; diagnose-only mode |

The two binaries share some helper patterns (`register_rpc_servers`,
`setup_trace_env`) but evolve independently. `llama-gpipe-profiler` is the
newer, task-stratified profiler targeting Pareto optimization (D4.11-D4.14).
`llama-pipeline-profiler` is stable and tied to existing B+6 gate scripts.
No flag day — both remain available.

## Build

```sh
# Standard build with RPC support
cmake -B build -DGGML_RPC=ON
cmake --build build --target llama-gpipe-profiler -j

# ROCm build (Romulus dual-GPU, AMD client)
cmake --build build-rocm-docker --target llama-gpipe-profiler -j

# CUDA build (NVIDIA client)
cmake --build build-cuda --target llama-gpipe-profiler -j
```

Requires `GGML_RPC` enabled for RPC endpoint profiling. Without RPC, the
binary profiles local-only configurations.

## CLI Reference

```
  -m, --model PATH           GGUF model path (required)
  -rpc, --rpc HOST:PORT      RPC endpoint (comma-separated for multiple)
  -ts, --tensor-split LIST   Per-device weight split (comma-separated floats)
  --tasks LIST               Tasks: pp, tg (comma-separated, default: pp,tg)
  -p, --n-prompt N           Prompt tokens for pp task (default: 512)
  -f, --file PATH            Prefill from prompt text file (overrides -p)
  -n, --n-gen N              Generation tokens for tg task (default: 128)
  -r, --repeat N             Repetitions per task (default: 5)
  --warmup, --no-warmup      Warmup run before profiling (default: enabled)
  -o, --output PATH          Heatmap output path (default: heatmap.json)
  --out-dir PATH             Artifact directory (default: ./profiler-out)
  --trace                    Enable sched/rpc/pipeline traces
  --server-telemetry         Request server telemetry (GGML_RPC_SERVER_TELEMETRY=1)
  -ngl, --n-gpu-layers N     GPU layers (default: 99)
  -sm, --split-mode MODE     Split strategy: none, layer, row, tensor (default: layer)
  -ctk, --cache-type-k TYPE  KV cache K type (default: q4_0)
  -ctv, --cache-type-v TYPE  KV cache V type (default: q4_0)
  -b, --batch-size N         Batch size (default: 512)
  -ub, --ubatch-size N       Micro-batch size (default: 512)
  -t, --threads N            Thread count (default: auto)
  --ctx-size N               Context size (default: 4096)
  --overlap-target N         B+6 gate threshold percent (default: 5)
  --spec-type draft-mtp       Enable MTP speculative decoding (TG-only)
  --spec-draft-n-max N        Max draft tokens per MTP step (default: 2)
  --spec-draft-n-min N        Min draft tokens per MTP step (default: 1)
  --model-draft PATH          Separate draft model for MTP (e.g., Gemma assistant)
  -h, --help                 Usage
```

## Usage Examples

### Local-only (single GPU, no RPC)

```sh
./build/bin/llama-gpipe-profiler \
  -m /models/Qwen3.6-35B-A3B.gguf \
  --tasks pp,tg \
  --out-dir ./profiler-out
```

### Single-GPU RPC server (Romulus: local ROCm + RPC NVIDIA)

```sh
GGML_RPC_SERVER_TELEMETRY=1 \
./build-rocm-docker/bin/llama-gpipe-profiler \
  -m <model-mount>/Qwen3.5-9B-MTP-Q4_K_M.gguf \
  --rpc 127.0.0.1:50051 \
  --tensor-split 50,50 \
  --server-telemetry \
  --tasks pp,tg \
  --repeat 3 \
  --warmup \
  --trace \
  --out-dir ./profiler-out
```

This is the canonical Romulus dual-GPU profile (7900 XTX local + 3060 Ti RPC).
Telemetry is returned in the `GRAPH_COMPUTE` response from the single-device
RPC server (`rpc-server -d CUDA0`). See [docs/D4.6-gpu-host-rpc-multidevice-test.md](../../docs/D4.6-gpu-host-rpc-multidevice-test.md).

### Multi-GPU RPC server (2+ GPUs on one endpoint)

```sh
GGML_RPC_SERVER_TELEMETRY=1 GGML_RPC_MULTIDEVICE=1 \
./build/bin/llama-gpipe-profiler \
  -m /models/Qwen3.6-35B-A3B.gguf \
  --rpc 192.168.1.10:50054 \
  --tensor-split 50,50 \
  --server-telemetry \
  --tasks pp,tg \
  --trace \
  --out-dir ./profiler-out
```

When the RPC server has 2+ GPUs (`rpc-server -d CUDA0,CUDA1`), telemetry is
returned in the `GRAPH_COMPUTE_ALL` response, which also enables server-side
multi-GPU scheduling.

### Multiple RPC endpoints (3+ GPUs total)

```sh
GGML_RPC_SERVER_TELEMETRY=1 \
./build/bin/llama-gpipe-profiler \
  -m /models/Qwen3.6-35B-A3B.gguf \
  --rpc gpu1:50051,gpu2:50052 \
  --tensor-split 33,33,34 \
  --server-telemetry \
  --tasks tg \
  --n-gen 256 \
  --repeat 5 \
  --out-dir ./profiler-out
```

### Generation-only profile (faster turnaround)

```sh
./build/bin/llama-gpipe-profiler \
  -m /models/Qwen3.6-35B-A3B.gguf \
  --tasks tg \
  --n-gen 128 \
  --repeat 10
```

### MTP speculative decoding profile (fused, e.g., Qwen NextN)

```sh
./build/bin/llama-gpipe-profiler \
  -m /models/Qwen3.6-35B-A3B-APEX-MTP.gguf \
  --tasks tg \
  --n-gen 128 \
  --repeat 5 \
  --spec-type draft-mtp \
  --spec-draft-n-max 8
```

### MTP with separate draft model (e.g., Gemma assistant)

```sh
./build/bin/llama-gpipe-profiler \
  -m /models/gemma-4-31B-it-Q4_K_M.gguf \
  --model-draft /models/gemma-4-31B-it-assistant-Q8_0.gguf \
  --tasks tg \
  --n-gen 128 \
  --repeat 5 \
  --spec-type draft-mtp \
  --spec-draft-n-max 8
```

**MTP TG mode:** When `--spec-type draft-mtp` is set, the profiler enables
`embeddings_nextn` on the target context and creates a draft context. It
measures per-token decode throughput **including** the nextn extraction cost
but **excluding** the speculative draft/accept loop (synthetic tokens can't
produce meaningful acceptance). This gives the MTP overhead — the pure cost
of having MTP enabled vs disabled.

For fused MTP (Qwen NextN), the draft context is created from the same model
(`llama_init_from_model` with `LLAMA_CONTEXT_TYPE_MTP`). For separate-draft
MTP (Gemma), pass `--model-draft` to load the assistant GGUF and the draft
context is created from that model instead.

## Output Artifacts

### `heatmap.json` (primary output)

Task-stratified heatmap JSON with per-layer timing, GPU metadata, and summary.
Schema version 1, additive-only evolution.

```json
{
  "schema_version": 1,
  "model": {
    "path": "/models/Qwen3.6-35B-A3B.gguf",
    "n_layers": 60,
    "param_count_b": 36.0
  },
  "tasks": {
    "pp": {
      "n_prompt_tokens": 512,
      "wall_ms": 1250.3,
      "tps": 409.6,
      "layers": [{ "idx": 0, "ms": 18.2, "gpu_id": 0 }],
      "kv": { "read_ms": 1.1, "write_ms": 0.8 }
    },
    "tg": {
      "n_gen_tokens": 128,
      "wall_ms": 3850.0,
      "tps": 33.2,
      "layers": [{ "idx": 0, "ms": 58.1, "gpu_id": 0 }],
      "kv": { "read_ms": 0.05, "write_ms": 0.03 }
    }
  },
  "gpu_metadata": [
    { "id": 0, "name": "NVIDIA RTX 3090",
      "backend": "CUDA", "vram_total_mib": 24576 }
  ],
  "summary": {
    "straggler_gpu": 1,
    "overlap_pct": 0.2,
    "gate_s5": "PASS",
    "gate_b6": "FAIL"
  }
}
```

Full schema: [docs/adr/0004b-profiler-architecture.md](../../docs/adr/0004b-profiler-architecture.md).

### Telemetry traces (`--out-dir/telemetry/`)

When `--trace` is on, raw per-event jsonl files land under the telemetry
subdirectory:

| File | Source | Content |
|------|--------|---------|
| `sched-trace.jsonl` | `GGML_SCHED_TRACE` | Per-split scheduler events |
| `rpc-trace.jsonl` | `GGML_RPC_TRACE` | Per-RPC-op timing and metadata |
| `pipeline-trace.jsonl` | `GGML_PIPELINE_TRACE` | Pipeline barrier events |
| `server-telemetry.jsonl` | RPC server | Per-device timing, layer assignments, device metadata |

## Server Telemetry

### How it works

1. Set `--server-telemetry` (or `GGML_RPC_SERVER_TELEMETRY=1`).
2. The RPC client includes a telemetry response buffer in every
   `GRAPH_COMPUTE` (single-device) and `GRAPH_COMPUTE_ALL` (multi-device)
   request.
3. The RPC server collects per-device timing and device metadata after each
   `ggml_backend_graph_compute()` call and returns it in the response.
4. The client writes the telemetry frame to `server-telemetry.jsonl` in the
   current working directory (same location as traces when `--trace` is on).

### 6-Field Telemetry Frame

| Field | Source | Description |
|-------|--------|-------------|
| `device_timings_us` | Scheduler | Per-device compute time (microseconds) |
| `layer_assignments` | Split output | Backend index per device |
| `copy_times_us` | PCIe copy | Peer-to-peer copy duration (per pair) |
| `device_meta` | Backend init | GPU name, VRAM, backend type, PCIe info |
| `kv_read_times_us` | KV cache read | Read timing per slot (Pareto optimizer input) |
| `kv_write_times_us` | KV cache write | Write timing per slot (Pareto optimizer input) |

### Graceful Degradation

When server telemetry is unavailable (pre-v4.4 RPC server, or
`GGML_RPC_SERVER_TELEMETRY` not set):

- The profiler uses only client-side traces for timing
- KV cache fields are omitted from the heatmap (`"estimated"` source)
- A stderr warning is emitted: `"server telemetry unavailable; heatmap kv fields omitted"`
- All other profiling functionality works normally

### Gating Condition

Telemetry works on **both** RPC compute paths:

| RPC server config | Compute path | Telemetry |
|-------------------|-------------|-----------|
| 1 GPU (`-d CUDA0`) | `GRAPH_COMPUTE` | Returns telemetry in response |
| 2+ GPUs (`-d CUDA0,CUDA1`) | `GRAPH_COMPUTE_ALL` | Returns telemetry in response |

A single-GPU RPC server (like Romulus: 3060 Ti with `-d CUDA0`) is fully
profiled — no multi-GPU RPC server required for basic profiler operation.

## Architecture

```
llama-gpipe-profiler  (exe)
  |
  +-- parse CLI args -> config
  +-- llama_backend_init()
  +-- load model (llama_model_load + llama_init)
  +-- register RPC endpoints (if --rpc)
  +-- setup trace env (if --trace)
  +-- for each task in config.tasks:
  |     repeat config.repeat times:
  |       if last repeat: enable trace capture
  |       run task (pp: batched decode / tg: serial decode)
  |       time wall clock
  +-- read server-telemetry.jsonl (if --server-telemetry)
  +-- synthesize heatmap:
  |     per-layer ms from server telemetry device_timings / layer_assignments
  |     pp/tg wall time from run loop
  |     gpu_metadata from server telemetry device_meta
  +-- write heatmap.json
  +-- llama_backend_free()
```

## Status

| Component | Status |
|-----------|--------|
| CLI surface | Complete |
| Local-only profiling | Complete |
| Single-device RPC telemetry | Complete |
| Multi-device RPC telemetry (`GRAPH_COMPUTE_ALL`) | Complete |
| Heatmap synthesis | Complete |
| Graceful degradation | Complete |
| KV cache timing (server slot-level) | Placeholder — zero-filled; waits on KV cache instrumentation |
| Pareto optimizer integration (D4.11-D4.14) | Planned — ticketed, not built |

## References

- [ADR-0004b: Profiler Binary Architecture](../../docs/adr/0004b-profiler-architecture.md)
- [D4.7: Profiler Research](../../docs/wayfinder/D4.7-profiler-research.md)
- [D4.6: Romulus RPC Multi-Device Test](../../docs/D4.6-gpu-host-rpc-multidevice-test.md)
- [RPC Server Telemetry in ggml-rpc.cpp](../../ggml/src/ggml-rpc/ggml-rpc.cpp) (`collect_telemetry`, `rpc_write_server_telemetry_jsonl`, server handler)
- [PIPELINE.md: Pipeline overlap documentation](../../PIPELINE.md)
- [docs/llama-pipeline-profiler/OVERVIEW.md: Existing profiler doc map](../../docs/llama-pipeline-profiler/OVERVIEW.md)
