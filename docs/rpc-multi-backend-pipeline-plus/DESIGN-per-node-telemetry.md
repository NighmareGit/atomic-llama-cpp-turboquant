# DESIGN: Per-node telemetry for hot-path profiling

**Date:** 2026-07-14
**Status:** Design complete — ready for implementation
**Parent:** [CURRENT-STATE-pipeline-flow.md](CURRENT-STATE-pipeline-flow.md) (current telemetry is per-device, not per-layer)

---

## 1. Motivation

Current RPC server telemetry captures per-device timing (one entry per `GRAPH_COMPUTE` call). The profiler heatmap `layers: []` is empty because there is no per-layer or per-node data. Operators cannot answer: "which transformer layer is the bottleneck?" or "should I move specific ops to the local GPU?"

Goal: **Per-graph-node timing** for every backend (RPC, local CUDA/ROCm, standalone) so the profiler can produce per-layer/per-op hot-path heatmaps. This enables informed split-mode and tensor-split placement decisions.

## 2. Decisions (grill-gate, 2026-07-14)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Per-node**, not per-layer | Future row/tensor split needs finer data; per-node survives split-mode changes |
| 2 | **Profiler collection only** (Phase 1); runtime-adaptive placement deferred | Scope control; prove data value before invasive scheduler changes |
| 3 | **Both RPC + local backends** | Romulus-local needs both 7900 XTX (local) and 3060 Ti (RPC) coverage |
| 4 | **Backend callback** (`ggml_backend_node_timing_cb`) | Only approach that works across CUDA/ROCm/CPU/RPC without backend-specific hacks |
| 5 | **GPU event-based** (CUDA events, non-blocking) | True GPU time without forcing synchronous execution |
| 6 | **Inline JSONL** in existing trace files | Self-describing, no protocol bump, profiler already parses these files |
| 7 | **Embed tensor names** (e.g. `l_out-17`) | Human-readable, no mapping table needed, debug-friendly |
| 8 | **`--telemetry` CLI group** for all flags | Replace hidden env vars; `--help` discoverable; env vars as fallback (hybrid deprecation) |
| 9 | **Sampling + optional aggregation** | `--telemetry-sample-interval N` (same knob); `--telemetry-aggregate` (server-side accumulation) |
| 10 | **Breadth-first** (Phase 1+2a+2b together) | Design callback interface once, wire all backends before profiler consumption |

## 3. Architecture

### 3.1 Backend timing callback

New optional callback in the backend interface:

```cpp
// ggml-backend.h
typedef void (*ggml_backend_node_timing_cb)(
    int backend_id,
    const char * node_name,
    uint64_t elapsed_us,   // GPU time between event-record pairs
    void * user_data);

// Set per-node timing callback (NULL = disabled)
void ggml_backend_set_node_timing_callback(
    ggml_backend_t backend,
    ggml_backend_node_timing_cb callback,
    void * user_data);
```

**CUDA backend** (and ROCm by analogy):
- Insert `cudaEventRecord(start)` before each node launch, `cudaEventRecord(stop)` after
- After `cudaStreamSynchronize` (or at graph end), read back `cudaEventElapsedTime` for each node
- Fire callback with `node_name` and `elapsed_us`

**CPU backend**: Nodes run synchronously — wrap each node with `clock_gettime`, fire callback inline.

**RPC backend**: On the **server side**, the rpc-server uses its own local CUDA/ROCm/CPU backend with the same callback mechanism. The server-side `graph_compute()` path already has the backend — the callback fires naturally.

### 3.2 Data transport

#### RPC path: server → client

Per-node timing rides in the existing `server-telemetry.jsonl`:

```json
{"event":"server_telemetry","ts_us":26037472484,"trace_id":1,
 "device_timings_us":[1082],
 "node_timings":[
   {"name":"l_out-0.attn_q","us":89},
   {"name":"l_out-0.attn_k","us":72},
   {"name":"l_out-0.attn_v","us":68},
   {"name":"l_out-0.ffn_gate","us":142},
   {"name":"l_out-0.ffn_up","us":201},
   {"name":"l_out-0.ffn_down","us":198}
 ],
 "device_meta":[{"name":"CUDA0","vram_mib":7841,"backend":"CUDA"}],
 "kv_read_times_us":[],"kv_write_times_us":[]}
```

The rpc-server's `collect_telemetry()` function buffers per-node data from the callback and emits it in the telemetry message.

#### Local path: client-side

Per-node timing for local backends (7900 XTX on romulus, or standalone single-GPU) is written to `sched-trace.jsonl` as an additional `node_timings` array on each split's trace entry:

```json
{"event":"graph_compute_async","split":2,"backend":2,
 "node_timings":[
   {"name":"l_out-14.attn_q","us":45},
   {"name":"l_out-14.attn_k","us":38}
 ],
 "elapsed_us":1250}
```

### 3.3 Profiler consumption

`tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp`:
- `parse_server_telemetry()` extracts `node_timings` from server-telemetry.jsonl
- `parse_sched_trace_for_client_timing()` extracts `node_timings` from sched-trace.jsonl
- Heatmap `layers` array is populated with per-node entries grouped by layer index (extracted from `l_out-N` prefix)
- Per-op-category aggregation: attn/ffn/norm/other, keyed by op name suffix
- MoE expert identification: `MUL_MAT_ID` nodes tagged with expert routing info

### 3.4 Sampling and aggregation

**Sampling** (Option B): `--telemetry-sample-interval N` — collect per-node timing only every Nth decode. Default 1 (every decode). Uses existing `TELEMETRY_SAMPLE_INTERVAL` mechanism extended to per-node data.

**Aggregation** (Option C): `--telemetry-aggregate` — server accumulates per-node timing (running average, count, min, max) across decodes. Each telemetry message carries the running aggregate, not raw per-decode data. Reduces data volume and post-processing cost. Mutually beneficial with sampling (aggregate every sample, report sampled aggregate).

## 4. CLI interface (`--telemetry` group)

All telemetry flags move from env vars to CLI args. Env vars remain as fallback with deprecation warning.

```
Telemetry options:
  --telemetry                    Enable telemetry collection (server + client)
  --telemetry-trace              Enable sched/rpc/pipeline trace files
  --telemetry-sample-interval N  Sample every Nth decode (default: 1)
  --telemetry-aggregate          Server-side timing aggregation (default: off)
  --telemetry-per-node-timing    Enable per-graph-node timing (default: on with --telemetry)
  --telemetry-file PATH          Telemetry output file (default: server-telemetry.jsonl)
  --telemetry-no-env-fallback    Disable env var fallback for telemetry flags

Deprecated env vars (still work, emit warning):
  GGML_RPC_SERVER_TELEMETRY=1          → use --telemetry
  GGML_RPC_SERVER_TELEMETRY_FILE=...   → use --telemetry-file
  GGML_SCHED_TRACE=1                   → use --telemetry-trace
  GGML_RPC_TRACE=1                     → use --telemetry-trace
  GGML_PIPELINE_TRACE=1                → use --telemetry-trace
```

Applications: `llama-server`, `rpc-server`, `llama-gpipe-profiler`.

## 5. Implementation phases

### Phase 1: Backend callback interface

| Step | File(s) | Work |
|------|---------|------|
| 1.1 | `ggml/include/ggml-backend.h` | Add `ggml_backend_node_timing_cb` typedef, `ggml_backend_set_node_timing_callback()` |
| 1.2 | `ggml/src/ggml-backend.cpp` | Store callback + user_data in `ggml_backend` struct |
| 1.3 | `ggml/src/ggml-cuda/ggml-cuda.cpp` | Insert `cudaEventRecord` pairs per node in graph_compute; fire callback after sync |
| 1.4 | `ggml/src/ggml-rocm/ggml-rocm.cpp` | Same as CUDA, HIP events |
| 1.5 | `ggml/src/ggml-cpu/ggml-cpu.cpp` | Wrap each node with `clock_gettime`, fire callback inline |

### Phase 2a: Local per-node timing → sched-trace

| Step | File(s) | Work |
|------|---------|------|
| 2a.1 | `ggml/src/ggml-backend.cpp` | In `ggml_backend_sched_compute_splits()`, register timing callback on each split's backend |
| 2a.2 | `ggml/src/ggml-backend.cpp` | Callback writes per-node entries into split's trace data, emitted in `sched_trace_emit_node_timings()` |
| 2a.3 | `ggml/src/ggml-backend.cpp` | Add `node_timings` array to sched-trace.jsonl format |

### Phase 2b: RPC per-node timing → server-telemetry

| Step | File(s) | Work |
|------|---------|------|
| 2b.1 | `ggml/src/ggml-rpc/ggml-rpc.cpp` | In rpc-server's `graph_compute()`, register timing callback on server-side backend |
| 2b.2 | `ggml/src/ggml-rpc/ggml-rpc.cpp` | Buffer per-node data in `collect_telemetry()`, emit in `rpc_msg_server_telemetry` |
| 2b.3 | `ggml/src/ggml-rpc/ggml-rpc.cpp` | Add `node_timings` array to `rpc_write_server_telemetry_jsonl()` |

### Phase 3: Profiler consumption

| Step | File(s) | Work |
|------|---------|------|
| 3.1 | `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp` | `parse_server_telemetry()` → extract `node_timings` |
| 3.2 | `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp` | `parse_sched_trace_for_client_timing()` → extract local `node_timings` |
| 3.3 | `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp` | Populate heatmap `layers[]` from merged per-node data |
| 3.4 | `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp` | Per-op-category aggregation (attn/ffn/norm) |

### Phase 4: CLI args + docs

| Step | File(s) | Work |
|------|---------|------|
| 4.1 | `common/common.cpp` | Add `--telemetry*` args to common params |
| 4.2 | `tools/server/server.cpp` | Wire `--telemetry*` flags |
| 4.3 | `tools/rpc/rpc-server.cpp` | Wire `--telemetry*` flags |
| 4.4 | `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp` | Convert env var flags to `--telemetry*` CLI args |
| 4.5 | `docs/rpc-multi-backend-pipeline-plus/TELEMETRY.md` | New doc: all flags, formats, examples |
| 4.6 | Existing docs | Update references from env vars to CLI flags |

### Phase 5: Sampling + aggregation

| Step | File(s) | Work |
|------|---------|------|
| 5.1 | `ggml/src/ggml-rpc/ggml-rpc.cpp` | Wire `--telemetry-sample-interval` to per-node collection |
| 5.2 | `ggml/src/ggml-rpc/ggml-rpc.cpp` | Server-side aggregation: running avg/min/max per node name |
| 5.3 | `ggml/src/ggml-backend.cpp` | Wire sampling interval for local backend callback |

## 6. Acceptance gates

| Gate | Metric | Baseline | Target |
|------|--------|----------|--------|
| G0 | Profiler heatmap `layers[]` populated | Empty (`[]`) | One entry per unique node name |
| G1 | Per-node timing on RPC backend | `device_timings_us: [N]` only | `node_timings` array with per-node `us` |
| G2 | Per-node timing on local ROCm backend | sched-trace split-level only | sched-trace with `node_timings` |
| G3 | Standalone single-GPU | No per-node data | sched-trace with `node_timings` |
| G4 | `--telemetry` in `--help` | Hidden env vars only | Published CLI group with descriptions |
| G5 | Throughput regression (timing OFF) | Baseline G | No regression (< 1%) |
| G6 | Throughput with timing ON (sampled) | N/A | Acceptable for profiling runs |

## 7. References

- Current telemetry: [CURRENT-STATE-pipeline-flow.md](CURRENT-STATE-pipeline-flow.md) §5
- RPC protocol: [RPC-PROTOCOL.md](RPC-PROTOCOL.md)
- Profiler: `tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp`
- Backend interface: `ggml/include/ggml-backend.h`
