# Telemetry

Per-node GPU timing for hot-path profiling. Replaces scattered environment variables
with a unified `--telemetry` CLI group.

**Design:** [DESIGN-per-node-telemetry.md](DESIGN-per-node-telemetry.md)

---

## CLI flags

All components (`llama-server`, `rpc-server`, `llama-gpipe-profiler`) accept these flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--telemetry` | bool | off | Enable telemetry collection (server + client) |
| `--telemetry-trace` | bool | off | Enable sched/rpc/pipeline trace output files |
| `--telemetry-sample-interval N` | int | 1 | Sample per-node timing every Nth decode step |
| `--telemetry-aggregate` | bool | off | Server-side timing aggregation (running avg/min/max) |
| `--telemetry-per-node-timing` | bool | on | Enable per-graph-node GPU timing |
| `--telemetry-file PATH` | string | (auto) | Telemetry output file path |
| `--telemetry-no-env-fallback` | bool | off | Suppress environment variable fallback |

### Quick examples

```sh
# llama-server with per-node timing and traces
llama-server -m model.gguf --rpc 192.168.1.100:50052 \
  --telemetry --telemetry-trace --telemetry-file /tmp/tel.jsonl

# rpc-server with telemetry, sampling every 4th decode
rpc-server -d CUDA0 -T \
  --telemetry-sample-interval 4 --telemetry-file /var/log/rpc-tel.jsonl

# Profiler: trace + telemetry output to custom directory
llama-gpipe-profiler -m model.gguf --rpc 192.168.1.100:50052 \
  --trace --server-telemetry --out-dir ./profiler-out \
  --telemetry-per-node-timing --telemetry-sample-interval 2
```

### Per-component notes

**`llama-server`**: Sets env vars before `llama_backend_init()` so RPC connection negotiation
picks up telemetry settings. Trace files land in the directory specified by `LLAMA_TRACE_DIR`.

**`rpc-server`**: Uses its own `rpc_server_params` struct (not `common_params`). The `-T` flag
is an alias for `--telemetry`.

**`llama-gpipe-profiler`**: Uses its own `profile_config` struct. The `--server-telemetry` flag
enables RPC server telemetry request; `--trace` enables trace file output. 
Per-node timing flags (`--telemetry-per-node-timing`, `--telemetry-sample-interval`) control
collection granularity.

---

## JSONL formats

### sched-trace.jsonl: `node_timings`

Per-node timing on each graph split, written when `--telemetry-trace` is active
or `GGML_SCHED_TRACE=1`:

```json
{"event":"graph_compute_async","split":2,"backend":2,
 "node_timings":[
   {"name":"l_out-14.attn_q","us":45},
   {"name":"l_out-14.attn_k","us":38},
   {"name":"l_out-14.attn_v","us":41},
   {"name":"l_out-14.attn_o","us":52},
   {"name":"l_out-14.ffn_gate","us":89},
   {"name":"l_out-14.ffn_up","us":124},
   {"name":"l_out-14.ffn_down","us":117}
 ],
 "elapsed_us":1250}
```

Tensor names follow the GGML naming convention: `l_out-N.op_type` for decoder layers,
`blk.N.op_type` for encoder blocks.

### server-telemetry.jsonl: `node_timings`

Per-node timing collected on the RPC server side, appended to `server-telemetry.jsonl`:

```json
{"event":"node_timings","ts_us":1712345678901234,"trace_id":1,"entries":[
  {"name":"l_out-0.attn_q","us":89},
  {"name":"l_out-0.attn_k","us":72},
  {"name":"l_out-0.attn_v","us":68},
  {"name":"l_out-0.ffn_gate","us":142},
  {"name":"l_out-0.ffn_up","us":201},
  {"name":"l_out-0.ffn_down","us":198}
]}
```

Entries are merged (running average) across multiple decode steps by the profiler.

#### Aggregated mode (`--telemetry-aggregate`)

When server-side aggregation is enabled, the server accumulates per-node stats across
decode steps and emits running aggregates instead of raw per-decode entries:

```json
{"event":"node_timings","ts_us":1712345678901234,"trace_id":1,"aggregated":true,"entries":[
  {"name":"l_out-0.attn_q","us":91,"avg_us":91,"min_us":87,"max_us":95,"count":16},
  {"name":"l_out-0.attn_k","us":73,"avg_us":73,"min_us":69,"max_us":77,"count":16},
  {"name":"l_out-0.ffn_gate","us":140,"avg_us":140,"min_us":135,"max_us":147,"count":16}
]}
```

- `us` / `avg_us`: running average (total_us / count)
- `min_us` / `max_us`: min/max observed across the sample period
- `count`: number of samples included in the aggregate
- `aggregated: true`: marker so parsers can distinguish aggregated vs raw format
- Aggregates reset after each emission (one sample period)

This reduces JSONL volume from O(N * decodes) to O(N * sample_periods).

### server-telemetry.jsonl: standard frame

Standard per-device telemetry frame (always emitted with `--telemetry`):

```json
{"event":"server_telemetry","ts_us":26037472484,"trace_id":1,
 "device_timings_us":[1082],
 "layer_assignments":[0],
 "copy_times_us":[],
 "device_meta":[{"name":"CUDA0","vram_mib":7841,"backend":"CUDA"}],
 "kv_read_times_us":[],"kv_write_times_us":[]}
```

---

## Architecture

```
  llama-server ──RPC──► rpc-server
      │                      │
      │  --telemetry-trace   │  --telemetry (-T)
      ▼                      ▼
  sched-trace.jsonl    server-telemetry.jsonl
  (per-node, per-split) (per-node + per-device)
      │                      │
      └──────────┬───────────┘
                 ▼
        llama-gpipe-profiler
        (heatmap.json: layers[] + layer_rollup[])
```

### Timing mechanism

| Backend | Method | Overhead |
|---------|--------|----------|
| CUDA / ROCm | `cudaEventRecord` pairs around `compute_forward` | < 2 us/node |
| CPU | `clock_gettime(CLOCK_MONOTONIC)` inline | negligible |
| RPC | Server-side CUDA/ROCm events, collected via callback | same as local |

CUDA graph replay mode skips per-node timing because a single `cudaGraphLaunch`
cannot be instrumented at the node level. Timing is active during graph capture
and non-graph mode.

### Sampling

`--telemetry-sample-interval N` controls how often per-node data is collected.
Set `N > 1` to reduce profiling overhead during production runs. The interval
applies to all backends uniformly:

- **RPC server**: per-node callback is skipped on non-sampled decodes (no event overhead)
- **Local sched-trace**: per-node callback and JSONL emit are gated by the same counter
- **Overall telemetry frame**: `collect_telemetry()` also respects the interval

---

## Environment variable fallback

The following env vars are still functional but emit a deprecation warning when
`--telemetry-no-env-fallback` is not set:

| Deprecated env var | New CLI flag |
|---------------------|--------------|
| `GGML_RPC_SERVER_TELEMETRY=1` | `--telemetry` |
| `GGML_RPC_SERVER_TELEMETRY_FILE=...` | `--telemetry-file` |
| `GGML_SCHED_TRACE=1` | `--telemetry-trace` |
| `GGML_RPC_TRACE=1` | `--telemetry-trace` |
| `GGML_PIPELINE_TRACE=1` | `--telemetry-trace` |

To suppress env var fallback: `--telemetry-no-env-fallback`.

---

## Heatmap output

The profiler produces `heatmap.json` with these new sections when per-node timing
is available:

```json
{
  "layers": [
    {"name": "l_out-0.attn_q", "us": 89.0, "ms": 0.089, "samples": 128},
    {"name": "l_out-0.attn_k", "us": 72.0, "ms": 0.072, "samples": 128}
  ],
  "layer_rollup": [
    {"idx": 0, "ms": 0.771, "us": 771, "n_nodes": 7,
     "nodes": ["l_out-0.attn_q", "l_out-0.attn_k", "l_out-0.attn_v",
               "l_out-0.attn_o", "l_out-0.ffn_gate", "l_out-0.ffn_up",
               "l_out-0.ffn_down"]}
  ],
  "op_categories": [
    {"category": "attn", "ms": 0.312, "us": 312, "n_nodes": 512,
     "ops": ["attn_q", "attn_k", "attn_v", "attn_o", "attn_rope"]},
    {"category": "ffn",  "ms": 0.459, "us": 459, "n_nodes": 384,
     "ops": ["ffn_gate", "ffn_up", "ffn_down"]},
    {"category": "norm", "ms": 0.055, "us": 55,  "n_nodes": 256,
     "ops": ["attn_norm", "ffn_norm"]},
    {"category": "other","ms": 0.012, "us": 12,  "n_nodes": 16,
     "ops": ["inp_embd", "output_norm", "output"]}
  ]
}
```

- `layers[]`: One entry per unique tensor name, with running-average timing
- `layer_rollup[]`: Per-transformer-layer aggregate (sum of all nodes in that layer)
- `op_categories[]`: Cross-layer aggregation by operation category (attn/ffn/norm/other).
  Category is determined by the op suffix: `attn_*` → attn, `ffn_*` → ffn,
  `*norm*` → norm, everything else → other. Useful for answering "how much time
  is spent in attention vs. feed-forward across the entire model?"

Layer index is extracted from `l_out-N.*` or `blk.N.*` tensor name patterns.

---

## Related docs

- [CURRENT-STATE-pipeline-flow.md](CURRENT-STATE-pipeline-flow.md) -- pipeline trace formats
- [RPC-PROTOCOL.md](RPC-PROTOCOL.md) -- RPC wire protocol and capabilities
- [DESIGN-per-node-telemetry.md](DESIGN-per-node-telemetry.md) -- design decisions and phases
- `tools/llama-gpipe-profiler/README.md` -- profiler usage
- `tools/llama-pipeline-profiler/TELEMETRY.md` -- legacy telemetry schemas (v1)
