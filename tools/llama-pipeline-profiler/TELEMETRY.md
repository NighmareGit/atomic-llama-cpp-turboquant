# Telemetry schemas

## sched-trace.jsonl

Per graph-split events from `GGML_SCHED_TRACE` ([`ggml-backend.cpp`](../../ggml/src/ggml-backend.cpp)):

```json
{"ts_us":123,"split":1,"backend":2,"copy":0,"phase":"split_total","elapsed_us":9623}
```

Phases: `input_wait_copy`, `graph_compute_async`, `event_record`, `split_total`.

When `GGML_PIPELINE_TRACE=1`, each sched line also includes `decode_id` (monotonic per `llama_decode`).

## pipeline-trace.jsonl

High-level pipeline events from `GGML_PIPELINE_TRACE` ([`ggml-backend.cpp`](../../ggml/src/ggml-backend.cpp)):

```json
{"ts_us":123,"event":"pipeline_barrier","decode_id":42,"copy_from":0,"copy_to":1,"n_copies":4,"elapsed_us":85}
```

Events: `pipeline_barrier`, `pipeline_barrier_sync` (fallback full sync path).

Set `GGML_PIPELINE_TRACE_FILE` to append; profiler and `BENCH_TRACE=1` benches default to `<telemetry>/pipeline-trace.jsonl`.

## rpc-trace.jsonl

Per RPC op from `GGML_RPC_TRACE` ([`ggml-rpc.cpp`](../../ggml/src/ggml-rpc/ggml-rpc.cpp)):

```json
{"ts_us":123,"fn":"send_rpc_cmd","phase":"send_recv","cmd":8,"bytes":16,"blocking":true,"elapsed_us":412}
```

## diagnose.json (v1)

Machine-readable output from `llama-pipeline-diagnose.sh`:

| Field | Type | Description |
|-------|------|-------------|
| `G_tps` | float | From `result.jsonl` |
| `stall_ratio` | float | `(input_wait + event_record) / split_total` |
| `straggler_backend` | string | Highest ms/tok backend id |
| `overlap_pct` | float | S5 assembly-line metric |
| `gate_s5` | string | PASS if `assembly_overlap_count > 0` |
| `gate_b6` | string | PASS if `overlap_pct >= overlap_target` |
| `overlap_efficiency` | float | serial_ms_per_tok / wall_ms_per_tok |
| `drain_flush_ms` | float | RPC drain preamble |
| `rpc_rtt_per_token` | float | blocking RPC ops / gen tokens |
| `graph_submit_count` | int | GRAPH_COMPUTE + GRAPH_RECOMPUTE |
| `topology_class` | string | `client_split` (Path C: `server_aggregated`) |
| `gpu_smell_flags` | array | See GATES.md |
| `path_c_reserved` | object | Placeholder for cross-endpoint copy counts |

## regression.jsonl

Append-only run history from `llama-pipeline-regression.sh` (profiler and bench stubs call this after diagnose):

```json
{"version":1,"ts_utc":"2026-06-28T12:00:00Z","label":"trace-f-2gpu-plus","git_sha":"1dfdf1b6f","client_kind":"native","G_tps":48.9,"gate_s5":"PASS","gate_b6":"FAIL","overlap_pct":0.6,"stall_ratio":0.68,"drain_flush_ms":1755.29}
```

Default path: `benches/path-b-plus/regression.jsonl`. Seed offline baselines with `scripts/llama-pipeline-import-baseline.sh`.

## trace sampling

`llama-pipeline-trace-sample.sh` writes `*.sample.jsonl` and `sample-meta.json` under telemetry. Profiler flag: `--trace-sample N` (default 10 in `--mode profile`).

## gpu/schema.json

Written by `gpu-telemetry-collect.sh`. NVIDIA CSV columns:

`timestamp_utc, power_w, util_gpu_pct, util_mem_pct, sm_clock_mhz, mem_clock_mhz, mem_used_mib, mem_total_mib, pcie_rx_mbs, pcie_tx_mbs`

ROCm JSONL: one JSON object per line with `ts` and `payload` from `rocm-smi --json`.