# Cluster 4-GPU primary bench summary

**Client:** romulus RX 7900 XTX (native ROCm `build-rocm-docker/bin/llama-server`, build `833ad4429`)  
**Workers:** remus 5060 `:50051`, romulus 3060 `127.0.0.1:50051`, Windows 5070 `192.168.8.21:50053`  
**Model:** `Qwen3.6-35B-A3B-APEX-I-Quality.gguf`  
**Flags:** `q8_0` KV, `GGML_PIPELINE_PLUS=1`, `ncmoe=none`, `ts=36,24,24,16`

## Gate runs (2026-06-27)

| Label | RESULT | Load | G (t/s) | Notes |
|-------|--------|------|---------|-------|
| trace-g-4gpu-primary | PASS | 85s | 43.0 | first stable 4-GPU |
| trace-g-4gpu-primary-r2 | PASS | 85s | 38.2 | gate 2/3 |
| trace-g-4gpu-primary-r3 | PASS | 85s | 43.1 | gate 3/3 |
| trace-g-4gpu-primary-trace | FAIL* | 85s | ~37 | *run-3 curl parser flake; telemetry valid |

Full logs: `../rpc-server-bench/trace-g-4gpu-primary*/` on romulus host.

## Model buffer split (load_tensors)

| Device | MiB |
|--------|-----|
| ROCm0 7900 | 3696 |
| RPC 3060 | 4381 |
| RPC 5060 | 8289 |
| RPC 5070 | 4986 |
| CPU mapped | 398 |

Load fan-out completes in ~64-85s. `sched_reserve`: `graph_splits=5`, `sched copies=4`.

## Trace hotpath (trace-g-4gpu-primary-trace)

Parsed via `pathb-hotpath-summary.sh` on romulus:

- `split_total_ms_sum` / 332 tok = **20.71 ms/tok**
- backend1 (5060): 9.63 ms/tok
- backend2 (3060): 5.51 ms/tok
- backend3 (5070): 5.41 ms/tok
- `SET_TENSOR_HASH`: 6478 ms / 280 calls (load phase)
- `assembly_overlap_count`: 1075

## Contrast: legacy 6600 4-GPU (DO NOT USE)

| Label | Stall point | Result |
|-------|-------------|--------|
| trace-g-4gpu-romulus-q8-pp1 | `initializing slots` after ~64s load | HANG (aborted) |
| trace-g-4gpu-full-trace (6600) | same | HANG |
| trace-g-4gpu-plus (Windows client + 6600) | `initializing slots` ~45min | RPC recv failed |

Replacing `:50052` (RX6600) with `:50053` (5070) fixes slot init.

## 3-GPU primary (Tier 0 control)

`pathb-romulus-3gpu-bench.sh primary-5060-3060` (7900 + 5060 + 3060):

| Label | RESULT | G (t/s) |
|-------|--------|---------|
| trace-g-3gpu-primary-r1 | PASS | ~29 |
| trace-g-3gpu-primary-r2 | FAIL* | ~41 | *curl run 3 |
| trace-g-3gpu-primary-r3 | PASS | ~35 |

## Retry (2026-07-02, SHA 9121d16d4, ts=25,12,25,38)

| Label | RESULT | Load | G (t/s) | Notes |
|-------|--------|------|---------|-------|
| trace-g-2gpu-retry-M35 | PASS | 56s | 46.4 avg | 2G baseline |
| trace-g-3gpu-3060-retry-M35 | PASS | 50s | 42.8 avg | 3060 third GPU |
| trace-g-3gpu-5070-retry-M35 | PASS | 80s | 44.7 avg | 5070 third GPU |
| trace-g-4gpu-primary-retry-M35 | PASS | 80s | 44.9 avg | **-3.2% vs 2G** |

4G vs 2G server regression **PASS** (<10%). June ~43 t/s baseline recovered. See [CLUSTER-4GPU-PRIMARY.md](../../../../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md#config-g-retry-2026-07-02-sha-9121d16d4).

## Romulus host snapshot (in git)

Meta/result files pulled to `romulus-host/` before cluster shutdown. Full server logs and jsonl telemetry remain on romulus at `rpc-patch/patch/bench-results/rpc-server-bench/`. Pull script: `rpc-patch/scripts/pull-romulus-bench-snapshot.sh`.

## Launchers

```bash
./rpc-patch/scripts/pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary
./rpc-patch/scripts/pathb-romulus-3gpu-bench.sh primary-5060-3060 trace-g-3gpu-primary-r1
BENCH_4GPU_PRESET=legacy-6600 ./rpc-patch/scripts/pathb-romulus-4gpu-bench.sh trace-g-4gpu-legacy-6600
```