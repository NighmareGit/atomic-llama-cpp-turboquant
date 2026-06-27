# Windows 5070 Ti benchmark artifacts

Logs from `scripts/cuda-windows-5070ti/rpc-server-bench.ps1` and matrix scripts.
Each run directory contains `result.meta`, `bench.result`, `server.log`, `server.log.err`,
`vram-calc.txt`, `gpu-monitor.log`, and nvidia-smi snapshots.

## Phase 1 (single-node CUDA)

| Label | Model | Result |
|-------|-------|--------|
| `20260627-042947` | gemma-4-E4B.i1-Q4_K_M | PASS |

## Config E (5070 Ti + remus 5060 Ti, `ts=50,50`)

| Label | Model | G (t/s) | Notes |
|-------|-------|---------|-------|
| `config-e-smoke-9b-v2` | Qwen3.5-9B-MTP-Q4_K_M | ~65 | canonical 9B run |
| `config-e-smoke-9b` | Qwen3.5-9B-MTP-Q4_K_M | FAIL | load timeout (obsolete) |
| `config-e-27b` | Qwen3.6-27B-Q5_K_M | ~23.6 | ctx=8192 |

Matrix summary: `config-e-matrix-summary.txt`

## Config F (5070 Ti + remus 5060 Ti + RX6600, `ts=30,12,58`)

| Label | Model | G (t/s) | ngl | ncmoe | Notes |
|-------|-------|---------|-----|-------|-------|
| `config-f-smoke-9b` | Qwen3.5-9B-MTP-Q4_K_M | ~41 | 99 | - | |
| `config-f-27b` | Qwen3.6-27B-Q5_K_M | ~17.5 | 99 | - | ctx=8192 |
| `config-f-31b` | Gemma-31B-it-quant | ~20.5 | 99 | - | full GPU |
| `config-f-35b-gpu` | Qwen3.5-35B-A3B-UD-Q4_K_XL | **~34.5** | 99 | - | preferred |
| `config-f-35b` | Qwen3.5-35B-A3B-UD-Q4_K_XL | ~15.7 | 99 | 8 | expert CPU offload |
| `config-f-36b-gpu` | Qwen3.6-35B-A3B-UD-IQ4_XS | **~33.6** | 99 | - | preferred |
| `config-f-36b` | Qwen3.6-35B-A3B-UD-IQ4_XS | ~14.7 | 99 | 8 | expert CPU offload |
| `config-f-36b-nl-gpu` | Qwen3.6-35B-A3B-UD-IQ4_NL_XL | **~40.3** | 99 | - | preferred |
| `config-f-36b-nl` | Qwen3.6-35B-A3B-UD-IQ4_NL_XL | ~13.8 | 99 | 8 | expert CPU offload |
| `config-f-48b-gpu` | qwen3-coder-next-reap-48b | **~11.3** | 33 | - | 33/49 layers GPU |
| `config-f-48b` | qwen3-coder-next-reap-48b | ~11 | 33 | 8 | same split |
| `config-f-80b` | Qwen3-Next-80B-A3B-Instruct-IQ4_NL | **~4.8** | 28 | 8 | 28/49 layers GPU |

### Profile runs (GPU underutilization investigation)

| Run | G (t/s) | Notes |
|-----|---------|-------|
| `profile-f-36b-nl-base` | 37.0 | 3-device F, ts=30,12,58 |
| `profile-e-36b-nl` | 42.4 | Config E, no 6600 |
| `profile-f-36b-nl-no6600` | **48.9** | 2-device, ts=50,50 - fastest |

See [PROFILING.md](../PROFILING.md) for bottleneck classification.

Matrix summary: `config-f-matrix-summary.txt`

## Path-B Plus trace matrix (2026-06-27)

Production default for 36B NL MoE: **2-device F** (`trace-f-2gpu-plus`).

| Label | Topology | Plus | G (t/s) | Notes |
|-------|----------|------|---------|-------|
| `trace-f-2gpu-plus` | 5070 + 5060 `ts=50,50` | 1 | **48.9** | **recommended** |
| `trace-f-3gpu-plus` | 5070 + 5060 + 6600 `ts=30,12,58` | 1 | 42.8 | topology-limited |
| `trace-f-3gpu-legacy` | 3-device | 0 | 39.3 | Plus=0 baseline |
| `s4-4b-2gpu-plus` | 2-device, gemma-4-E4B | 1 | 44.2 | S4 correctness smoke |

Run: `pathb-trace-runbook.ps1 -Runs trace-f-2gpu-plus`. Tracking: [rpc-patch/docs/rpc-path-b-plus-tracking.md](../../../rpc-patch/docs/rpc-path-b-plus-tracking.md).

## Config G primary: 4-GPU cluster (romulus client, 2026-06-27)

Stable topology: 7900 + 3060 + 5060 + 5070 (no RX6600). Artifacts on romulus host under `rpc-patch/patch/bench-results/rpc-server-bench/`.

| Label | G (t/s) | Load | Result |
|-------|---------|------|--------|
| `trace-g-4gpu-primary` | 43.0 | 85s | PASS |
| `trace-g-4gpu-primary-r2` | 38.2 | 85s | PASS |
| `trace-g-4gpu-primary-r3` | 43.1 | 85s | PASS |
| `trace-g-4gpu-primary-trace` | ~37 | 85s | trace captured |

Summary: [rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md](../../../rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md).  
Doc: [../CLUSTER-4GPU-PRIMARY.md](../CLUSTER-4GPU-PRIMARY.md).

### Legacy / failed 4-GPU (RX6600 -- do not use)

| Label | Result |
|-------|--------|
| `trace-g-4gpu-plus` (Windows client) | HANG at slot init |
| `trace-g-4gpu-smoke` | smoke only |

Details and offload guidance: [../MULTI-NODE.md](../MULTI-NODE.md)