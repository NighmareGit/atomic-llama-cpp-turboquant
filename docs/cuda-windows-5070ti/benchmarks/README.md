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

Matrix summary: `config-f-matrix-summary.txt`

Details and offload guidance: [../MULTI-NODE.md](../MULTI-NODE.md)