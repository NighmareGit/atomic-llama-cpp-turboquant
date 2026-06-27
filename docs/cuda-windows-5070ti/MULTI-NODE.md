# Multi-node Path B: Windows 5070 Ti + remus RPC

**Config E:** native `llama-server.exe` (RTX 5070 Ti) + remus RTX 5060 Ti RPC (`:50051`)

**Config F:** Config E + remus RX 6600 RPC (`:50052`) -- **gen straggler** on 3-GPU; **slot-init hang** on 4-GPU with 6600.

**Config G primary (4-GPU stable):** romulus 7900 client + 3060 + 5060 + Windows 5070 (`:50053`). **No RX6600.** G ~40 t/s. See [CLUSTER-4GPU-PRIMARY.md](CLUSTER-4GPU-PRIMARY.md).

**Path-B Plus production default (Windows client, 36B NL MoE):** Config F as **2-device** -- single `:50051`, `-ts 50,50`. G=**48.9 t/s** (`trace-f-2gpu-plus`).

**Path-B Plus 4-GPU (romulus client):** Config G primary -- `pathb-romulus-4gpu-bench.sh`. Windows must run `pathb-rpc-server.ps1` on `:50053`.

## Topology

| Device | Role | VRAM |
|--------|------|------|
| Windows CUDA0 | llama-server client | RTX 5070 Ti 16 GB |
| RPC0 | remus `pathb-rpc-remus` | RTX 5060 Ti 16 GB |

```
--rpc 192.168.8.176:50051
  RPC0 = remus 5060 Ti    CUDA0 = Windows 5070 Ti
  -ts 50,50               --fit-target 1024,1024
```

Models on `D:\models` only. RPC worker does not mount GGUFs.

## Prerequisites

- Portable build: `scripts/cuda-windows-5070ti/build.ps1`
- WSL with `sshpass`, `python3`, `nc`
- remus: `~/docker/Atomic-Llama-Remus-PathB/` image `atomic-llama-remus-pathb-rpc:latest`
- **Port 50051:** stop `rx6600-rpc` before PathB (both default to 50051)

```powershell
.\scripts\cuda-windows-5070ti\pathb-remus-rpc.ps1 start   # via WSL + sshpass
# or manually on remus: docker stop rx6600-rpc; docker compose up -d
```

## Scripts

| Script | Purpose |
|--------|---------|
| `invoke-wsl.ps1` | WSL bridge (`D:\` -> `/mnt/d/`, SSH env) |
| `pathb-remus-rpc.ps1` | remus 5060 Ti RPC lifecycle (Config E) |
| `pathb-remus-multi-rpc.ps1` | remus 5060 Ti + RX6600 RPC (Config F) |
| `pathb-vram-calc.ps1` | VRAM/ts planner (`config-e` / `config-f`) |
| `rpc-server-bench.ps1` | Single hybrid bench run |
| `pathb-config-e-matrix.ps1` | Config E preset loop (9b / 27b / 31b) |
| `pathb-config-f-matrix.ps1` | Config F preset loop (9b through 80b) |
| `pathb-rpc-server.ps1` | Windows 5070 as RPC worker `:50053` (Config G) |
| `pathb-romulus-4gpu-bench.sh` | romulus 4-GPU primary launcher (WSL + sshpass) |
| `pathb-romulus-3gpu-bench.sh` | romulus 3-GPU subset launcher |

Benchmark index: [benchmarks/README.md](benchmarks/README.md)

## Quick bench

```powershell
.\scripts\cuda-windows-5070ti\rpc-server-bench.ps1 `
  -Label config-e-smoke-9b `
  -Config config-e `
  -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf" `
  -TensorSplit "50,50" -Ctx 4096
```

## Results (2026-06-27)

### Config E smoke 9B (`ts=50,50`, ctx=4096, q8_0/turbo3)

| Metric | Value |
|--------|-------|
| Load | ~15 s |
| VRAM RPC0 / CUDA0 weights | 2163 / 2878 MiB |
| G (t/s) avg | ~65 (runs 62-67) |
| Log dir | `benchmarks/config-e-smoke-9b-v2/` |

Compare Config B (Romulus 7900+remus): remus-smoke-9b **G=88.3** with `ts=15,85`.

### Config E 27B (`Qwen3.6-27B-Q5_K_M`, ctx=8192)

| Metric | Value |
|--------|-------|
| Load | ~70 s |
| VRAM RPC0 / CUDA0 weights | 8641 / 9121 MiB |
| G (t/s) avg | ~23.6 |
| Log dir | `benchmarks/config-e-27b/` |

Compare Config B 27B: **G=30.9** (`ts=15,85`, 7900+5060).

## Config F topology

| Device | Role | VRAM |
|--------|------|------|
| RPC0 | remus `pathb-rpc-remus` | RTX 5060 Ti 16 GB |
| RPC1 | remus `rx6600-rpc` | RX 6600 8 GB |
| CUDA0 | llama-server client | RTX 5070 Ti 16 GB |

```
--rpc 192.168.8.176:50051,192.168.8.176:50052
  RPC0 = remus 5060 Ti    RPC1 = remus RX6600    CUDA0 = Windows 5070 Ti
  -ts 30,12,58             --fit-target 1024,1024,1024
```

Both RPC workers run simultaneously. `pathb-rpc-remus` and `rx6600-rpc` both default to port 50051 if started alone; Config F uses **50051 (5060) + 50052 (6600)**.

```powershell
.\scripts\cuda-windows-5070ti\pathb-remus-multi-rpc.ps1 start
.\scripts\cuda-windows-5070ti\rpc-server-bench.ps1 `
  -Label config-f-smoke-9b `
  -Config config-f `
  -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf" `
  -TensorSplit "30,12,58" -Ctx 4096
```

Matrix loop:

```powershell
.\scripts\cuda-windows-5070ti\pathb-config-f-matrix.ps1 -Preset 9b,27b,31b,35b,48b,80b -EnsureRemusRpc
```

### Config F smoke 9B (`ts=30,12,58`, ctx=4096, q8_0/turbo3)

| Metric | Value |
|--------|-------|
| Load | ~15 s |
| VRAM RPC0 / RPC1 / CUDA0 weights | 1423 / 494 / 3124 MiB |
| G (t/s) avg | ~40.7 (runs 36-43) |
| Log dir | `benchmarks/config-f-smoke-9b/` |

Compare Config E 9B: **G~65**. Extra RPC hop and weak 6600 leg reduce throughput vs dual-GPU Config E.

### Config F 27B (`Qwen3.6-27B-Q5_K_M`, ctx=8192)

| Metric | Value |
|--------|-------|
| Load | ~55 s |
| VRAM RPC0 / RPC1 / CUDA0 weights | 5273 / 2066 / 10423 MiB |
| G (t/s) avg | ~17.5 |
| Log dir | `benchmarks/config-f-27b/` |

Compare Config E 27B: **G~23.6**. Triple split offloads more to 5070 Ti but 6600 RPC bandwidth limits gen speed.

### Config F 31B (`Gemma-31B-it-quant.gguf`, ctx=8192, ngl=99)

| Metric | Value |
|--------|-------|
| Load | ~35 s |
| VRAM RPC0 / RPC1 / CUDA0 weights | 3499 / 1286 / 7036 MiB |
| G (t/s) avg | ~20.5 |
| Log dir | `benchmarks/config-f-31b/` |

Dense 31B (~11.6 GB file) fits fully on GPU across all three devices with `ngl=99`; no CPU layer offload needed.

### Config F 35B MoE - full GPU (`ngl=99`, no `--n-cpu-moe`)

| Metric | Value |
|--------|-------|
| Load | ~70 s |
| VRAM RPC0 / RPC1 / CUDA0 weights | 6623 / 2511 / 11552 MiB |
| CPU mapped | ~515 MiB (mmap only) |
| G (t/s) avg | **~34.5** |
| Log dir | `benchmarks/config-f-35b-gpu/` |

With `--n-cpu-moe 8` (older run): G ~15.7 (`config-f-35b/`). Active experts fit in 38.5 GB GPU; expert CPU offload is unnecessary and halves throughput.

### Config F 36B MoE - full GPU (`ngl=99`, no `--n-cpu-moe`)

| Metric | Value |
|--------|-------|
| Load | ~60 s |
| G (t/s) avg | **~33.6** |
| Log dir | `benchmarks/config-f-36b-gpu/` |

With `--n-cpu-moe 8` (older run): G ~14.7 (`config-f-36b/`).

### Config F 36B MoE NL - full GPU (`ngl=99`, no `--n-cpu-moe`)

| Metric | Value |
|--------|-------|
| Load | ~60 s |
| G (t/s) avg | **~40.3** |
| Log dir | `benchmarks/config-f-36b-nl-gpu/` |

### Config F 48B MoE (`qwen3-coder-next-reap-48b`, ctx=4096, `ngl=33`)

| Metric | Value |
|--------|-------|
| Load | ~65 s |
| GPU layers | 33/49 |
| VRAM RPC0 / RPC1 / CUDA0 weights | 5652 / 2324 / 10780 MiB |
| CPU mapped (layer offload) | ~9543 MiB |
| G (t/s) avg | **~11.3** |
| Log dir | `benchmarks/config-f-48b-gpu/` |

48B exceeds combined GPU; needs `-ngl 33` (~33% layer offload to host RAM). `--n-cpu-moe 8` optional (same split either way).

### Config F 80B MoE (`Qwen3-Next-80B-A3B-Instruct-IQ4_NL`, ctx=4096, `ngl=28`, `--n-cpu-moe 8`)

| Metric | Value |
|--------|-------|
| Load | ~95 s |
| GPU layers | 28/49 |
| VRAM RPC0 / RPC1 / CUDA0 weights | 7989 / 2662 / 13556 MiB |
| CPU mapped (layers + experts) | ~18806 MiB |
| G (t/s) avg | **~4.8** |
| Log dir | `benchmarks/config-f-80b/` |

80B needs aggressive offload: `-ngl 28` (40% layers to RAM) plus `--n-cpu-moe 8` for inactive experts.

### Offload cheat sheet (Config F)

| Model class | Fits full GPU? | Recommended flags |
|-------------|----------------|-------------------|
| <=31B dense | yes | `-ngl 99`, no `--n-cpu-moe` |
| 35B/36B MoE (A3B) | yes (active experts) | `-ngl 99`, no `--n-cpu-moe` |
| 48B MoE | no (~28 GB file) | `-ngl 33`, optional `--n-cpu-moe 8` |
| 80B MoE (~42 GB file) | no | `-ngl 28`, `--n-cpu-moe 8` |

VRAM budget: **38.5 GB** combined GPU. Use `--n-cpu-moe` only when load fails or model file exceeds GPU pool; it is not needed for 35B/36B A3B on this topology.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Failed to create server socket` on remus | Another RPC holds :50051 (`docker stop rx6600-rpc`) |
| Load timeout in bench script | Logs go to `server.log.err`; fixed in `rpc-server-bench.ps1` |
| Garbled vram-calc output | Fixed GGUF parser in `pathb-72b-vram-calc.py` |
| CRLF on WSL bash scripts | Run `scripts/cuda-windows-5070ti/fix-sh-crlf.sh` via WSL |