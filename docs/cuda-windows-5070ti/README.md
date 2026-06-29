# Windows CUDA native build - RTX 5070 Ti

Native Windows 11 build collateral for **Path B** (`Path-B-Event-Support`, RPC v4.2.2).
Same binaries and build directory as Linux rpc-patch benches (`build-cuda-b-bin/`).

This is **not** a separate project - it complements [rpc-patch/README.md](../../rpc-patch/README.md).
Linux Docker deploy under `rpc-patch/deploy/` is unchanged.

## Repository layout

```
atomic-llama-cpp-turboquant/
├── build-cuda-b-bin/              # shared with rpc-patch (Linux + Windows)
│   ├── bin/                       # MSVC build tree
│   └── portable/                  # copy-to-node folder (exes + DLLs)
├── docs/cuda-windows-5070ti/      # <-- Windows 5070 Ti docs + benchmarks
│   ├── README.md
│   ├── BUILD.md
│   ├── MULTI-NODE.md              # Config E/F topology + results
│   ├── CLUSTER-4GPU-PRIMARY.md    # Stable 4-GPU (7900+3060+5060+5070)
│   └── benchmarks/                # see benchmarks/README.md
├── scripts/cuda-windows-5070ti/   # <-- Windows 5070 Ti scripts
│   ├── build.ps1
│   ├── smoke-llama-server.ps1
│   ├── rpc-server-bench.ps1       # Config E/F hybrid bench
│   ├── pathb-remus-rpc.ps1        # WSL -> remus 5060 RPC (Config E)
│   ├── pathb-remus-multi-rpc.ps1  # WSL -> 5060 + RX6600 RPC (Config F, experimental)
│   ├── pathb-rpc-server.ps1       # Windows 5070 RPC worker :50053 (Config G)
│   ├── invoke-wsl.ps1             # WSL + SSH bridge
│   ├── pathb-vram-calc.ps1        # VRAM/ts planner
│   ├── pathb-config-e-matrix.ps1  # Config E preset loop
│   └── pathb-config-f-matrix.ps1  # Config F preset loop (through 80b)
├── scripts/                       # B+6 gate (JUPITER ops)
│   ├── b6-gate-jupiter-start-rpc-task.ps1
│   └── b6-gate-jupiter-rebuild-rpc.cmd
└── rpc-patch/                     # Path B source docs, Linux bench harnesses
```

## Hardware (this host)

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5070 Ti (~16 GB) |
| OS | Windows 11 Pro x64 |
| CUDA Toolkit | 12.8 (`CUDA_PATH`) |
| Driver | Game Ready / Studio (nvidia-smi for monitoring) |
| Portable arch | `86-real;120a-real` on JUPITER (`-Profile all`); Triton uses `86-real` only |
| RPC worker | JUPITER `192.168.8.21:50053` (Config G RPC2) |

## Quick start

From repo root (any cwd; scripts resolve paths automatically):

```powershell
.\scripts\cuda-windows-5070ti\build.ps1
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1
```

Build details: [BUILD.md](BUILD.md)

**Config G RPC worker (romulus 4-GPU client):**
```powershell
.\scripts\b6-gate-jupiter-start-rpc-task.ps1   # schtask PathB-Jupiter-RPC-50053
# or foreground: .\scripts\cuda-windows-5070ti\pathb-rpc-server.ps1
```

4-GPU `-ts` on romulus: `25,12,25,38` (RPC0 5060 / RPC1 3060 / RPC2 5070 / ROCm0 7900). See [CLUSTER-4GPU-PRIMARY.md](CLUSTER-4GPU-PRIMARY.md).

If `rpc-server.exe` shows a Windows **abort()** dialog during load/gen, rebuild with `120a-real` arch ([RPC-BUG-HUNT.md](RPC-BUG-HUNT.md#5070-ti-rpc-server-abort-2026-06-29)).

## Phase 1 scope

- Single-node `llama-server` on local CUDA GPU
- Portable `build-cuda-b-bin/portable/` for copy to other Windows nodes (3070/3090/5070 Ti)
- Smoke uses **>= 4B chat** GGUFs only (skips `FIM/`, embedding, VL)

Latest PASS: `benchmarks/20260627-042947/` - `gemma-4-E4B.i1-Q4_K_M.gguf`.

## Path B / RPC context

For protocol changes, tensor-split presets, and Linux matrix benches see:

- [rpc-patch/patch/HANDOVER.md](../../rpc-patch/patch/HANDOVER.md)
- [rpc-patch/docs/RPC_PATH_AB_OPTIMIZATION_REPORT.md](../../rpc-patch/docs/RPC_PATH_AB_OPTIMIZATION_REPORT.md)
- [rpc-patch/scripts/rpc-server-bench-matrix.sh](../../rpc-patch/scripts/rpc-server-bench-matrix.sh) (blueprint for later phases)

## Phase 2: Multi-node RPC (Config E / F)

Windows 5070 Ti client + remus RPC workers. See [MULTI-NODE.md](MULTI-NODE.md).

**Config E** (5070 Ti + remus 5060 Ti):

```powershell
.\scripts\cuda-windows-5070ti\pathb-remus-rpc.ps1 start
.\scripts\cuda-windows-5070ti\rpc-server-bench.ps1 -Label config-e-smoke-9b -Config config-e `
  -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf"
```

Latest PASS: `benchmarks/config-e-smoke-9b-v2/` (9B, G~65 t/s), `benchmarks/config-e-27b/` (27B, G~24 t/s).

**Config F** (5070 Ti + remus 5060 Ti + RX6600 on :50052):

```powershell
.\scripts\cuda-windows-5070ti\pathb-remus-multi-rpc.ps1 start
.\scripts\cuda-windows-5070ti\rpc-server-bench.ps1 -Label config-f-smoke-9b -Config config-f `
  -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf"
```

Latest PASS: through 80B MoE on Config F. Highlights: `config-f-35b-gpu/` (G~34.5, full GPU),
`config-f-48b-gpu/` (48B, G~11.3, ngl=33), `config-f-80b/` (80B, G~4.8, ngl=28 + ncmoe=8).
See [MULTI-NODE.md](MULTI-NODE.md) offload cheat sheet.