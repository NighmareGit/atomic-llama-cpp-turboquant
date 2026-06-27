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
│   └── benchmarks/
├── scripts/cuda-windows-5070ti/   # <-- Windows 5070 Ti scripts
│   ├── build.ps1
│   └── smoke-llama-server.ps1
└── rpc-patch/                     # Path B source docs, Linux bench harnesses
```

## Hardware (this host)

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5070 Ti (~16 GB) |
| OS | Windows 11 Pro x64 |
| CUDA Toolkit | 12.8 (`CUDA_PATH`) |
| Driver | Game Ready / Studio (nvidia-smi for monitoring) |
| Portable arch | `86-real` (3070/3090 Ampere) + `120a-real` (5070 Ti Blackwell) |

## Quick start

From repo root (any cwd; scripts resolve paths automatically):

```powershell
.\scripts\cuda-windows-5070ti\build.ps1
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1
```

Build details: [BUILD.md](BUILD.md)

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

## Later phases (not yet)

- Multi-node RPC (`rpc-server` worker + remote client) - see Config B in rpc-patch README
- Windows matrix benches modeled on `rpc-server-bench-matrix.sh`