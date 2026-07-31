# Build: Windows CUDA (Path B / Path-B-Plus)

Shared build doc for all Windows CUDA nodes. Host-specific ops live in lateral folders:

- [cuda-windows-5070ti](../cuda-windows-5070ti/README.md) -- RTX 5070 Ti client / `:50053` worker
- [cuda-windows-triton](../cuda-windows-triton/README.md) -- RTX 3090+3070, `:50054` RPC spike worker

Artifact tree is **one per repo clone:** `build-cuda-b-bin/portable/`

## Prerequisites

- Windows 11 x64
- Visual Studio 2022 (Desktop development with C++)
- CUDA Toolkit **12.8** (not 13.x CTK)
- Ninja
- NVIDIA driver + [VC++ 2022 x64 redist](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

## Build

```powershell
.\scripts\cuda-windows\build.ps1
.\scripts\cuda-windows\build.ps1 -Profile triton    # Ampere only (3090+3070)
.\scripts\cuda-windows\build.ps1 -Profile all       # 86-real + 120a-real (5070 Ti portable)
```

Legacy forwarder (same binary):

```powershell
.\scripts\cuda-windows-5070ti\build.ps1
```

| Flag | Effect |
|------|--------|
| `-Profile triton` | `-DCMAKE_CUDA_ARCHITECTURES=86-real` |
| `-Profile all` / `5070ti` | `86-real;120a-real` (default) |
| `-Clean` | Remove `build-cuda-b-bin` before configure |
| `-Reconfigure` | Refresh CMake cache |
| `-SkipGgml` | Skip ggml target |
| `-SkipBuild` | Only assemble `portable/` from existing `bin/` |

## Output

```
build-cuda-b-bin/portable/
  llama-server.exe
  rpc-server.exe
  llama-pipeline-profiler.exe
  ggml*.dll
  cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll
```

Copy `portable/` to any Windows CUDA node. Run `rpc-server.exe` from the portable cwd.

## CMake reference

```
-DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_SCHED_MAX_COPIES=4
-DGGML_NATIVE=OFF
-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
-DGGML_CUDA_CUB_3DOT2=ON
-DCMAKE_CUDA_ARCHITECTURES=<profile-dependent>
```

General notes: [docs/build.md](../build.md). Linux parity: [rpc-patch/README.md](../../rpc-patch/README.md).