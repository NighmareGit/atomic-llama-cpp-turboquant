# Build: cuda-windows-5070ti

## Prerequisites

- Windows 11 x64
- Visual Studio 2022 (Professional or Community) - Desktop development with C++
- CUDA Toolkit **12.8** (not 13.x CTK; driver may report CUDA 13.x capability)
- Ninja
- Models on host, e.g. `D:\models`

## Build

```powershell
.\scripts\cuda-windows-5070ti\build.ps1
```

| Flag | Effect |
|------|--------|
| `-Clean` | Remove `build-cuda-b-bin` before configure |
| `-Reconfigure` | Refresh CMake cache |
| `-SkipGgml` | Skip ggml target (resume after ggml already built) |
| `-SkipBuild` | Only assemble `portable/` from existing `bin/` |

Resume after MSVC PDB errors:

```powershell
.\scripts\cuda-windows-5070ti\build.ps1 -Reconfigure -SkipGgml
```

## Output

```
build-cuda-b-bin/portable/
  llama-server.exe
  rpc-server.exe
  ggml*.dll
  cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll
```

Copy `portable/` to another Windows node. Target needs NVIDIA driver + [VC++ 2022 x64 redist](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## CMake reference

Aligned with Path B Linux `build-cuda-b-bin` (see [rpc-patch/README.md](../../rpc-patch/README.md)) plus portable multi-GPU arch:

```
-DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_SCHED_MAX_COPIES=4
-DGGML_NATIVE=OFF
-DCMAKE_CUDA_ARCHITECTURES=86-real;120a-real
-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
-DGGML_CUDA_CUB_3DOT2=ON
```

General CUDA/Windows notes: [docs/build.md](../build.md)

## Smoke test

```powershell
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1 -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf"
```

Logs: `docs/cuda-windows-5070ti/benchmarks/<timestamp>/`

Monitor VRAM: `nvidia-smi -l 2`