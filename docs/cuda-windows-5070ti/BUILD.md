# Build: cuda-windows-5070ti

Canonical build documentation: [../cuda-windows/BUILD.md](../cuda-windows/BUILD.md)

```powershell
.\scripts\cuda-windows-5070ti\build.ps1
# equivalent:
.\scripts\cuda-windows\build.ps1 -Profile all
```

## 5070 Ti CUDA arch

JUPITER (RTX 5070 Ti, compute 12.0) **must** ship `120a-real` kernels in `ggml-cuda.dll`.

| Profile | `CMAKE_CUDA_ARCHITECTURES` | Deploy target |
|---------|---------------------------|---------------|
| `all` / `5070ti` (default here) | `86-real;120a-real` | **JUPITER** `:50053` |
| `triton` | `86-real` | triton 3090/3070 only |

**Never copy a Triton-only portable build to JUPITER.** HELLO and `GET_DEVICE_MEMORY` can succeed; `GRAPH_COMPUTE` on 5070 Ti may fail and trigger a Windows `abort()` dialog from `rpc-server.exe` (see [RPC-BUG-HUNT.md](RPC-BUG-HUNT.md#5070-ti-rpc-server-abort-2026-06-29)).

Verify after configure:
```powershell
findstr CMAKE_CUDA_ARCHITECTURES build-cuda-b-bin\CMakeCache.txt
# expect: 86-real;120a-real
```

### RPC-only redeploy (after arch fix)

Full `build.ps1` also builds `llama-pipeline-profiler` (uses `ggml_backend_dev_memory` for `--validate-rpc` so it links under `GGML_BACKEND_DL`). Romulus ROCm profiler remains the B+6 gate client. For JUPITER worker only:

```powershell
.\scripts\cuda-windows-5070ti\build.ps1 -Reconfigure -SkipBuild   # refresh arch in cache only
scripts\b6-gate-jupiter-rebuild-rpc.cmd                            # -t rpc-server + copy portable
.\scripts\b6-gate-jupiter-start-rpc-task.ps1                       # restart :50053 schtask
```

`b6-gate-jupiter-start-rpc-task.ps1` warns if `CMakeCache.txt` lacks `120a`.

## Blackwell native build (5070 Ti sm_120)

For a from-scratch Windows build with CUDA 12.9.2 nvcc (bypasses VS CUDA 13.x integration):

```powershell
.\blackwell-windows-build-guide\build-blackwell.ps1
```

Full steps: [blackwell-windows-build-guide/README.md](../../blackwell-windows-build-guide/README.md).

**CUDA `-INFINITY` fix (2026-07-09)**: `ggml/src/ggml-cuda/common.cuh` uses `neg_inf_f32()` bit-casts instead of `-INFINITY` (MSVC/nvcc #221-D). Required for stable `llama-server` on Blackwell. After pulling the fix, **clean rebuild** - stale `ggml-cuda.dll` can still crash.

## Smoke test

```powershell
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1 -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf"
```

Smoke resolves binaries in order: `portable/` -> `bin/Release/` -> `bin/`. Ninja Multi-Config output lives in `build-cuda-b-bin/bin/Release/`.

Logs: `docs/cuda-windows-5070ti/benchmarks/<timestamp>/`

Latest PASS: `benchmarks/20260709-100701/` (Qwen3.5-9B-MTP, SMOKE_OK).

Monitor VRAM: `nvidia-smi -l 2`

## Binary layout

```
build-cuda-b-bin/bin/Release/   # Ninja Multi-Config (local dev + smoke fallback)
  llama-server.exe
  ggml-cuda.dll                 # must include 120a-real for 5070 Ti
  ...

build-cuda-b-bin/portable/      # copy-to-node folder (RPC worker deploy)
  rpc-server.exe                # Config G worker :50053
  ggml-cuda.dll
  llama-server.exe              # Config E/F Windows client
  ...
```

Triton nodes receive a **separate** portable built with `-Profile triton` (JUPITER build host copies to `\\triton\C$\backup\pathb-portable`).