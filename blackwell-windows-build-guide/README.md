# Blackwell (sm_120) Windows Build Guide

## Overview

This guide documents the complete, reproducible process for building llama.cpp with CUDA support for NVIDIA Blackwell architecture (RTX 5070 Ti, sm_120) on Windows.

**Target Hardware**: RTX 5070 Ti (Blackwell, sm_120, 16GB GDDR7)
**Target OS**: Windows 11
**Build Output Directory**: `build-cuda-b-bin/` (next to this guide)

## Prerequisites

### Required Software

| Software | Version | Download/Path |
|----------|---------|---------------|
| CUDA Toolkit | 12.9.2 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |
| MSVC (VS 2022) | 19.44.35214.0 | Professional Edition |
| CMake | 4.3+ | via `cmake` command (in PATH) |
| Ninja | 1.12+ | via `ninja` command (in PATH) |
| Git | Latest | For cloning llama.cpp |

### Optional but Recommended
- PowerShell 7+ (for scripting)
- PowerShell Modules: `PSReadLine`, `TerminalGui`

### System Requirements
- 32GB+ RAM (64GB recommended for large models)
- 50GB+ free disk space for build
- NVIDIA driver >= 560 (for Blackwell support)

---

## Step 1: Install CUDA Toolkit 12.9.2

### From Local Package
If you have the CUDA 12.9.2 installer at `H:\downloads\`:
```powershell
# Run the installer
Start-Process "H:\downloads\CUDA_Toolkit_12.9.2_Windows.exe" -Wait -Verb RunAs

# Verify installation
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" --version
```

### Expected Output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on_...
Cuda compilation tools, release 12.9, V12.9.xx
```

### Verify GPU Detection
```powershell
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" --list-gpu-arch
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" --list-gpu-info
```

You should see `sm_120` listed as a supported architecture.

---

## Step 2: Set Up MSVC Environment

### Locate vcvarsall.bat
```powershell
# Professional Edition
$VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"

# Community Edition (if applicable)
# $VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

Test-Path $VCVARS
```

### Verify MSVC Version
```powershell
cmd /c 'call "%VCVARS%" x64 && cl.exe' 2>&1 | Select-String "Version"
```

Expected: `Version 19.44.35214.0` or similar (19.xx = VS 2022)

---

## Step 3: Clone llama.cpp (If Not Already Cloned)

```powershell
cd D:\projects\
if (-not (Test-Path "atomic-llama-cpp-turboquant")) {
    git clone https://github.com/ggml-org/llama.cpp.git atomic-llama-cpp-turboquant
}
cd atomic-llama-cpp-turboquant
```

---

## Step 4: Configure Build with CMake

### The Critical Build Command

VS 2022's built-in CUDA 13.3 integration has compatibility issues with Blackwell sm_120. **You must explicitly specify CUDA 12.9.2's nvcc**.

```powershell
# Set variables
$VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
$CUDA_NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
$BUILD_DIR = "build-cuda-b-bin"

# Run cmake via cmd wrapper (required for vcvarsall.bat)
cmd /c "call `"$VCVARS`" x64 && cmake -B $BUILD_DIR -G `"Ninja Multi-Config`" -DCMAKE_CUDA_COMPILER=`"$CUDA_NVCC`" -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=ON -DGGML_CUDA_MMQ=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DCMAKE_BUILD_TYPE=Release"
```

### CMake Flags Explained

| Flag | Purpose |
|------|---------|
| `-G "Ninja Multi-Config"` | Ninja build system with Debug/Release/RelWithDebInfo configs |
| `-DCMAKE_CUDA_COMPILER=...` | **Critical**: Forces use of CUDA 12.9.2 nvcc, bypasses VS 13.3 integration |
| `-DGGML_CUDA=ON` | Enable CUDA backend |
| `-DGGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=ON` | Force 32-bit compute for CUBLAS (Blackwell stability) |
| `-DGGML_CUDA_MMQ=ON` | Enable MMQ (mixed precision) kernels |
| `-DGGML_CUDA_F16=ON` | Enable FP16 support |
| `-DGGML_CUDA_K_QUANTS=ON` | Enable K-quants support |
| `-DCMAKE_BUILD_TYPE=Release` | Release build (optimized) |

### Expected CMake Output
```
-- The C compiler identification: MSVC 19.44.35214.0
-- The CXX compiler identification: MSVC 19.44.35214.0
-- The CUDA compiler identification: NVIDIA 12.9.x
-- Detecting CUDA GPU architecture: sm_120
-- Found CUDA: 12.9.x
-- Configuring done (xx seconds)
-- Generating done (x seconds)
-- Build files have been written to: D:/projects/.../build-cuda-b-bin
```

**Verify**: You should see `sm_120` in the CMake output. If you see `sm_90` or `sm_80`, the CUDA compiler path is wrong.

---

## Step 5: Build

### Build Command
```powershell
cmake --build build-cuda-b-bin --config Release --parallel
```

### Expected Output
```
[1/672] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/...
...
[672/672] Linking CUDA executable bin\Release\llama-server.exe
```

### Build Duration
- First build: ~5-15 minutes (depending on CPU)
- Incremental build: ~30 seconds

### Verification
```powershell
# Check build artifacts exist
Test-Path "build-cuda-b-bin\bin\Release\ggml-cuda.dll"      # Should be ~368 MB
Test-Path "build-cuda-b-bin\bin\Release\llama-server.exe"   # Should be ~10 KB
Test-Path "build-cuda-b-bin\bin\Release\llama-cli.exe"      # Should exist
Test-Path "build-cuda-b-bin\bin\Release\llama-server-impl.dll"  # Should be ~12 MB
```

---

## Step 6: Verify Build

### Check nvcc Architecture
```powershell
cd build-cuda-b-bin\bin\Release

# Verify ggml-cuda.dll was compiled for sm_120
# Use cuobjdump if available
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cuobjdump.exe" -arch build-cuda-b-bin\bin\Release\ggml-cuda.dll
```

Expected output should include `sm_120`.

### Quick Test with llama-cli
```powershell
.\llama-cli.exe -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -n 32 -ngl 99 -c 1024 -t 2 --no-warmup -p "Hello, how"
```

Expected: Should generate text at 60-120 tokens/second with no errors.

---

## Step 7: Smoke Test llama-server

Ninja Multi-Config places binaries under `build-cuda-b-bin/bin/Release/` (not `portable/` until copied). The smoke script checks both paths.

From repo root:

```powershell
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1 -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf"
```

Expected: `SMOKE_OK` in `docs/cuda-windows-5070ti/benchmarks/<timestamp>/`.

Latest PASS (2026-07-09): `docs/cuda-windows-5070ti/benchmarks/20260709-100701/` (Qwen3.5-9B-MTP).

### Recommended production flags (5070 Ti, single slot, MTP)

```powershell
cd build-cuda-b-bin\bin\Release
.\llama-server.exe -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf `
  -c 20480 -ngl 99 -np 1 --no-kv-unified `
  -ctk turbo3 -ctv q8_0 -ctkd f16 -ctvd f16 `
  --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.6 `
  -b 2048 -ub 1024 --flash-attn on --host 0.0.0.0 --port 8080
```

Post-fix MTP 20k validation: ~128 t/s generation, ~91% draft acceptance. Logs in `blackwell-sm120-server-crash-investigation/mtp-20k-test/`.

**Note**: Qwen3.5 thinking models may put answers in `reasoning_content`; raise `max_tokens` for usable `content` in API responses.

---

## Troubleshooting

### Problem 1: "CUDA compiler not found"
**Cause**: CMake can't find nvcc or VS CUDA integration conflicts.

**Solution**: Explicitly set CMAKE_CUDA_COMPILER:
```powershell
-DCCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
```

### Problem 2: "sm_120 not recognized"
**Cause**: Using CUDA toolkit older than 12.9.

**Solution**: Install CUDA 12.9.2. Check with:
```powershell
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" --version
```

### Problem 3: llama-server silent crash (RESOLVED 2026-07-09)
**Cause**: MSVC/nvcc `#221-D` from `-INFINITY` -> `-((float)(1e+300))` in CUDA softmax/MoE kernels. Multi-slot server decoding hit bad sentinel values; `llama-cli` was less affected.

**Solution**: Fix is in-tree (`neg_inf_f32()` in `ggml/src/ggml-cuda/common.cuh`). **Clean rebuild** required - stale `ggml-cuda.dll` can still BEX64 after the source fix.

```powershell
Remove-Item -Recurse -Force build-cuda-b-bin
.\blackwell-windows-build-guide\build-blackwell.ps1
```

See [../blackwell-sm120-server-crash-investigation/README.md](../blackwell-sm120-server-crash-investigation/README.md) (CLOSED).

### Problem 4: MSVC #221-D warnings during compile
**Cause**: Same `-INFINITY` expansion as Problem 3.

**Solution**: Already applied in this repo. If porting from upstream:
- Use `neg_inf_f32()` / `neg_inf_f32_host()` from `ggml/src/ggml-cuda/common.cuh`
- Update `softmax.cu`, `topk-moe.cu`, `cross-entropy-loss.cu`

### Problem 5: RPC linker errors (unresolved external symbols)
**Cause**: RPC sources not injected outside GGML_BACKEND_DL conditional.

**Solution**: See `ggml/src/CMakeLists.txt` - move RPC source injection to proper scope.

### Problem 6: Build hangs at step 672/672
**Cause**: Final link step taking long with MSVC.

**Solution**: Wait 5-10 minutes for final linking. If truly stuck, check for memory pressure.

### Problem 7: "Ninja: build stopped: subcommand failed"
**Cause**: Stale build artifacts.

**Solution**: Clean rebuild:
```powershell
Remove-Item -Recurse -Force build-cuda-b-bin
# Re-run Step 4 and Step 5
```

---

## Reproducible Build Script

Canonical script: [build-blackwell.ps1](build-blackwell.ps1). Inline copy for reference:

```powershell
# build-blackwell.ps1
# Usage: .\build-blackwell.ps1

$ErrorActionPreference = "Stop"

$VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
$CUDA_NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
$BUILD_DIR = "build-cuda-b-bin"

Write-Host "=== Blackwell (sm_120) Build ===" -ForegroundColor Cyan
Write-Host "CUDA nvcc: $CUDA_NVCC" -ForegroundColor Yellow
Write-Host "MSVC vcvars: $VCVARS" -ForegroundColor Yellow
Write-Host ""

# Verify prerequisites
if (-not (Test-Path $CUDA_NVCC)) {
    Write-Error "CUDA nvcc not found at: $CUDA_NVCC"
    exit 1
}
if (-not (Test-Path $VCVARS)) {
    Write-Error "vcvarsall.bat not found at: $VCVARS"
    exit 1
}

# Clean build directory
if (Test-Path $BUILD_DIR) {
    Write-Host "Cleaning existing build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BUILD_DIR
}

# Configure
Write-Host "Configuring with CMake..." -ForegroundColor Yellow
cmd /c "call `"$VCVARS`" x64 && cmake -B $BUILD_DIR -G `"Ninja Multi-Config`" -DCMAKE_CUDA_COMPILER=`"$CUDA_NVCC`" -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=ON -DGGML_CUDA_MMQ=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DCMAKE_BUILD_TYPE=Release"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed"
    exit 1
}

# Build
Write-Host ""
Write-Host "Building (this may take 5-15 minutes)..." -ForegroundColor Yellow
cmake --build $BUILD_DIR --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit 1
}

# Verify
Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host "Artifacts: $($BUILD_DIR)\bin\Release\" -ForegroundColor Green
Get-ChildItem "$BUILD_DIR\bin\Release\*.exe", "$BUILD_DIR\bin\Release\*.dll" | Format-Table Name, Length -AutoSize
```

---

## Related Documentation

| Doc | Status |
|-----|--------|
| [blackwell-sm120-server-crash-investigation/](../blackwell-sm120-server-crash-investigation/README.md) | **CLOSED** - `-INFINITY` fix, post-fix verification |
| [docs/blackwell/README.md](../docs/blackwell/README.md) | Cross-platform Blackwell guide |
| [docs/cuda-windows-5070ti/README.md](../docs/cuda-windows-5070ti/README.md) | 5070 Ti ops, RPC, benchmarks |
| [docs/cuda-windows-5070ti/BUILD.md](../docs/cuda-windows-5070ti/BUILD.md) | Shared `build.ps1` + arch profiles |

Investigation archive (reference only):

- [INVESTIGATION.md](../blackwell-sm120-server-crash-investigation/INVESTIGATION.md) - full report + closure summary
- [CODE-CHANGES.md](../blackwell-sm120-server-crash-investigation/CODE-CHANGES.md) - CUDA + CMake diffs
- [TEST-RESULTS.md](../blackwell-sm120-server-crash-investigation/TEST-RESULTS.md) - pre/post-fix matrix + MTP 20k

---

*Last updated: 2026-07-09*
*Build verified: 672/672 steps; llama-server smoke SMOKE_OK; MTP 20k PASS*
