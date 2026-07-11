# build-blackwell.ps1
# Reproducible Blackwell (sm_120) build script for Windows
# Usage: .\blackwell-windows-build-guide\build-blackwell.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Blackwell (sm_120) Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
$CUDA_NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
$BUILD_DIR = "build-cuda-b-bin"

Write-Host "[1/5] Verifying prerequisites..." -ForegroundColor Yellow

# Verify CUDA
if (-not (Test-Path $CUDA_NVCC)) {
    Write-Error "CUDA nvcc not found at: $CUDA_NVCC"
    Write-Error "Install CUDA Toolkit 12.9.2 first."
    exit 1
}
$nvccVersion = & $CUDA_NVCC --version 2>&1 | Select-String "release 12"
Write-Host "  CUDA nvcc: $CUDA_NVCC" -ForegroundColor Green
Write-Host "  CUDA version: $($nvccVersion.Trim())" -ForegroundColor Green

# Verify MSVC
if (-not (Test-Path $VCVARS)) {
    Write-Error "vcvarsall.bat not found at: $VCVARS"
    Write-Error "Install Visual Studio 2022 with C++ workload first."
    exit 1
}
Write-Host "  MSVC vcvars: $VCVARS" -ForegroundColor Green

# Verify CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "cmake not found in PATH"
    exit 1
}
Write-Host "  CMake: $(cmake --version | Select-String -Pattern '(\d+\.\d+\.\d+)').`$1" -ForegroundColor Green

# Verify Ninja
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Error "ninja not found in PATH"
    exit 1
}
Write-Host "  Ninja: $(ninja --version)" -ForegroundColor Green

Write-Host ""

# Clean build directory
Write-Host "[2/5] Cleaning build directory..." -ForegroundColor Yellow
if (Test-Path $BUILD_DIR) {
    Write-Host "  Removing existing $BUILD_DIR..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $BUILD_DIR -ErrorAction SilentlyContinue
    Write-Host "  Done." -ForegroundColor Green
} else {
    Write-Host "  No existing build to clean." -ForegroundColor Gray
}

Write-Host ""

# Configure
Write-Host "[3/5] Configuring with CMake..." -ForegroundColor Yellow
$configArgs = @(
    "-B", $BUILD_DIR,
    "-G", "Ninja Multi-Config",
    "-DCMAKE_CUDA_COMPILER=$CUDA_NVCC",
    "-DGGML_CUDA=ON",
    "-DGGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=ON",
    "-DGGML_CUDA_MMQ=ON",
    "-DGGML_CUDA_F16=ON",
    "-DGGML_CUDA_K_QUANTS=ON",
    "-DCMAKE_BUILD_TYPE=Release"
)

Write-Host "  Running: cmd /c 'call vcvarsall x64 && cmake $($configArgs -join " ")'" -ForegroundColor Gray
cmd /c "call `"$VCVARS`" x64 && cmake @($configArgs)"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed"
    exit 1
}
Write-Host "  Configuration successful." -ForegroundColor Green

Write-Host ""

# Build
Write-Host "[4/5] Building (this may take 5-15 minutes)..." -ForegroundColor Yellow
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Starting build..." -ForegroundColor Gray

$buildStart = Get-Date
cmake --build $BUILD_DIR --config Release --parallel
$buildExit = $LASTEXITCODE

$buildDuration = (Get-Date) - $buildStart
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Build completed in $($buildDuration.TotalSeconds.ToString('0.0'))s" -ForegroundColor Gray

if ($buildExit -ne 0) {
    Write-Error "Build failed (exit code: $buildExit)"
    exit 1
}

Write-Host ""

# Verify
Write-Host "[5/5] Verifying build artifacts..." -ForegroundColor Yellow
$artifacts = @(
    "ggml-cuda.dll",
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama-cli.exe",
    "llama-quantize.exe",
    "llama-server.exe"
)

$allPresent = $true
foreach ($art in $artifacts) {
    $path = Join-Path $BUILD_DIR "bin\Release\$art"
    if (Test-Path $path) {
        $size = (Get-Item $path).Length
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "  [OK] $art ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $art" -ForegroundColor Red
        $allPresent = $false
    }
}

Write-Host ""
if ($allPresent) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Build Complete - All artifacts present" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run tests:" -ForegroundColor Cyan
    Write-Host "  cd $BUILD_DIR\bin\Release" -ForegroundColor Gray
    Write-Host "  .\llama-cli.exe -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -n 32 -ngl 99 -c 1024 -t 2 --no-warmup -p 'Hello'" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " Build Failed - Missing artifacts" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
}
