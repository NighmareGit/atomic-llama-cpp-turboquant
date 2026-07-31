# Portable Windows CUDA build for Path B / Path-B-Plus.
# Canonical script. Host forwarders: cuda-windows-5070ti/build.ps1, build-cuda-windows-pathb.ps1
param(
    [ValidateSet("all", "5070ti", "triton")]
    [string]$Profile = "all",
    [switch]$Clean,
    [switch]$SkipBuild,
    [switch]$Reconfigure,
    [switch]$SkipGgml,
    [string]$VsEdition = "Professional"
)

$ErrorActionPreference = "Stop"
$CollateralRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $CollateralRoot "..\..")).Path
$BuildDir = Join-Path $RepoRoot "build-cuda-b-bin"
$PortableDir = Join-Path $BuildDir "portable"

$CudaArch = switch ($Profile) {
    "triton" { "86-real" }
    default  { "86-real;120a-real" }
}

function Get-BinDir {
    param([string]$Root)
    $release = Join-Path $Root "bin\Release"
    if (Test-Path (Join-Path $release "llama-server.exe")) { return $release }
    $plain = Join-Path $Root "bin"
    if (Test-Path (Join-Path $plain "llama-server.exe")) { return $plain }
    return $null
}

function Find-VcVarsAll {
    param([string]$Edition)
    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\$Edition\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    throw "vcvarsall.bat not found. Install VS 2022 with C++ workload."
}

function Invoke-CmdBuild {
    param([string]$VcVars, [string]$Command)
    $bat = Join-Path $env:TEMP "llama-cuda-build.cmd"
    @"
@echo off
call "$VcVars" x64
cd /d "$RepoRoot"
$Command
"@ | Set-Content -Path $bat -Encoding ASCII
    cmd /c $bat
    if ($LASTEXITCODE -ne 0) { throw "Build command failed (exit $LASTEXITCODE)" }
}

function Copy-CudaRuntimeDlls {
    param([string]$Dest, [string]$CudaPath)
    $patterns = @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
    $searchDirs = @(
        (Join-Path $CudaPath "bin"),
        (Join-Path $CudaPath "bin\x64"),
        (Join-Path $CudaPath "lib")
    )
    $copied = @{}
    foreach ($dir in $searchDirs) {
        if (-not (Test-Path $dir)) { continue }
        foreach ($pat in $patterns) {
            Get-ChildItem -Path $dir -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
                if (-not $copied.ContainsKey($_.Name)) {
                    Copy-Item $_.FullName -Destination $Dest -Force
                    Write-Host "Bundled $($_.Name)"
                    $copied[$_.Name] = $true
                }
            }
        }
    }
    if ($copied.Count -eq 0) {
        throw "No CUDA runtime DLLs found under $CudaPath"
    }
}

Write-Host "=== cuda-windows build (profile=$Profile arch=$CudaArch) ==="
Write-Host "Repo: $RepoRoot"

Write-Host "=== Preflight ==="
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi not found"
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
}
if (-not (Test-Path (Join-Path $cudaPath "bin\nvcc.exe"))) {
    throw "nvcc not found at $cudaPath\bin\nvcc.exe"
}
Write-Host "CUDA_PATH=$cudaPath"
& (Join-Path $cudaPath "bin\nvcc.exe") --version

$vcvars = Find-VcVarsAll -Edition $VsEdition
Write-Host "Using $vcvars"

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Removing $BuildDir"
    Remove-Item $BuildDir -Recurse -Force
}

$nvcc = (Join-Path $cudaPath "bin\nvcc.exe").Replace('\', '/')
$cmakeArgs = @(
    "-S", ".",
    "-B", "build-cuda-b-bin",
    "-G", "Ninja Multi-Config",
    "-DCMAKE_CUDA_COMPILER=`"$nvcc`"",
    "-DGGML_CUDA=ON",
    "-DGGML_RPC=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DGGML_SCHED_MAX_COPIES=4",
    "-DGGML_CUDA_CUB_3DOT2=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_OPENSSL=OFF",
    "-DCMAKE_CXX_FLAGS_RELEASE=/FS",
    "-DCMAKE_C_FLAGS_RELEASE=/FS"
)
$cmakeLine = "cmake " + ($cmakeArgs -join " ")
$MaxJobs = [Math]::Min(8, [Math]::Max(1, $env:NUMBER_OF_PROCESSORS - 1))

if (-not $SkipBuild) {
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if ($Reconfigure -and (Test-Path $cacheFile)) {
        Write-Host "Reconfigure: refreshing CMake cache"
        Invoke-CmdBuild -VcVars $vcvars -Command $cmakeLine
    } elseif (-not (Test-Path $cacheFile)) {
        Write-Host "=== CMake configure ==="
        Invoke-CmdBuild -VcVars $vcvars -Command $cmakeLine
    } else {
        Write-Host "CMake cache exists (use -Reconfigure or -Clean to refresh)"
    }

    $jobs = $MaxJobs
    if (-not $SkipGgml) {
        Write-Host "=== Build ggml (Release, -j $jobs, /FS) ==="
        Invoke-CmdBuild -VcVars $vcvars -Command "cmake --build build-cuda-b-bin --config Release -j $jobs -t ggml"
    }
    Write-Host "=== Build llama-server rpc-server llama-pipeline-profiler (Release, -j $jobs) ==="
    Invoke-CmdBuild -VcVars $vcvars -Command "cmake --build build-cuda-b-bin --config Release -j $jobs -t llama-server rpc-server llama-pipeline-profiler"
}

$BinDir = Get-BinDir -Root $BuildDir
if (-not $BinDir) {
    throw "llama-server.exe not found under $BuildDir\bin or bin\Release"
}

Write-Host "=== Assemble portable folder ==="
if (Test-Path $PortableDir) { Remove-Item $PortableDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $PortableDir | Out-Null

$runtimeExt = @(".exe", ".dll")
Get-ChildItem $BinDir | Where-Object {
    $runtimeExt -contains $_.Extension.ToLower()
} | Copy-Item -Destination $PortableDir -Force

Copy-CudaRuntimeDlls -Dest $PortableDir -CudaPath $cudaPath

Write-Host "=== Portable manifest ==="
Get-ChildItem $PortableDir | Sort-Object Name | Format-Table Name, @{N='MiB';E={[math]::Round($_.Length/1MB,2)}} -AutoSize

Write-Host "BUILD_OK"
Write-Host "  build tree: $BinDir"
Write-Host "  portable:   $PortableDir"