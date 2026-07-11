$ErrorActionPreference = "Stop"

$cl = "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
$nvcc = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe"
$mt = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/mt.exe"
$rc = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe"
$cmake = "C:/Program Files/CMake/bin/cmake.exe"
$buildDir = "D:/projects/atomic-llama-cpp-5070ti/atomic-llama-cpp-turboquant/build-cuda"

if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
New-Item -ItemType Directory $buildDir | Out-Null
Set-Location $buildDir

Write-Host "Configuring with Ninja + MSVC..."
Write-Host "  cl=$cl"
Write-Host "  nvcc=$nvcc"
Write-Host "  mt=$mt"
Write-Host "  rc=$rc"

$args = @(
    "..",
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$cl",
    "-DCMAKE_CXX_COMPILER=$cl",
    "-DCMAKE_CUDA_COMPILER=$nvcc",
    "-DCMAKE_CUDA_ARCHITECTURES=120",
    "-DGGML_CUDA=ON",
    "-DCMAKE_MT=$mt",
    "-DCMAKE_RC_COMPILER=$rc"
)

& $cmake $args
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure FAILED"
    exit 1
}

Write-Host "Configure OK. Building..."
& $cmake --build . -j 16
