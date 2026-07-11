$ErrorActionPreference = "Stop"

$vcBat = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
$buildDir = "D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-vs-cuda"
$cmakeExe = "C:\Program Files\CMake\bin\cmake.exe"
$msbuildExe = "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"

if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
New-Item -ItemType Directory -Path $buildDir | Out-Null

# Run vcvarsall, then cmake, in the same cmd session
$cmdScript = @'
@"%vcBat%" amd64 >nul 2>&1
cd /d "%buildDir%"
%cmakeExe% .. -G "Visual Studio 17 2022" -A x64 -T "host=x64,cuda=%cudaDir%" -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA=ON
@if %ERRORLEVEL% neq 0 (echo CONFIGURE FAILED; exit /b 1)
echo CONFIGURE OK; building...
%cmakeExe% --build . --config Release --target llama
'@

$cmdScript = $cmdScript -f @(
    @{ 'vcBat' = $vcBat },
    @{ 'buildDir' = $buildDir },
    @{ 'cmakeExe' = $cmakeExe },
    @{ 'cudaDir' = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9' },
    @{ 'msbuildExe' = $msbuildExe }
)

# Actually use proper variable substitution
$cmdBody = @"
@"$vcBat" amd64 >nul 2>&1
cd /d "$buildDir"
"$cmakeExe" .. -G "Visual Studio 17 2022" -A x64 -T "host=x64,cuda=$cudaDir" -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA=ON
@if %ERRORLEVEL% neq 0 (echo CONFIGURE FAILED; exit /b 1)
echo CONFIGURE OK; building...
"$cmakeExe" --build . --config Release --target llama
"@

$scriptPath = "$buildDir\_build.cmd"
Set-Content -Path $scriptPath -Value $cmdBody -Encoding ASCII
Set-Location $buildDir
Start-Process cmd.exe -ArgumentList "/c", "_build.cmd" -Wait -NoNewWindow -PassThru
