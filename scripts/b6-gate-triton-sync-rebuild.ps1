# Run on triton: fetch latest Path-B-Event-Support-Pipeline-Plus, rebuild portable, restart RPC.
$ErrorActionPreference = "Stop"
$Repo = "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus"

if (-not (Test-Path $Repo)) { throw "Repo missing: $Repo" }
Set-Location $Repo

$GiteaUrl = "http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git"
$Branch = "Path-B-Event-Support-Pipeline-Plus"

Write-Host "=== git fetch + reset ==="
git stash push -u -m "b6-gate-triton-sync" 2>$null | Out-Null
if (-not (git remote | Select-String -Pattern '^gitea$' -Quiet)) {
    git remote add gitea $GiteaUrl
} else {
    git remote set-url gitea $GiteaUrl
}
git fetch gitea --prune
git checkout $Branch 2>$null
if ($LASTEXITCODE -ne 0) {
    git checkout -b $Branch "gitea/$Branch"
}
git reset --hard "gitea/$Branch"
$sha = git rev-parse --short HEAD
Write-Host "HEAD=$sha"

$portable = Join-Path $Repo "build-cuda-b-bin\portable\rpc-server.exe"
$vcvars = @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($vcvars) {
    Write-Host "=== build portable (triton profile) ==="
    & "$Repo\scripts\cuda-windows\build.ps1" -Profile triton -Clean
} else {
    Write-Host "=== no VS 2022; expect portable from JUPITER scp ==="
}

if (-not (Test-Path $portable)) { throw "rpc-server.exe missing: $portable (build on JUPITER or install VS 2022)" }
Write-Host "PORTABLE_OK $portable"

Write-Host "=== restart RPC :50054 ==="
& "$Repo\scripts\cuda-windows-triton\pathb-rpc-server.ps1" -Restart

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -match '^192\.168\.8\.' } |
    Select-Object -First 1).IPAddress
Write-Host "TRITON_SYNC_REBUILD_OK sha=$sha rpc=${ip}:50054"