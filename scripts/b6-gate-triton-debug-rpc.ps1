$ErrorActionPreference = "Continue"
$candidates = @(
    "C:\backup\lcuda\rpc-server.exe",
    "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus\build-cuda\bin\rpc-server.exe"
)
foreach ($exe in $candidates) {
    if (-not (Test-Path $exe)) { continue }
    Write-Host "=== $exe ==="
    & $exe --help 2>&1 | Select-Object -First 15
    Write-Host ""
}
Get-Process rpc-server -ErrorAction SilentlyContinue | Format-Table Id,Path
Get-NetTCPConnection -LocalPort 50054 -ErrorAction SilentlyContinue | Format-Table LocalAddress,LocalPort,State