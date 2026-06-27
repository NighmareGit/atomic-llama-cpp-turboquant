# Start Windows 5070 Ti as Path B RPC worker on :50053 (Config G RPC2).
param(
    [string]$Port = "50053",
    [string]$HostBind = "0.0.0.0",
    [string]$Device = "CUDA0",
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$RpcExe = Join-Path $Root "build-cuda-b-bin\portable\rpc-server.exe"

if ($Stop) {
    Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Stopped rpc-server processes"
    exit 0
}

if (-not (Test-Path $RpcExe)) {
    throw "Missing $RpcExe - run scripts\cuda-windows-5070ti\build.ps1 first"
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Port $Port already listening (PID $($existing.OwningProcess))"
    exit 0
}

Write-Host "Starting rpc-server $Device on ${HostBind}:$Port"
Start-Process -FilePath $RpcExe -ArgumentList @("-H", $HostBind, "-p", $Port, "-d", $Device) -WindowStyle Hidden
Start-Sleep -Seconds 2

$check = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $check) {
    throw "rpc-server failed to bind :$Port"
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -match '^192\.168\.' } | Select-Object -First 1).IPAddress
Write-Host "RPC worker up on :$Port (LAN IP: $ip)"
Write-Host "Config G endpoint third hop: ${ip}:$Port"