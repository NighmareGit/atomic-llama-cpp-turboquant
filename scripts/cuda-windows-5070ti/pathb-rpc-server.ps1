# Start Windows 5070 Ti as Path B RPC worker on :50053 (Config G RPC2).
param(
    [string]$Port = "50053",
    [string]$HostBind = "0.0.0.0",
    [string]$Device = "CUDA0",
    [switch]$Stop,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$CollateralRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $CollateralRoot "..\..")).Path

function Resolve-RpcPortableDir {
    $candidates = @(
        (Join-Path $RepoRoot "build-cuda-b-bin\portable"),
        "D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\portable"
    )
    foreach ($dir in $candidates) {
        $exe = Join-Path $dir "rpc-server.exe"
        if (Test-Path $exe) {
            return $dir
        }
    }
    return $null
}

function Get-RpcListenState {
    param([string]$ListenPort)
    Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue
}

$PortableDir = Resolve-RpcPortableDir
if (-not $PortableDir) {
    throw "rpc-server.exe not found - run scripts\cuda-windows-5070ti\build.ps1 first"
}
$RpcExe = Join-Path $PortableDir "rpc-server.exe"

if ($Stop -or $Restart) {
    Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Stopped rpc-server processes"
    if ($Stop) { exit 0 }
    Start-Sleep -Seconds 1
}

$existing = Get-RpcListenState -ListenPort $Port
if ($existing -and -not $Restart) {
    $proc = Get-Process -Id $existing.OwningProcess -ErrorAction SilentlyContinue
    Write-Host "Port $Port already listening on $($existing.LocalAddress) (PID $($existing.OwningProcess))"
    if ($proc) {
        Write-Host "  path: $($proc.Path)"
    }
    if ($existing.LocalAddress -ne "0.0.0.0") {
        Write-Host "warning: not bound to 0.0.0.0 - romulus cannot reach this worker; use -Restart"
    }
    exit 0
}

Write-Host "Starting rpc-server $Device from $PortableDir on ${HostBind}:$Port"
$procInfo = Start-Process -FilePath $RpcExe `
    -ArgumentList @("-H", $HostBind, "-p", $Port, "-d", $Device) `
    -WorkingDirectory $PortableDir `
    -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3

$check = Get-RpcListenState -ListenPort $Port
if (-not $check) {
    throw "rpc-server failed to bind :$Port (PID $($procInfo.Id) may have exited - run from portable cwd)"
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -match '^192\.168\.8\.' } |
    Select-Object -First 1).IPAddress
if (-not $ip) {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -match '^192\.168\.' -and $_.IPAddress -notmatch '\.(159|126)\.' } |
        Select-Object -First 1).IPAddress
}
Write-Host "RPC worker up on $($check.LocalAddress):$Port (PID $($check.OwningProcess), LAN IP: $ip)"
Write-Host "Config G third hop: ${ip}:$Port"