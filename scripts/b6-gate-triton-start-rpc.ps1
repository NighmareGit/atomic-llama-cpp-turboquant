# Start existing rpc-server on triton :50054 (no VS build required).
$ErrorActionPreference = "Stop"

function Ensure-FirewallPort {
    param([string]$Name, [string]$DisplayName, [int]$Port)
    $rule = Get-NetFirewallRule -Name $Name -ErrorAction SilentlyContinue
    if (-not $rule) {
        New-NetFirewallRule -Name $Name -DisplayName $DisplayName `
            -Enabled True -Direction Inbound -Protocol TCP -Action Allow `
            -LocalPort $Port -Profile Any | Out-Null
    } else {
        Set-NetFirewallRule -Name $Name -Enabled True -Profile Any | Out-Null
    }
}

Ensure-FirewallPort "PathB-RPC-50054-In-TCP" "Path B RPC worker (50054)" 50054
netsh advfirewall firewall add rule name="PathB-RPC-50054-In-TCP" dir=in action=allow protocol=TCP localport=50054 profile=any 2>$null | Out-Null

$candidates = @(
    "C:\backup\lcuda\rpc-server.exe",
    "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus\build-cuda-b-bin\portable\rpc-server.exe",
    "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus\build-cuda\bin\rpc-server.exe"
)
$RpcExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $RpcExe) { throw "rpc-server.exe not found" }
$PortableDir = Split-Path $RpcExe -Parent

Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "Starting $RpcExe on 0.0.0.0:50054 CUDA0"
$log = "C:\Users\nightmare\b6-triton-rpc.log"
$proc = Start-Process -FilePath $RpcExe `
    -ArgumentList @("-H", "0.0.0.0", "-p", "50054", "-d", "CUDA0", "-c") `
    -WorkingDirectory $PortableDir `
    -RedirectStandardOutput $log `
    -RedirectStandardError "${log}.err" `
    -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 5

if ($proc.HasExited) {
    $tail = @(Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 30)
    throw "rpc-server exited (code $($proc.ExitCode)). Log: $tail"
}

$listen = Get-NetTCPConnection -LocalPort 50054 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) { throw "rpc-server failed to bind :50054 (PID $($proc.Id))" }

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -match '^192\.168\.8\.' } |
    Select-Object -First 1).IPAddress
Write-Host "TRITON_RPC_OK pid=$($proc.Id) endpoint=${ip}:50054 exe=$RpcExe"