# Triton RTX 3070 RPC worker on :50055 (CUDA1). Separate process from :50054 3090.
$ErrorActionPreference = "Stop"
$Repo = "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus"
$RpcDir = $null
foreach ($candidate in @(
    "C:\backup\pathb-portable",
    (Join-Path $Repo "build-cuda-b-bin\portable"),
    "C:\backup\lcuda"
)) {
    if (Test-Path (Join-Path $candidate "rpc-server.exe")) {
        $RpcDir = $candidate
        break
    }
}
if (-not $RpcDir) { throw "rpc-server.exe not found" }
$RpcExe = Join-Path $RpcDir "rpc-server.exe"

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
Ensure-FirewallPort "PathB-RPC-50055-In-TCP" "Path B RPC worker (50055 3070)" 50055

$conns = Get-NetTCPConnection -LocalPort 50055 -State Listen -ErrorAction SilentlyContinue
if ($conns) {
    foreach ($c in $conns) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

$taskName = "PathB-Triton-RPC-50055"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
cmd /c "schtasks /Delete /TN $taskName /F" | Out-Null
$ErrorActionPreference = $prevEap
$tr = "cmd /c cd /d `"$RpcDir`" && `"$RpcExe`" -H 0.0.0.0 -p 50055 -d CUDA1 -c"
$create = cmd /c "schtasks /Create /TN $taskName /TR `"$tr`" /SC ONLOGON /RL HIGHEST /F"
if ($LASTEXITCODE -ne 0) { throw "schtasks create failed: $create" }
cmd /c "schtasks /Run /TN $taskName" | Out-Null
Start-Sleep -Seconds 5

$listen = Get-NetTCPConnection -LocalPort 50055 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) { throw "rpc-server not listening on :50055 (CUDA1)" }
$proc = Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "$RpcDir*" } |
    Sort-Object StartTime -Descending |
    Select-Object -First 1
Write-Host "TRITON_RPC_3070_OK task=$taskName device=CUDA1 endpoint=192.168.8.23:50055 exe=$($proc.Path)"