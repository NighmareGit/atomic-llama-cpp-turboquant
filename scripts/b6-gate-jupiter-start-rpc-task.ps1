# JUPITER 5070 Ti RPC worker :50053 (persistent schtask).
$ErrorActionPreference = "Stop"
$RpcDir = "D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\portable"
if (-not (Test-Path (Join-Path $RpcDir "rpc-server.exe"))) {
    throw "missing $RpcDir\rpc-server.exe"
}
$cmakeCache = "D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\CMakeCache.txt"
if (Test-Path $cmakeCache) {
    $archLine = Select-String -Path $cmakeCache -Pattern "^CMAKE_CUDA_ARCHITECTURES:" | Select-Object -First 1
    if ($archLine -and $archLine.Line -notmatch "120a") {
        Write-Warning "JUPITER rpc-server built without 120a-real (5070 Ti). Rebuild: scripts\cuda-windows-5070ti\build.ps1 -Reconfigure. Triton-only (86-real) can abort() on graph_compute."
    }
}
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
Ensure-FirewallPort "PathB-RPC-50053-In-TCP" "Path B RPC worker (50053 5070 Ti)" 50053

Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$taskName = "PathB-Jupiter-RPC-50053"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
cmd /c "schtasks /Delete /TN $taskName /F" | Out-Null
$ErrorActionPreference = $prevEap
$tr = "cmd /c cd /d `"$RpcDir`" && `"$RpcExe`" -H 0.0.0.0 -p 50053 -d CUDA0 -c"
$create = cmd /c "schtasks /Create /TN $taskName /TR `"$tr`" /SC ONLOGON /RL HIGHEST /F"
if ($LASTEXITCODE -ne 0) { throw "schtasks create failed: $create" }
cmd /c "schtasks /Run /TN $taskName" | Out-Null
Start-Sleep -Seconds 5

$listen = Get-NetTCPConnection -LocalPort 50053 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) { throw "rpc-server not listening on :50053" }
$proc = Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Select-Object -First 1
Write-Host "JUPITER_RPC_TASK_OK task=$taskName exe=$($proc.Path) endpoint=192.168.8.21:50053"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader