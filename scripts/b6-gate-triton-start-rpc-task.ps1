# Register a local scheduled task so rpc-server survives SSH disconnect.
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
if (-not $RpcDir) { throw "rpc-server.exe not found under backup, repo portable, or lcuda" }
$RpcExe = Join-Path $RpcDir "rpc-server.exe"
if (-not (Test-Path $RpcExe)) { throw "missing $RpcExe" }

Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$taskName = "PathB-Triton-RPC-50054"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
cmd /c "schtasks /Delete /TN $taskName /F" | Out-Null
$ErrorActionPreference = $prevEap
$tr = "cmd /c cd /d `"$RpcDir`" && `"$RpcExe`" -H 0.0.0.0 -p 50054 -d CUDA0 -c"
$create = cmd /c "schtasks /Create /TN $taskName /TR `"$tr`" /SC ONLOGON /RL HIGHEST /F"
if ($LASTEXITCODE -ne 0) { throw "schtasks create failed: $create" }
$run = cmd /c "schtasks /Run /TN $taskName"
if ($LASTEXITCODE -ne 0) { throw "schtasks run failed: $run" }
Start-Sleep -Seconds 5

$listen = Get-NetTCPConnection -LocalPort 50054 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) { throw "rpc-server not listening on :50054 after task start" }
$proc = Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Select-Object -First 1
Write-Host "TRITON_RPC_TASK_OK task=$taskName exe=$($proc.Path) endpoint=192.168.8.23:50054"