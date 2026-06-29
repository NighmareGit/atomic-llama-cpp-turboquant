# Graceful cluster shutdown (remus + romulus docker RPC, triton schtasks).
# JUPITER is NOT stopped -- shutdown the workstation locally when done.
param(
    [string]$RomulusPass = "12345",
    [string]$RemusPass = "12345",
    [string]$TritonHost = "192.168.8.23",
    [string]$TritonUser = "nightmare",
    [string]$TritonPass = "12345",
    [string]$TritonRepo = "C:/projects/atomic-llama-cpp-turboquant/Path-B-Event-Support-Pipeline-Plus"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ClusterScript = (Join-Path $RepoRoot "rpc-patch/scripts/pathb-cluster-up.sh") -replace '\\', '/'
$WslRepo = "/mnt/$($RepoRoot.Substring(0,1).ToLower())/$($RepoRoot.Substring(3) -replace '\\','/')"

Write-Host "=== stop remus + romulus pathb-rpc docker ==="
wsl -e bash -lc "cd '$WslRepo/rpc-patch/scripts' && PATHB_REMUS_SSH_PASS='$RemusPass' PATHB_ROMULUS_SSH_PASS='$RomulusPass' ./pathb-cluster-up.sh stop"

Write-Host "=== stop hung profilers on romulus ==="
wsl -e bash -lc "sshpass -p '$RomulusPass' ssh -o StrictHostKeyChecking=no hunter@192.168.8.108 'pkill -f llama-pipeline-profiler 2>/dev/null || true'"

Write-Host "=== stop triton rpc-server schtasks ==="
& wsl -e sshpass -p $TritonPass ssh -o StrictHostKeyChecking=no "${TritonUser}@${TritonHost}" `
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TritonRepo/scripts/b6-gate-triton-stop-rpc-task.ps1"
if ($LASTEXITCODE -ne 0) { throw "triton rpc stop failed (exit $LASTEXITCODE)" }

Write-Host "=== cluster shutdown complete (JUPITER :50053 not touched) ==="
Write-Host "Verify: remus/romulus :50051 refused; triton :50054 no rpc-server; stop JUPITER schtask locally if needed."