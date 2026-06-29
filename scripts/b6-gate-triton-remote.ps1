# Run cmd/git on triton over SSH. Use wsl -e sshpass + cmd.exe (not nested bash -lc).
param(
    [Parameter(Position = 0)]
    [ValidateSet("sync", "status", "rpc", "audit")]
    [string]$Action = "status",
    [string]$TritonHost = "192.168.8.23",
    [string]$TritonUser = "nightmare",
    [string]$TritonPass = "12345",
    [string]$RemoteRepo = "C:/projects/atomic-llama-cpp-turboquant/Path-B-Event-Support-Pipeline-Plus"
)

$ErrorActionPreference = "Stop"

function Invoke-TritonCmd {
    param([string]$RemoteCmd)
    & wsl -e sshpass -p $TritonPass ssh -o StrictHostKeyChecking=no "${TritonUser}@${TritonHost}" cmd.exe /c $RemoteCmd
    if ($LASTEXITCODE -ne 0) { throw "triton cmd failed (exit $LASTEXITCODE): $RemoteCmd" }
}

switch ($Action) {
    "sync" {
        Write-Host "=== triton git sync (cmd) ==="
        Invoke-TritonCmd "$RemoteRepo/scripts/b6-gate-triton-git-sync.cmd"
    }
    "status" {
        Write-Host "=== triton git status (cmd) ==="
        Invoke-TritonCmd "$RemoteRepo/scripts/b6-gate-triton-git-status.cmd"
    }
    "rpc" {
        Write-Host "=== triton RPC restart (schtask) ==="
        & wsl -e sshpass -p $TritonPass ssh -o StrictHostKeyChecking=no "${TritonUser}@${TritonHost}" `
            powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$RemoteRepo/scripts/b6-gate-triton-start-rpc-task.ps1"
        if ($LASTEXITCODE -ne 0) { throw "triton rpc restart failed (exit $LASTEXITCODE)" }
    }
    "audit" {
        Write-Host "=== triton audit (cmd) ==="
        Invoke-TritonCmd "$RemoteRepo/scripts/b6-gate-triton-audit.cmd"
    }
}