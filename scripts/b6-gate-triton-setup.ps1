# Remote triton setup: audit, sync hint, start :50054 RPC worker.
param(
    [string]$TritonHost = "192.168.8.23",
    [string]$TritonUser = "nightmare",
    [string]$TritonPass = "12345",
    [string]$RemoteRepo = "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus",
    [switch]$AuditOnly,
    [switch]$StartRpc
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Invoke-TritonCmd {
    param([string]$ScriptBlock)
    $sec = ConvertTo-SecureString $TritonPass -AsPlainText -Force
    $cred = New-Object System.Management.Automation.PSCredential($TritonUser, $sec)
    Invoke-Command -ComputerName $TritonHost -Credential $cred -ScriptBlock ([scriptblock]::Create($ScriptBlock))
}

Write-Host "=== triton audit ($TritonHost) ==="
try {
    $gpus = Invoke-TritonCmd "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"
    Write-Host "GPUs:"
    $gpus | ForEach-Object { Write-Host "  $_" }

    $repoOk = Invoke-TritonCmd "Test-Path '$RemoteRepo'"
    Write-Host "Repo exists: $repoOk"
    if ($repoOk) {
        $sha = Invoke-TritonCmd "cd '$RemoteRepo'; git rev-parse --short HEAD 2>`$null; if (-not `$?) { 'no-git' }"
        $portable = Invoke-TritonCmd "Test-Path '$RemoteRepo\build-cuda-b-bin\portable\rpc-server.exe'"
        Write-Host "git_sha: $sha"
        Write-Host "portable rpc-server: $portable"
    }
} catch {
    Write-Host "WinRM failed: $_"
    Write-Host "Fallback: run audit manually on triton or enable WinRM."
    exit 1
}

if ($AuditOnly) { exit 0 }

if ($StartRpc) {
    Invoke-TritonCmd @"
Set-Location '$RemoteRepo'
if (Test-Path '.\scripts\cuda-windows-triton\pathb-rpc-server.ps1') {
    .\scripts\cuda-windows-triton\pathb-rpc-server.ps1 -Restart
} elseif (Test-Path '.\scripts\cuda-windows-5070ti\pathb-rpc-server.ps1') {
    .\scripts\cuda-windows-5070ti\pathb-rpc-server.ps1 -Port 50054 -Restart
} else {
    throw 'pathb-rpc-server.ps1 not found on triton repo'
}
"@
    Write-Host "RPC start requested on triton :50054"
}