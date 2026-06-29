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
$Remote = Join-Path $PSScriptRoot "b6-gate-triton-remote.ps1"

Write-Host "=== triton ($TritonHost) via SSH+cmd ==="
& $Remote -Action audit -TritonHost $TritonHost -TritonUser $TritonUser -TritonPass $TritonPass

if ($AuditOnly) { exit 0 }

if ($StartRpc) {
    & $Remote -Action rpc -TritonHost $TritonHost -TritonUser $TritonUser -TritonPass $TritonPass
    Write-Host "RPC start requested on triton :50054"
}