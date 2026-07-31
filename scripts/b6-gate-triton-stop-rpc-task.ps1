# Stop triton rpc-server schtasks (:50054, :50055).
$ErrorActionPreference = "Stop"

Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

foreach ($taskName in @("PathB-Triton-RPC-50054", "PathB-Triton-RPC-50055")) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    cmd /c "schtasks /End /TN $taskName" | Out-Null
    cmd /c "schtasks /Delete /TN $taskName /F" | Out-Null
    $ErrorActionPreference = $prevEap
}

$left = Get-Process -Name "rpc-server" -ErrorAction SilentlyContinue
if ($left) { throw "rpc-server still running after stop" }
Write-Host "TRITON_RPC_STOP_OK tasks=PathB-Triton-RPC-50054,PathB-Triton-RPC-50055"