# Trace matrix for RPC bug hunt (requires rebuilt llama-server with GGML_*_TRACE).
param(
    [string[]]$Runs = @("trace-f-3gpu", "trace-f-2gpu"),
    [string]$ModelsRoot = "D:\models"
)

$ErrorActionPreference = "Stop"
$Bench = Join-Path $PSScriptRoot "rpc-server-bench.ps1"
$model = Get-ChildItem -Path $ModelsRoot -Recurse -Filter "Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $model) { throw "Missing model" }

$Matrix = @{
    "trace-f-3gpu" = @{ Config = "config-f"; TensorSplit = "30,12,58"; Rpc = "" }
    "trace-f-2gpu" = @{ Config = "config-f"; TensorSplit = "50,50"; Rpc = "192.168.8.176:50051" }
    "trace-e-2gpu" = @{ Config = "config-e"; TensorSplit = "50,50"; Rpc = "" }
}

if ($Runs.Count -eq 1 -and $Runs[0] -match ',') {
    $Runs = $Runs[0] -split ',' | ForEach-Object { $_.Trim() }
}

$summary = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..\docs\cuda-windows-5070ti\benchmarks")).Path "trace-matrix-summary.txt"
"" | Set-Content $summary -Encoding utf8

foreach ($label in $Runs) {
    if (-not $Matrix.ContainsKey($label)) { Write-Warning "unknown $label"; continue }
    $cfg = $Matrix[$label]
    Write-Host "=== $label ==="
    $args = @{
        Label = $label; Config = $cfg.Config; ModelPath = $model.FullName
        TensorSplit = $cfg.TensorSplit; Ctx = 4096; Ctk = "q4_0"; Ctv = "q4_0"; Ngl = 99
        GenTokens = 128; Runs = 1; LoadTimeout = 1200
        Profile = $true; Trace = $true; SchedDebug = $true; EnsurePathbRpc = $true
    }
    if ($cfg.Rpc) { $args.RpcEndpoint = $cfg.Rpc }
    try {
        & $Bench @args
        "PASS $label" | Tee-Object -FilePath $summary -Append
    } catch {
        "FAIL $label $_" | Tee-Object -FilePath $summary -Append
    }
}
Write-Host "summary -> $summary"