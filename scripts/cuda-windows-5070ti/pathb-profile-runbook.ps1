# Profile matrix orchestrator for GPU underutilization investigation.
param(
    [string[]]$Runs = @("profile-f-36b-nl-base"),
    [string]$ModelsRoot = "D:\models",
    [switch]$EnsureRemusRpc,
    [switch]$SchedDebug
)

$ErrorActionPreference = "Stop"
$Bench = Join-Path $PSScriptRoot "rpc-server-bench.ps1"
$ModelFile = "Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf"
$model = Get-ChildItem -Path $ModelsRoot -Recurse -Filter $ModelFile -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $model) { throw "Missing $ModelFile under $ModelsRoot" }

$Matrix = @{
    "profile-f-36b-nl-base"    = @{ Config = "config-f"; TensorSplit = "30,12,58"; Rpc = ""; Extra = "" }
    "profile-e-36b-nl"         = @{ Config = "config-e"; TensorSplit = "50,50"; Rpc = ""; Extra = "" }
    "profile-f-36b-nl-ts1080"  = @{ Config = "config-f"; TensorSplit = "12,8,80"; Rpc = ""; Extra = "" }
    "profile-f-36b-nl-no6600"  = @{ Config = "config-f"; TensorSplit = "50,50"; Rpc = "192.168.8.176:50051"; Extra = "" }
}

if ($EnsureRemusRpc) {
    & (Join-Path $PSScriptRoot "pathb-remus-multi-rpc.ps1") start
}

$summary = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..\docs\cuda-windows-5070ti\benchmarks")).Path "profile-matrix-summary.txt"
"" | Set-Content $summary -Encoding utf8

if ($Runs.Count -eq 1 -and $Runs[0] -match ',') {
    $Runs = $Runs[0] -split ',' | ForEach-Object { $_.Trim() }
}

foreach ($label in $Runs) {
    if (-not $Matrix.ContainsKey($label)) { Write-Warning "unknown run $label"; continue }
    $cfg = $Matrix[$label]
    Write-Host "=== $label ==="
    $args = @{
        Label = $label
        Config = $cfg.Config
        ModelPath = $model.FullName
        TensorSplit = $cfg.TensorSplit
        Ctx = 4096; Ctk = "q4_0"; Ctv = "q4_0"; Ngl = 99
        GenTokens = 256; Runs = 1; LoadTimeout = 1200
        Profile = $true
        EnsurePathbRpc = $true
    }
    if ($cfg.Rpc) { $args.RpcEndpoint = $cfg.Rpc }
    if ($cfg.Extra) { $args.ExtraArgs = $cfg.Extra }
    if ($SchedDebug) { $args.SchedDebug = $true }
    try {
        & $Bench @args
        "PASS $label" | Tee-Object -FilePath $summary -Append
    } catch {
        "FAIL $label $_" | Tee-Object -FilePath $summary -Append
    }
}
Write-Host "summary -> $summary"