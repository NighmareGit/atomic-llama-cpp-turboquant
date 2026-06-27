# Config F matrix: Windows 5070 Ti + remus 5060 Ti (:50051) + RX6600 (:50052)
param(
    [string[]]$Preset = @("9b", "27b", "31b", "35b", "36b"),
    [string]$ModelsRoot = "D:\models",
    [string]$TensorSplit = "30,12,58",
    [switch]$EnsureRemusRpc
)

$ErrorActionPreference = "Stop"
$Bench = Join-Path $PSScriptRoot "rpc-server-bench.ps1"

$Presets = @{
    "9b"  = @{ File = "Qwen3.5-9B-MTP-Q4_K_M.gguf";           Ctx = 4096; Ctk = "q8_0"; Ctv = "turbo3"; Ngl = 99; NcMoe = 0; Timeout = 120 }
    "27b" = @{ File = "Qwen3.6-27B-Q5_K_M.gguf";               Ctx = 8192; Ctk = "q8_0"; Ctv = "turbo3"; Ngl = 99; NcMoe = 0; Timeout = 900 }
    "31b" = @{ File = "Gemma-31B-it-quant.gguf";               Ctx = 8192; Ctk = "q8_0"; Ctv = "turbo3"; Ngl = 99; NcMoe = 0; Timeout = 900 }
    "35b" = @{ File = "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";       Ctx = 4096; Ctk = "q4_0"; Ctv = "q4_0";   Ngl = 99; NcMoe = 0; Timeout = 1200 }
    "36b" = @{ File = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf";        Ctx = 4096; Ctk = "q4_0"; Ctv = "q4_0";   Ngl = 99; NcMoe = 0; Timeout = 1200 }
    "36b-nl" = @{ File = "Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf";  Ctx = 4096; Ctk = "q4_0"; Ctv = "q4_0";   Ngl = 99; NcMoe = 0; Timeout = 1200 }
    "48b" = @{ File = "qwen3-coder-next-reap-48b-a3b-q4_k_m.gguf"; Ctx = 4096; Ctk = "q4_0"; Ctv = "q4_0"; Ngl = 33; NcMoe = 0; Timeout = 1800 }
    "80b" = @{ File = "Qwen3-Next-80B-A3B-Instruct-IQ4_NL.gguf"; Ctx = 4096; Ctk = "q4_0"; Ctv = "q4_0"; Ngl = 28; NcMoe = 8; Timeout = 2400 }
}

if ($EnsureRemusRpc) {
    & (Join-Path $PSScriptRoot "pathb-remus-multi-rpc.ps1") start
}

$summary = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..\docs\cuda-windows-5070ti\benchmarks")).Path "config-f-matrix-summary.txt"
"" | Set-Content $summary -Encoding utf8

foreach ($p in $Preset) {
    if (-not $Presets.ContainsKey($p)) { Write-Warning "unknown preset $p"; continue }
    $cfg = $Presets[$p]
    $model = Get-ChildItem -Path $ModelsRoot -Recurse -Filter $cfg.File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $model) {
        "SKIP $p missing $($cfg.File)" | Tee-Object -FilePath $summary -Append
        continue
    }
    $label = "config-f-$p"
    Write-Host "=== matrix $label ==="
    $benchArgs = @{
        Label = $label; Config = "config-f"; ModelPath = $model.FullName
        TensorSplit = $TensorSplit; Ctx = $cfg.Ctx; Ctk = $cfg.Ctk; Ctv = $cfg.Ctv
        Ngl = $cfg.Ngl; LoadTimeout = $cfg.Timeout
    }
    if ($cfg.NcMoe -gt 0) { $benchArgs.NcMoe = $cfg.NcMoe }
    try {
        & $Bench @benchArgs
        "PASS $label" | Tee-Object -FilePath $summary -Append
    } catch {
        "FAIL $label $_" | Tee-Object -FilePath $summary -Append
    }
}
Write-Host "summary -> $summary"