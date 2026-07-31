# Config E matrix: Windows 5070 Ti + remus 5060 Ti RPC
param(
    [string[]]$Preset = @("9b", "27b"),
    [string]$ModelsRoot = "D:\models",
    [string]$TensorSplit = "50,50",
    [switch]$EnsureRemusRpc
)

$ErrorActionPreference = "Stop"
$Bench = Join-Path $PSScriptRoot "rpc-server-bench.ps1"
$RemusRpc = Join-Path $PSScriptRoot "pathb-remus-rpc.ps1"

$Presets = @{
    "9b"  = @{ File = "Qwen3.5-9B-MTP-Q4_K_M.gguf"; Ctx = 4096; Ctk = "q8_0"; Ctv = "turbo3"; Timeout = 120 }
    "27b" = @{ File = "Qwen3.6-27B-Q5_K_M.gguf";      Ctx = 8192; Ctk = "q8_0"; Ctv = "turbo3"; Timeout = 900 }
    "31b" = @{ File = "Gemma-31B-it-quant.gguf";      Ctx = 8192; Ctk = "q8_0"; Ctv = "turbo3"; Timeout = 900 }
}

if ($EnsureRemusRpc) {
    & (Join-Path $PSScriptRoot "invoke-wsl.ps1") -BashCommand @"
export SSHPASS="\${SSHPASS:-\${PATHB_REMUS_SSH_PASS:-\${PATHB_ROMULUS_SSH_PASS}}}"
sshpass -e ssh -o StrictHostKeyChecking=accept-new hunter@192.168.8.176 'docker stop rx6600-rpc 2>/dev/null || true'
cd /mnt/d/projects/atomic-llama-cpp-5070ti/atomic-llama-cpp-turboquant && ./rpc-patch/scripts/pathb-remus-rpc.sh start
"@
}

$summary = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..\docs\cuda-windows-5070ti\benchmarks")).Path "config-e-matrix-summary.txt"
"" | Set-Content $summary -Encoding utf8

foreach ($p in $Preset) {
    if (-not $Presets.ContainsKey($p)) { Write-Warning "unknown preset $p"; continue }
    $cfg = $Presets[$p]
    $model = Get-ChildItem -Path $ModelsRoot -Recurse -Filter $cfg.File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $model) {
        "SKIP $p missing $($cfg.File)" | Tee-Object -FilePath $summary -Append
        continue
    }
    $label = "config-e-$p"
    Write-Host "=== matrix $label ==="
    try {
        & $Bench -Label $label -Config config-e -ModelPath $model.FullName `
            -TensorSplit $TensorSplit -Ctx $cfg.Ctx -Ctk $cfg.Ctk -Ctv $cfg.Ctv -LoadTimeout $cfg.Timeout
        "PASS $label" | Tee-Object -FilePath $summary -Append
    } catch {
        "FAIL $label $_" | Tee-Object -FilePath $summary -Append
    }
}
Write-Host "summary -> $summary"
