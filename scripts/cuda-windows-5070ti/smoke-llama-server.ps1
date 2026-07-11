# Phase 1: local llama-server CUDA smoke (5070 Ti Windows host, single node).
param(
    [string]$ModelPath = "",
    [string]$ModelsRoot = "D:\models",
    [int]$Port = 8080,
    [int]$MaxTokens = 64,
    [double]$MinGiB = 3.5
)

$ErrorActionPreference = "Stop"
$CollateralRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $CollateralRoot "..\..")).Path
$Portable = Join-Path $RepoRoot "build-cuda-b-bin\portable"
$Release  = Join-Path $RepoRoot "build-cuda-b-bin\bin\Release"
$BinDir   = if (Test-Path (Join-Path $Portable "llama-server.exe")) { $Portable }
            elseif (Test-Path (Join-Path $Release "llama-server.exe")) { $Release }
            else { Join-Path $RepoRoot "build-cuda-b-bin\bin" }
$BenchDir = Join-Path $RepoRoot "docs\cuda-windows-5070ti\benchmarks"
$Stamp    = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir   = Join-Path $BenchDir $Stamp

function Find-SmokeModel {
    param([string]$Root, [double]$MinSizeGiB)
    $minBytes = [int64]($MinSizeGiB * 1GB)
    $excludePath = '(\\FIM\\|\\embeddings\\|embedding|vl-embedding|-VL-|\\vision\\)'
    $excludeName = '(?i)(embed|mmproj|vision)'

    $preferred = @(
        "Qwen3.5-4B-Q4_K_M.gguf",
        "gemma-4-E4B.i1-Q4_K_M.gguf",
        "Qwen3.5-9B-MTP-Q4_K_M.gguf",
        "gemma-4-12b-it-Q4_K_M.gguf"
    )
    foreach ($name in $preferred) {
        $hit = Get-ChildItem -Path $Root -Recurse -Filter $name -ErrorAction SilentlyContinue |
            Where-Object { $_.Length -ge $minBytes -and $_.FullName -notmatch $excludePath } |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }

    $fallback = Get-ChildItem -Path $Root -Recurse -Filter "*.gguf" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Length -ge $minBytes -and
            $_.FullName -notmatch $excludePath -and
            $_.Name -notmatch $excludeName
        } |
        Sort-Object Length |
        Select-Object -First 1
    if ($fallback) { return $fallback.FullName }
    throw "No >= ${MinSizeGiB} GiB chat GGUF found under $Root (FIM/embedding/VL excluded)"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$serverExe = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    throw "Run scripts/cuda-windows-5070ti/build.ps1 first. Missing $serverExe"
}

if (-not $ModelPath) {
    $ModelPath = Find-SmokeModel -Root $ModelsRoot -MinSizeGiB $MinGiB
}
Write-Host "Model: $ModelPath"
Write-Host "Logs:  $LogDir"

nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-before.txt")

Write-Host "=== Binary checks ==="
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
cmd /c "`"$serverExe`" --version 2>&1" | Tee-Object -FilePath (Join-Path $LogDir "version.txt")
cmd /c "`"$serverExe`" --help 2>&1" | Select-String "cache-type-k" | Tee-Object -FilePath (Join-Path $LogDir "help-cache-type.txt")
$ErrorActionPreference = $prevEap

$env:GGML_PIPELINE_PLUS = "1"
$env:GGML_PIPELINE_MULTI_BACKEND_SEQ = "1"
$env:GGML_RPC_DUAL_SOCKET = "0"
$env:GGML_SCHED_WAVEFRONT_DISPATCH = "0"

$serverLog = Join-Path $LogDir "server.log"
$serverArgs = @(
    "-m", $ModelPath,
    "-ngl", "99",
    "-c", "2048",
    "-ctk", "turbo3",
    "-ctv", "turbo3",
    "--host", "127.0.0.1",
    "--port", "$Port"
)

Write-Host "=== Starting llama-server ==="
$proc = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -PassThru `
    -RedirectStandardOutput $serverLog -RedirectStandardError "${serverLog}.err" `
    -WorkingDirectory $BinDir -WindowStyle Hidden

$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        if ($proc.HasExited) {
            Get-Content $serverLog -Tail 40
            throw "llama-server exited early (code $($proc.ExitCode))"
        }
    }
}
if (-not $ready) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Get-Content $serverLog -Tail 40
    throw "Server not healthy after 240s"
}

nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-loaded.txt")

$body = @{
    messages = @(@{ role = "user"; content = "Say hi in one short sentence." })
    max_tokens = $MaxTokens
} | ConvertTo-Json -Depth 5 -Compress

Write-Host "=== Chat completion ==="
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
$resp | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $LogDir "completion.json")
Write-Host ($resp.choices[0].message.content)

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-after.txt")

@"
smoke=PASS
model=$ModelPath
port=$Port
timestamp=$Stamp
"@ | Set-Content (Join-Path $LogDir "result.meta")

Write-Host "SMOKE_OK -> $LogDir"