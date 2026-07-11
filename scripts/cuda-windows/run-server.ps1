# Launcher for Windows CUDA llama-server (Path-B+ multi-backend pipeline).
# Mirrors romulus-local-up.sh behavior: sets pipeline env vars, detects MTP,
# constructs server args, and launches with correct configuration.
#
# usage:
#   .\scripts\cuda-windows\run-server.ps1                          # default model, 128k ctx
#   .\scripts\cuda-windows\run-server.ps1 -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf"
#   .\scripts\cuda-windows\run-server.ps1 -Ctx 4096 -CTK q8_0 -CTV turbo3
#   .\scripts\cuda-windows\run-server.ps1 -MTP off                 # disable speculative decoding
#   .\scripts\cuda-windows\run-server.ps1 -Stop                    # kill existing server

param(
    [string]$ModelPath = "",
    [string]$ModelsRoot = "D:\models",
    [int]$Ctx = 131072,
    [string]$CTK = "q8_0",
    [string]$CTV = "turbo3",
    [string]$MTP = "auto",
    [string]$ListenHost = "0.0.0.0",
    [int]$Port = 8080,
    [string]$Rpc = "127.0.0.1:50051",
    [string]$TensorSplit = "",
    [int]$Ngl = 99,
    [int]$DraftMax = 16,
    [int]$DraftMin = 0,
    [switch]$Stop,
    [switch]$NoPreflight
)

# --- Helper functions ---
function Resolve-Model {
    param([string]$Path, [string]$Root)
    if ($Path -and (Test-Path $Path)) { return $Path }
    $preferred = @(
        "Qwen3.5-4B-Q4_K_M.gguf",
        "gemma-4-E4B.i1-Q4_K_M.gguf",
        "Qwen3.5-9B-MTP-Q4_K_M.gguf",
        "gemma-4-12b-it-Q4_K_M.gguf"
    )
    foreach ($name in $preferred) {
        $hit = Get-ChildItem -Path $Root -Recurse -Filter $name -ErrorAction SilentlyContinue |
            Where-Object { $_.Length -ge 2GB -and $_.Name -notmatch '(embed|mmproj|vision)'} |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    Write-Host "warning: no preferred model found under $Root" -ForegroundColor Yellow
    return ""
}

$ErrorActionPreference = "Stop"
$CollateralRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $CollateralRoot "..\..")).Path

# --- Environment variables (critical for Pipeline+ multi-backend) ---
$env:GGML_PIPELINE_PLUS = "1"
$env:GGML_PIPELINE_MULTI_BACKEND_SEQ = "1"
$env:GGML_RPC_DUAL_SOCKET = "0"
$env:GGML_SCHED_WAVEFRONT_DISPATCH = "0"

# --- Stop existing server ---
if ($Stop) {
    Write-Host "=== stopping llama-server ==="
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "STOP_OK"
    exit 0
}

# --- Find server binary ---
$Portable = Join-Path $RepoRoot "build-cuda-b-bin\portable"
$ReleaseBin = Join-Path $RepoRoot "build-cuda-b-bin\bin\Release"

$serverExe = $null
if (Test-Path (Join-Path $Portable "llama-server.exe")) {
    $serverExe = Join-Path $Portable "llama-server.exe"
} elseif (Test-Path (Join-Path $ReleaseBin "llama-server.exe")) {
    $serverExe = Join-Path $ReleaseBin "llama-server.exe"
}

if (-not $serverExe) {
    Write-Host "error: llama-server.exe not found" -ForegroundColor Red
    Write-Host "hint: run .\scripts\cuda-windows-5070ti\build.ps1 first" -ForegroundColor Yellow
    exit 1
}

Write-Host "=== cuda-windows run-server (Pipeline+) ==="
Write-Host "  server: $serverExe"

# --- Resolve model ---
$ModelPath = Resolve-Model -Path $ModelPath -Root $ModelsRoot
Write-Host "  model:  $ModelPath"

# --- Auto-detect MTP ---
$Spec = $MTP
if ($Spec -eq "auto") {
    if ($ModelPath -match 'MTP|NextN|nextn|UDT') {
        $Spec = "draft-mtp"
    } else {
        $Spec = "none"
    }
}
Write-Host "  mtp:    $Spec"

# --- Tensor split ---
$TS = $TensorSplit
if (-not $TS) { $TS = "50,50" }
Write-Host "  ts:     $TS"

# --- Build server args ---
$ARGS = @(
    "-m", $ModelPath,
    "-c", "$Ctx",
    "-ngl", "$Ngl",
    "-ctk", $CTK,
    "-ctv", $CTV,
    "-fa", "on",
    "--host", $ListenHost,
    "--port", "$Port",
    "--parallel", "1",
    "-np", "1",
    "--cont-batching",
    "--split-mode", "layer",
    "-ts", $TS,
    "--rpc", $Rpc,
    "--fit", "off",
    "--fit-target", "1024,1024",
    "--metrics",
    "--slots",
    "--log-timestamps",
    "--log-prefix",
    "--reasoning", "off"
)

if ($Spec -ne "none") {
    $ARGS += @(
        "--spec-type", $Spec,
        "--spec-draft-n-max", "$DraftMax",
        "--spec-draft-n-min", "$DraftMin"
    )
}

Write-Host ""
Write-Host "=== plan ===" -ForegroundColor Cyan
Write-Host "  ctx=${Ctx} ctk=${CTK} ctv=${CTV} mtp=${Spec}"
Write-Host "  ts=${TS} ngl=${Ngl} rpc=${Rpc}"
Write-Host "  listen=${Host}:${Port}"
Write-Host "  pipeline_plus=1 multi_backend_seq=1"
Write-Host ""
Write-Host "  health: http://${Host}:${Port}/health" -ForegroundColor Green
Write-Host "  chat:   http://${Host}:${Port}/v1/chat/completions" -ForegroundColor Green
Write-Host ""

# --- Launch ---
$LogDir = Join-Path $RepoRoot "docs\cuda-windows-5070ti\benchmarks"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $LogDir "run-$Stamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$serverLog = Join-Path $LogDir "server.log"
$serverErr = Join-Path $LogDir "server.log.err"

$proc = Start-Process -FilePath $serverExe -ArgumentList $ARGS -PassThru `
    -RedirectStandardOutput $serverLog -RedirectStandardError $serverErr `
    -WorkingDirectory (Split-Path $serverExe -Parent) -WindowStyle Hidden

Write-Host "=== Waiting for health check ==="
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        if ($proc.HasExited) {
            Get-Content $serverLog -Tail 40
            Write-Host "llama-server exited early (code $($proc.ExitCode))" -ForegroundColor Red
            Write-Host "See logs: $serverLog" -ForegroundColor Red
            exit 1
        }
    }
}

if (-not $ready) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Get-Content $serverLog -Tail 40
    Write-Host "Server not healthy after 240s" -ForegroundColor Red
    exit 1
}

nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-loaded.txt")

Write-Host ""
Write-Host "SERVER_OK (pid=$($proc.Id), url=http://${Host}:${Port})" -ForegroundColor Green
Write-Host "Logs: $serverLog"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow

# Wait for user interrupt
try {
    Wait-Process -Id $proc.Id -ErrorAction SilentlyContinue
} catch {
    # Server already exited
}

Write-Host "=== Server stopped ==="
$proc = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($proc) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }

nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-after.txt")
Write-Host "DONE"
