# Hybrid bench: WSL remus RPC + GPU monitor, native Windows llama-server (Config E/F).
param(
    [string]$Label = "",
    [string]$Config = "config-e",
    [string]$ModelPath = "",
    [string]$ModelsRoot = "D:\models",
    [string]$RpcEndpoint = "",
    [string]$TensorSplit = "",
    [int]$Ctx = 4096,
    [string]$Ctk = "q8_0",
    [string]$Ctv = "turbo3",
    [int]$Ngl = 99,
    [int]$NcMoe = 0,
    [string]$FitTarget = "",
    [string]$ExtraArgs = "",
    [int]$Port = 8081,
    [int]$GenTokens = 64,
    [int]$Runs = 3,
    [int]$LoadTimeout = 600,
    [switch]$Profile,
    [switch]$Trace,
    [switch]$SchedDebug,
    [switch]$StartRemusRpc,
    [switch]$StopRemusRpc,
    [switch]$EnsurePathbRpc,
    [switch]$MonitorRocm,
    [string]$RemusIp = "192.168.8.176",
    [string]$RemusPass = ""
)

$ErrorActionPreference = "Stop"
$CollateralRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $CollateralRoot "..\..")).Path
$Portable = Join-Path $RepoRoot "build-cuda-b-bin\portable"
$BinDir = if (Test-Path (Join-Path $Portable "llama-server.exe")) { $Portable } else { Join-Path $RepoRoot "build-cuda-b-bin\bin" }
$BenchRoot = Join-Path $RepoRoot "docs\cuda-windows-5070ti\benchmarks"
$Stamp = if ($Label) { $Label } else { Get-Date -Format "yyyyMMdd-HHmmss" }
$LogDir = Join-Path $BenchRoot $Stamp
$InvokeWsl = Join-Path $CollateralRoot "invoke-wsl.ps1"

function Invoke-WslBench {
    param([string]$Cmd)
    & $InvokeWsl -BashCommand $Cmd -RemusIp $RemusIp -RemusPass $RemusPass
}

switch ($Config) {
    "config-e" {
        if (-not $RpcEndpoint) { $RpcEndpoint = "${RemusIp}:50051" }
        if (-not $TensorSplit) { $TensorSplit = "50,50" }
        if (-not $FitTarget) { $FitTarget = "1024,1024" }
    }
    "config-f" {
        if (-not $RpcEndpoint) { $RpcEndpoint = "${RemusIp}:50051,${RemusIp}:50052" }
        if (-not $TensorSplit) { $TensorSplit = "30,12,58" }
        if (-not $FitTarget) { $FitTarget = "1024,1024,1024" }
        $MonitorRocm = $true
    }
    default { throw "Unknown -Config $Config (use config-e or config-f)" }
}

if (-not $ModelPath) {
    $candidates = @(
        "Qwen3.5-9B-MTP-Q4_K_M.gguf",
        "Qwen3.5-4B-Q4_K_M.gguf",
        "gemma-4-E4B.i1-Q4_K_M.gguf"
    )
    foreach ($name in $candidates) {
        $hit = Get-ChildItem -Path $ModelsRoot -Recurse -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { $ModelPath = $hit.FullName; break }
    }
    if (-not $ModelPath) { throw "No default model found under $ModelsRoot" }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Meta = Join-Path $LogDir "result.meta"
$Result = Join-Path $LogDir "bench.result"
$ServerLog = Join-Path $LogDir "server.log"
$ServerLogErr = Join-Path $LogDir "server.log.err"
$GpuLog = Join-Path $LogDir "gpu-monitor.log"
$TelDir = Join-Path $LogDir "telemetry"
$PhaseLog = Join-Path $TelDir "phase.log"
$serverExe = Join-Path $BinDir "llama-server.exe"

function Log-Meta([string]$Line) {
    if ($Line -match '^[\x00-\x08\x0B\x0C\x0E-\x1F]*$') { return }
    Write-Host $Line
    Add-Content -Path $Meta -Value $Line -Encoding utf8
}

if ($Profile -or $Trace) {
    New-Item -ItemType Directory -Force -Path $TelDir | Out-Null
    if ($GenTokens -lt 128) { $GenTokens = 256 }
    if ($Runs -lt 1) { $Runs = 1 }
}
if ($Trace -or $Profile) {
    if (-not $env:GGML_PIPELINE_PLUS) { $env:GGML_PIPELINE_PLUS = "1" }
}
if ($Trace) {
    $rpcTrace = Join-Path $TelDir "rpc-trace.jsonl"
    $schedTrace = Join-Path $TelDir "sched-trace.jsonl"
    "" | Set-Content -Path $rpcTrace -Encoding utf8 -NoNewline
    "" | Set-Content -Path $schedTrace -Encoding utf8 -NoNewline
    $env:GGML_RPC_TRACE = "1"
    $env:GGML_SCHED_TRACE = "1"
    $env:GGML_RPC_TRACE_FILE = $rpcTrace
    $env:GGML_SCHED_TRACE_FILE = $schedTrace
    Log-Meta "trace: rpc=$rpcTrace sched=$schedTrace"
}
if (-not (Test-Path $serverExe)) { throw "Missing $serverExe - run build.ps1 first" }

function Invoke-VramCalc {
    param([string]$WslModelPath)
    $wslErr = "/mnt/" + ((Join-Path $LogDir "vram-calc.err") -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/'
    $wslErr = $wslErr.ToLower()
    $out = Invoke-WslBench "python3 rpc-patch/scripts/pathb-72b-vram-calc.py --config $Config --gguf '$WslModelPath' --ctx $Ctx 2>'$wslErr'"
    $out | Set-Content -Path $vramCalcOut -Encoding utf8
    $errLocal = Join-Path $LogDir "vram-calc.err"
    if (Test-Path $errLocal) {
        Get-Content $errLocal -Encoding utf8 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^WARN:' } | ForEach-Object { Log-Meta $_ }
    }
}

Log-Meta "=== $Stamp ==="
Log-Meta "config=$Config endpoint=$RpcEndpoint model=$ModelPath"
$ncMoeMeta = if ($NcMoe -gt 0) { "$NcMoe" } else { "none" }
Log-Meta "ctx=$Ctx ctk=$Ctk ctv=$Ctv ngl=$Ngl ncmoe=$ncMoeMeta ts=$TensorSplit fitt=$FitTarget"
Log-Meta "extra=$ExtraArgs"

nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-before.txt") | Out-Null

# VRAM calc via WSL
$wslModel = "/mnt/" + ($ModelPath -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/'
$wslModel = $wslModel.ToLower()
$vramCalcOut = Join-Path $LogDir "vram-calc.txt"
Invoke-VramCalc -WslModelPath $wslModel
if (Test-Path $vramCalcOut) {
    Get-Content $vramCalcOut -Encoding utf8 | ForEach-Object { Write-Host $_ }
}

if ($EnsurePathbRpc -or $StartRemusRpc) {
    $rp = if ($RemusPass) { $RemusPass } elseif ($env:PATHB_REMUS_SSH_PASS) { $env:PATHB_REMUS_SSH_PASS } else { "12345" }
    Invoke-WslBench "sshpass -p $rp ssh -o StrictHostKeyChecking=accept-new hunter@${RemusIp} 'docker stop rx6600-rpc 2>/dev/null || true'"
}
if ($StartRemusRpc -or $EnsurePathbRpc) {
    if ($Config -eq "config-f") {
        Invoke-WslBench "chmod +x rpc-patch/scripts/pathb-remus-multi-rpc-win.sh rpc-patch/scripts/pathb-remus-rpc.sh rpc-patch/scripts/pathb-remus-rx6600-rpc.sh; ./rpc-patch/scripts/pathb-remus-multi-rpc-win.sh start"
    } else {
        Invoke-WslBench "chmod +x rpc-patch/scripts/pathb-remus-rpc.sh; ./rpc-patch/scripts/pathb-remus-rpc.sh start"
    }
}

foreach ($ep in ($RpcEndpoint -split ',')) {
    $hostPart = ($ep -split ':')[0]
    $portPart = ($ep -split ':')[1]
    $tn = Test-NetConnection -ComputerName $hostPart -Port $portPart -WarningAction SilentlyContinue
    if (-not $tn.TcpTestSucceeded) {
        throw "RPC not reachable: $ep"
    }
    Log-Meta "RPC reachable: $ep"
}

function Write-Phase([string]$Phase) {
    if (-not $Profile) { return }
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $PhaseLog -Value "$ts $Phase" -Encoding utf8
    Log-Meta "phase=$Phase"
}

$monDur = $LoadTimeout + 180
$monRocmFlag = if ($MonitorRocm) { "1" } else { "0" }
$wslGpuLog = "/mnt/" + ($GpuLog -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/'
$wslGpuLog = $wslGpuLog.ToLower()
$wslTelDir = if ($Profile) {
    "/mnt/" + ($TelDir -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/'
    ($("/mnt/" + ($TelDir -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/').ToLower())
} else { "" }

$rp = if ($RemusPass) { $RemusPass } elseif ($env:PATHB_REMUS_SSH_PASS) { $env:PATHB_REMUS_SSH_PASS } else { "12345" }

if ($Profile) {
    Write-Phase "LOAD_START"
    $monJobs = @()
    $cpuMon = Join-Path $CollateralRoot "pathb-cpu-monitor-win.ps1"
    $monJobs += Start-Job -FilePath $cpuMon -ArgumentList $TelDir, $monDur, 500, 0
} else {
    $monJob = Start-Job -ScriptBlock {
        param($InvokeWsl, $WslGpuLog, $Dur, $Ip, $Pass, $Rocm)
        & $InvokeWsl -BashCommand "chmod +x rpc-patch/scripts/pathb-gpu-monitor-win.sh; PATHB_MONITOR_REMUS_ROCM=$Rocm ./rpc-patch/scripts/pathb-gpu-monitor-win.sh '$WslGpuLog' $Dur" -RemusIp $Ip -RemusPass $Pass
    } -ArgumentList $InvokeWsl, $wslGpuLog, $monDur, $RemusIp, $rp, $monRocmFlag
}

if ($SchedDebug) { $env:GGML_SCHED_DEBUG = "1" }

$serverArgs = @(
    "--rpc", $RpcEndpoint,
    "-m", $ModelPath,
    "-ngl", "$Ngl",
    "-c", "$Ctx",
    "-ctk", $Ctk,
    "-ctv", $Ctv,
    "-sm", "layer",
    "-ts", $TensorSplit,
    "--fit", "off",
    "--fit-target", $FitTarget,
    "--verbose",
    "-lv", "4",
    "--no-warmup",
    "-np", "1",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
if ($NcMoe -gt 0) {
    $serverArgs += @("--n-cpu-moe", "$NcMoe")
}
if ($ExtraArgs) {
    $serverArgs += ($ExtraArgs -split '\s+')
}

function Get-ServerLogText {
    $parts = @()
    foreach ($p in @($ServerLogErr, $ServerLog)) {
        if (Test-Path $p) {
            $parts += Get-Content $p -Raw -Encoding utf8 -ErrorAction SilentlyContinue
        }
    }
    return ($parts -join "`n")
}

if ($Profile) {
    $monJobs += Start-Job -ScriptBlock {
        param($InvokeWsl, $WslTel, $Dur, $Ip, $Pass, $Rocm)
        & $InvokeWsl -BashCommand "sed -i 's/\r$//' rpc-patch/scripts/pathb-profile-gpu-monitor.sh 2>/dev/null; chmod +x rpc-patch/scripts/pathb-profile-gpu-monitor.sh; PATHB_MONITOR_REMUS_ROCM=$Rocm ./rpc-patch/scripts/pathb-profile-gpu-monitor.sh '$WslTel' $Dur" -RemusIp $Ip -RemusPass $Pass
    } -ArgumentList $InvokeWsl, $wslTelDir, $monDur, $RemusIp, $rp, $monRocmFlag
}

Log-Meta "starting llama-server: $serverExe $($serverArgs -join ' ')"
$proc = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -PassThru `
    -RedirectStandardOutput $ServerLog -RedirectStandardError $ServerLogErr `
    -WorkingDirectory $BinDir -WindowStyle Hidden

$ready = $false
for ($i = 0; $i -lt [math]::Ceiling($LoadTimeout / 5); $i++) {
    Start-Sleep -Seconds 5
    $logText = Get-ServerLogText
    if ($logText -match "model loaded|server is listening") {
        $ready = $true
        Log-Meta "server ready at $($i * 5)s"
        if ($Profile) { Write-Phase "LOAD_END" }
        break
    }
    if ($proc.HasExited) {
        Get-Content $ServerLogErr -Tail 40 -Encoding utf8 -ErrorAction SilentlyContinue
        throw "llama-server exited during load (code $($proc.ExitCode))"
    }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200 -and $logText -match "llama_server") {
            $ready = $true
            Log-Meta "server ready at $($i * 5)s (health ok)"
            if ($Profile) { Write-Phase "LOAD_END" }
            break
        }
    } catch { }
    if ($i % 6 -eq 0 -and $i -gt 0) { Log-Meta "poll $($i*5)s: still loading..." }
}

if (-not $ready) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Get-Content $ServerLogErr -Tail 30 -Encoding utf8 -ErrorAction SilentlyContinue
    throw "Load timeout after ${LoadTimeout}s"
}

nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-loaded.txt") | Out-Null

Select-String -Path @($ServerLogErr, $ServerLog) -Pattern 'MiB|load_tensors|offload|assign|buffer|tensor|RPC[0-9]|CUDA0|memory breakdown|layer|device' -CaseSensitive:$false -ErrorAction SilentlyContinue |
    ForEach-Object { $_.Line } |
    Where-Object { $_ -notmatch '^[\|\\/\-]+$' -and $_ -notmatch '^\s*$' } |
    Select-Object -Last 40 |
    ForEach-Object { Log-Meta $_ }

Log-Meta "benchmark $Runs runs x $GenTokens tokens..."
Set-Content -Path $Result -Value ""
$prompt = "The quick brown fox jumps over the lazy dog. Explain in detail the history of computing from Babbage to modern GPUs."
for ($run = 1; $run -le $Runs; $run++) {
    if ($Profile) { Write-Phase "GEN_RUN_$run" }
    $body = @{
        messages = @(@{ role = "user"; content = $prompt })
        max_tokens = $GenTokens
    } | ConvertTo-Json -Depth 5 -Compress
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
    $t = $resp.timings
    $preview = $resp.choices[0].message.content
    if ($preview.Length -gt 80) { $preview = $preview.Substring(0, 80) }
    $line = "run=$run P=$([math]::Round($t.prompt_per_second,1)) G=$([math]::Round($t.predicted_per_second,1)) preview=$preview"
    Log-Meta $line
    Add-Content -Path $Result -Value $line
}

if ($Profile) { Write-Phase "GEN_END" }

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
if ($Profile) {
    # Let telemetry jobs flush GEN-window samples
    Start-Sleep -Seconds 45
} else {
    Start-Sleep -Seconds 2
}
if ($Profile) {
    foreach ($j in $monJobs) {
        Stop-Job $j -ErrorAction SilentlyContinue
        Remove-Job $j -Force -ErrorAction SilentlyContinue
    }
    $parse = Join-Path $CollateralRoot "pathb-profile-parse.ps1"
    if (Test-Path $parse) { & $parse -ProfileDir $LogDir }
    if ($Trace) {
        $traceParse = Join-Path $CollateralRoot "pathb-rpc-trace-parse.ps1"
        if (Test-Path $traceParse) { & $traceParse -TraceDir $TelDir }
    }
} elseif ($Trace) {
    Start-Sleep -Seconds 2
    $traceParse = Join-Path $CollateralRoot "pathb-rpc-trace-parse.ps1"
    if (Test-Path $traceParse) { & $traceParse -TraceDir $TelDir }
} else {
    Stop-Job $monJob -ErrorAction SilentlyContinue
    Remove-Job $monJob -Force -ErrorAction SilentlyContinue
}

nvidia-smi | Tee-Object -FilePath (Join-Path $LogDir "nvidia-smi-after.txt") | Out-Null
Invoke-WslBench "./rpc-patch/scripts/pathb-remus-rpc.sh logs" | Out-File (Join-Path $LogDir "rpc-remus.log")

if ($StopRemusRpc) {
    if ($Config -eq "config-f") {
        Invoke-WslBench "./rpc-patch/scripts/pathb-remus-multi-rpc-win.sh stop"
    } else {
        Invoke-WslBench "./rpc-patch/scripts/pathb-remus-rpc.sh stop"
    }
}

Log-Meta "RESULT=PASS"
Log-Meta "logs=$LogDir"
Write-Host "BENCH_OK -> $LogDir"