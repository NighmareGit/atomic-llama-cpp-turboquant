# Parse profile telemetry -> TDP-normalized summary + cross-path verdict.
param(
    [Parameter(Mandatory = $true)]
    [string]$ProfileDir
)

$ErrorActionPreference = "Stop"
$TelDir = Join-Path $ProfileDir "telemetry"
$OutFile = Join-Path $ProfileDir "profile-summary.txt"

if (-not (Test-Path $TelDir)) {
    throw "No telemetry dir: $TelDir"
}

$Tdp = @{
    "5070" = 300
    "5060" = 175
    "6600" = 140
}

function Get-DutyCycle {
    param([double[]]$Powers, [double]$Tdp, [double]$ThresholdPct = 20)
    if ($Powers.Count -eq 0) { return 0 }
    $thresh = $Tdp * ($ThresholdPct / 100)
    $active = ($Powers | Where-Object { $_ -ge $thresh }).Count
    return [math]::Round(100 * $active / $Powers.Count, 1)
}

$lines = @()
$lines += "=== profile summary: $(Split-Path $ProfileDir -Leaf) ==="
$lines += "generated: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"

# Bench result
$bench = Join-Path $ProfileDir "bench.result"
if (Test-Path $bench) {
    $lines += ""
    $lines += "[throughput]"
    Get-Content $bench | ForEach-Object { $lines += "  $_" }
}

# Windows 5070 from query CSV
$winCsv = Join-Path $TelDir "win-5070-dmon.csv"
if (Test-Path $winCsv) {
    $rows = Import-Csv $winCsv -ErrorAction SilentlyContinue
    if ($rows) {
        $powers = @($rows | ForEach-Object { [double]($_.power_w -replace ' W','' -replace ' ','') } | Where-Object { $_ -gt 0 })
        $utils = @($rows | ForEach-Object { [double]$_.util_gpu_pct } | Where-Object { $_ -ge 0 })
        $lines += ""
        $lines += "[win-5070] samples=$($rows.Count)"
        if ($powers.Count -gt 0) {
            $lines += "  power: min=$([math]::Round(($powers | Measure-Object -Minimum).Minimum,1))W max=$([math]::Round(($powers | Measure-Object -Maximum).Maximum,1))W avg=$([math]::Round(($powers | Measure-Object -Average).Average,1))W"
            $lines += "  duty_cycle_20pct_tdp=$(Get-DutyCycle $powers $Tdp['5070'])%"
            $lines += "  peak_pct_tdp=$([math]::Round(100 * ($powers | Measure-Object -Maximum).Maximum / $Tdp['5070'],1))%"
        }
        if ($utils.Count -gt 0) {
            $lines += "  util_gpu: max=$([math]::Round(($utils | Measure-Object -Maximum).Maximum,1))% avg=$([math]::Round(($utils | Measure-Object -Average).Average,1))%"
        }
    }
}

# remus 5060 dmon - parse power from CSV if present
$remusCsv = Join-Path $TelDir "remus-5060-dmon.csv"
if (Test-Path $remusCsv) {
    $content = Get-Content $remusCsv -Raw
    $pwrMatches = [regex]::Matches($content, '(?m)^\s*0\s+(\d+(?:\.\d+)?)')
    $powers = @($pwrMatches | ForEach-Object { [double]$_.Groups[1].Value })
    $lines += ""
    $lines += "[remus-5060] power_samples=$($powers.Count)"
    if ($powers.Count -gt 0) {
        $lines += "  power: min=$([math]::Round(($powers | Measure-Object -Minimum).Minimum,1))W max=$([math]::Round(($powers | Measure-Object -Maximum).Maximum,1))W avg=$([math]::Round(($powers | Measure-Object -Average).Average,1))W"
        $lines += "  duty_cycle_20pct_tdp=$(Get-DutyCycle $powers $Tdp['5060'])%"
        $lines += "  peak_pct_tdp=$([math]::Round(100 * ($powers | Measure-Object -Maximum).Maximum / $Tdp['5060'],1))%"
    }
}

# remus 6600 rocm jsonl
$rocmJsonl = Join-Path $TelDir "remus-6600-rocm.jsonl"
if (Test-Path $rocmJsonl) {
    $useVals = @()
    $pwrVals = @()
    Get-Content $rocmJsonl | ForEach-Object {
        if ($_ -match '"GPU use \(%\)":\s*"(\d+(?:\.\d+)?)"') { $useVals += [double]$Matches[1] }
        if ($_ -match '"Average Graphics Package Power \(W\)":\s*"(\d+(?:\.\d+)?)"') { $pwrVals += [double]$Matches[1] }
    }
    $lines += ""
    $lines += "[remus-6600] samples=$($useVals.Count)"
    if ($pwrVals.Count -gt 0) {
        $lines += "  power: max=$([math]::Round(($pwrVals | Measure-Object -Maximum).Maximum,1))W avg=$([math]::Round(($pwrVals | Measure-Object -Average).Average,1))W"
        $lines += "  duty_cycle_20pct_tdp=$(Get-DutyCycle $pwrVals $Tdp['6600'])%"
    }
    if ($useVals.Count -gt 0) {
        $lines += "  gpu_use: max=$([math]::Round(($useVals | Measure-Object -Maximum).Maximum,1))% avg=$([math]::Round(($useVals | Measure-Object -Average).Average,1))%"
    }
}

# RAM LOAD vs GEN from phase.log timestamps
$phaseLog = Join-Path $TelDir "phase.log"
$loadEnd = $null
$genEnd = $null
if (Test-Path $phaseLog) {
    Get-Content $phaseLog | ForEach-Object {
        if ($_ -match '^(\S+)\s+LOAD_END') { $loadEnd = [datetime]::ParseExact($Matches[1], 'yyyy-MM-ddTHH:mm:ssZ', $null).ToUniversalTime() }
        if ($_ -match '^(\S+)\s+GEN_END') { $genEnd = [datetime]::ParseExact($Matches[1], 'yyyy-MM-ddTHH:mm:ssZ', $null).ToUniversalTime() }
    }
}

$memCsv = Join-Path $TelDir "win-cpu-memory.csv"
if (Test-Path $memCsv) {
    $memRows = Import-Csv $memCsv | ForEach-Object {
        $row = $_
        $row | Add-Member -NotePropertyName uts -NotePropertyValue ([datetime]::ParseExact($_.timestamp, 'yyyy-MM-ddTHH:mm:ssZ', $null).ToUniversalTime()) -Force
        $row
    }
    $loadRows = if ($loadEnd) { $memRows | Where-Object { $_.uts -lt $loadEnd } } else { @() }
    $genRows = if ($loadEnd) { $memRows | Where-Object { $_.uts -ge $loadEnd } } else { $memRows }
    $lines += ""
    $lines += "[ram]"
    if ($loadRows) {
        $avail = @($loadRows | ForEach-Object { [double]$_.available_mb })
        $pps = @($loadRows | ForEach-Object { [double]$_.pages_per_sec })
        $lines += "  LOAD: min_avail=$([math]::Round(($avail | Measure-Object -Minimum).Minimum))MB max_pages_sec=$([math]::Round(($pps | Measure-Object -Maximum).Maximum,1))"
    }
    if ($genRows) {
        $avail = @($genRows | ForEach-Object { [double]$_.available_mb })
        $pps = @($genRows | ForEach-Object { [double]$_.pages_per_sec })
        $lines += "  GEN: min_avail=$([math]::Round(($avail | Measure-Object -Minimum).Minimum))MB max_pages_sec=$([math]::Round(($pps | Measure-Object -Maximum).Maximum,1))"
    }
}

# NIC headroom
$nicCsv = Join-Path $TelDir "win-nic.csv"
if (Test-Path $nicCsv) {
    $nicRows = Import-Csv $nicCsv
    $bytes = @($nicRows | ForEach-Object { [double]$_.bytes_total_per_sec })
    if ($bytes.Count -gt 0) {
        $peakMbps = [math]::Round(($bytes | Measure-Object -Maximum).Maximum * 8 / 1MB, 2)
        $lines += ""
        $lines += "[nic] peak_mbps=$peakMbps (2.5G ceiling=2500)"
        $lines += "  nic_headroom_pct=$([math]::Round(100 * (1 - $peakMbps / 2500), 1))"
        $lines += "  nic_ok=$($peakMbps -lt 500)"
    }
}

# Cross-path verdict (simplified)
$lines += ""
$lines += "[verdict]"
$lines += "  Primary classification requires scheduler trace + E-vs-F delta."
$lines += "  If all GPU duty_cycle < 20% during GEN and NIC/PCIe headroom > 99%: orchestration/RPC stall likely."

$lines | Set-Content $OutFile -Encoding utf8
$lines | ForEach-Object { Write-Host $_ }
Write-Host "summary -> $OutFile"