# Windows host CPU + RAM telemetry during profile bench.
param(
    [Parameter(Mandatory = $true)]
    [string]$TelemetryDir,
    [int]$DurationSec = 600,
    [int]$IntervalMs = 500,
    [int]$ServerPid = 0
)

$ErrorActionPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path $TelemetryDir | Out-Null

$cpuOut = Join-Path $TelemetryDir "win-cpu-memory.csv"
$procOut = Join-Path $TelemetryDir "win-llama-server.csv"
$nicOut = Join-Path $TelemetryDir "win-nic.csv"
$gpuOut = Join-Path $TelemetryDir "win-5070-dmon.csv"

"timestamp,phase,available_mb,pages_per_sec,pages_input_per_sec" | Set-Content $cpuOut -Encoding utf8
"timestamp,cpu_pct,threads,working_set_mb" | Set-Content $procOut -Encoding utf8
"timestamp,bytes_total_per_sec" | Set-Content $nicOut -Encoding utf8
"timestamp,power_w,util_gpu_pct,util_mem_pct,sm_clock_mhz,mem_clock_mhz,mem_used_mib,mem_total_mib" | Set-Content $gpuOut -Encoding utf8

$end = (Get-Date).AddSeconds($DurationSec)
while ((Get-Date) -lt $end) {
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $avail = (Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
    $pps = (Get-Counter '\Memory\Pages/sec' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
    $pin = (Get-Counter '\Memory\Pages Input/sec' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
    $phase = "MON"
    if ($ServerPid -gt 0) {
        $p = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
        if ($p) { $phase = "GEN" }
    }
    Add-Content $cpuOut "$ts,$phase,$([math]::Round($avail)),$([math]::Round($pps,2)),$([math]::Round($pin,2))"

    if ($ServerPid -gt 0) {
        $p = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
        if ($p) {
            Add-Content $procOut "$ts,$([math]::Round($p.CPU,2)),$($p.Threads.Count),$([math]::Round($p.WorkingSet64/1MB,1))"
        }
    }

    $nic = (Get-Counter '\Network Interface(*)\Bytes Total/sec' -ErrorAction SilentlyContinue).CounterSamples |
        Where-Object { $_.InstanceName -notmatch 'isatap|loopback|teredo|vEthernet|WSL' } |
        Measure-Object -Property CookedValue -Sum
    if ($nic) {
        Add-Content $nicOut "$ts,$([math]::Round($nic.Sum,0))"
    }

    $gq = & nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    if ($gq) {
        $v = ($gq -split ',') | ForEach-Object { $_.Trim() }
        if ($v.Count -ge 7) {
            Add-Content $gpuOut "$ts,$($v[0]),$($v[1]),$($v[2]),$($v[3]),$($v[4]),$($v[5]),$($v[6])"
        }
    }

    Start-Sleep -Milliseconds $IntervalMs
}