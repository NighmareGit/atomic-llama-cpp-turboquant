# Parse GGML_RPC_TRACE + GGML_SCHED_TRACE jsonl into stall budget summary.
param(
    [Parameter(Mandatory = $true)]
    [string]$TraceDir
)

$ErrorActionPreference = "Stop"
$rpcFile = Join-Path $TraceDir "rpc-trace.jsonl"
$schedFile = Join-Path $TraceDir "sched-trace.jsonl"
$outFile = Join-Path $TraceDir "trace-summary.txt"

$lines = @()
$lines += "=== trace summary ==="
$lines += "dir=$TraceDir"

# Matches rpc_cmd enum in ggml/src/ggml-rpc/ggml-rpc.cpp
$cmdNames = @{
    0  = "ALLOC_BUFFER"
    1  = "GET_ALIGNMENT"
    2  = "GET_MAX_SIZE"
    3  = "BUFFER_GET_BASE"
    4  = "FREE_BUFFER"
    5  = "BUFFER_CLEAR"
    6  = "SET_TENSOR"
    7  = "SET_TENSOR_HASH"
    8  = "GET_TENSOR"
    9  = "COPY_TENSOR"
    10 = "GRAPH_COMPUTE"
    11 = "GET_DEVICE_MEMORY"
    12 = "INIT_TENSOR"
    13 = "GET_ALLOC_SIZE"
    14 = "HELLO"
    15 = "DEVICE_COUNT"
    16 = "GRAPH_RECOMPUTE"
    17 = "SET_TENSOR_BATCH"
    18 = "EVENT_RECORD"
    19 = "COPY_TENSOR_PEER"
}

if (Test-Path $rpcFile) {
    $rpcRows = Get-Content $rpcFile | ForEach-Object {
        try { $_ | ConvertFrom-Json } catch { $null }
    } | Where-Object { $_ }
    $lines += ""
    $lines += "[rpc] events=$($rpcRows.Count)"
    if ($rpcRows.Count -gt 0) {
        $byCmd = $rpcRows | Group-Object cmd
        foreach ($g in ($byCmd | Sort-Object Name)) {
            $name = if ($cmdNames.ContainsKey([int]$g.Name)) { $cmdNames[[int]$g.Name] } else { "cmd$($g.Name)" }
            $us = @($g.Group | ForEach-Object { [int64]$_.elapsed_us })
            $lines += "  $name count=$($g.Count) total_ms=$([math]::Round(($us | Measure-Object -Sum).Sum/1000,2)) avg_us=$([math]::Round(($us | Measure-Object -Average).Average,0))"
        }
        $blocking = $rpcRows | Where-Object { $_.blocking -eq "true" }
        $lines += "  blocking_events=$($blocking.Count) blocking_ms=$([math]::Round((@($blocking | ForEach-Object { [int64]$_.elapsed_us }) | Measure-Object -Sum).Sum/1000,2))"
        $drain = $rpcRows | Where-Object { $_.fn -match "drain|flush" }
        if ($drain) {
            $lines += "  drain_flush_ms=$([math]::Round((@($drain | ForEach-Object { [int64]$_.elapsed_us }) | Measure-Object -Sum).Sum/1000,2))"
        }
        $drainCopy = $rpcRows | Where-Object { $_.phase -eq "drain_copy" }
        if ($drainCopy) {
            $lines += "  drain_copy_count=$($drainCopy.Count) drain_copy_ms=$([math]::Round((@($drainCopy | ForEach-Object { [int64]$_.elapsed_us }) | Measure-Object -Sum).Sum/1000,2))"
        }
        $hello = $rpcRows | Where-Object { $_.phase -eq "hello" }
        if ($hello) {
            foreach ($h in $hello) {
                $ep = if ($h.endpoint) { $h.endpoint } else { "?" }
                $lines += "  hello endpoint=$ep minor=$($h.minor) peer_copy=$($h.peer_copy)"
            }
        }
        $copyIssue = $rpcRows | Where-Object { $_.phase -eq "copy_issue" }
        if ($copyIssue) {
            $defer = @($copyIssue | Where-Object { $_.defer -eq "true" })
            $peer = @($copyIssue | Where-Object { $_.peer_copy -eq "true" })
            $lines += "  copy_issue_count=$($copyIssue.Count) defer_count=$($defer.Count) peer_copy_count=$($peer.Count)"
        }
    }
} else {
    $lines += ""
    $lines += "[rpc] missing $rpcFile"
}

if (Test-Path $schedFile) {
    $schedRows = Get-Content $schedFile | ForEach-Object {
        try { $_ | ConvertFrom-Json } catch { $null }
    } | Where-Object { $_ }
    $lines += ""
    $lines += "[sched] events=$($schedRows.Count)"
    if ($schedRows.Count -gt 0) {
        $splits = $schedRows | Where-Object { $_.phase -eq "split_total" }
        $lines += "  split_total_count=$($splits.Count)"
        if ($splits.Count -gt 0) {
            $us = @($splits | ForEach-Object { [int64]$_.elapsed_us })
            $lines += "  split_total_ms_sum=$([math]::Round(($us | Measure-Object -Sum).Sum/1000,2)) avg_us=$([math]::Round(($us | Measure-Object -Average).Average,0))"
            $byBackend = $splits | Group-Object backend
            foreach ($g in $byBackend) {
                $bu = @($g.Group | ForEach-Object { [int64]$_.elapsed_us })
                $lines += "    backend$($g.Name) splits=$($g.Count) ms=$([math]::Round(($bu | Measure-Object -Sum).Sum/1000,2))"
            }
        }
        foreach ($phase in @("input_wait_copy", "graph_compute_async", "event_record")) {
            $p = $schedRows | Where-Object { $_.phase -eq $phase }
            if ($p) {
                $pu = @($p | ForEach-Object { [int64]$_.elapsed_us })
                $lines += "  ${phase}_ms=$([math]::Round(($pu | Measure-Object -Sum).Sum/1000,2))"
            }
        }
        $splitEv = @($schedRows | Where-Object { $_.phase -eq "split_total" })
        if ($splitEv.Count -ge 2) {
            $overlap = 0
            $pairs = 0
            for ($i = 0; $i -lt $splitEv.Count; $i++) {
                $a = $splitEv[$i]
                $aStart = [int64]$a.ts_us
                $aEnd = $aStart + [int64]$a.elapsed_us
                for ($j = $i + 1; $j -lt $splitEv.Count; $j++) {
                    $b = $splitEv[$j]
                    if ($a.backend -eq $b.backend) { continue }
                    $bStart = [int64]$b.ts_us
                    if ($bStart -ge $aStart -and $bStart -lt $aEnd) {
                        $overlap++
                    }
                    $pairs++
                }
            }
            $pct = if ($pairs -gt 0) { [math]::Round(100.0 * $overlap / $pairs, 1) } else { 0 }
            $lines += "  assembly_overlap_count=$overlap pairs=$pairs overlap_pct=$pct"
        }
        $byCopy = $splitEv | Group-Object copy
        if ($byCopy.Count -gt 1) {
            foreach ($g in $byCopy) {
                $lines += "    copy$($g.Name) splits=$($g.Count)"
            }
        }
    }
} else {
    $lines += ""
    $lines += "[sched] missing $schedFile"
}

$lines | Set-Content $outFile -Encoding utf8
$lines | ForEach-Object { Write-Host $_ }
Write-Host "summary -> $outFile"