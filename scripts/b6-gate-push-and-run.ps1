# Push b6 scripts to romulus and run a profiler label.
# Primary repo: D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant
param(
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [string]$RomulusHost = "hunter@192.168.8.108",
    [string]$RomulusPass = "12345",
    [string]$RemoteRoot = "/home/hunter/atomic-llama-cpp-turboquant"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function ConvertTo-WslPath {
    param([string]$WinPath)
    $p = $WinPath -replace '\\', '/'
    if ($p -match '^([A-Za-z]):/(.*)$') {
        return "/mnt/$($Matches[1].ToLower())/$($Matches[2])"
    }
    return $p
}

$WslRepo = ConvertTo-WslPath $RepoRoot

$files = @(
    "scripts/b6-gate-run-remote.sh",
    "scripts/b6-gate-profiler-romulus.sh",
    "scripts/b6-gate-ts-sweep-4gpu.sh",
    "scripts/b6-gate-diagnose-runs.sh",
    "scripts/llama-pipeline-profiler-cluster.sh",
    "benches/path-b-plus/prompts/profiler-reasoning-long.txt"
)

Write-Host "=== scp b6 collateral to romulus (from $RepoRoot) ==="
foreach ($rel in $files) {
    $local = Join-Path $RepoRoot $rel
    if (-not (Test-Path $local)) { throw "missing: $local" }
    $wslLocal = "$WslRepo/$($rel -replace '\\','/')"
    $remoteDir = "$RemoteRoot/$(Split-Path $rel -Parent)".Replace("\", "/")
    wsl -e bash -lc "sshpass -p '$RomulusPass' ssh -o StrictHostKeyChecking=no $RomulusHost mkdir -p '$remoteDir'"
    wsl -e bash -lc "sshpass -p '$RomulusPass' scp -o StrictHostKeyChecking=no '$wslLocal' ${RomulusHost}:${RemoteRoot}/$($rel -replace '\\','/')"
}

Write-Host "=== run $Label on romulus ==="
wsl -e bash -lc "sshpass -p '$RomulusPass' ssh -o StrictHostKeyChecking=no $RomulusHost 'chmod +x $RemoteRoot/scripts/b6-gate-run-remote.sh $RemoteRoot/scripts/b6-gate-ts-sweep-4gpu.sh $RemoteRoot/scripts/b6-gate-diagnose-runs.sh && bash $RemoteRoot/scripts/b6-gate-run-remote.sh $Label'"