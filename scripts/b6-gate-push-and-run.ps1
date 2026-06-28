# Push b6 scripts to romulus and run a profiler label.
param(
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [string]$RomulusHost = "hunter@192.168.8.108",
    [string]$RomulusPass = "12345",
    [string]$RemoteRoot = "/home/hunter/atomic-llama-cpp-turboquant"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WslRepo = "/mnt/c/Users/nightmare/.grok/worktrees/atomic-llama-cpp-5070ti-atomic-llama-cpp-turboquant/path-b-plus-5070ti-triton"

$files = @(
    "scripts/b6-gate-run-remote.sh",
    "scripts/b6-gate-profiler-romulus.sh",
    "benches/path-b-plus/prompts/profiler-reasoning-long.txt"
)

Write-Host "=== scp b6 collateral to romulus ==="
foreach ($rel in $files) {
    $local = Join-Path $RepoRoot $rel
    if (-not (Test-Path $local)) { throw "missing: $local" }
    $remoteDir = "$RemoteRoot/$(Split-Path $rel -Parent)".Replace("\", "/")
    wsl -e bash -lc "sshpass -p '$RomulusPass' ssh -o StrictHostKeyChecking=no $RomulusHost mkdir -p '$remoteDir'"
    wsl -e bash -lc "sshpass -p '$RomulusPass' scp -o StrictHostKeyChecking=no '$($local -replace '\\','/')' ${RomulusHost}:${RemoteRoot}/$($rel -replace '\\','/')"
}

Write-Host "=== run $Label on romulus ==="
wsl -e bash -lc "sshpass -p '$RomulusPass' ssh -o StrictHostKeyChecking=no $RomulusHost 'chmod +x $RemoteRoot/scripts/b6-gate-run-remote.sh && bash $RemoteRoot/scripts/b6-gate-run-remote.sh $Label'"