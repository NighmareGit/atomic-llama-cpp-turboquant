# Bridge PowerShell -> WSL bash with repo paths and remus SSH env.
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$BashCommand,
    [string]$RemusPass = "",
    [string]$RemusIp = "192.168.8.176"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$WslRepo = "/mnt/" + ($RepoRoot -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/'
$WslRepo = $WslRepo.ToLower()

$pass = if ($RemusPass) { $RemusPass } elseif ($env:PATHB_REMUS_SSH_PASS) { $env:PATHB_REMUS_SSH_PASS } else { "12345" }

$exports = @(
    "export LLAMA_TURBOQUANT_ROOT='$WslRepo'",
    "export PATHB_REMUS_SSH_PASS='$pass'",
    "export REMUS_RPC_IP='$RemusIp'",
    "export MODELS_ROOT='/mnt/d/models'",
    "cd '$WslRepo'"
) -join "; "

$full = "$exports; $BashCommand"
& wsl.exe bash -lc $full