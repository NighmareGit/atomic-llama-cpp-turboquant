# Forwarder to canonical cuda-windows build (5070 Ti profile).
param(
    [switch]$Clean,
    [switch]$SkipBuild,
    [switch]$Reconfigure,
    [switch]$SkipGgml,
    [string]$VsEdition = "Professional"
)
$forwarder = Join-Path $PSScriptRoot "..\cuda-windows\build.ps1"
& $forwarder -Profile all @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }