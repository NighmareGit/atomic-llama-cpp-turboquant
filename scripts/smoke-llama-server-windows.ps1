# Forwarder - use scripts/cuda-windows-5070ti/smoke-llama-server.ps1
& (Join-Path $PSScriptRoot "cuda-windows-5070ti\smoke-llama-server.ps1") @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }