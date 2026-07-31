param(
    [ValidateSet("config-e","config-f")][string]$Config = "config-e",
    [string]$ModelPath = "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf",
    [int]$Ctx = 4096
)
$wslModel = "/mnt/" + ($ModelPath -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/'
$wslModel = $wslModel.ToLower()
$errFile = Join-Path $env:TEMP "pathb-vram-calc.err"
$wslErr = "/mnt/" + ($errFile -replace '^([A-Za-z]):\\', '$1/') -replace '\\', '/'
$wslErr = $wslErr.ToLower()
& "$PSScriptRoot\invoke-wsl.ps1" -BashCommand "python3 rpc-patch/scripts/pathb-72b-vram-calc.py --config $Config --gguf '$wslModel' --ctx $Ctx 2>'$wslErr'"
if (Test-Path $errFile) {
    Get-Content $errFile -Encoding utf8 | Where-Object { $_ -match '^WARN:' } | Write-Host -ForegroundColor Yellow
}