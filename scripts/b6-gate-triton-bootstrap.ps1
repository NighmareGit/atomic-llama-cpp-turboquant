# Run on triton (Admin PowerShell): firewall :22/:50054, build portable, start RPC :50054.
$ErrorActionPreference = "Stop"
$Repo = "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus"

function Ensure-FirewallPort {
    param([string]$Name, [string]$DisplayName, [int]$Port)
    $rule = Get-NetFirewallRule -Name $Name -ErrorAction SilentlyContinue
    if (-not $rule) {
        New-NetFirewallRule -Name $Name -DisplayName $DisplayName `
            -Enabled True -Direction Inbound -Protocol TCP -Action Allow `
            -LocalPort $Port -Profile Any | Out-Null
    } else {
        Set-NetFirewallRule -Name $Name -Enabled True -Profile Any | Out-Null
    }
}

Set-Service sshd -StartupType Automatic
if ((Get-Service sshd).Status -ne "Running") { Start-Service sshd }
Ensure-FirewallPort "OpenSSH-Server-In-TCP" "OpenSSH Server (sshd)" 22
Ensure-FirewallPort "PathB-RPC-50054-In-TCP" "Path B RPC worker (50054)" 50054

if (-not (Test-Path $Repo)) { throw "Repo missing: $Repo" }
Set-Location $Repo

$portable = Join-Path $Repo "build-cuda-b-bin\portable\rpc-server.exe"
if (-not (Test-Path $portable)) {
    Write-Host "=== build portable (triton profile) ==="
    & "$Repo\scripts\cuda-windows\build.ps1" -Profile triton
}

if (-not (Test-Path $portable)) { throw "rpc-server.exe missing after build" }

Write-Host "=== start :50054 ==="
& "$Repo\scripts\cuda-windows-triton\pathb-rpc-server.ps1" -Restart

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -match '^192\.168\.8\.' } |
    Select-Object -First 1).IPAddress
Write-Host "TRITON_BOOTSTRAP_OK ssh=:22 rpc=${ip}:50054"