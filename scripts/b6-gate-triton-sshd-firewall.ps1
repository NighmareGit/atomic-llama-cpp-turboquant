# Run on triton: start OpenSSH Server and allow inbound :22 (B+6 remote access).
$ErrorActionPreference = "Stop"

$svc = Get-Service sshd -ErrorAction SilentlyContinue
if (-not $svc) {
    throw "sshd service not found - install OpenSSH Server capability first"
}

Set-Service sshd -StartupType Automatic
if ($svc.Status -ne "Running") {
    Start-Service sshd
}

$rule = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
if (-not $rule) {
    New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (sshd)" `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
} else {
    Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP" | Out-Null
}

$listen = Get-NetTCPConnection -LocalPort 22 -State Listen -ErrorAction SilentlyContinue
if (-not $listen) {
    throw "sshd not listening on :22 after start"
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -match '^192\.168\.8\.' } |
    Select-Object -First 1).IPAddress
Write-Host "TRITON_SSHD_OK sshd=$($svc.Status) listen=:22 ip=$ip"