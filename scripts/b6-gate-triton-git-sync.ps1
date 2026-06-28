# Run on triton: fetch Path-B-Event-Support-Pipeline-Plus from gitea.
$ErrorActionPreference = "Stop"
$Repo = "C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus"
$GiteaUrl = "http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git"
$Branch = "Path-B-Event-Support-Pipeline-Plus"

if (-not (Test-Path $Repo)) { throw "Repo missing: $Repo" }
Set-Location $Repo

git stash push -u -m "b6-gate-triton-sync" 2>$null | Out-Null
if (-not (git remote | Select-String -Pattern '^gitea$' -Quiet)) {
    git remote add gitea $GiteaUrl
} else {
    git remote set-url gitea $GiteaUrl
}
git fetch gitea --prune
git checkout $Branch 2>$null
if ($LASTEXITCODE -ne 0) {
    git checkout -b $Branch "gitea/$Branch"
}
git reset --hard "gitea/$Branch"
Write-Host "TRITON_GIT_SYNC_OK sha=$(git rev-parse --short HEAD)"