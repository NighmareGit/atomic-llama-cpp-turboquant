#!/usr/bin/env bash
# Jupiter: git sync (gitea) + rpc-server rebuild + restart :50053.
#
# usage:
#   B6_JUPITER_PASS=... bash scripts/b6-gate-jupiter-sync-rebuild.sh
#
# env:
#   B6_JUPITER_HOST   default 192.168.8.21
#   B6_JUPITER_USER   default nightmare
#   B6_JUPITER_REPO   default D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant
#   B6_JUPITER_BRANCH default Path-B-Event-Support-Pipeline-Plus
#   B6_JUPITER_GITEA  default http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JUPITER_HOST="${B6_JUPITER_HOST:-192.168.8.21}"
JUPITER_USER="${B6_JUPITER_USER:-nightmare}"
JUPITER_REPO="${B6_JUPITER_REPO:-D:\\projects\\atomic-llama-cpp-5070ti\\atomic-llama-cpp-turboquant}"
BRANCH="${B6_JUPITER_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
GITEA="${B6_JUPITER_GITEA:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"

ssh_jupiter() {
    if command -v sshpass &>/dev/null && [[ -n "${B6_JUPITER_PASS:-}" ]]; then
        sshpass -p "$B6_JUPITER_PASS" ssh -o StrictHostKeyChecking=accept-new \
            "${JUPITER_USER}@${JUPITER_HOST}" "$@"
    else
        ssh -o StrictHostKeyChecking=accept-new "${JUPITER_USER}@${JUPITER_HOST}" "$@"
    fi
}

echo "=== jupiter git sync + rpc rebuild ==="
ssh_jupiter "powershell -NoProfile -ExecutionPolicy Bypass -Command \"
  Set-Location '${JUPITER_REPO}'
  if (Get-Command git -ErrorAction SilentlyContinue) {
    git remote get-url gitea 2>\\\$null; if (-not \\\$?) { git remote add gitea '${GITEA}' }
    git fetch gitea --prune
    git checkout -f -B '${BRANCH}' 'gitea/${BRANCH}'
    git reset --hard 'gitea/${BRANCH}'
    Write-Host ('JUPITER_GIT_SHA=' + (git rev-parse --short HEAD))
  } else {
    Write-Warning 'git not in PATH; skipping sync'
  }
  & '${JUPITER_REPO}\\scripts\\b6-gate-jupiter-rebuild-rpc.cmd'
\""

B6_JUPITER_USER="$JUPITER_USER" B6_JUPITER_PASS="${B6_JUPITER_PASS:-}" \
    bash "${ROOT}/scripts/b6-gate-jupiter-remote.sh" start-rpc

echo "=== post-rebuild validate (romulus) ==="
ROMULUS_HOST="${B6_ROMULUS_HOST:-user@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "
  cd /home/hunter/atomic-llama-cpp-turboquant
  LD_LIBRARY_PATH=build-rocm-docker/bin:/opt/rocm/lib \
    build-rocm-docker/bin/llama-pipeline-profiler --validate-rpc \
    -rpc '192.168.8.21:50053' -ts 50 2>&1 | tail -8
"

echo "JUPITER_SYNC_REBUILD_DONE"