#!/usr/bin/env bash
# Archive existing checkout and fresh-clone at canonical cluster path.
# Remote policy: github primary (internet), gitea LAN fallback; sync to newest tip.
#
# usage (on any Linux node):
#   bash scripts/cluster-node-fresh-clone.sh
#
# env:
#   CLUSTER_REPO     default ~/projects/atomic-llama-cpp-turboquant
#   GIT_BRANCH       default Path-B-Event-Support-Pipeline-Plus
#   GITHUB_URL       default https://github.com/NighmareGit/atomic-llama-cpp-turboquant.git
#   GITEA_URL        default http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git
#   ARCHIVE_LEGACY   default 1 — also archive ~/atomic-llama-cpp-turboquant if present (romulus)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/rpc-patch/scripts/pathb-cluster-git-remotes.sh"

CLUSTER_REPO="${CLUSTER_REPO:-$HOME/projects/atomic-llama-cpp-turboquant}"
ARCHIVE_LEGACY="${ARCHIVE_LEGACY:-1}"
STAMP="$(date +%Y%m%d-%H%M%S)"

archive_dir() {
    local src="$1"
    if [[ -d "$src" ]]; then
        local dst="${src}.archived-${STAMP}"
        echo "=== archive ${src} -> ${dst} ==="
        mv "$src" "$dst"
    fi
}

mkdir -p "$(dirname "$CLUSTER_REPO")"

if [[ -d "$CLUSTER_REPO" ]]; then
    archive_dir "$CLUSTER_REPO"
fi

if [[ "$ARCHIVE_LEGACY" == "1" && -d "$HOME/atomic-llama-cpp-turboquant" ]]; then
    archive_dir "$HOME/atomic-llama-cpp-turboquant"
fi

CLONE_URL="$(pathb_cluster_clone_url)"
BRANCH="$(pathb_cluster_git_branch)"

echo "=== clone ${CLONE_URL} -> ${CLUSTER_REPO} (branch ${BRANCH}) ==="
git clone --branch "$BRANCH" "$CLONE_URL" "$CLUSTER_REPO"
cd "$CLUSTER_REPO"
pathb_cluster_sync_repo

echo "CLUSTER_FRESH_CLONE_OK sha=$(git rev-parse --short HEAD) path=${CLUSTER_REPO}"