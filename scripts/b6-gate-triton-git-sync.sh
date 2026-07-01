#!/usr/bin/env bash
# Run on triton: fetch Path-B-Event-Support-Pipeline-Plus from gitea (or origin).
# Repairs incomplete .git trees (missing objects/) seen on ZFS /home clones.
#
# usage: b6-gate-triton-git-sync.sh
#
# env:
#   B6_TRITON_REPO     default ~/projects/atomic-llama-cpp-turboquant
#   B6_TRITON_BRANCH   default Path-B-Event-Support-Pipeline-Plus
#   B6_TRITON_GITEA    default http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git

set -euo pipefail

REPO="${B6_TRITON_REPO:-${HOME}/projects/atomic-llama-cpp-turboquant}"
BRANCH="${B6_TRITON_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
GITEA_URL="${B6_TRITON_GITEA:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"

[[ -d "$REPO" ]] || { echo "repo missing: $REPO" >&2; exit 1; }
cd "$REPO"

git_ok() {
    [[ -d .git/objects ]] && git rev-parse --is-inside-work-tree &>/dev/null
}

if ! git_ok; then
    echo "WARN: repairing broken .git (missing objects/ or invalid repo)" >&2
    stamp="$(date +%s)"
    [[ -d .git ]] && mv .git ".git.broken-${stamp}"
    git init -q
    git remote add gitea "$GITEA_URL" 2>/dev/null || git remote set-url gitea "$GITEA_URL"
fi

git stash push -u -m "b6-gate-triton-sync" 2>/dev/null || true

if ! git remote | grep -q '^gitea$'; then
    git remote add gitea "$GITEA_URL"
else
    git remote set-url gitea "$GITEA_URL"
fi

git fetch gitea --prune
if ! git show-ref --verify --quiet "refs/remotes/gitea/${BRANCH}"; then
    echo "remote branch gitea/${BRANCH} missing" >&2
    exit 1
fi

git checkout -f -B "$BRANCH" "gitea/${BRANCH}"
git reset --hard "gitea/${BRANCH}"

sha="$(git rev-parse --short HEAD)"
echo "TRITON_GIT_SYNC_OK sha=${sha} repo=${REPO}"