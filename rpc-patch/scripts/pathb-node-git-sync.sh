#!/usr/bin/env bash
# Shared: sync a remote host checkout to newest tip (github primary, gitea fallback).
# Requires pathb_node_init_ssh / pathb_node_remote from pathb-node-deploy-sync.sh.
#
# env:
#   GIT_BRANCH          default Path-B-Event-Support-Pipeline-Plus
#   GITHUB_URL          default https://github.com/NighmareGit/atomic-llama-cpp-turboquant.git
#   GITEA_URL           default http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git
#   PATHB_NODE_GIT_REPO default ~/projects/atomic-llama-cpp-turboquant

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pathb-cluster-git-remotes.sh
source "${_SCRIPT_DIR}/pathb-cluster-git-remotes.sh"

pathb_node_git_branch() {
    pathb_cluster_git_branch
}

pathb_node_git_url() {
    pathb_cluster_clone_url
}

pathb_node_git_remote() {
    pathb_cluster_pick_remote
}

# pathb_node_git_sync [remote_repo_path]
pathb_node_git_sync() {
    local repo="${1:-${PATHB_NODE_GIT_REPO:-\$HOME/projects/atomic-llama-cpp-turboquant}}"
    local branch clone_url github_url gitea_url
    branch="$(pathb_cluster_git_branch)"
    github_url="$(pathb_cluster_github_url)"
    gitea_url="$(pathb_cluster_gitea_url)"
    clone_url="$(pathb_cluster_clone_url)"

    echo "=== git sync: ${PATHB_NODE_SSH}:${repo} -> ${branch} (github primary, gitea fallback) ==="
    pathb_node_remote "set -euo pipefail
repo='${repo}'
branch='${branch}'
clone_url='${clone_url}'
github_url='${github_url}'
gitea_url='${gitea_url}'
helpers='${repo}/rpc-patch/scripts/pathb-cluster-git-remotes.sh'

if [[ ! -d \"\${repo}/.git\" ]]; then
    git clone --branch \"\${branch}\" \"\${clone_url}\" \"\${repo}\"
fi
cd \"\${repo}\"

if [[ -f \"\${helpers}\" ]]; then
    # shellcheck source=/dev/null
    source \"\${helpers}\"
    pathb_cluster_sync_repo
else
    # Bootstrap before helpers exist on very old trees
    if git remote | grep -qx github; then git remote set-url github \"\${github_url}\"
    else git remote add github \"\${github_url}\"; fi
    if git remote | grep -qx gitea; then git remote set-url gitea \"\${gitea_url}\"
    else git remote add gitea \"\${gitea_url}\"; fi
    git fetch github --prune 2>/dev/null || true
    git fetch gitea --prune 2>/dev/null || true
    pick=github
    if git show-ref --verify --quiet refs/remotes/github/\${branch}; then pick=github
    elif git show-ref --verify --quiet refs/remotes/gitea/\${branch}; then pick=gitea
    else echo 'error: no remote branch' >&2; exit 1; fi
    git checkout \"\${branch}\" 2>/dev/null || git checkout -b \"\${branch}\" \"\${pick}/\${branch}\"
    git reset --hard \"\${pick}/\${branch}\"
    echo GIT_SYNC_OK remote=\${pick} sha=\$(git rev-parse --short HEAD) \$(git log -1 --oneline)
fi"
}

# pathb_node_git_remote_tip  -> prints short sha of newest reachable remote tip
pathb_node_git_remote_tip() {
    local branch tip_g tip_t
    branch="$(pathb_cluster_git_branch)"
    tip_g="$(pathb_cluster_ls_remote_tip "$(pathb_cluster_github_url)" "$branch" | cut -c1-9)"
    tip_t="$(pathb_cluster_ls_remote_tip "$(pathb_cluster_gitea_url)" "$branch" | cut -c1-9)"
    if [[ -n "$tip_g" ]]; then
        echo "$tip_g"
    elif [[ -n "$tip_t" ]]; then
        echo "$tip_t"
    fi
}