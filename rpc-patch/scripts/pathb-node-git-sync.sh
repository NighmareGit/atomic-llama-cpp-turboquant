#!/usr/bin/env bash
# Shared: ensure a remote host checkout tracks the Path B gitea branch.
# Requires pathb_node_init_ssh / pathb_node_remote from pathb-node-deploy-sync.sh.
#
# env:
#   GIT_BRANCH   (default Path-B-Event-Support-Pipeline-Plus)
#   GIT_URL      (default gitea on romulus)
#   GIT_REMOTE   (default gitea)
#   PATHB_NODE_GIT_REPO  (default ~/atomic-llama-cpp-turboquant)

pathb_node_git_branch() {
    echo "${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
}

pathb_node_git_url() {
    echo "${GIT_URL:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"
}

pathb_node_git_remote() {
    echo "${GIT_REMOTE:-gitea}"
}

# pathb_node_git_sync [remote_repo_path]
pathb_node_git_sync() {
    local repo="${1:-${PATHB_NODE_GIT_REPO:-\$HOME/atomic-llama-cpp-turboquant}}"
    local branch
    local url
    local remote
    branch="$(pathb_node_git_branch)"
    url="$(pathb_node_git_url)"
    remote="$(pathb_node_git_remote)"

    echo "=== git sync: ${PATHB_NODE_SSH}:${repo} -> ${branch} ==="
    pathb_node_remote "set -euo pipefail
repo='${repo}'
branch='${branch}'
url='${url}'
remote='${remote}'
if [[ ! -d \"\${repo}/.git\" ]]; then
    git clone \"\${url}\" \"\${repo}\"
fi
cd \"\${repo}\"
if git remote | grep -qx \"\${remote}\"; then
    git remote set-url \"\${remote}\" \"\${url}\"
else
    git remote add \"\${remote}\" \"\${url}\"
fi
git fetch \"\${remote}\" --prune
if git show-ref --verify --quiet \"refs/heads/\${branch}\"; then
    git checkout \"\${branch}\"
else
    git checkout -b \"\${branch}\" \"\${remote}/\${branch}\"
fi
git reset --hard \"\${remote}/\${branch}\"
echo GIT_SYNC_OK: \$(git rev-parse --short HEAD) \$(git log -1 --oneline)"
}

# pathb_node_git_remote_tip  -> prints short sha of branch tip on gitea
pathb_node_git_remote_tip() {
    local branch url remote
    branch="$(pathb_node_git_branch)"
    url="$(pathb_node_git_url)"
    remote="$(pathb_node_git_remote)"
    pathb_node_remote "git ls-remote '${url}' 'refs/heads/${branch}' | awk '{print \$1}' | cut -c1-9"
}