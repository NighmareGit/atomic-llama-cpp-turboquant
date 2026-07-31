#!/usr/bin/env bash
# Shared git remote policy for cluster nodes:
#   github  = primary (internet)
#   gitea   = LAN mirror / offline fallback
# Pick whichever remote branch tip is ahead when both are reachable.
#
# shellcheck shell=bash
# Source from other scripts: source "$(dirname "$0")/../rpc-patch/scripts/pathb-cluster-git-remotes.sh"

pathb_cluster_git_branch() {
    echo "${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
}

pathb_cluster_github_url() {
    echo "${GITHUB_URL:-https://github.com/NighmareGit/atomic-llama-cpp-turboquant.git}"
}

pathb_cluster_gitea_url() {
    echo "${GITEA_URL:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"
}

# pathb_cluster_ls_remote_tip <url> <branch>  -> full sha or empty
pathb_cluster_ls_remote_tip() {
    local url="$1" branch="$2"
    git ls-remote "$url" "refs/heads/${branch}" 2>/dev/null | awk '{print $1; exit}'
}

# pathb_cluster_ensure_remotes  (run inside repo)
# Sets remotes github + gitea; fetches what is reachable.
pathb_cluster_ensure_remotes() {
    local github_url gitea_url
    github_url="$(pathb_cluster_github_url)"
    gitea_url="$(pathb_cluster_gitea_url)"

    if git remote | grep -qx github; then
        git remote set-url github "$github_url"
    else
        git remote add github "$github_url"
    fi

    if git remote | grep -qx gitea; then
        git remote set-url gitea "$gitea_url"
    else
        git remote add gitea "$gitea_url"
    fi

    PATHB_CLUSTER_GIT_FETCHED_GITHUB=0
    PATHB_CLUSTER_GIT_FETCHED_GITEA=0

    if git fetch github --prune 2>/dev/null; then
        PATHB_CLUSTER_GIT_FETCHED_GITHUB=1
    else
        echo "warn: github fetch failed (offline?); will try gitea" >&2
    fi

    if git fetch gitea --prune 2>/dev/null; then
        PATHB_CLUSTER_GIT_FETCHED_GITEA=1
    else
        echo "warn: gitea fetch failed (romulus down?)" >&2
    fi
}

# pathb_cluster_pick_remote  (run inside repo after ensure_remotes)
# Prints remote name to reset to: github or gitea
pathb_cluster_pick_remote() {
    local branch ref_g ref_t
    branch="$(pathb_cluster_git_branch)"
    ref_g="github/${branch}"
    ref_t="gitea/${branch}"

    local have_g=0 have_t=0
    git show-ref --verify --quiet "refs/remotes/${ref_g}" && have_g=1
    git show-ref --verify --quiet "refs/remotes/${ref_t}" && have_t=1

    if [[ "$have_g" -eq 1 && "$have_t" -eq 1 ]]; then
        local sha_g sha_t
        sha_g="$(git rev-parse "$ref_g")"
        sha_t="$(git rev-parse "$ref_t")"
        if [[ "$sha_g" == "$sha_t" ]]; then
            echo github
            return 0
        fi
        if git merge-base --is-ancestor "$ref_t" "$ref_g" 2>/dev/null; then
            echo "warn: github ahead of gitea ($sha_g vs $sha_t)" >&2
            echo github
            return 0
        fi
        if git merge-base --is-ancestor "$ref_g" "$ref_t" 2>/dev/null; then
            echo "warn: gitea ahead of github ($sha_t vs $sha_g) — push github to restore mirror" >&2
            echo gitea
            return 0
        fi
        echo "error: github and gitea diverged on ${branch} ($sha_g vs $sha_t)" >&2
        return 1
    fi

    if [[ "$have_g" -eq 1 ]]; then
        echo github
        return 0
    fi
    if [[ "$have_t" -eq 1 ]]; then
        echo gitea
        return 0
    fi

    echo "error: neither github nor gitea reachable for ${branch}" >&2
    return 1
}

# pathb_cluster_clone_url  -> pick URL for initial clone (ls-remote probe, no local repo)
pathb_cluster_clone_url() {
    local branch github_url gitea_url tip_g tip_t
    branch="$(pathb_cluster_git_branch)"
    github_url="$(pathb_cluster_github_url)"
    gitea_url="$(pathb_cluster_gitea_url)"

    tip_g="$(pathb_cluster_ls_remote_tip "$github_url" "$branch")"
    tip_t="$(pathb_cluster_ls_remote_tip "$gitea_url" "$branch")"

    if [[ -n "$tip_g" && -n "$tip_t" ]]; then
        if [[ "$tip_g" == "$tip_t" ]]; then
            printf '%s\n' "$github_url"
            return 0
        fi
        # Prefer github when both reachable but differ (mirror lag); user can force gitea.
        if [[ "${GIT_CLONE_PREFER:-github}" == "gitea" ]]; then
            printf '%s\n' "$gitea_url"
        else
            echo "warn: clone tip mismatch github=${tip_g:0:9} gitea=${tip_t:0:9}; using github" >&2
            printf '%s\n' "$github_url"
        fi
        return 0
    fi

    if [[ -n "$tip_g" ]]; then
        printf '%s\n' "$github_url"
        return 0
    fi
    if [[ -n "$tip_t" ]]; then
        echo "warn: github unreachable; cloning from gitea" >&2
        printf '%s\n' "$gitea_url"
        return 0
    fi

    echo "error: cannot reach github or gitea for ${branch}" >&2
    return 1
}

# pathb_cluster_sync_repo  (run inside repo)
pathb_cluster_sync_repo() {
    local branch remote
    branch="$(pathb_cluster_git_branch)"
    pathb_cluster_ensure_remotes
    remote="$(pathb_cluster_pick_remote)" || return 1

    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        git checkout "$branch"
    else
        git checkout -b "$branch" "${remote}/${branch}"
    fi
    git reset --hard "${remote}/${branch}"
    echo "GIT_SYNC_OK remote=${remote} sha=$(git rev-parse --short HEAD) $(git log -1 --oneline)"
}