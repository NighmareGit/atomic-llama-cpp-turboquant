#!/usr/bin/env bash
# Shared: rsync/scp deploy collateral from repo to a Path B node ~/docker/<stack>/.
# Sourced by pathb-remus-deploy-sync.sh, pathb-romulus-rpc.sh, etc.
#
# env:
#   PATHB_NODE_SSH        e.g. hunter@192.168.8.176
#   PATHB_NODE_SSH_PASS   optional sshpass password
#   PATHB_NODE_DOCKER_DIR optional default remote dir (second arg to deploy_sync still wins)

pathb_node_init_ssh() {
    SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new)
    SCP_CMD=(scp -o StrictHostKeyChecking=accept-new)
    RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new"
    if [[ -n "${PATHB_NODE_SSH_PASS:-}" ]] && command -v sshpass >/dev/null; then
        SSH_CMD=(sshpass -p "$PATHB_NODE_SSH_PASS" ssh -o StrictHostKeyChecking=accept-new)
        SCP_CMD=(sshpass -p "$PATHB_NODE_SSH_PASS" scp -o StrictHostKeyChecking=accept-new)
        RSYNC_SSH="sshpass -p $PATHB_NODE_SSH_PASS ssh -o StrictHostKeyChecking=accept-new"
    fi
}

pathb_node_remote() {
    "${SSH_CMD[@]}" "$PATHB_NODE_SSH" "$@"
}

# pathb_node_deploy_sync <local_deploy_dir> [remote_dir] [file glob...]
pathb_node_deploy_sync() {
    local src="$1"
    local dst="${2:-${PATHB_NODE_DOCKER_DIR:-}}"
    shift 2 || true
    local files=("$@")

    if [[ -z "$dst" ]]; then
        echo "ERROR: remote deploy dir missing (pass as 2nd arg or set PATHB_NODE_DOCKER_DIR)" >&2
        return 1
    fi

    if [[ ! -d "$src" ]]; then
        echo "ERROR: deploy source missing: $src" >&2
        return 1
    fi

    echo "=== deploy sync: $src -> $PATHB_NODE_SSH:$dst ==="
    pathb_node_remote "mkdir -p $dst"

    if command -v rsync >/dev/null 2>&1; then
        rsync -avz --delete \
            --exclude 'staging/' \
            --exclude 'staging/**' \
            -e "$RSYNC_SSH" \
            "${src}/" "${PATHB_NODE_SSH}:${dst}/"
    else
        if [[ ${#files[@]} -eq 0 ]]; then
            files=(build.sh build-rpc.sh build-server.sh docker-compose.yml Dockerfile Dockerfile.runtime Dockerfile.rpc Dockerfile.server README.md)
        fi
        for name in "${files[@]}"; do
            if [[ -f "${src}/${name}" ]]; then
                "${SCP_CMD[@]}" "${src}/${name}" "${PATHB_NODE_SSH}:${dst}/"
            fi
        done
    fi

    pathb_node_remote "find ${dst} -maxdepth 1 -name '*.sh' -exec sed -i 's/\\r$//' {} + 2>/dev/null || true"
    pathb_node_remote "chmod +x ${dst}/build.sh ${dst}/build-rpc.sh ${dst}/build-server.sh ${dst}/build-from-bin.sh 2>/dev/null || true"
    pathb_node_remote "grep -h 'GIT_BRANCH=' ${dst}/build.sh ${dst}/build-rpc.sh 2>/dev/null | head -3 || true"

    local branch="${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
    local url="${GIT_URL:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"
    pathb_node_remote "tip=\$(git ls-remote '${url}' 'refs/heads/${branch}' 2>/dev/null | awk '{print \$1}' | cut -c1-9); echo GITEA_TIP: \${branch}@\${tip:-unknown}"

    echo "DEPLOY_SYNC_OK"
}