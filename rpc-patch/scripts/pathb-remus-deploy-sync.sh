#!/usr/bin/env bash
# Shared: rsync/scp deploy collateral from repo to remus ~/docker/<stack>/.
# Sourced by pathb-remus-rpc.sh and pathb-remus-rx6600-rpc.sh.

pathb_remus_init_ssh() {
    SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new)
    SCP_CMD=(scp -o StrictHostKeyChecking=accept-new)
    RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new"
    if [[ -n "${PATHB_REMUS_SSH_PASS:-}" ]] && command -v sshpass >/dev/null; then
        SSH_CMD=(sshpass -p "$PATHB_REMUS_SSH_PASS" ssh -o StrictHostKeyChecking=accept-new)
        SCP_CMD=(sshpass -p "$PATHB_REMUS_SSH_PASS" scp -o StrictHostKeyChecking=accept-new)
        RSYNC_SSH="sshpass -p $PATHB_REMUS_SSH_PASS ssh -o StrictHostKeyChecking=accept-new"
    fi
}

pathb_remus_remote() {
    "${SSH_CMD[@]}" "$REMUS_SSH" "$@"
}

# pathb_remus_deploy_sync <local_deploy_dir> <remote_dir> [file glob...]
pathb_remus_deploy_sync() {
    local src="$1"
    local dst="$2"
    shift 2
    local files=("$@")

    if [[ ! -d "$src" ]]; then
        echo "ERROR: deploy source missing: $src" >&2
        return 1
    fi

    echo "=== deploy sync: $src -> $REMUS_SSH:$dst ==="
    pathb_remus_remote "mkdir -p $dst"

    if command -v rsync >/dev/null 2>&1; then
        rsync -avz --delete \
            --exclude 'staging/' \
            --exclude 'staging/**' \
            -e "$RSYNC_SSH" \
            "${src}/" "${REMUS_SSH}:${dst}/"
    else
        if [[ ${#files[@]} -eq 0 ]]; then
            files=(build.sh build-rpc.sh build-server.sh docker-compose.yml Dockerfile Dockerfile.runtime Dockerfile.rpc Dockerfile.server README.md)
        fi
        for name in "${files[@]}"; do
            if [[ -f "${src}/${name}" ]]; then
                "${SCP_CMD[@]}" "${src}/${name}" "${REMUS_SSH}:${dst}/"
            fi
        done
    fi

    pathb_remus_remote "find ${dst} -maxdepth 1 -name '*.sh' -exec sed -i 's/\\r$//' {} + 2>/dev/null || true"
    pathb_remus_remote "chmod +x ${dst}/build.sh ${dst}/build-rpc.sh ${dst}/build-server.sh 2>/dev/null || true"
    pathb_remus_remote "grep -h 'GIT_BRANCH=' ${dst}/build.sh ${dst}/build-rpc.sh 2>/dev/null | head -3 || true"
    echo "DEPLOY_SYNC_OK"
}