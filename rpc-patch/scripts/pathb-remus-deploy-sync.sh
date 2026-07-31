#!/usr/bin/env bash
# Shared: rsync/scp deploy collateral from repo to remus ~/docker/<stack>/.
# Sourced by pathb-remus-rpc.sh and pathb-remus-rx6600-rpc.sh.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pathb-node-deploy-sync.sh
source "${_SCRIPT_DIR}/pathb-node-deploy-sync.sh"

pathb_remus_init_ssh() {
    PATHB_NODE_SSH="${PATHB_NODE_SSH:-${REMUS_SSH:-}}"
    PATHB_NODE_SSH_PASS="${PATHB_NODE_SSH_PASS:-${PATHB_REMUS_SSH_PASS:-}}"
    pathb_node_init_ssh
}

pathb_remus_remote() {
    PATHB_NODE_SSH="${PATHB_NODE_SSH:-${REMUS_SSH:-}}"
    pathb_node_remote "$@"
}

pathb_remus_deploy_sync() {
    PATHB_NODE_SSH="${PATHB_NODE_SSH:-${REMUS_SSH:-}}"
    pathb_node_deploy_sync "$@"
}