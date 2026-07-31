#!/usr/bin/env bash
# Disk/docker preflight for Path B bench runs. Whitelist cleanup only.
#
# usage:
#   pathb-disk-preflight.sh [--cleanup] [--remote] [--min-gb 20]
#
# --cleanup  remove known bench containers (never infra images)
# --remote   also check remus via SSH

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MIN_GB="${PATHB_MIN_ROOT_GB:-20}"
DO_CLEANUP=0
DO_REMOTE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cleanup) DO_CLEANUP=1; shift ;;
        --remote)  DO_REMOTE=1; shift ;;
        --min-gb)  MIN_GB="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

REMUS_SSH="${PATHB_REMUS_SSH:-user@remus.local}"
REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new)
if [[ -n "$REMUS_SSH_PASS" ]] && command -v sshpass >/dev/null; then
    SSH_CMD=(sshpass -p "$REMUS_SSH_PASS" ssh -o StrictHostKeyChecking=accept-new)
fi

BENCH_CONTAINERS=(bench-rpc bench-llama pathb-longgen512)

check_root() {
    local host_label="$1"
    local df_out avail_gb
    df_out=$(df -BG / | awk 'NR==2 {print $4}')
    avail_gb=${df_out%G}
    echo "[$host_label] root free: ${avail_gb} GB"
    if [[ "$avail_gb" -lt "$MIN_GB" ]]; then
        echo "ERROR: $host_label root free ${avail_gb}GB < ${MIN_GB}GB threshold" >&2
        return 1
    fi
    docker system df 2>/dev/null || true
}

cleanup_bench() {
    local host_label="$1"
    echo "[$host_label] cleaning bench containers: ${BENCH_CONTAINERS[*]}"
    docker rm -f "${BENCH_CONTAINERS[@]}" 2>/dev/null || true
    if [[ "${PATHB_CONFIG:-}" == "remus" ]]; then
        docker rm -f pathb-rpc 2>/dev/null || true
    fi
}

echo "=== pathb-disk-preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
check_root "romulus"
[[ "$DO_CLEANUP" == "1" ]] && cleanup_bench "romulus"

if [[ "$DO_REMOTE" == "1" ]]; then
    "${SSH_CMD[@]}" "$REMUS_SSH" bash -s "$MIN_GB" <<'REMOTE'
min_gb="$1"
df_out=$(df -BG / | awk 'NR==2 {print $4}')
avail_gb=${df_out%G}
echo "[remus] root free: ${avail_gb} GB"
if [[ "$avail_gb" -lt "$min_gb" ]]; then
    echo "ERROR: remus root free ${avail_gb}GB < ${min_gb}GB" >&2
    exit 1
fi
docker system df 2>/dev/null || true
REMOTE
fi

echo "=== preflight OK ==="