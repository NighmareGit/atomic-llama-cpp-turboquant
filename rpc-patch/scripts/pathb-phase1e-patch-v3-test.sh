#!/usr/bin/env bash
# Phase 1e: test uncommitted PATCH fix (ggml/src/ggml-rpc.h RPC_PROTO_PATCH_VERSION=3)
# usage: ./rpc-patch/scripts/pathb-phase1e-patch-v3-test.sh [--skip-push] [--skip-rebuild] [--smoke-only]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SKIP_PUSH=0
SKIP_REBUILD=0
SMOKE_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-push) SKIP_PUSH=1 ;;
        --skip-rebuild) SKIP_REBUILD=1 ;;
        --smoke-only) SMOKE_ONLY=1 ;;
    esac
done

log() { echo "=== $* ==="; }

# 1. Verify local PATCH
patch_ver=$(grep -E '^#define RPC_PROTO_PATCH_VERSION' ggml/src/ggml-rpc.h | awk '{print $3}')
log "Step 1: ggml/src/ggml-rpc.h RPC_PROTO_PATCH_VERSION=${patch_ver}"
[[ "$patch_ver" == "3" ]] || { echo "FAIL: expected PATCH_VERSION 3" >&2; exit 1; }

# 2. Source cluster creds (never echo secrets)
if [[ -f .scratch/cluster-access.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .scratch/cluster-access.env
    set +a
    log "Step 2: sourced .scratch/cluster-access.env"
else
    echo "WARN: .scratch/cluster-access.env missing" >&2
fi

export GIT_BRANCH=test/patch-v3-src-header
export GIT_URL="${GIT_URL:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"

# 3. Push test branch
if [[ "$SKIP_PUSH" == "0" ]]; then
    log "Step 3: push test branch to gitea"
    git checkout Path-B-Event-Support-Pipeline-Plus
    git checkout -B test/patch-v3-src-header
    if ! git diff --quiet ggml/src/ggml-rpc.h 2>/dev/null || git status --porcelain ggml/src/ggml-rpc.h | grep -q .; then
        git add ggml/src/ggml-rpc.h
        git commit -m "fix: align ggml/src/ggml-rpc.h PATCH_VERSION to 3"
    else
        echo "ggml/src/ggml-rpc.h unchanged vs index — skip commit"
    fi
    git push -u gitea test/patch-v3-src-header
fi

# 4-7. Cluster + rebuild
if [[ "$SKIP_REBUILD" == "0" ]]; then
    log "Step 5: cluster start"
    "$SCRIPT_DIR/pathb-cluster-up.sh" start

    log "Step 6: remus RPC rebuild"
    GIT_BRANCH="$GIT_BRANCH" "$SCRIPT_DIR/pathb-remus-rpc.sh" rebuild

    log "Step 7: romulus RPC rebuild"
    GIT_BRANCH="$GIT_BRANCH" "$SCRIPT_DIR/pathb-romulus-rpc.sh" rebuild

    log "Step 8: romulus ROCm client rebuild"
    SSH_BASE=(ssh -o StrictHostKeyChecking=no)
    if [[ -n "${PATHB_ROMULUS_SSH_PASS:-}" ]] && command -v sshpass >/dev/null; then
        SSH_BASE=(sshpass -p "$PATHB_ROMULUS_SSH_PASS" ssh -o StrictHostKeyChecking=no)
    fi
    "${SSH_BASE[@]}" "${PATHB_ROMULUS_SSH:-hunter@${ROMULUS_RPC_IP:-192.168.8.108}}" \
        "GIT_BRANCH='${GIT_BRANCH}' bash -s" < "$SCRIPT_DIR/pathb-romulus-rocm-build-remote.sh"
fi

# 9. Smoke
log "Step 9: 2-GPU fox smoke trace-g-2gpu-fd7e-retest"
BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_TRACE=0 \
    "$SCRIPT_DIR/pathb-romulus-2gpu-bench.sh" trace-g-2gpu-fd7e-retest \
    | tee /tmp/phase1e-smoke.log

if [[ "$SMOKE_ONLY" == "1" ]]; then
    log "Done (--smoke-only)"
    exit 0
fi

# 10. TSC matrix D1-D2
log "Step 10: TSC matrix D1 (2-GPU) Plus on/off"
for plus in 1 0; do
    GGML_PIPELINE_PLUS=$plus BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_TRACE=0 \
        "$SCRIPT_DIR/pathb-romulus-2gpu-bench.sh" "trace-g-2gpu-tsc-plus${plus}" \
        | tee "/tmp/phase1e-2gpu-plus${plus}.log"
done

log "Step 10: TSC matrix D2 (3-GPU) Plus on/off"
for plus in 1 0; do
    GGML_PIPELINE_PLUS=$plus BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TRACE=0 \
        "$SCRIPT_DIR/pathb-romulus-3gpu-bench.sh" primary-5060-3060 "trace-g-3gpu-tsc-plus${plus}" \
        | tee "/tmp/phase1e-3gpu-plus${plus}.log"
done

# 11. D4 optional 4-GPU
log "Step 11: probe Jupiter :50053"
if nc -zv 192.168.8.21 50053 2>&1 | tee /tmp/phase1e-jupiter-probe.log; then
    log "Step 11: TSC matrix D4 (4-GPU) Plus on/off"
    for plus in 1 0; do
        GGML_PIPELINE_PLUS=$plus BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TRACE=0 \
            "$SCRIPT_DIR/pathb-romulus-4gpu-bench.sh" "trace-g-4gpu-tsc-plus${plus}" \
            | tee "/tmp/phase1e-4gpu-plus${plus}.log"
    done
else
    echo "SKIP: Jupiter :50053 not reachable"
fi

log "Phase 1e complete — logs in /tmp/phase1e-*.log"