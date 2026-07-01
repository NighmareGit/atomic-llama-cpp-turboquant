#!/usr/bin/env bash
# Sync B+12 (or workspace) to romulus, rebuild profiler (ggml-base path), run B+12 bisects.
#
# usage (from dev host):
#   bash scripts/b6-gate-romulus-sync-rebuild.sh [--bisect both|canonical-romulus|no-get-defer] [--no-bisect]
#   bash scripts/b6-gate-romulus-sync-rebuild.sh --sync-only
#   bash scripts/b6-gate-romulus-sync-rebuild.sh --rebuild-only
#
# usage (on romulus):
#   ROMULUS_LOCAL=1 bash scripts/b6-gate-romulus-sync-rebuild.sh --rebuild-only
#
# env:
#   B6_ROMULUS_HOST       default hunter@192.168.8.108
#   B6_ROMULUS_PASS       default 12345
#   B6_ROMULUS_REPO       default /home/hunter/atomic-llama-cpp-turboquant
#   SYNC_MODE             rsync (default) | git
#   GIT_BRANCH            Path-B-Event-Support-Pipeline-Plus
#   GIT_REMOTE            gitea

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-hunter@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
SYNC_MODE="${SYNC_MODE:-rsync}"
GIT_BRANCH="${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
GIT_REMOTE="${GIT_REMOTE:-gitea}"
BUILD_DIR=build-rocm-docker

BISECT_MODE="both"
DO_SYNC=1
DO_REBUILD=1
DO_BISECT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bisect)
            BISECT_MODE="${2:?--bisect requires argument}"
            shift 2
            ;;
        --no-bisect) DO_BISECT=0; shift ;;
        --sync-only) DO_REBUILD=0; DO_BISECT=0; shift ;;
        --rebuild-only) DO_SYNC=0; DO_BISECT=0; shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

scp_romulus() {
    local src="$1" dst="$2"
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" scp -o StrictHostKeyChecking=no "$src" "${ROMULUS_HOST}:${dst}"
    else
        scp -o StrictHostKeyChecking=no "$src" "${ROMULUS_HOST}:${dst}"
    fi
}

remote_body() {
    local action="$1"
    case "$action" in
        git-sync)
            cd "$ROMULUS_REPO"
            if git remote | grep -qx "$GIT_REMOTE"; then
                :
            else
                git remote add "$GIT_REMOTE" "http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git" 2>/dev/null || true
            fi
            git fetch "$GIT_REMOTE" --prune
            git checkout "$GIT_BRANCH" 2>/dev/null || git checkout -b "$GIT_BRANCH" "$GIT_REMOTE/$GIT_BRANCH"
            git reset --hard "$GIT_REMOTE/$GIT_BRANCH"
            echo "ROMULUS_GIT_SHA=$(git rev-parse --short HEAD)"
            ;;
        rebuild)
            cd "$ROMULUS_REPO"
            # ggml-base includes "ggml-rpc.h" from ggml/src/ (PRIVATE .), not ggml/include/.
            if [[ -f ggml/include/ggml-rpc.h ]]; then
                cp -f ggml/include/ggml-rpc.h ggml/src/ggml-rpc.h
            fi
            # Force ggml-base regen after git reset or rsync (b15b lesson).
            touch ggml/src/ggml-backend.cpp ggml/src/ggml-rpc/ggml-rpc.cpp \
                ggml/src/ggml-rpc/transport.h ggml/include/ggml-rpc.h ggml/src/ggml-rpc.h
            if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
                cmake -S . -B "$BUILD_DIR" \
                    -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
                    -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4 \
                    -DLLAMA_BUILD_TOOLS=ON
            fi
            cmake --build "$BUILD_DIR" --target ggml-rpc ggml-base llama-pipeline-profiler -j"$(nproc)"
            test -x "${BUILD_DIR}/bin/llama-pipeline-profiler"
            "${BUILD_DIR}/bin/llama-pipeline-profiler" --validate-rpc \
                -rpc "192.168.8.23:50054" -ts "50,50"
            echo "ROMULUS_REBUILD_OK sha=$(git rev-parse --short HEAD)"
            ;;
        bisect)
            cd "$ROMULUS_REPO"
            export LD_LIBRARY_PATH="${ROMULUS_REPO}/${BUILD_DIR}/bin:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            run_one() {
                local bisect="$1" out_suffix="$2"
                echo "=== romulus bisect: ${bisect} -> ${out_suffix} ==="
                PROFILER_OUT_DIR="${ROMULUS_REPO}/benches/path-b-plus/${out_suffix}" \
                    bash scripts/b6-gate-bisect-run.sh "$bisect"
            }
            case "${BISECT_MODE}" in
                both)
                    run_one canonical-romulus "b6-2gpu-f-triton-n384-romulus-native-b12"
                    run_one no-get-defer "b6-2gpu-f-triton-n384-romulus-native-no-get-defer"
                    ;;
                canonical-romulus)
                    run_one canonical-romulus "b6-2gpu-f-triton-n384-romulus-native-b12"
                    ;;
                no-get-defer)
                    run_one no-get-defer "b6-2gpu-f-triton-n384-romulus-native-no-get-defer"
                    ;;
                *)
                    echo "error: unknown BISECT_MODE: ${BISECT_MODE}" >&2
                    exit 1
                    ;;
            esac
            ;;
    esac
}

rsync_sources() {
    local files=(
        ggml/include/ggml-rpc.h
        ggml/src/ggml-backend.cpp
        ggml/src/ggml-rpc/ggml-rpc.cpp
        scripts/b6-gate-bisect-run.sh
        scripts/b6-gate-b14-wavefront-spike.sh
        scripts/b6-gate-b14-n384-compare.sh
        scripts/b6-gate-b15-l4-layer-spread-spike.sh
        scripts/b6-gate-5gpu-deploy.sh
        scripts/b6-gate-phase1c-assembly-line.sh
        scripts/b6-gate-phase1c-l1-hash-defer-spike.sh
        scripts/b6-gate-phase1c-l4-n384-confirm.sh
        scripts/b6-gate-phase1c-moe-light-offload-spike.sh
        tools/llama-pipeline-profiler/llama-pipeline-profiler.cpp
        scripts/b6-gate-5gpu-production-env.sh
        scripts/b6-gate-phase0-assembly-bounds.py
        scripts/b6-gate-phase0-assembly-bounds.sh
        scripts/b6-gate-overlap-serial-audit.py
        rpc-patch/scripts/pathb-rpc-vram-preflight.py
        rpc-patch/scripts/pathb-72b-vram-calc.py
        scripts/b6-gate-profiler-romulus.sh
        scripts/b6-gate-romulus-sync-rebuild.sh
        scripts/b6-gate-romulus-b12-bisect-bg.sh
        scripts/b6-gate-run-remote.sh
        scripts/llama-pipeline-profiler-cluster.sh
    )
    echo "=== rsync B+12 sources to ${ROMULUS_HOST}:${ROMULUS_REPO} ==="
    for rel in "${files[@]}"; do
        local src="${ROOT}/${rel}"
        if [[ ! -f "$src" ]]; then
            echo "error: missing local file: ${src}" >&2
            exit 1
        fi
        ssh_romulus "mkdir -p ${ROMULUS_REPO}/$(dirname "$rel")"
        scp_romulus "$src" "${ROMULUS_REPO}/${rel}"
    done
    # ggml-base compiles against ggml/src/ggml-rpc.h (include path PRIVATE .).
    scp_romulus "${ROOT}/ggml/include/ggml-rpc.h" "${ROMULUS_REPO}/ggml/src/ggml-rpc.h"
    local sha
    sha="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "LOCAL_GIT_SHA=${sha} (rsync overlay; romulus tree may differ until git reset)"
}

if [[ "${ROMULUS_LOCAL:-0}" == "1" ]]; then
    [[ "$DO_SYNC" -eq 1 ]] && { [[ "$SYNC_MODE" == "git" ]] && remote_body git-sync || true; }
    [[ "$DO_REBUILD" -eq 1 ]] && remote_body rebuild
    [[ "$DO_BISECT" -eq 1 ]] && remote_body bisect
    exit 0
fi

if [[ "$DO_SYNC" -eq 1 ]]; then
    case "$SYNC_MODE" in
        rsync) rsync_sources ;;
        git)   ssh_romulus "ROMULUS_REPO='${ROMULUS_REPO}' GIT_BRANCH='${GIT_BRANCH}' GIT_REMOTE='${GIT_REMOTE}' bash -s" <<< "$(declare -f remote_body); remote_body git-sync" ;;
        *)
            echo "error: SYNC_MODE must be rsync or git" >&2
            exit 1
            ;;
    esac
fi

if [[ "$DO_REBUILD" -eq 1 ]]; then
    echo "=== romulus rebuild ==="
    ssh_romulus "set -euo pipefail; ROMULUS_REPO='${ROMULUS_REPO}' BUILD_DIR='${BUILD_DIR}' bash -s" <<< "$(declare -f remote_body); remote_body rebuild"
fi

if [[ "$DO_BISECT" -eq 1 ]]; then
    echo "=== romulus bisect (${BISECT_MODE}) ==="
    ssh_romulus "set -euo pipefail; ROMULUS_REPO='${ROMULUS_REPO}' BUILD_DIR='${BUILD_DIR}' BISECT_MODE='${BISECT_MODE}' bash -s" <<< "$(declare -f remote_body); remote_body bisect"
fi

pull_results() {
    local dirs=(
        b6-2gpu-f-triton-n384-romulus-native-b12
        b6-2gpu-f-triton-n384-romulus-native-no-get-defer
    )
    for d in "${dirs[@]}"; do
        local remote="${ROMULUS_REPO}/benches/path-b-plus/${d}"
        local local="${ROOT}/benches/path-b-plus/${d}"
        if ssh_romulus "test -d '${remote}'"; then
            echo "=== pull ${d} ==="
            mkdir -p "$local"
            if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
                sshpass -p "$ROMULUS_PASS" scp -o StrictHostKeyChecking=no -r \
                    "${ROMULUS_HOST}:${remote}/"* "$local/" 2>/dev/null || true
            else
                scp -o StrictHostKeyChecking=no -r "${ROMULUS_HOST}:${remote}/"* "$local/" 2>/dev/null || true
            fi
        fi
    done
}

if [[ "$DO_BISECT" -eq 1 ]]; then
    pull_results
    echo "=== B+12 romulus workflow done ==="
    for d in b6-2gpu-f-triton-n384-romulus-native-b12 b6-2gpu-f-triton-n384-romulus-native-no-get-defer; do
        if [[ -f "${ROOT}/benches/path-b-plus/${d}/summary.md" ]]; then
            echo "--- ${d} ---"
            head -20 "${ROOT}/benches/path-b-plus/${d}/summary.md"
        fi
    done
fi