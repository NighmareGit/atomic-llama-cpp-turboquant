#!/usr/bin/env bash
set -euo pipefail

GIT_BRANCH="${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
GIT_URL="${GIT_URL:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"
GIT_REMOTE="${GIT_REMOTE:-gitea}"
REPO=/home/hunter/atomic-llama-cpp-turboquant

cd "$REPO"
if git remote | grep -qx "$GIT_REMOTE"; then
    git remote set-url "$GIT_REMOTE" "$GIT_URL"
else
    git remote add "$GIT_REMOTE" "$GIT_URL"
fi
git fetch "$GIT_REMOTE" --prune
git checkout "$GIT_BRANCH" 2>/dev/null || git checkout -b "$GIT_BRANCH" "$GIT_REMOTE/$GIT_BRANCH"
git reset --hard "$GIT_REMOTE/$GIT_BRANCH"
echo "ROCM_GIT_SYNC: $(git rev-parse --short HEAD) $(git log -1 --oneline)"
BUILD=build-rocm-docker
LOG=/tmp/rocm-docker-build.log

docker run --rm --entrypoint rm \
    -v "/home/hunter/atomic-llama-cpp-turboquant:/src" \
    llama-rocm-patched -rf "/src/${BUILD}" 2>/dev/null || rm -rf "$BUILD"

cmake -S . -B "$BUILD" \
    -DGGML_HIP=ON -DGGML_RPC=ON -DGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_SCHED_MAX_COPIES=4

cmake --build "$BUILD" --target llama-server -j"$(nproc)" 2>&1 | tee "$LOG"
mkdir -p "$BUILD/bin"
find "$BUILD" -name '*.so*' -exec cp -n {} "$BUILD/bin/" \; 2>/dev/null || true
test -x "$BUILD/bin/llama-server"
echo ROCM_BUILD_OK