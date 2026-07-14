#!/usr/bin/env bash
# Build Docker rpc-server image from LOCAL source tree (not gitea clone).
# Use this when you need the latest source (e.g., telemetry support) without
# pushing to gitea first.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
STAGING="${DIR}/staging"
REPO_ROOT="${REPO_ROOT:-$(cd "$DIR/../../.." && pwd)}"
PROXY_IP="${PROXY_IP:-192.168.8.108}"
MIN_GB="${MIN_ROOT_GB:-25}"

avail=$(df -BG / | awk 'NR==2 {print $4}')
avail_gb=${avail%G}
echo "root free: ${avail_gb} GB (min ${MIN_GB} GB)"
if [[ "$avail_gb" -lt "$MIN_GB" ]]; then
    echo "ERROR: insufficient root disk" >&2
    exit 1
fi

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi missing" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rm -rf "$STAGING"
mkdir -p "$STAGING/lib"

echo "=== GPU compile stage (CUDA 12.8, sm_86 for RTX 3060 Ti) ==="
echo "=== source: ${REPO_ROOT} ==="
echo "=== source commit: $(cd "$REPO_ROOT" && git rev-parse --short HEAD) $(cd "$REPO_ROOT" && git log -1 --oneline) ==="

docker run --rm --gpus=all \
    -v "${REPO_ROOT}:/tmp/src:ro" \
    -v "${DIR}:/workspace" \
    -w /workspace \
    nvidia/cuda:12.8.0-devel-ubuntu22.04 \
    bash -c "
set -e
timeout 1 bash -c 'cat < /dev/null > /dev/tcp/${PROXY_IP}/3128' 2>/dev/null && \
  echo 'Acquire::http::Proxy \"http://${PROXY_IP}:3128\";' > /etc/apt/apt.conf.d/01proxy || true
apt-get update && apt-get install -y git cmake ninja-build build-essential libopenblas-dev libomp-dev
cp -a /tmp/src /tmp/build-src
cd /tmp/build-src
git config --global --add safe.directory /tmp/build-src
echo ggml source commit: \$(git rev-parse --short HEAD) \$(git log -1 --oneline)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
rm -rf build-docker
cmake -S . -B build-docker -G Ninja \
    -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES='86-real' \
    -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DGGML_CUDA_MMQ=ON \
    -DGGML_SCHED_MAX_COPIES=4
cmake --build build-docker --config Release --target rpc-server -- -j\$(nproc)
cp build-docker/bin/rpc-server /workspace/staging/
find build-docker -name '*.so*' -exec cp -v {} /workspace/staging/lib/ \;
echo COMPILE_OK
"

echo "=== runtime image ==="
docker build -f "${DIR}/Dockerfile.runtime" -t atomic-llama-romulus-pathb-rpc:latest "${DIR}"

docker builder prune -f >/dev/null 2>&1 || true
docker images atomic-llama-romulus-pathb-rpc:latest
df -h /
echo "BUILD_OK"
