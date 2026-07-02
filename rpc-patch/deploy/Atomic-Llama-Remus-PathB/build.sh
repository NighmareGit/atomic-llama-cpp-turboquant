#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
STAGING="${DIR}/staging"
PROXY_IP="${PROXY_IP:-192.168.8.108}"
GIT_BRANCH="${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
GIT_URL="${GIT_URL:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"
GIT_COMMIT="${GIT_COMMIT:-}"
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

echo "=== GPU compile stage (CUDA 12.8, sm_120a for RTX 5060 Ti) ==="
docker run --rm --gpus=all \
    -v "${DIR}:/workspace" \
    -w /workspace \
    nvidia/cuda:12.8.0-devel-ubuntu22.04 \
    bash -c "
set -e
timeout 1 bash -c 'cat < /dev/null > /dev/tcp/${PROXY_IP}/3128' 2>/dev/null && \
  echo 'Acquire::http::Proxy \"http://${PROXY_IP}:3128\";' > /etc/apt/apt.conf.d/01proxy || true
apt-get update && apt-get install -y git cmake ninja-build build-essential libopenblas-dev libomp-dev
rm -rf /tmp/src && git clone '${GIT_URL}' /tmp/src
cd /tmp/src && git checkout '${GIT_BRANCH}'
[ -n '${GIT_COMMIT}' ] && git checkout '${GIT_COMMIT}' || true
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
cmake -S . -B build -G Ninja \
    -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES='120a-real' \
    -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DGGML_CUDA_MMQ=ON \
    -DGGML_SCHED_MAX_COPIES=4
cmake --build build --config Release --target rpc-server llama-server -- -j\$(nproc)
cp build/bin/rpc-server build/bin/llama-server /workspace/staging/
find build -name '*.so*' -exec cp -v {} /workspace/staging/lib/ \;
echo COMPILE_OK
"

echo "=== runtime image ==="
docker build -f "${DIR}/Dockerfile.runtime" -t atomic-llama-remus-pathb-rpc:latest "${DIR}"

docker builder prune -f >/dev/null 2>&1 || true
docker images atomic-llama-remus-pathb-rpc:latest
df -h /
echo "BUILD_OK"