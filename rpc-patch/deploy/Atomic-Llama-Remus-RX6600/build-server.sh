#!/usr/bin/env bash
# Build the llama-server only image for RX 6600 on remus.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
STAGING="${DIR}/staging"
PROXY_IP="${PROXY_IP:-192.168.8.108}"
GIT_BRANCH="${GIT_BRANCH:-Path-B-Event-Support-Pipeline-Plus}"
GIT_URL="${GIT_URL:-http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git}"
MIN_GB="${MIN_ROOT_GB:-25}"

avail=$(df -BG / | awk 'NR==2 {print $4}')
avail_gb=${avail%G}
echo "root free: ${avail_gb} GB (min ${MIN_GB} GB)"
if [[ "$avail_gb" -lt "$MIN_GB" ]]; then
    echo "ERROR: insufficient root disk" >&2
    exit 1
fi

command -v rocm-smi >/dev/null || { echo "ERROR: rocm-smi missing on host" >&2; exit 1; }
rocm-smi --showproductname --showmeminfo vram | cat

rm -rf "$STAGING"
mkdir -p "$STAGING/lib"

echo "=== Compile llama-server for gfx1030 (RX 6600) using rocm 6.4 ==="
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    -v "${DIR}:/workspace" \
    -w /workspace \
    rocm/dev-ubuntu-22.04:6.4-complete \
    bash -c "
set -e
timeout 1 bash -c 'cat < /dev/null > /dev/tcp/${PROXY_IP}/3128' 2>/dev/null && \
  echo 'Acquire::http::Proxy \"http://${PROXY_IP}:3128\";' > /etc/apt/apt.conf.d/01proxy || true
apt-get update && apt-get install -y git cmake ninja-build build-essential
rm -rf /tmp/src && git clone '${GIT_URL}' /tmp/src
cd /tmp/src && git checkout '${GIT_BRANCH}'
cmake -S . -B build -G Ninja \
    -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS='gfx1030' \
    -DGGML_HIP_ROCWMMA_FATTN=ON
cmake --build build --config Release --target llama-server -- -j\$(nproc)
cp build/bin/llama-server /workspace/staging/
find build -name '*.so*' -exec cp -v {} /workspace/staging/lib/ \;
echo COMPILE_OK
"

echo "=== Build runtime image ==="
docker build -f "${DIR}/Dockerfile.server" -t atomic-llama-remus-rx6600-server:latest "${DIR}"

docker builder prune -f >/dev/null 2>&1 || true
docker images atomic-llama-remus-rx6600-server:latest
df -h /
echo "BUILD_SERVER_OK"
