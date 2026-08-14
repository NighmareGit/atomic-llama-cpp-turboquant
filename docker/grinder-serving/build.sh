#!/usr/bin/env bash
# Build the grinder-serving hub image, pinned to good-prototype HEAD.
#
# Usage (from the fork root):
#   ./docker/grinder-serving/build.sh          # pin to current good-prototype HEAD
#   SOURCE_COMMIT=abc123 ./docker/grinder-serving/build.sh  # pin explicitly
#
# Produces:
#   grinder-serving:<commit>   (the durable, reproducible tag)
#   grinder-serving:latest     (convenience alias — the moving "stable" pointer)
#
# Requires: docker with buildkit, the fork checked out at good-prototype.
set -euo pipefail

FORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FORK_ROOT"

# K1 — read-only source: never modify tracked files. We only read the commit.
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git rev-parse good-prototype)}"
SHORT_COMMIT="${SOURCE_COMMIT:0:9}"

IMAGE_TAG="grinder-serving:${SHORT_COMMIT}"
IMAGE_LATEST="grinder-serving:latest"

echo "[build.sh] pinning to good-prototype @ ${SOURCE_COMMIT}"
echo "[build.sh] tagging ${IMAGE_TAG} (+ ${IMAGE_LATEST})"

# Build from a lean tar context: the tracked .dockerignore excludes build*/
# but NOT worktrees/ (84GB of full repo copies), which balloons the context
# to 90+GB and fills the disk. Piping a tar that excludes worktrees (and the
# other big non-source dirs) keeps the context to ~3GB. This respects the
# no-edit rule on the tracked .dockerignore.
tar --exclude='./worktrees' --exclude='./tools/ui/node_modules' \
    --exclude='./build*' --exclude='./benches' -cf - . 2>/dev/null \
  | docker build \
      --build-arg SOURCE_COMMIT="${SOURCE_COMMIT}" \
      -t "${IMAGE_TAG}" \
      -t "${IMAGE_LATEST}" \
      -f docker/grinder-serving/Dockerfile \
      -

echo
echo "[build.sh] built:"
echo "  ${IMAGE_TAG}"
echo "  ${IMAGE_LATEST}"
echo
echo "[build.sh] verify on the 3060 Ti (GPU 0):"
echo "  docker run --rm --gpus device=0 -p 8094:8080 \\"
echo "    -v <model-mount>/models:/models \\"
echo "    ${IMAGE_LATEST} \\"
echo "    -m <project-root>/Qwen3.5-9B-MTP-Q4_K_M.gguf \\"
echo "    --host 0.0.0.0 --port 8080 -ngl 999 -fa --ctx-size 8192 \\"
echo "    -p 'Building a mobile app can be done in 15 steps:'"
