#!/usr/bin/env bash
# Live VRAM + tensor-split preflight for Path B RPC bench/deploy.
#
# usage:
#   ./scripts/pathb-rpc-vram-preflight.sh --preset b6-4gpu-g-triton --gguf /mnt/models/foo.gguf
#   ./scripts/pathb-rpc-vram-preflight.sh --preset b6-2gpu-f-triton --gguf foo.gguf --strict
#
# See: rpc-patch/scripts/pathb-rpc-vram-preflight.py

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "${ROOT}/rpc-patch/scripts/pathb-rpc-vram-preflight.py" "$@"