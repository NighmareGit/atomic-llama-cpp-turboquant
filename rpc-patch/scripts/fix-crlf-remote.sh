#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/home/hunter/atomic-llama-cpp-turboquant}"
find "$ROOT/rpc-patch/scripts" -name '*.sh' -exec sed -i 's/\r$//' {} +
sed -i 's/\r$//' "$ROOT/../bench-5gpu.sh" 2>/dev/null || sed -i 's/\r$//' /home/hunter/bench-5gpu.sh 2>/dev/null || true
head -1 "$ROOT/rpc-patch/scripts/rpc-server-bench.sh" | od -c | head -2
echo FIX_CRLF_OK