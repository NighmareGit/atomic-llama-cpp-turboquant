#!/usr/bin/env bash
# Run on romulus in background: B+11 dual-socket bisect on 4-GPU gate (canonical + no-dual-socket).
# Requires proto 4.4 rpc-server on ALL endpoints (remus :50051, jupiter :50053).
#
# usage: bash scripts/b6-gate-romulus-b11-4gpu-bisect-bg.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/benches/path-b-plus/b11-4gpu-bisect-logs"
mkdir -p "$LOG_DIR"

run_bisect() {
    local bisect="$1" out="$2"
    local log="${LOG_DIR}/${out}.log"
    echo "=== START ${bisect} -> ${out} $(date -u +%FT%TZ) ===" | tee -a "$log"
    (
        export B6_GATE_PRESET=b6-4gpu-g
        bash "${ROOT}/scripts/b6-gate-bisect-run.sh" "${bisect}"
    ) >>"$log" 2>&1
    echo "=== DONE ${bisect} $(date -u +%FT%TZ) ===" | tee -a "$log"
}

# B+11 ON arm: dual-socket explicit (requires proto 4.4 on all rpc-servers).
export GGML_RPC_DUAL_SOCKET=1
run_bisect canonical-romulus b6-4gpu-g-n384-romulus-native-b11
run_bisect no-dual-socket b6-4gpu-g-n384-romulus-native-no-dual-socket