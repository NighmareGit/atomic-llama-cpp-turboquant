#!/usr/bin/env bash
# Run on romulus in background: both B+12 bisects (canonical + no-get-defer).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PROFILER_LOCAL=1
export PROFILER_BIN="${ROOT}/build-rocm-docker/bin/llama-pipeline-profiler"
export PROFILER_SKIP_VALIDATE=1
export BENCH_GEN_TOKENS=384
export GGML_PIPELINE_PLUS=1

run_bisect() {
    local bisect="$1" out="$2"
    echo "=== START ${bisect} -> ${out} $(date -u +%FT%TZ) ==="
    PROFILER_OUT_DIR="${ROOT}/benches/path-b-plus/${out}" \
        bash "${ROOT}/scripts/b6-gate-bisect-run.sh" "${bisect}"
    echo "=== DONE ${bisect} $(date -u +%FT%TZ) ==="
}

run_bisect canonical-romulus b6-2gpu-f-triton-n384-romulus-native-b12
run_bisect no-get-defer b6-2gpu-f-triton-n384-romulus-native-no-get-defer
echo "ALL_BISECTS_DONE $(date -u +%FT%TZ)"