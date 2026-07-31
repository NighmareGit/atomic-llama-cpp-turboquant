#!/usr/bin/env bash
# B+6 gate profiler presets on romulus (see rpc-patch/docs/b6-gate/PLAN.md).
#
# Cluster GPUs (live inventory: scripts/b6-gate-cluster-gpu-inventory.sh):
#   romulus 192.168.8.108 — 7900 XTX (ROCm client) + 3060 Ti (docker :50051)
#   remus   192.168.8.176 — 5060 Ti (:50051) + RX6600 (:50052, unused)
#   triton  192.168.8.23  — 3090 (:50054) + 3070 (:50055)
# Linux 5-GPU = remus(1) + romulus(2) + triton(2); jupiter skipped
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:-}"
shift || true

if [[ -z "$LABEL" || "$LABEL" == "-h" || "$LABEL" == "--help" ]]; then
    echo "usage: $0 <label> [extra profiler args...]"
    echo "labels:"
    echo "  b6-2gpu-f          remus 5060 RPC (gate baseline)"
    echo "  b6-2gpu-f-triton   triton 3090 RPC (spike, same client)"
    echo "  b6-4gpu-g          4-GPU jupiter :50053 (DEPRECATED — use triton)"
    echo "  b6-4gpu-g-triton   4-GPU canonical gate (triton :50054 as RPC2)"
    echo "  b6-3gpu-g          3-GPU Linux (remus 5060 + romulus 3060 + 7900)"
    echo "  b6-3gpu-g-triton   3-GPU A/B (remus 5060 + triton 3090 + 7900)"
    echo "  b6-5gpu-g          5-GPU Linux profiler (baseline; dual-socket off)"
    echo "  b6-5gpu-g-prod     5-GPU production preset (dual-socket ON, see b6-gate-5gpu-production-env.sh)"
    echo "  b6-2gpu-f-plus0    2-GPU remus, GGML_PIPELINE_PLUS=0"
    echo "  b6-2gpu-jupiter    2-GPU remus CUDA + JUPITER 5070 :50053"
    echo "  b6-2gpu-jupiter-plus0  same, GGML_PIPELINE_PLUS=0"
    echo ""
    echo "OFF bisects: bash scripts/b6-gate-bisect-run.sh <no-partial|no-async-copy|no-defer|no-get-defer|no-dual-socket|canonical-romulus>"
    exit 0
fi

export PATHB_ROMULUS_SSH_PASS="${PATHB_ROMULUS_SSH_PASS:-12345}"
export PROFILER_BIN="${PROFILER_BIN:-${ROOT}/build-rocm-docker/bin/llama-pipeline-profiler}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-${ROOT}/build-rocm-docker/bin:/opt/rocm/lib}"
export BENCH_MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
export BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-384}"
export BENCH_PROMPT_FILE="${BENCH_PROMPT_FILE:-${ROOT}/benches/path-b-plus/prompts/profiler-reasoning-long.txt}"
export PROFILER_MODE="${PROFILER_MODE:-trace}"
# Run on romulus via SSH when dev host has no local ROCm profiler build.
if [[ -x "${PROFILER_BIN}" || -f "${PROFILER_BIN}" ]]; then
    export PROFILER_LOCAL=1
else
    export PROFILER_LOCAL=0
    export PATHB_ROMULUS_SSH="${PATHB_ROMULUS_SSH:-user@192.168.8.108}"
    export PATHB_ROMULUS_REPO="${PATHB_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
    export PATHB_ROMULUS_PROFILER="${PATHB_ROMULUS_PROFILER:-${PATHB_ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler}"
    if [[ -n "${PROFILER_OUT_DIR:-}" ]]; then
        export PATHB_ROMULUS_OUT="${PATHB_ROMULUS_OUT:-${PATHB_ROMULUS_REPO}/benches/path-b-plus/$(basename "$PROFILER_OUT_DIR")}"
    fi
fi
export BENCH_TRACE=1
export B6_CLIENT_KIND="${B6_CLIENT_KIND:-rocm-native}"

case "$LABEL" in
    b6-2gpu-f)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-2gpu-f-triton)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.23:50054}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-4gpu-g)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053}"
        # ts order: RPC0 remus 5060, RPC1 romulus 3060 docker (127.0.0.1), RPC2 JUPITER 5070, ROCm0 7900
        export BENCH_TS="${BENCH_TS:-25,12,25,38}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-4gpu-g-triton)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054}"
        # ts order: RPC0 remus 5060, RPC1 romulus 3060 docker, RPC2 triton 3090, ROCm0 7900 (VRAM ~22/11/34/33)
        export BENCH_TS="${BENCH_TS:-22,11,34,33}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-3gpu-g)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051}"
        export BENCH_TS="${BENCH_TS:-50,28,22}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-3gpu-g-triton)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,192.168.8.23:50054}"
        export BENCH_TS="${BENCH_TS:-30,35,35}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-5gpu-g|b6-5gpu-g-prod)
        # shellcheck source=scripts/b6-gate-5gpu-production-env.sh
        source "${ROOT}/scripts/b6-gate-5gpu-production-env.sh"
        if [[ "$LABEL" == "b6-5gpu-g-prod" ]]; then
            b6_5gpu_production_env
        else
            b6_5gpu_base_env
        fi
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-2gpu-f-plus0)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS=0
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-2gpu-jupiter)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.21:50053}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    b6-2gpu-jupiter-plus0)
        export BENCH_RPC_ENDPOINT="${BENCH_RPC_ENDPOINT:-192.168.8.21:50053}"
        export BENCH_TS="${BENCH_TS:-50,50}"
        export GGML_PIPELINE_PLUS=0
        export PROFILER_OUT_DIR="${PROFILER_OUT_DIR:-${ROOT}/benches/path-b-plus/${LABEL}}"
        ;;
    *)
        echo "error: unknown label: $LABEL" >&2
        exit 1
        ;;
esac

cd "$ROOT"
exec bash scripts/llama-pipeline-profiler-cluster.sh "$LABEL" \
    -ctk q8_0 -ctv q8_0 -ngl 99 --no-warmup \
    --with-gpu-telemetry --trace-sample 5 \
    --regression-file "${ROOT}/benches/path-b-plus/regression.jsonl" \
    "$@"