#!/bin/bash
# Multi-GPU profiler run for KV-cache fix verification
# Usage: bash run-profiler.sh
set -euo pipefail

MODEL="/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
PROFILER="build-rocm-docker/bin/llama-gpipe-profiler"
OUTDIR="profiler-out"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$OUTDIR"

echo "=== GPipe Profiler: Multi-GPU KV-cache fix ==="
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
echo ""

run_profile() {
  local label="$1"
  local env_vars="$2"
  local rpc_arg="$3"
  local out="$OUTDIR/${TIMESTAMP}-${label}"

  echo "========== $label =========="
  echo "env: $env_vars"
  echo "rpc: $rpc_arg"
  echo "output: $out"

  # Run profiler with --trace for detailed pipeline/sched/rpc traces
  # n-gen=256 for meaningful TG profiling; n-prompt=64 for quick PP
  env $env_vars $PROFILER \
    -m "$MODEL" \
    $rpc_arg \
    --tasks pp,tg \
    -p 64 \
    -n 256 \
    -r 3 \
    --repeat 3 \
    --warmup \
    --trace \
    --server-telemetry \
    --gpipe-stages 2 \
    -ngl 99 \
    --ctx-size 2048 \
    --out-dir "${out}" \
    2>&1 | tee "${out}-log.txt"

  echo "  -> artifacts in ${out}/"
  echo ""
}

# ===== 1-GPU: ROCm only, PIPELINE_PLUS=1 =====
run_profile "1gpu-pplus" \
  "GGML_PIPELINE_PLUS=1" ""

# ===== 2-GPU: ROCm+RPC(local 3060Ti), PIPELINE_PLUS=1 =====
run_profile "2gpu-pplus" \
  "GGML_PIPELINE_PLUS=1 GGML_RPC_ENDPOINTS=127.0.0.1:50051" \
  "--rpc 127.0.0.1:50051"

# ===== 3-GPU: ROCm+RPC(local)+RPC(remus), PIPELINE_PLUS=1 =====
run_profile "3gpu-pplus" \
  "GGML_PIPELINE_PLUS=1 GGML_RPC_ENDPOINTS=127.0.0.1:50051,192.168.8.22:50052" \
  "--rpc 127.0.0.1:50051,192.168.8.22:50052"

# ===== 1-GPU: ROCm only, NO pipeline-plus (baseline) =====
run_profile "1gpu-noplus" "" ""

# ===== 2-GPU: ROCm+RPC, NO pipeline-plus (baseline) =====
run_profile "2gpu-noplus" \
  "GGML_RPC_ENDPOINTS=127.0.0.1:50051" \
  "--rpc 127.0.0.1:50051"

# ===== 3-GPU: ROCm+2xRPC, NO pipeline-plus (baseline) =====
run_profile "3gpu-noplus" \
  "GGML_RPC_ENDPOINTS=127.0.0.1:50051,192.168.8.22:50052" \
  "--rpc 127.0.0.1:50051,192.168.8.22:50052"

echo ""
echo "=== Profiler complete ==="
echo "Results in: $OUTDIR/${TIMESTAMP}-*/"
echo ""
echo "Summary:"
for d in "$OUTDIR/${TIMESTAMP}-"*/; do
  label=$(basename "$d" | sed "s/${TIMESTAMP}-//")
  if [ -f "${d}result.json" ] || [ -f "${d}summary.json" ]; then
    echo "  $label: $(ls ${d}*.json 2>/dev/null | wc -l) result files"
  else
    echo "  $label: log only (check ${d}../*-log.txt)"
  fi
done
