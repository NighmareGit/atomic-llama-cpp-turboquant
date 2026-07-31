#!/usr/bin/env bash
# Run the PPLUS multi-GPU garble falsification matrix on triton.
# Each case starts a fresh llama-server with chosen env, runs the detector,
# kills the server, records a single verdict line.
#
# Usage: scripts/triton-pplus-falsification-matrix.sh [--model PATH] [--smoke]
#   --smoke : run a single short case (0.8B model, 32 tokens) for sanity
#
# Output: benches/pplus-garble/matrix-<timestamp>.log (and stdout)
# Exit: 0 if all cases completed (verdict codes in log), 1 on infra failure.

set -euo pipefail

cd "$(dirname "$0")/.."

SMOKE=0
MODEL="/mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)    SMOKE=1; shift;;
        --model)    MODEL="$2"; shift 2;;
        -h|--help)  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
        *)          echo "unknown arg: $1" >&2; exit 2;;
    esac
done

TS=$(date +%Y%m%d-%H%M%S)
OUTDIR="benches/pplus-garble"
mkdir -p "$OUTDIR"
OUTLOG="$OUTDIR/matrix-$TS.log"

# matrix spec: LABEL | PLUS | extra env (space-separated) | tokens
# Each row produces one verdict line in $OUTLOG.
if [[ $SMOKE -eq 1 ]]; then
    MODEL="/mnt/980pro/models/Qwen3.5-0.8B.Q4_K_M.gguf"
    ROWS=(
        "green-plus0|0||32"
        "red-plus1|1||32"
    )
else
    ROWS=(
        "green-plus0-control|0||64"
        "red-plus1-baseline|1||64"
        "h1-barrier-partial-off|1|GGML_PIPELINE_BARRIER_PARTIAL=0|64"
        "h5-pipeline-depth-2|1|GGML_SCHED_PIPELINE_DEPTH=2|64"
        "h6-multi-backend-seq|1|GGML_PIPELINE_MULTI_BACKEND_SEQ=1|64"
    )
fi

exec > >(tee -a "$OUTLOG") 2>&1
echo "=== PPLUS garble falsification matrix ==="
echo "timestamp: $TS"
echo "model:     $MODEL"
echo "node:      $(hostname)"
echo "tip:       $(git rev-parse --short HEAD)"
echo "gpus:      $(nvidia-smi -L | tr '\n' '|')"
echo "outlog:    $OUTLOG"
echo

PORT_BASE=8080
for i in "${!ROWS[@]}"; do
    IFS='|' read -r LABEL PLUS EXTRA TOKENS <<< "${ROWS[$i]}"
    PORT=$((PORT_BASE + i))
    echo "--- case $((i+1))/${#ROWS[@]}: label=$LABEL plus=$PLUS extra='$EXTRA' tokens=$TOKENS ---"

    # Build env exports
    ENV_EXPORTS=()
    if [[ -n "$EXTRA" ]]; then
        for kv in $EXTRA; do
            ENV_EXPORTS+=("$kv")
        done
    fi

    # Run the loop with the extra env pre-exported. The loop script uses
    # `env "${SERVER_ENV[@]}" binary ...` so inherited env propagates.
    VERDICT_LINE=""
    set +e
    VERDICT_LINE=$(env "${ENV_EXPORTS[@]}" \
        bash scripts/triton-pplus-garble-loop.sh \
            --plus "$PLUS" \
            --model "$MODEL" \
            --port "$PORT" \
            --tokens "$TOKENS" \
            --seed 42 \
            --prompt "What is the capital of France?" \
            --expect "Paris" \
            --label "$LABEL" \
            --binary ./build-cuda-fresh/bin/llama-server \
        2>&1 | grep -E "^\[pplus-garble\]")
    RC=$?
    set -e

    # Last matching line is the verdict
    echo "$VERDICT_LINE" | tail -1
    echo "    exit_code=$RC"
    echo
done

echo "=== matrix complete ==="
echo "summary log: $OUTLOG"
