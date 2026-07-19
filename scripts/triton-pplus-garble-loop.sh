#!/usr/bin/env bash
# Phase 1+2 repro loop for the PPLUS multi-GPU garble bug (triton, dual CUDA 3090+3070).
# Starts llama-server with GGML_PIPELINE_PLUS, runs the garble detector, kills the server,
# and exits with the detector's code (0=CLEAN/PASS, 1=GARBLED/FAIL).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- defaults ----
PLUS=1
MODEL="/mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf"
TS="75,25"
PORT=8080
PROMPT="What is the capital of France?"
EXPECT="Paris"
TOKENS=64
SEED=42
SINGLE_GPU=0
NGL=99
CTX=4096
NO_WARMUP=1        # --no-warmup passed to llama-server by default
JINJA=1            # --jinja passed to llama-server by default
BINARY="${ROOT}/build-cuda-fresh/bin/llama-server"
KEEP_SERVER=0
LABEL=""

usage() {
    cat <<EOF
usage: $(basename "$0") [options]
  --plus 0|1              GGML_PIPELINE_PLUS (default 1)
  --model PATH            GGUF path (default /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf)
  --ts A,B                split ratio (default 75,25)
  --port N                server port (default 8080)
  --prompt TEXT           user prompt
  --expect SUBSTR         expected substring (semantic assert; detector TODO)
  --tokens N              max tokens to generate (default 64)
  --seed N                sampling seed (default 42)
  --single-gpu            use only CUDA0 (CUDA_VISIBLE_DEVICES=0, -ts 100)
  --ngl N                 gpu layers (default 99)
  --ctx N                 context size (default 4096)
  --no-warmup             pass --no-warmup to llama-server (default on)
  --jinja                 pass --jinja to llama-server (default on)
  --binary PATH           llama-server binary
  --keep-server           do not kill server after the run
  --label LABEL           label printed in the verdict line
  -h, --help              show this help
EOF
}

# ---- arg parse ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plus)      PLUS="$2";       shift 2 ;;
        --model)     MODEL="$2";      shift 2 ;;
        --ts)        TS="$2";         shift 2 ;;
        --port)      PORT="$2";       shift 2 ;;
        --prompt)    PROMPT="$2";     shift 2 ;;
        --expect)    EXPECT="$2";     shift 2 ;;
        --tokens)    TOKENS="$2";     shift 2 ;;
        --seed)      SEED="$2";       shift 2 ;;
        --single-gpu) SINGLE_GPU=1;   shift ;;
        --ngl)       NGL="$2";        shift 2 ;;
        --ctx)       CTX="$2";        shift 2 ;;
        --no-warmup) NO_WARMUP=1;     shift ;;
        --jinja)     JINJA=1;         shift ;;
        --binary)    BINARY="$2";     shift 2 ;;
        --keep-server) KEEP_SERVER=1; shift ;;
        --label)     LABEL="$2";      shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "error: unknown arg $1" >&2; usage >&2; exit 1 ;;
    esac
done

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${ROOT}/benches/pplus-garble"
mkdir -p "$LOGDIR"
LOGPATH="${LOGDIR}/${STAMP}-plus${PLUS}.log"

# ---- sanity: GPU presence ----
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
echo "$GPU_COUNT GPU(s) detected:"
nvidia-smi -L 2>/dev/null || true
if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "error: no GPU found (nvidia-smi)" >&2
    exit 1
fi
if [[ "$SINGLE_GPU" -eq 0 && "$GPU_COUNT" -lt 2 ]]; then
    echo "error: need >=2 GPUs for multi-GPU (found ${GPU_COUNT}); use --single-gpu for single GPU" >&2
    exit 1
fi

# ---- binary check ----
if [[ ! -x "$BINARY" ]]; then
    echo "error: llama-server binary not found or not executable: ${BINARY}" >&2
    exit 1
fi

# ---- pre-clean any server on $PORT ----
pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
sleep 1

# ---- server env + args ----
export GGML_PIPELINE_PLUS="${PLUS}"
SERVER_ENV=()
if [[ "$SINGLE_GPU" -eq 1 ]]; then
    SERVER_ENV+=(CUDA_VISIBLE_DEVICES=0)
    TS="100"
fi

SERVER_ARGS=(
    -m "$MODEL"
    -ngl "$NGL"
    -c "$CTX"
    --host 127.0.0.1
    --port "$PORT"
    --split-mode layer
    -ts "$TS"
    --metrics
    --log-timestamps
    --log-prefix
)
[[ "$NO_WARMUP" -eq 1 ]] && SERVER_ARGS+=(--no-warmup)
[[ "$JINJA"     -eq 1 ]] && SERVER_ARGS+=(--jinja)

# ---- lifecycle helpers ----
SERVER_PID=""
kill_server() {
    [[ -z "${SERVER_PID:-}" ]] && return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 1
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
            sleep 1
        fi
    fi
    SERVER_PID=""
}
cleanup() {
    if [[ "$KEEP_SERVER" -ne 1 ]]; then
        kill_server
    fi
}
trap cleanup EXIT

# ---- start server ----
echo "starting llama-server (PLUS=${PLUS} ts=${TS} port=${PORT}) ..."
echo "  binary: ${BINARY}"
echo "  log:    ${LOGPATH}"
# shellcheck disable=SC2086
env "${SERVER_ENV[@]}" "$BINARY" "${SERVER_ARGS[@]}" >"$LOGPATH" 2>&1 &
SERVER_PID=$!

# ---- wait for readiness (up to 120s) ----
wait_max=120
waited=0
while [[ $waited -lt $wait_max ]]; do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "server ready at ${waited}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "error: llama-server exited during load" >&2
        tail -40 "$LOGPATH" >&2 || true
        exit 2
    fi
    sleep 1
    waited=$((waited + 1))
done
if [[ $waited -ge $wait_max ]]; then
    echo "error: server not ready after ${wait_max}s (port ${PORT})" >&2
    tail -40 "$LOGPATH" >&2 || true
    exit 2
fi

# ---- run detector ----
# TODO: loop-check-garble.sh currently accepts only 4 args (PORT PROMPT TOKENS SEED).
# The EXPECT (--expect) semantic-assert arg is not yet wired into the detector.
# Once the detector fix lands, pass "$EXPECT" as the 5th arg here.
DETECTOR="${ROOT}/rpc-patch/patch/loop-check-garble.sh"
DETECTOR_RC=0
bash "$DETECTOR" "$PORT" "$PROMPT" "$TOKENS" "$SEED" || DETECTOR_RC=$?

# ---- verdict ----
if [[ "$DETECTOR_RC" -eq 0 ]]; then
    VERDICT="CLEAN"
else
    VERDICT="GARBLED"
fi
echo "[pplus-garble] PLUS=${PLUS} label=${LABEL} verdict=${VERDICT} model=$(basename "$MODEL") ts=${TS} log=${LOGPATH}"

exit "$DETECTOR_RC"
