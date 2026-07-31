#!/usr/bin/env bash
# Path B Config A test runner: ROCm client + CUDA RPC worker.
# Polls in-container logs during model load, validates output (no death-loops).
#
# usage:
#   pathb-run-test.sh --label NAME --model PATH [options]
#
# options:
#   --load-timeout SEC   default 180 (<36B), use 600 for 72B+
#   --gen-timeout SEC    default 300 (<36B), use 600 for 72B+
#   --prompt TEXT
#   --n-predict N        default 64
#   --ctk TYPE           default q4_0  (allowed: f32,f16,bf16,q8_0,q4_0,...)
#   --ctv TYPE           default q4_0
#   --ngl N              default 99
#   --split-mode MODE    default layer
#   --tensor-split TS    e.g. 3,1 (ROCm,RPC proportions); empty = omit flag
#   --ctx N              default 4096; omit flag if 0
#   --extra-args STR     extra llama-cli args
#
# env:
#   PATHB_RPC_CONTAINER  default pathb-rpc
#   PATHB_BIN_ROCM       default $ROOT/build-rocm-docker/bin
#   PATHB_LOG_DIR        default patch/bench-results/pathb-runs
#   PATHB_ROCM_IMAGE     default llama-rocm-patched
#   PATHB_RPC_ENDPOINT   default 127.0.0.1:50051

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"

LABEL=""
MODEL=""
LOAD_TIMEOUT=180
GEN_TIMEOUT=300
PROMPT="The quick brown fox jumps over the lazy dog."
N_PREDICT=64
CTK="q4_0"
CTV="q4_0"
NGL=99
SPLIT_MODE="layer"
TENSOR_SPLIT=""
CTX=4096
EXTRA_ARGS=""

PATHB_RPC_CONTAINER="${PATHB_RPC_CONTAINER:-pathb-rpc}"
PATHB_BIN_ROCM="${PATHB_BIN_ROCM:-${TQ}/build-rocm-docker/bin}"
PATHB_LOG_DIR="${PATHB_LOG_DIR:-${RPC_PATCH_ROOT}/patch/bench-results/pathb-runs}"
PATHB_ROCM_IMAGE="${PATHB_ROCM_IMAGE:-llama-rocm-patched}"
PATHB_RPC_ENDPOINT="${PATHB_RPC_ENDPOINT:-127.0.0.1:50051}"

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --label) LABEL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --load-timeout) LOAD_TIMEOUT="$2"; shift 2 ;;
        --gen-timeout) GEN_TIMEOUT="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --n-predict) N_PREDICT="$2"; shift 2 ;;
        --ctk) CTK="$2"; shift 2 ;;
        --ctv) CTV="$2"; shift 2 ;;
        --ngl) NGL="$2"; shift 2 ;;
        --split-mode) SPLIT_MODE="$2"; shift 2 ;;
        --tensor-split) TENSOR_SPLIT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$LABEL" && -n "$MODEL" ]] || usage 1
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }
[[ -x "${PATHB_BIN_ROCM}/llama-cli" ]] || { echo "llama-cli missing: ${PATHB_BIN_ROCM}/llama-cli" >&2; exit 1; }

mkdir -p "$PATHB_LOG_DIR"
LOG="${PATHB_LOG_DIR}/${LABEL}.log"
META="${PATHB_LOG_DIR}/${LABEL}.meta"
LOG_RAW="${PATHB_LOG_DIR}/${LABEL}.raw"

: >"$LOG"
: >"$META"

echo "=== $LABEL ===" | tee -a "$META"
echo "model=$MODEL" | tee -a "$META"
echo "rpc=$PATHB_RPC_ENDPOINT ngl=$NGL split=$SPLIT_MODE ts=${TENSOR_SPLIT:-auto} ctk=$CTK ctv=$CTV" | tee -a "$META"
echo "load_timeout=${LOAD_TIMEOUT}s gen_timeout=${GEN_TIMEOUT}s" | tee -a "$META"
date -u +"%Y-%m-%dT%H:%M:%SZ" | tee -a "$META"

if ! docker ps --format '{{.Names}}' | grep -qx "$PATHB_RPC_CONTAINER"; then
    echo "ERROR: RPC container '$PATHB_RPC_CONTAINER' not running. Start with: scripts/pathb-start-rpc.sh" | tee -a "$META"
    exit 1
fi

CONTAINER="pathb-${LABEL}"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# Optional debug env (host -> container)
DOCKER_DEBUG_ENV=()
[[ -n "${GGML_SCHED_DEBUG:-}" ]] && DOCKER_DEBUG_ENV+=(-e "GGML_SCHED_DEBUG=${GGML_SCHED_DEBUG}")
[[ -n "${GGML_RPC_DEBUG:-}" ]] && DOCKER_DEBUG_ENV+=(-e "GGML_RPC_DEBUG=${GGML_RPC_DEBUG}")

# Pass prompt/model via env to avoid shell-quoting issues in docker -c
cid=$(docker run -d --name "$CONTAINER" --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri --group-add video --network host \
    -v "${PATHB_BIN_ROCM}:/app/bin:ro" -v /mnt/models:/mnt/models:ro \
    -e LD_LIBRARY_PATH=/app/bin \
    "${DOCKER_DEBUG_ENV[@]}" \
    -e PATHB_PROMPT="$PROMPT" \
    -e PATHB_MODEL="$MODEL" \
    -e PATHB_RPC="$PATHB_RPC_ENDPOINT" \
    -e PATHB_N_PREDICT="$N_PREDICT" \
    -e PATHB_NGL="$NGL" \
    -e PATHB_CTK="$CTK" \
    -e PATHB_CTV="$CTV" \
    -e PATHB_SPLIT_MODE="$SPLIT_MODE" \
    -e PATHB_TENSOR_SPLIT="$TENSOR_SPLIT" \
    -e PATHB_CTX="$CTX" \
    -e PATHB_EXTRA_ARGS="$EXTRA_ARGS" \
    -e PATHB_GEN_TIMEOUT="$GEN_TIMEOUT" \
    "$PATHB_ROCM_IMAGE" \
    -c 'exec > /tmp/run.log 2>&1
echo START-$(date +%s)
TS_ARGS=""
[[ -n "$PATHB_TENSOR_SPLIT" ]] && TS_ARGS="-ts $PATHB_TENSOR_SPLIT"
CTX_ARGS=""
[[ "$PATHB_CTX" != "0" ]] && CTX_ARGS="-c $PATHB_CTX"
timeout "$PATHB_GEN_TIMEOUT" stdbuf -oL -eL /app/bin/llama-cli \
    --rpc "$PATHB_RPC" -m "$PATHB_MODEL" -p "$PATHB_PROMPT" -n "$PATHB_N_PREDICT" \
    -ngl "$PATHB_NGL" -ctk "$PATHB_CTK" -ctv "$PATHB_CTV" -sm "$PATHB_SPLIT_MODE" \
    $TS_ARGS $CTX_ARGS --single-turn --simple-io --no-conversation $PATHB_EXTRA_ARGS
echo EXIT-$?-$(date +%s)')

echo "container=$cid" | tee -a "$META"

start=$(date +%s)
while true; do
    elapsed=$(( $(date +%s) - start ))
    if ! docker ps -q --filter "name=^${CONTAINER}$" | grep -q .; then
        echo "container exited at ${elapsed}s" | tee -a "$META"
        break
    fi
    docker exec "$CONTAINER" sh -c 'test -f /tmp/run.log && tail -c 65536 /tmp/run.log' 2>/dev/null \
        | strings | grep -vE '^[\\|/-]+$' | tail -12 >"$LOG" || true

    if grep -qE 'Prompt:.*t/s|Generation:.*t/s' "$LOG" 2>/dev/null; then
        echo "inference done at ${elapsed}s" | tee -a "$META"
        break
    fi
    if [[ "$elapsed" -ge "$LOAD_TIMEOUT" ]]; then
        if grep -qE 'available commands:' "$LOG" 2>/dev/null && ! grep -qE 'Generation:' "$LOG"; then
            echo "TIMEOUT: model loaded but no generation (interactive hang?)" | tee -a "$META"
        else
            echo "TIMEOUT: load phase ${elapsed}s" | tee -a "$META"
        fi
        tail -8 "$LOG" | tee -a "$META"
        docker logs "$PATHB_RPC_CONTAINER" 2>&1 | tail -5 | tee -a "$META"
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        echo "RESULT=TIMEOUT" | tee -a "$META"
        exit 2
    fi
    echo "poll ${elapsed}s: $(tail -1 "$LOG" 2>/dev/null | head -c 100)" | tee -a "$META"
    sleep 10
done

for _ in $(seq 1 30); do
    docker ps -q --filter "name=^${CONTAINER}$" | grep -q . || break
    sleep 2
done

docker cp "${CONTAINER}:/tmp/run.log" "$LOG_RAW" 2>/dev/null || true
if [[ -f "$LOG_RAW" ]]; then
    strings "$LOG_RAW" | grep -vE '^[\\|/-]+$' >"$LOG" || true
fi
if [[ ! -s "$LOG" ]]; then
    docker logs "$CONTAINER" 2>&1 | strings | grep -vE '^[\\|/-]+$' >"$LOG" || true
fi

exit_code=$(docker inspect "$CONTAINER" --format '{{.State.ExitCode}}' 2>/dev/null || echo -1)
echo "exit_code=$exit_code" | tee -a "$META"

fail=0
if grep -qE 'Generation: 1000000|Generation: 0\.0 t/s' "$LOG"; then
    echo "FAIL: bogus generation metric" | tee -a "$META"
    fail=1
fi
if ! grep -qE 'Generation: [0-9]' "$LOG"; then
    echo "FAIL: no generation throughput line" | tee -a "$META"
    fail=1
fi
if grep -qE 'abort|Remote RPC server crashed|malformed response' "$LOG"; then
    echo "FAIL: RPC abort" | tee -a "$META"
    fail=1
fi
if awk '/^> /{p=1} p' "$LOG" | sort | uniq -c | awk '$1>20{exit 1}'; then
    :
else
    echo "FAIL: possible death-loop output" | tee -a "$META"
    fail=1
fi

grep -E 'Prompt:|Generation:|EXIT-' "$LOG" | tee -a "$META"
tail -20 "$LOG" | tee -a "$META"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

if [[ "$fail" -ne 0 || "$exit_code" != "0" ]]; then
    echo "RESULT=FAIL" | tee -a "$META"
    exit 1
fi
echo "RESULT=PASS" | tee -a "$META"