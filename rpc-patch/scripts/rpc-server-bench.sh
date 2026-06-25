#!/usr/bin/env bash
# Single-instance llama-server (ROCm) <-> rpc-server (CUDA) benchmark.
# Ensures at most one rpc + one llama-server; distinct GPUs.
#
# usage:
#   rpc-server-bench.sh <variant> <label>
#
# variants:
#   a1        Path A1 image binaries (llama-rpc-cuda + llama-rocm-patched)
#   a1a2      Path A1+A2 image binaries (llama-rpc-cuda-a2 + llama-rocm-patched)
#   pathb     Path B git workspace builds (build-cuda-b-bin + build-rocm-docker)
#
# env:
#   BENCH_MODEL, BENCH_CTX, BENCH_CTK, BENCH_CTV, BENCH_NGL, BENCH_TS, BENCH_LOAD_TIMEOUT
#   BENCH_PORT=8081, BENCH_RPC_PORT=50051, BENCH_GEN_TOKENS=64
#   BENCH_EXTRA (extra llama-server args), BENCH_NCMOE, BENCH_NP (parallel slots, default 1 for large)
#   BENCH_NO_WARMUP=1 adds --no-warmup (recommended for >27B Path B)

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
VARIANT="${1:?variant required: a1|a1a2|pathb}"
LABEL="${2:?label required}"

MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf}"
CTX="${BENCH_CTX:-8192}"
CTK="${BENCH_CTK:-q8_0}"
CTV="${BENCH_CTV:-turbo3}"
NGL="${BENCH_NGL:-99}"
TS="${BENCH_TS:-1,1}"
SPLIT_MODE="${BENCH_SPLIT_MODE:-layer}"
LOAD_TIMEOUT="${BENCH_LOAD_TIMEOUT:-180}"
GEN_TOKENS="${BENCH_GEN_TOKENS:-64}"
PORT="${BENCH_PORT:-8081}"
RPC_PORT="${BENCH_RPC_PORT:-50051}"
RUNS="${BENCH_RUNS:-3}"
LOG_DIR="${BENCH_LOG_DIR:-${RPC_PATCH_ROOT}/patch/bench-results/rpc-server-bench}"
EXTRA="${BENCH_EXTRA:-}"
NCMOE="${BENCH_NCMOE:-}"
NP="${BENCH_NP:-1}"
NO_WARMUP="${BENCH_NO_WARMUP:-1}"

SERVER_EXTRA=()
[[ "$NO_WARMUP" == "1" ]] && SERVER_EXTRA+=(--no-warmup)
[[ -n "$NP" ]] && SERVER_EXTRA+=(-np "$NP")
[[ -n "$NCMOE" ]] && SERVER_EXTRA+=(--n-cpu-moe "$NCMOE")
[[ -n "$EXTRA" ]] && read -r -a _extra <<<"$EXTRA" && SERVER_EXTRA+=("${_extra[@]}")

RPC_NAME="bench-rpc"
LLAMA_NAME="bench-llama"

mkdir -p "$LOG_DIR"
META="${LOG_DIR}/${LABEL}.meta"
RESULT="${LOG_DIR}/${LABEL}.result"
: >"$META"

log() { echo "$*" | tee -a "$META"; }

cleanup() {
    docker rm -f "$LLAMA_NAME" "$RPC_NAME" 2>/dev/null || true
}

case "$VARIANT" in
    a1)
        CUDA_IMAGE="llama-rpc-cuda"
        CUDA_BIN="/app/build/bin"
        ROCM_IMAGE="llama-rocm-patched"
        ROCM_BIN="/app/llama.cpp/build/bin"
        PROTO="A1 (v4.1.0 image)"
        ;;
    a1a2)
        CUDA_IMAGE="llama-rpc-cuda-a2"
        CUDA_BIN="/app/build/bin"
        ROCM_IMAGE="llama-rocm-patched"
        ROCM_BIN="/app/llama.cpp/build/bin"
        PROTO="A1+A2 (v4.1.0 image)"
        ;;
    pathb)
        CUDA_IMAGE="llama-rpc-cuda-a2"
        CUDA_BIN_HOST="${TQ}/build-cuda-b-bin/bin"
        ROCM_IMAGE="llama-rocm-patched"
        ROCM_BIN_HOST="${TQ}/build-rocm-docker/bin"
        PROTO="Path B (v4.2.2 workspace)"
        ;;
    *)
        echo "unknown variant: $VARIANT" >&2
        exit 1
        ;;
esac

log "=== $LABEL ==="
log "variant=$VARIANT proto=$PROTO"
log "model=$MODEL ctx=$CTX ctk=$CTK ctv=$CTV ngl=$NGL ts=$TS ncmoe=${NCMOE:-none} np=$NP no_warmup=$NO_WARMUP load_timeout=${LOAD_TIMEOUT}s"
log "server_extra=${SERVER_EXTRA[*]:-none}"
date -u +"%Y-%m-%dT%H:%M:%SZ" | tee -a "$META"

cleanup

# NVIDIA 3060 Ti worker only
log "starting rpc-server (CUDA device 0)..."
if [[ "$VARIANT" == "pathb" ]]; then
    docker run -d --name "$RPC_NAME" \
        --gpus "device=0" --network host \
        -v "${CUDA_BIN_HOST}:/app/bin:ro" \
        -e LD_LIBRARY_PATH=/app/bin \
        -e CUDA_VISIBLE_DEVICES=0 \
        "$CUDA_IMAGE" \
        bash -c "/app/bin/rpc-server -H 0.0.0.0 -p ${RPC_PORT} -d CUDA0" >>"$META" 2>&1
else
    docker run -d --name "$RPC_NAME" \
        --gpus "device=0" --network host \
        -e CUDA_VISIBLE_DEVICES=0 \
        --entrypoint "${CUDA_BIN}/rpc-server" \
        "$CUDA_IMAGE" \
        -H 0.0.0.0 -p "${RPC_PORT}" -d CUDA0 >>"$META" 2>&1
fi
sleep 2

# AMD 7900 XTX client only
log "starting llama-server (ROCm HIP device 0)..."
if [[ "$VARIANT" == "pathb" ]]; then
    docker run -d --name "$LLAMA_NAME" --entrypoint bash \
        --device=/dev/kfd --device=/dev/dri --group-add video --network host \
        -v "${ROCM_BIN_HOST}:/app/bin:ro" -v /mnt/models:/mnt/models:ro \
        -e LD_LIBRARY_PATH=/app/bin -e HIP_VISIBLE_DEVICES=0 \
        "$ROCM_IMAGE" \
        -c "exec > /tmp/server.log 2>&1; /app/bin/llama-server \
            --rpc 127.0.0.1:${RPC_PORT} -m ${MODEL} -ngl ${NGL} -c ${CTX} \
            -ctk ${CTK} -ctv ${CTV} -sm ${SPLIT_MODE} -ts ${TS} \
            --host 127.0.0.1 --port ${PORT} ${SERVER_EXTRA[*]}" >>"$META" 2>&1
else
    docker run -d --name "$LLAMA_NAME" --entrypoint bash \
        --device=/dev/kfd --device=/dev/dri --group-add video --network host \
        -v /mnt/models:/mnt/models:ro \
        -e HIP_VISIBLE_DEVICES=0 \
        "$ROCM_IMAGE" \
        -c "exec > /tmp/server.log 2>&1; ${ROCM_BIN}/llama-server \
            --rpc 127.0.0.1:${RPC_PORT} -m ${MODEL} -ngl ${NGL} -c ${CTX} \
            -ctk ${CTK} -ctv ${CTV} -sm ${SPLIT_MODE} -ts ${TS} \
            --host 127.0.0.1 --port ${PORT} ${SERVER_EXTRA[*]}" >>"$META" 2>&1
fi

log "polling server health (max ${LOAD_TIMEOUT}s)..."
start=$(date +%s)
ready=0
while true; do
    elapsed=$(( $(date +%s) - start ))
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        log "server ready at ${elapsed}s"
        break
    fi
    if ! docker ps -q --filter "name=^${LLAMA_NAME}$" | grep -q .; then
        log "ERROR: llama-server container exited during load"
        docker cp "${LLAMA_NAME}:/tmp/server.log" "${LOG_DIR}/${LABEL}-server.log" 2>/dev/null || \
            docker logs "$LLAMA_NAME" >"${LOG_DIR}/${LABEL}-server.log" 2>&1 || true
        tail -30 "${LOG_DIR}/${LABEL}-server.log" | tee -a "$META"
        cleanup
        exit 2
    fi
    if [[ "$elapsed" -ge "$LOAD_TIMEOUT" ]]; then
        log "TIMEOUT: server not healthy after ${LOAD_TIMEOUT}s"
        docker exec "$LLAMA_NAME" sh -c 'tail -c 32768 /tmp/server.log' 2>/dev/null \
            | strings | grep -vE '^[\\|/-]+$' | tail -15 | tee -a "$META"
        cleanup
        exit 2
    fi
    if (( elapsed % 30 == 0 && elapsed > 0 )); then
        log "poll ${elapsed}s: still loading..."
        docker exec "$LLAMA_NAME" sh -c 'strings /tmp/server.log | grep -vE "^[\\\\|/-]+$" | tail -2' 2>/dev/null | tee -a "$META" || true
    fi
    sleep 5
done

if [[ "$ready" -ne 1 ]]; then
    cleanup
    exit 2
fi

# warmup
curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8}" >/dev/null || true

log "benchmark ${RUNS} runs..."
: >"$RESULT"
for i in $(seq 1 "$RUNS"); do
    out=$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"The quick brown fox jumps over the lazy dog.\"}],\"max_tokens\":${GEN_TOKENS}}" \
        2>"${LOG_DIR}/${LABEL}-curl.err") || {
        log "FAIL run $i: curl error"
        cat "${LOG_DIR}/${LABEL}-curl.err" | tee -a "$META"
        cleanup
        exit 1
    }
    line=$(python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d.get('timings') or {}
c=d.get('choices',[{}])[0].get('message',{}).get('content','')[:80]
print(f\"run=$i P={t.get('prompt_per_second',0):.1f} G={t.get('predicted_per_second',0):.1f} preview={c!r}\")
" <<<"$out")
    log "$line"
    echo "$line" >>"$RESULT"
    # death-loop check
    if python3 -c "
import json,sys
d=json.load(sys.stdin)
c=d.get('choices',[{}])[0].get('message',{}).get('content','')
print(1 if len(c)>500 and c.count(c[:20])>5 else 0)
" <<<"$out" | grep -q 1; then
        log "WARN: possible garbage loop in output"
    fi
done

docker cp "${LLAMA_NAME}:/tmp/server.log" "${LOG_DIR}/${LABEL}-server.log" 2>/dev/null || true
docker logs "$RPC_NAME" >"${LOG_DIR}/${LABEL}-rpc.log" 2>&1 || true

cleanup
log "RESULT=PASS"
log "results in $RESULT"