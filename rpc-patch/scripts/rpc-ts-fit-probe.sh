#!/usr/bin/env bash
# Probe tensor-split / fit / n-cpu-moe with GPU memory monitoring.
# Single bench-rpc + bench-llama; logs VRAM during load via rocm-smi/nvidia-smi.
#
# usage: rpc-ts-fit-probe.sh <label>
# env: PROBE_MODEL, PROBE_CTX, PROBE_CTK, PROBE_CTV, PROBE_TS, PROBE_NGL,
#      PROBE_FIT (on|off), PROBE_FITT (e.g. 512,1024), PROBE_NCMOE,
#      PROBE_EXTRA (extra llama-server args), PROBE_LOAD_TIMEOUT

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"

LABEL="${1:?label required}"
MODEL="${PROBE_MODEL:-/mnt/models/Qwen3.5-27B-Q5_K_M.gguf}"
CTX="${PROBE_CTX:-4096}"
CTK="${PROBE_CTK:-q4_0}"
CTV="${PROBE_CTV:-q4_0}"
TS="${PROBE_TS:-10,90}"
NGL="${PROBE_NGL:-99}"
FIT="${PROBE_FIT:-off}"
FITT="${PROBE_FITT:-}"
NCMOE="${PROBE_NCMOE:-}"
EXTRA="${PROBE_EXTRA:-}"
LOAD_TIMEOUT="${PROBE_LOAD_TIMEOUT:-600}"
PORT="${PROBE_PORT:-8081}"
RPC_PORT="${PROBE_RPC_PORT:-50051}"
LOG_DIR="${PROBE_LOG_DIR:-${RPC_PATCH_ROOT}/patch/bench-results/rpc-ts-probe}"
VARIANT="${PROBE_VARIANT:-a1}"

RPC_NAME="bench-rpc"
LLAMA_NAME="bench-llama"

mkdir -p "$LOG_DIR"
META="${LOG_DIR}/${LABEL}.meta"
GPULOG="${LOG_DIR}/${LABEL}.gpu"
SERVERLOG="${LOG_DIR}/${LABEL}-server.log"
: >"$META"
: >"$GPULOG"

log() { echo "$*" | tee -a "$META"; }

cleanup() {
    [[ -n "${MON_PID:-}" ]] && kill "$MON_PID" 2>/dev/null || true
    docker rm -f "$LLAMA_NAME" "$RPC_NAME" 2>/dev/null || true
}
trap cleanup EXIT

gpu_poll() {
    while true; do
        {
            echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
            rocm-smi --showmeminfo vram 2>/dev/null | grep -E 'GPU\[|VRAM' || rocm-smi 2>/dev/null | head -3
            nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
        } >>"$GPULOG"
        sleep 3
    done
}

case "$VARIANT" in
    a1)
        CUDA_IMAGE="llama-rpc-cuda-a2"
        CUDA_ENTRY="/app/build/bin/rpc-server"
        ROCM_IMAGE="llama-rocm-patched"
        ROCM_BIN="/app/llama.cpp/build/bin/llama-server"
        ;;
    pathb)
        CUDA_IMAGE="llama-rpc-cuda-a2"
        CUDA_BIN_HOST="${TQ}/build-cuda-b-bin/bin"
        ROCM_IMAGE="llama-rocm-patched"
        ROCM_BIN_HOST="${TQ}/build-rocm-docker/bin"
        ROCM_BIN="/app/bin/llama-server"
        ;;
    *) echo "unknown PROBE_VARIANT=$VARIANT" >&2; exit 1 ;;
esac

log "=== $LABEL ==="
log "model=$MODEL ctx=$CTX ctk=$CTK ctv=$CTV ts=$TS ngl=$NGL fit=$FIT fitt=${FITT:-none} ncmoe=${NCMOE:-none}"
log "extra=$EXTRA variant=$VARIANT"

FIT_ARGS=()
[[ "$FIT" == "on" || "$FIT" == "off" ]] && FIT_ARGS+=(--fit "$FIT")
[[ -n "$FITT" ]] && FIT_ARGS+=(--fit-target "$FITT")
[[ -n "$NCMOE" ]] && FIT_ARGS+=(--n-cpu-moe "$NCMOE")

cleanup
gpu_poll &
MON_PID=$!

log "starting rpc-server..."
if [[ "$VARIANT" == "pathb" ]]; then
    docker run -d --name "$RPC_NAME" --gpus "device=0" --network host \
        -v "${CUDA_BIN_HOST}:/app/bin:ro" -e LD_LIBRARY_PATH=/app/bin -e CUDA_VISIBLE_DEVICES=0 \
        "$CUDA_IMAGE" bash -c "/app/bin/rpc-server -H 0.0.0.0 -p ${RPC_PORT} -d CUDA0" >>"$META" 2>&1
else
    docker run -d --name "$RPC_NAME" --gpus "device=0" --network host \
        -e CUDA_VISIBLE_DEVICES=0 --entrypoint "$CUDA_ENTRY" "$CUDA_IMAGE" \
        -H 0.0.0.0 -p "$RPC_PORT" -d CUDA0 >>"$META" 2>&1
fi
sleep 2

SERVER_CMD="${ROCM_BIN} --rpc 127.0.0.1:${RPC_PORT} -m ${MODEL} -c ${CTX} -ctk ${CTK} -ctv ${CTV} \
    -sm layer -ts ${TS} -ngl ${NGL} ${FIT_ARGS[*]} --host 127.0.0.1 --port ${PORT} --no-warmup -np 1 ${EXTRA}"

log "starting llama-server: $SERVER_CMD"
if [[ "$VARIANT" == "pathb" ]]; then
    docker run -d --name "$LLAMA_NAME" --entrypoint bash \
        --device=/dev/kfd --device=/dev/dri --group-add video --network host \
        -v "${ROCM_BIN_HOST}:/app/bin:ro" -v /mnt/models:/mnt/models:ro \
        -e LD_LIBRARY_PATH=/app/bin -e HIP_VISIBLE_DEVICES=0 "$ROCM_IMAGE" \
        -c "exec > /tmp/server.log 2>&1; ${SERVER_CMD}" >>"$META" 2>&1
else
    docker run -d --name "$LLAMA_NAME" --entrypoint bash \
        --device=/dev/kfd --device=/dev/dri --group-add video --network host \
        -v /mnt/models:/mnt/models:ro -e HIP_VISIBLE_DEVICES=0 "$ROCM_IMAGE" \
        -c "exec > /tmp/server.log 2>&1; ${SERVER_CMD}" >>"$META" 2>&1
fi

start=$(date +%s)
ready=0
while true; do
    elapsed=$(( $(date +%s) - start ))
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        log "ready at ${elapsed}s"
        break
    fi
    if ! docker ps -q --filter "name=^${LLAMA_NAME}$" | grep -q .; then
        log "FAIL: container exited at ${elapsed}s"
        docker cp "${LLAMA_NAME}:/tmp/server.log" "$SERVERLOG" 2>/dev/null || true
        grep -E "device_info|ROCm|RPC|tensor|split|fit|alloc_tensor|memory breakdown|layer" "$SERVERLOG" 2>/dev/null | tail -25 | tee -a "$META" || true
        exit 2
    fi
    if [[ "$elapsed" -ge "$LOAD_TIMEOUT" ]]; then
        log "TIMEOUT ${LOAD_TIMEOUT}s"
        docker exec "$LLAMA_NAME" sh -c 'tail -c 16384 /tmp/server.log' 2>/dev/null | strings | tail -15 | tee -a "$META"
        exit 2
    fi
    sleep 5
done

docker cp "${LLAMA_NAME}:/tmp/server.log" "$SERVERLOG" 2>/dev/null || true

# extract device + memory info
{
    echo "--- device_info ---"
    grep -E "device_info|ROCm0|RPC0|CPU" "$SERVERLOG" | head -10
    echo "--- fit ---"
    grep -iE "llama_params_fit|fit params|fit-target" "$SERVERLOG" | head -15
    echo "--- memory breakdown ---"
    grep -i "memory breakdown" "$SERVERLOG" | head -10
    echo "--- errors ---"
    grep -iE "alloc_tensor|OOM|error|failed" "$SERVERLOG" | tail -5
} | tee -a "$META"

# peak VRAM from poll log
python3 - <<'PY' "$GPULOG" | tee -a "$META" || true
import re, sys
path = sys.argv[1]
text = open(path).read()
rocm = [int(m.group(1)) for m in re.finditer(r'VRAM Total Used Memory \(B\):\s*(\d+)', text)]
nv = [int(m.group(1)) for m in re.finditer(r'^0,\s*(\d+)\s*MiB', text, re.M)]
if rocm:
    print(f"rocm_peak_vram_MiB={max(rocm)//(1024*1024)}")
else:
    print("rocm_peak_vram_MiB=unknown")
if nv:
    print(f"nvidia_peak_MiB={max(nv)}")
else:
    print("nvidia_peak_MiB=unknown")
PY

if [[ "$ready" -eq 1 ]]; then
    out=$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' 2>/dev/null) || true
    if [[ -n "$out" ]]; then
        python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d.get('timings') or {}
print(f\"gen={t.get('predicted_per_second',0):.1f} prompt={t.get('prompt_per_second',0):.1f}\")
" <<<"$out" | tee -a "$META" || log "WARN: could not parse inference timings"
    else
        log "WARN: inference curl failed"
    fi
fi

log "RESULT=PASS logs=$LOG_DIR/${LABEL}.*"