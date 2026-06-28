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
#   BENCH_PROMPTS_FILE=json  (default: single fox prompt)
#   BENCH_VERBOSE_LV=4       (-lv N for per-device load allocation logs)
#   BENCH_EXTRACT_VRAM=1     (grep MiB/allocation lines from server log into .meta)
#   BENCH_NO_WARMUP=1 adds --no-warmup (recommended for >27B Path B)
#   BENCH_RPC_MODE=local|remote|multi  (default local)
#   BENCH_RPC_HOST=remus.local        (remote single worker)
#   BENCH_RPC_ENDPOINT=host1:50051,host2:50051  (multi; overrides BENCH_RPC_HOST)
#   BENCH_RPC_WAIT=5                   (seconds after remote RPC expected up)
#   PATHB_CUDA_DISABLE_GRAPHS=1        (GGML_CUDA_DISABLE_GRAPHS on rpc-server workers)
#   BENCH_TRACE=1                      (GGML_RPC_TRACE + GGML_SCHED_TRACE under ${LOG_DIR}/${LABEL}/telemetry/)
#   GGML_PIPELINE_PLUS=1               (default when BENCH_TRACE=1)

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
PROMPTS_FILE="${BENCH_PROMPTS_FILE:-}"
VERBOSE_LV="${BENCH_VERBOSE_LV:-}"
EXTRACT_VRAM="${BENCH_EXTRACT_VRAM:-0}"
CURL_TIMEOUT="${BENCH_CURL_TIMEOUT:-300}"
RPC_MODE="${BENCH_RPC_MODE:-local}"
RPC_HOST="${BENCH_RPC_HOST:-${REMUS_RPC_IP:-192.168.8.176}}"
RPC_WAIT="${BENCH_RPC_WAIT:-5}"
if [[ -n "${BENCH_RPC_ENDPOINT:-}" ]]; then
    RPC_ENDPOINT="$BENCH_RPC_ENDPOINT"
    if [[ "$BENCH_RPC_ENDPOINT" == *","* ]]; then
        RPC_MODE="multi"
    fi
elif [[ "$RPC_MODE" == "remote" ]]; then
    RPC_ENDPOINT="${RPC_HOST}:${RPC_PORT}"
else
    RPC_ENDPOINT="127.0.0.1:${RPC_PORT}"
fi

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

BENCH_TRACE="${BENCH_TRACE:-0}"
TELEMETRY_DIR="${LOG_DIR}/${LABEL}/telemetry"
if [[ "$BENCH_TRACE" == "1" ]]; then
    mkdir -p "$TELEMETRY_DIR"
    export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
    : >"${TELEMETRY_DIR}/rpc-trace.jsonl"
    : >"${TELEMETRY_DIR}/sched-trace.jsonl"
fi

log() { echo "$*" | tee -a "$META"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"

capture_rpc_artifacts() {
    local reason="${1:-unknown}"
    local stamp
    stamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local snap="${LOG_DIR}/${LABEL}-gpu-snapshot.log"
    local rpc_local="${LOG_DIR}/${LABEL}-rpc-local.log"
    local rpc_remus="${LOG_DIR}/${LABEL}-rpc-remus.log"
    local docker_ps="${LOG_DIR}/${LABEL}-docker-ps.log"

    log "--- artifact capture (${reason}) @ ${stamp} ---"

    {
        echo "=== ${stamp} reason=${reason} ==="
        echo "[docker ps -a]"
        docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null \
            | rg -i 'bench|pathb|rpc|llama' || docker ps -a 2>/dev/null || true
    } >"$docker_ps" 2>/dev/null || true
    cat "$docker_ps" | tee -a "$META" || true

    {
        echo "=== ${stamp} reason=${reason} ==="
        echo "[romulus-rocm]"
        timeout 5 rocm-smi --showmeminfo vram 2>/dev/null | head -8 || rocm-smi 2>/dev/null | head -4 || true
        echo "[romulus-nvidia]"
        timeout 5 nvidia-smi 2>/dev/null || true
        echo "[remus-nvidia]"
        SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
        if [[ -n "${PATHB_REMUS_SSH_PASS:-}" ]] && command -v sshpass >/dev/null; then
            SSH_CMD=(sshpass -p "${PATHB_REMUS_SSH_PASS}" ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "hunter@${REMUS_IP}")
        fi
        timeout 12 "${SSH_CMD[@]}" nvidia-smi 2>/dev/null || echo "remus nvidia-smi failed"
    } >"$snap" 2>/dev/null || true
    cat "$snap" | tee -a "$META" || true

    if docker ps -aq --filter "name=^pathb-rpc$" | grep -q .; then
        docker logs --tail=400 pathb-rpc >"$rpc_local" 2>&1 || true
    elif docker ps -aq --filter "name=^${RPC_NAME}$" | grep -q .; then
        docker logs --tail=400 "$RPC_NAME" >"$rpc_local" 2>&1 || true
    else
        echo "no local rpc container (pathb-rpc or ${RPC_NAME})" >"$rpc_local"
    fi
    log "--- rpc-local (tail) ---"
    tail -40 "$rpc_local" | tee -a "$META" || true

    {
        echo "endpoint=${RPC_ENDPOINT:-none} mode=${RPC_MODE}"
        PATHB_REMUS_LOG_TAIL=400 PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}" \
            "${SCRIPT_DIR}/pathb-remus-rpc.sh" logs 2>/dev/null || true
        echo "--- logs-since ---"
        PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}" \
            "${SCRIPT_DIR}/pathb-remus-rpc.sh" logs-since 2>/dev/null || true
    } >"$rpc_remus" 2>&1 || true
    log "--- rpc-remus (tail) ---"
    tail -40 "$rpc_remus" | tee -a "$META" || true

    docker cp "${LLAMA_NAME}:/tmp/server.log" "${LOG_DIR}/${LABEL}-server.log" 2>/dev/null || true
    if [[ -f "${LOG_DIR}/${LABEL}-server.log" ]]; then
        log "--- server log tail (infer) ---"
        tail -15 "${LOG_DIR}/${LABEL}-server.log" | tee -a "$META" || true
    fi
}

LLAMA_NATIVE=0
LLAMA_PID=""
LLAMA_LOG=""

llama_log_path() {
    if [[ "$LLAMA_NATIVE" == "1" ]]; then
        echo "$LLAMA_LOG"
    else
        echo "/tmp/server.log"
    fi
}

llama_running() {
    if [[ "$LLAMA_NATIVE" == "1" ]]; then
        [[ -n "$LLAMA_PID" ]] && kill -0 "$LLAMA_PID" 2>/dev/null
    else
        docker ps -q --filter "name=^${LLAMA_NAME}$" | grep -q .
    fi
}

llama_log_has() {
    local pat="$1"
    if [[ "$LLAMA_NATIVE" == "1" ]]; then
        grep -q "$pat" "$LLAMA_LOG" 2>/dev/null
    else
        docker exec "$LLAMA_NAME" sh -c "grep -q \"$pat\" /tmp/server.log 2>/dev/null" 2>/dev/null
    fi
}

llama_log_tail() {
    local n="${1:-15}"
    if [[ "$LLAMA_NATIVE" == "1" ]]; then
        tail -"$n" "$LLAMA_LOG" 2>/dev/null || true
    else
        docker exec "$LLAMA_NAME" sh -c "tail -c 32768 /tmp/server.log" 2>/dev/null \
            | strings | grep -vE '^[\\|/-]+$' | tail -"$n" || true
    fi
}

llama_copy_log() {
    local dst="${LOG_DIR}/${LABEL}-server.log"
    if [[ "$LLAMA_NATIVE" == "1" ]]; then
        cp -f "$LLAMA_LOG" "$dst" 2>/dev/null || true
    else
        docker cp "${LLAMA_NAME}:/tmp/server.log" "$dst" 2>/dev/null || true
    fi
}

cleanup() {
    if [[ -n "$LLAMA_PID" ]]; then
        kill "$LLAMA_PID" 2>/dev/null || true
        wait "$LLAMA_PID" 2>/dev/null || true
        LLAMA_PID=""
    fi
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
        _cuda_sync="${TQ}/build-cuda-b-bin-sync/bin"
        _cuda_rebuild="${TQ}/build-cuda-b-bin-rebuild/bin"
        if [[ -x "${_cuda_sync}/rpc-server" ]]; then
            CUDA_BIN_HOST="${_cuda_sync}"
        elif [[ -x "${_cuda_rebuild}/rpc-server" ]]; then
            CUDA_BIN_HOST="${_cuda_rebuild}"
        else
            CUDA_BIN_HOST="${TQ}/build-cuda-b-bin/bin"
        fi
        ROCM_IMAGE="llama-rocm-patched"
        ROCM_BIN_HOST="${TQ}/build-rocm-docker/bin"
        ROCM_LD_PATH="${BENCH_ROCM_LD_PATH:-/opt/rocm-7.2.3/lib:/opt/rocm/lib}"
        if [[ "${BENCH_ROCM_NATIVE:-1}" == "1" && -x "${ROCM_BIN_HOST}/llama-server" ]]; then
            LLAMA_NATIVE=1
        fi
        PROTO="Path B (v4.2.2 workspace)"
        ;;
    *)
        echo "unknown variant: $VARIANT" >&2
        exit 1
        ;;
esac

log "=== $LABEL ==="
log "variant=$VARIANT proto=$PROTO rpc_mode=$RPC_MODE endpoint=$RPC_ENDPOINT trace=$BENCH_TRACE"
if [[ "$BENCH_TRACE" == "1" ]]; then
    log "telemetry=$TELEMETRY_DIR"
fi
log "model=$MODEL ctx=$CTX ctk=$CTK ctv=$CTV ngl=$NGL ts=$TS ncmoe=${NCMOE:-none} np=$NP no_warmup=$NO_WARMUP load_timeout=${LOAD_TIMEOUT}s"
log "server_extra=${SERVER_EXTRA[*]:-none}"
log "prompts_file=${PROMPTS_FILE:-default-fox} verbose_lv=${VERBOSE_LV:-default} extract_vram=$EXTRACT_VRAM"
df -h / | tee -a "$META"
date -u +"%Y-%m-%dT%H:%M:%SZ" | tee -a "$META"

cleanup

if [[ "$RPC_MODE" == "local" ]]; then
    log "starting rpc-server (CUDA device 0, local)..."
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
else
    log "using remote/multi RPC at $RPC_ENDPOINT (no local bench-rpc)"
    IFS=',' read -ra _rpc_hosts <<<"$RPC_ENDPOINT"
    for ep in "${_rpc_hosts[@]}"; do
        host="${ep%%:*}"
        port="${ep##*:}"
        if nc -zv -w 3 "$host" "$port" >>"$META" 2>&1; then
            log "RPC reachable: $host:$port"
        else
            log "WARN: RPC not reachable yet: $host:$port"
        fi
    done
    sleep "$RPC_WAIT"
fi

# AMD 7900 XTX client only
if [[ "$LLAMA_NATIVE" == "1" ]]; then
    log "starting llama-server (native ROCm HIP device 0)..."
else
    log "starting llama-server (ROCm HIP device 0)..."
fi
EXTRA_HOSTS=()
[[ "$RPC_MODE" != "local" ]] && EXTRA_HOSTS+=(--add-host "remus.local:${REMUS_RPC_IP:-192.168.8.176}")

if [[ "$VARIANT" == "pathb" && "$LLAMA_NATIVE" == "1" ]]; then
    LLAMA_LOG="${LOG_DIR}/${LABEL}-server.log"
    : >"$LLAMA_LOG"
    export LD_LIBRARY_PATH="${ROCM_BIN_HOST}:${ROCM_LD_PATH}"
    export HIP_VISIBLE_DEVICES=0
    if [[ "$BENCH_TRACE" == "1" ]]; then
        export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
        export GGML_RPC_TRACE=1
        export GGML_SCHED_TRACE=1
        export GGML_PIPELINE_TRACE=1
        export GGML_RPC_TRACE_FILE="${TELEMETRY_DIR}/rpc-trace.jsonl"
        export GGML_SCHED_TRACE_FILE="${TELEMETRY_DIR}/sched-trace.jsonl"
        export GGML_PIPELINE_TRACE_FILE="${TELEMETRY_DIR}/pipeline-trace.jsonl"
    fi
    "${ROCM_BIN_HOST}/llama-server" \
        --rpc "${RPC_ENDPOINT}" -m "${MODEL}" -ngl "${NGL}" -c "${CTX}" \
        -ctk "${CTK}" -ctv "${CTV}" -sm "${SPLIT_MODE}" -ts "${TS}" \
        --host 127.0.0.1 --port "${PORT}" "${SERVER_EXTRA[@]}" >>"$LLAMA_LOG" 2>&1 &
    LLAMA_PID=$!
    log "native llama-server pid=${LLAMA_PID} log=${LLAMA_LOG}"
elif [[ "$VARIANT" == "pathb" ]]; then
    LLAMA_VOLUMES=(-v "${ROCM_BIN_HOST}:/app/bin:ro" -v /mnt/models:/mnt/models:ro)
    LLAMA_TRACE_ENV=()
    if [[ "$BENCH_TRACE" == "1" ]]; then
        LLAMA_VOLUMES+=(-v "${TELEMETRY_DIR}:/telemetry:rw")
        LLAMA_TRACE_ENV=(
            -e "GGML_PIPELINE_PLUS=${GGML_PIPELINE_PLUS:-1}"
            -e GGML_RPC_TRACE=1
            -e GGML_SCHED_TRACE=1
            -e GGML_PIPELINE_TRACE=1
            -e GGML_RPC_TRACE_FILE=/telemetry/rpc-trace.jsonl
            -e GGML_SCHED_TRACE_FILE=/telemetry/sched-trace.jsonl
            -e GGML_PIPELINE_TRACE_FILE=/telemetry/pipeline-trace.jsonl
        )
    fi
    docker run -d --name "$LLAMA_NAME" --entrypoint bash \
        --device=/dev/kfd --device=/dev/dri --group-add video --network host \
        "${EXTRA_HOSTS[@]}" \
        "${LLAMA_VOLUMES[@]}" \
        -e LD_LIBRARY_PATH=/app/bin:${ROCM_LD_PATH} -e HIP_VISIBLE_DEVICES=0 \
        ${GGML_RPC_DEBUG:+-e GGML_RPC_DEBUG=${GGML_RPC_DEBUG}} \
        ${GGML_SCHED_DEBUG:+-e GGML_SCHED_DEBUG=${GGML_SCHED_DEBUG}} \
        "${LLAMA_TRACE_ENV[@]}" \
        "$ROCM_IMAGE" \
        -c "exec > /tmp/server.log 2>&1; /app/bin/llama-server \
            --rpc ${RPC_ENDPOINT} -m ${MODEL} -ngl ${NGL} -c ${CTX} \
            -ctk ${CTK} -ctv ${CTV} -sm ${SPLIT_MODE} -ts ${TS} \
            --host 127.0.0.1 --port ${PORT} ${SERVER_EXTRA[*]}" >>"$META" 2>&1
else
    docker run -d --name "$LLAMA_NAME" --entrypoint bash \
        --device=/dev/kfd --device=/dev/dri --group-add video --network host \
        "${EXTRA_HOSTS[@]}" \
        -v /mnt/models:/mnt/models:ro \
        -e HIP_VISIBLE_DEVICES=0 \
        "$ROCM_IMAGE" \
        -c "exec > /tmp/server.log 2>&1; ${ROCM_BIN}/llama-server \
            --rpc ${RPC_ENDPOINT} -m ${MODEL} -ngl ${NGL} -c ${CTX} \
            -ctk ${CTK} -ctv ${CTV} -sm ${SPLIT_MODE} -ts ${TS} \
            --host 127.0.0.1 --port ${PORT} ${SERVER_EXTRA[*]}" >>"$META" 2>&1
fi

log "polling server health (max ${LOAD_TIMEOUT}s)..."
start=$(date +%s)
ready=0
while true; do
    elapsed=$(( $(date +%s) - start ))
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        if llama_log_has "model loaded"; then
            ready=1
            log "server ready at ${elapsed}s (model loaded)"
            break
        fi
    fi
    if ! llama_running; then
        log "ERROR: llama-server exited during load"
        llama_copy_log
        tail -30 "${LOG_DIR}/${LABEL}-server.log" | tee -a "$META"
        capture_rpc_artifacts "load_exit"
        cleanup
        exit 2
    fi
    if [[ "$elapsed" -ge "$LOAD_TIMEOUT" ]]; then
        log "TIMEOUT: server not healthy after ${LOAD_TIMEOUT}s"
        llama_log_tail 15 | tee -a "$META"
        capture_rpc_artifacts "load_timeout"
        cleanup
        exit 2
    fi
    if (( elapsed % 30 == 0 && elapsed > 0 )); then
        log "poll ${elapsed}s: still loading..."
        llama_log_tail 2 | tee -a "$META" || true
        if [[ "$EXTRACT_VRAM" == "1" ]]; then
            rg -i 'MiB|load_tensors|offload|assign|buffer|tensor split|RPC[0-9]|ROCm' \
                "$(llama_log_path)" 2>/dev/null | tail -8 | tee -a "$META" || true
        fi
    fi
    sleep 5
done

if [[ "$ready" -ne 1 ]]; then
    cleanup
    exit 2
fi

llama_copy_log
if [[ "$EXTRACT_VRAM" == "1" && -f "${LOG_DIR}/${LABEL}-server.log" ]]; then
    log "--- vram allocation excerpt (load) ---"
    rg -i 'MiB|load_tensors|offload|assign|buffer|tensor|RPC[0-9]|ROCm|CPU|layer' \
        "${LOG_DIR}/${LABEL}-server.log" 2>/dev/null \
        | rg -v 'progress|████' | tail -40 | tee -a "$META" || true
fi

# optional client warmup (skip when BENCH_NO_WARMUP=1; large RPC splits can OOM here)
if [[ "$NO_WARMUP" != "1" ]]; then
    curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8}" >/dev/null || true
fi

log "benchmark ${RUNS} runs..."
: >"$RESULT"

bench_one() {
    local run_idx="$1"
    local prompt_id="$2"
    local prompt_text="$3"
    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': int(sys.argv[2]),
}))
" "$prompt_text" "$GEN_TOKENS")
    out=$(curl -sf --max-time "$CURL_TIMEOUT" "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        2>"${LOG_DIR}/${LABEL}-curl.err") || {
        local http_code="?"
        if llama_running; then
            http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
                "http://127.0.0.1:${PORT}/v1/chat/completions" -H "Content-Type: application/json" -d "$payload" 2>/dev/null || echo '?')
        else
            log "WARN: llama-server exited during generation"
            llama_log_tail 20 | tee -a "$META" || true
        fi
        log "FAIL run $run_idx prompt=$prompt_id: curl error (http=${http_code})"
        cat "${LOG_DIR}/${LABEL}-curl.err" 2>/dev/null | tee -a "$META" || true
        capture_rpc_artifacts "gen_fail_run${run_idx}_${prompt_id}"
        return 1
    }
    line=$(python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d.get('timings') or {}
c=d.get('choices',[{}])[0].get('message',{}).get('content','')[:200]
pid=sys.argv[1]
ri=sys.argv[2]
print(f\"run={ri} prompt={pid} P={t.get('prompt_per_second',0):.1f} G={t.get('predicted_per_second',0):.1f} preview={c!r}\")
" "$prompt_id" "$run_idx" <<<"$out")
    log "$line"
    echo "$line" >>"$RESULT"
    if python3 -c "
import json,sys
d=json.load(sys.stdin)
c=d.get('choices',[{}])[0].get('message',{}).get('content','')
bad = len(c)>500 and len(c)>20 and c.count(c[:20])>5
garbled = sum(1 for ch in c if ord(ch)>0x3000) > len(c)*0.15
print(1 if bad or garbled else 0)
" <<<"$out" | grep -q 1; then
        log "WARN: possible garbage loop or garbled output (prompt=$prompt_id)"
    fi
    return 0
}

if [[ -n "$PROMPTS_FILE" && -f "$PROMPTS_FILE" ]]; then
    mapfile -t _prompt_rows < <(python3 -c "
import json, sys
for p in json.load(open(sys.argv[1])):
    print(p['id'] + '\t' + p['content'].replace('\n', '\\n'))
" "$PROMPTS_FILE")
    run_idx=0
    for row in "${_prompt_rows[@]}"; do
        pid="${row%%$'\t'*}"
        ptext="${row#*$'\t'}"
        ptext="${ptext//\\n/$'\n'}"
        for _ in $(seq 1 "$RUNS"); do
            run_idx=$((run_idx + 1))
            bench_one "$run_idx" "$pid" "$ptext" || { cleanup; exit 1; }  # artifacts captured in bench_one
        done
    done
else
    for i in $(seq 1 "$RUNS"); do
        bench_one "$i" "fox" "The quick brown fox jumps over the lazy dog." || { cleanup; exit 1; }  # artifacts in bench_one
    done
fi

llama_copy_log
if [[ "$RPC_MODE" == "local" ]]; then
    docker logs "$RPC_NAME" >"${LOG_DIR}/${LABEL}-rpc.log" 2>&1 || true
else
    capture_rpc_artifacts "pass_final"
    {
        echo "remote rpc endpoint=$RPC_ENDPOINT"
        cat "${LOG_DIR}/${LABEL}-rpc-local.log" 2>/dev/null || true
        echo "--- remus ---"
        cat "${LOG_DIR}/${LABEL}-rpc-remus.log" 2>/dev/null || true
    } >"${LOG_DIR}/${LABEL}-rpc.log" 2>&1 || true
fi
df -h / | tee -a "$META"

if [[ "$BENCH_TRACE" == "1" && -f "${TELEMETRY_DIR}/rpc-trace.jsonl" ]]; then
    log "parsing trace telemetry..."
    "${SCRIPT_DIR}/pathb-rpc-trace-parse.sh" "$TELEMETRY_DIR" 2>&1 | tee -a "$META" || true
    "${SCRIPT_DIR}/pathb-hotpath-summary.sh" "$TELEMETRY_DIR" 2>&1 | tee -a "$META" || true
fi

cleanup
log "RESULT=PASS"
log "results in $RESULT"