#!/usr/bin/env bash
# Reproduce ROCm+RPC segfault under gdb on romulus (run ON romulus host).
set -euo pipefail

ROOT=/home/hunter/atomic-llama-cpp-turboquant
BIN="${ROOT}/build-rocm-docker/bin"
MODEL="${BENCH_MODEL:-/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf}"
PRESET="${BENCH_GDB_PRESET:-2gpu}"
case "$PRESET" in
    4gpu|4gpu-full|4gpu-primary)
        RPC="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053}"
        TS="${BENCH_TS:-36,24,24,16}"
        ;;
    4gpu-legacy-6600)
        RPC="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,192.168.8.176:50052,127.0.0.1:50051}"
        TS="${BENCH_TS:-28,12,28,32}"
        ;;
    3gpu|3gpu-primary)
        RPC="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051,127.0.0.1:50051}"
        TS="${BENCH_TS:-50,28,22}"
        ;;
    *)
        RPC="${BENCH_RPC_ENDPOINT:-192.168.8.176:50051}"
        TS="${BENCH_TS:-50,50}"
        ;;
esac
CTK="${BENCH_CTK:-q8_0}"
CTV="${BENCH_CTV:-q8_0}"
LOG=/tmp/rocm-gdb-repro.log

export LD_LIBRARY_PATH="${BIN}:/opt/rocm-7.2.3/lib:/opt/rocm/lib"
export HIP_VISIBLE_DEVICES=0
ulimit -c unlimited

: >"$LOG"
echo "=== gdb repro $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "model=$MODEL rpc=$RPC ts=$TS ctk=$CTK ctv=$CTV" | tee -a "$LOG"

gdb -batch \
    -ex "set pagination off" \
    -ex "handle SIGPIPE nostop noprint pass" \
    -ex "run" \
    -ex "bt full" \
    -ex "thread apply all bt full" \
    -ex "quit" \
    --args "${BIN}/llama-server" \
        --rpc "${RPC}" -m "${MODEL}" -ngl 99 -c 4096 \
        -ctk "${CTK}" -ctv "${CTV}" -sm layer -ts "${TS}" \
        --host 127.0.0.1 --port 8081 \
        --no-warmup -np 1 --fit off --verbose -lv 4 --reasoning off \
    >>"$LOG" 2>&1 || true

echo "=== gdb done exit=$? ===" | tee -a "$LOG"
tail -80 "$LOG"