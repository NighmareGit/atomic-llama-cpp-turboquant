#!/usr/bin/env bash
# D4.1 Romulus Dual-GPU Baseline Benchmark
# Runs 3 models on 7900 XTX (client) + 3060 Ti (RPC server) with tracing.
#
# usage: bash scripts/d41-baseline-bench.sh
#
# Output: /tmp/d41-baseline/<model>/  (raw traces, logs, results)
#   traces/   - GGML_SCHED_TRACE + GGML_RPC_TRACE JSONL files
#   server.log - llama-server log
#   results.json - parsed benchmark results
#   gpu-mon.csv - GPU utilization during run

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${ROOT}/build-rocm-docker/bin/llama-server"
BENCH_DIR="/tmp/d41-baseline"
RPC_ENDPOINT="127.0.0.1:50051"

# Default settings
CTK="q8_0"
CTV="turbo3"
CTX=8192
GEN_TOKENS=128
RUNS=3

declare -A MODELS
MODELS=(
    ["9b-mtp"]="/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf"
    ["llama8b"]="/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
    ["35b-mtp"]="/mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"
)

declare -A MODEL_TS
MODEL_TS=(
    ["9b-mtp"]="45,55"
    ["llama8b"]="45,55"
    ["35b-mtp"]="50,50"
)

declare -A MODEL_CTX
MODEL_CTX=(
    ["9b-mtp"]="8192"
    ["llama8b"]="8192"
    ["35b-mtp"]="4096"
)

ensure_rpc() {
    if ! nc -z 127.0.0.1 50051 2>/dev/null; then
        echo "ERROR: RPC server not running on :50051. Start with:"
        echo "  docker compose -f scripts/romulus-pathb/docker-compose.yml up -d"
        exit 1
    fi
    echo "RPC server OK on :50051"
}

bench_model() {
    local label="$1"
    local model="${MODELS[$label]}"
    local ts="${MODEL_TS[$label]}"
    local ctx="${MODEL_CTX[$label]}"
    local outdir="${BENCH_DIR}/${label}"

    if [[ -z "$model" ]]; then
        echo "ERROR: unknown model label: $label"
        return 1
    fi

    mkdir -p "${outdir}/traces"

    echo ""
    echo "============================================================"
    echo "  Model: ${label} ($(basename $model))"
    echo "  ctx=${ctx} ts=${ts} ctk=${CTK} ctv=${CTV}"
    echo "============================================================"

    # Determine mtp setting
    local mtp_flag=""
    if [[ "$model" == *MTP* ]]; then
        mtp_flag="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1"
    fi

    # 1. Run llama-bench for standardized measurements
    echo ""
    echo "--- llama-bench (standardized) ---"
    ${ROOT}/build-rocm-docker/bin/llama-bench \
        -m "$model" \
        -rpc "$RPC_ENDPOINT" \
        -ngl 99 \
        -ts "$ts" \
        -sm layer \
        -ctk "$CTK" \
        -ctv "$CTV" \
        -p 512 -n 128 -r "$RUNS" \
        -o json 2>&1 | tee "${outdir}/llama-bench.json"

    # 2. Start llama-server with tracing
    echo ""
    echo "--- Starting llama-server with tracing ---"
    export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm/lib"
    export HIP_VISIBLE_DEVICES=0
    export GGML_PIPELINE_PLUS=1
    export GGML_PIPELINE_MULTI_BACKEND_SEQ=1
    export GGML_RPC_TRACE=1
    export GGML_SCHED_TRACE=1
    export GGML_RPC_TRACE_FILE="${outdir}/traces/rpc-trace.jsonl"
    export GGML_SCHED_TRACE_FILE="${outdir}/traces/sched-trace.jsonl"

    local server_log="${outdir}/server.log"
    local port=8082

    # Kill any existing server on our port
    local old_pid
    old_pid=$(lsof -ti:${port} 2>/dev/null || true)
    [[ -n "$old_pid" ]] && kill "$old_pid" 2>/dev/null || true
    sleep 1

    ${SERVER} \
        -m "$model" \
        -ngl 99 \
        -c "$ctx" \
        -ctk "$CTK" \
        -ctv "$CTV" \
        -fa on \
        --host 127.0.0.1 \
        --port "$port" \
        --parallel 1 -np 1 \
        --cont-batching \
        --split-mode layer \
        -ts "$ts" \
        --rpc "$RPC_ENDPOINT" \
        --fit off \
        --metrics \
        --slots \
        --log-timestamps \
        --log-prefix \
        --reasoning off \
        --no-warmup \
        ${mtp_flag} \
        > "$server_log" 2>&1 &
    local server_pid=$!
    echo "  llama-server PID: ${server_pid} (port ${port})"

    # Wait for server ready
    local wait_max=180
    local waited=0
    while true; do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "  Server ready at ${waited}s"
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "  ERROR: llama-server exited during load"
            tail -30 "$server_log"
            return 1
        fi
        if [[ "$waited" -ge "$wait_max" ]]; then
            echo "  ERROR: timeout waiting for server (${wait_max}s)"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    # 3. Run trace-capture benchmarks
    echo ""
    echo "--- Trace-capture benchmarks ---"

    # 3a. Short prompt
    echo "  Run 1: short prompt (fox)"
    curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"The quick brown fox jumps over the lazy dog.\"}],\"max_tokens\":${GEN_TOKENS}}" \
        > "${outdir}/traces/run1-fox.json" 2>/dev/null || echo "  FAIL: run1"

    sleep 1

    # 3b. Longer prompt for PP stress
    # Generate a 1500-token prompt
    local long_prompt
    long_prompt=$(python3 -c "
import random
random.seed(42)
words = ['the','quick','brown','fox','jumps','over','lazy','dog','lorem','ipsum','dolor','sit','amet','consectetur','adipiscing','elit','sed','do','eiusmod','tempor']
tokens = ' '.join(random.choices(words, k=300))
print(tokens)
" 2>/dev/null)

    echo "  Run 2: long prompt (~300 words)"
    curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${long_prompt}\"}],\"max_tokens\":${GEN_TOKENS}}" \
        > "${outdir}/traces/run2-long.json" 2>/dev/null || echo "  FAIL: run2"

    sleep 1

    # Wait a moment for traces to flush
    sleep 2

    # Capture GPU utilization snapshots during inference
    {
        echo "timestamp,gpu,memory_used_mib,util_pct"
        for i in 1 2 3; do
            local ts_now
            ts_now=$(date +%s)
            if command -v rocm-smi &>/dev/null; then
                rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[0\]" | head -1 | \
                    awk -v ts="$ts_now" '{print ts",7900 XTX,"($5/1024/1024)",vram"}'
            fi
            if command -v nvidia-smi &>/dev/null; then
                nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
                    head -1 | awk -v ts="$ts_now" -F', ' '{print ts",3060 Ti,"$1","$2}'
            fi
            sleep 3
        done
    } > "${outdir}/gpu-mon.csv"

    # 4. Stop server
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "  Server stopped"

    # 5. Parse results
    echo ""
    echo "--- Parsing results ---"
    python3 -c "
import json, os, sys

label = sys.argv[1]
outdir = sys.argv[2]

results = {
    'model': label,
    'model_path': '${model}',
    'config': {
        'ctx': ${ctx},
        'ctk': '${CTK}',
        'ctv': '${CTV}',
        'ts': '${ts}',
        'rpc': '${RPC_ENDPOINT}'
    }
}

# Parse llama-bench json
bench_file = os.path.join(outdir, 'llama-bench.json')
if os.path.exists(bench_file):
    with open(bench_file) as f:
        bench_lines = [l for l in f if l.startswith('{')]
    bench_results = []
    for line in bench_lines:
        try:
            bench_results.append(json.loads(line))
        except:
            pass
    results['llama_bench'] = bench_results

# Parse trace files
for trace_file in ['traces/rpc-trace.jsonl', 'traces/sched-trace.jsonl']:
    tf = os.path.join(outdir, trace_file)
    if os.path.exists(tf):
        trace_lines = [l for l in open(tf) if l.strip()]
        results[trace_file.replace('/', '_').replace('.jsonl','') + '_count'] = len(trace_lines)
        # Collect first/last few lines
        with open(tf) as f:
            all_lines = [l.strip() for l in f if l.strip()]
        results[trace_file.replace('/', '_').replace('.jsonl','') + '_sample'] = all_lines[:5] if all_lines else []

# Parse server log for timing info
server_log = os.path.join(outdir, 'server.log')
if os.path.exists(server_log):
    with open(server_log) as f:
        content = f.read()
    # Extract timing from run results
    import re
    pp_times = re.findall(r'pp\s+(\d+\.?\d*)\s+t/s', content)
    tg_times = re.findall(r'tg\s+(\d+\.?\d*)\s+t/s', content)
    if pp_times:
        results['server_pp_t_s'] = [float(x) for x in pp_times]
    if tg_times:
        results['server_tg_t_s'] = [float(x) for x in tg_times]
    
    # Look for timings in the completion response
    # Also extract from the JSON response files
    for run_file in ['traces/run1-fox.json', 'traces/run2-long.json']:
        rf = os.path.join(outdir, run_file)
        if os.path.exists(rf):
            try:
                with open(rf) as f:
                    data = json.load(f)
                if 'timings' in data:
                    results[run_file.replace('/', '_') + '_timings'] = data['timings']
            except:
                pass

# GPU monitoring
gpu_file = os.path.join(outdir, 'gpu-mon.csv')
if os.path.exists(gpu_file):
    with open(gpu_file) as f:
        results['gpu_snapshots'] = f.read().strip().split('\\n')

# Trace size info
for trace_file in ['traces/rpc-trace.jsonl', 'traces/sched-trace.jsonl']:
    tf = os.path.join(outdir, trace_file)
    if os.path.exists(tf):
        sz = os.path.getsize(tf)
        results[trace_file.replace('/', '_') + '_bytes'] = sz

# Trace summary stats (count of each event type)
rpc_file = os.path.join(outdir, 'traces/rpc-trace.jsonl')
if os.path.exists(rpc_file):
    with open(rpc_file) as f:
        rpc_types = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                et = ev.get('event_type', ev.get('type', 'unknown'))
                rpc_types[et] = rpc_types.get(et, 0) + 1
            except:
                pass
    results['rpc_event_types'] = rpc_types

sched_file = os.path.join(outdir, 'traces/sched-trace.jsonl')
if os.path.exists(sched_file):
    with open(sched_file) as f:
        sched_types = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                et = ev.get('event', ev.get('type', 'unknown'))
                sched_types[et] = sched_types.get(et, 0) + 1
            except:
                pass
    results['sched_event_types'] = sched_types

with open(os.path.join(outdir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to {outdir}/results.json')
" "$label" "$outdir"

    echo "  Done: ${label}"
}

# ---- Main ----
mkdir -p "$BENCH_DIR"

ensure_rpc

# Run models sequentially
bench_model "9b-mtp"
bench_model "llama8b"
bench_model "35b-mtp"

echo ""
echo "============================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "  Results: ${BENCH_DIR}/"
echo "============================================================"
