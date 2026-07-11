#!/usr/bin/env bash
# Run D4.1 benchmark for a single model with trace capture.
# Usage: bash scripts/d41-per-model.sh <model-label>
#   model-label: 9b-mtp | llama8b | 35b-mtp
#
# This script:
# 1. Runs llama-bench (standardized PP/TG)
# 2. Starts llama-server with GGML_SCHED_TRACE + GGML_RPC_TRACE
# 3. Sends curl requests for timing
# 4. Captures GPU utilization snapshots
# 5. Parses results into results.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${ROOT}/build-rocm-docker/bin/llama-server"
BENCH_DIR="/tmp/d41-baseline"
RPC="127.0.0.1:50051"
PORT=8082

# Model configs
case "${1:-}" in
    9b-mtp)
        MODEL="/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf"
        TS="45,55"
        CTX=8192
        CTK="q8_0"
        CTV="turbo3"
        MTP="--spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-n-min 0"
        LABEL="9b-mtp"
        ;;
    llama8b)
        MODEL="/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
        TS="45,55"
        CTX=8192
        CTK="q8_0"
        CTV="turbo3"
        MTP=""
        LABEL="llama8b"
        ;;
    35b-mtp)
        MODEL="/mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"
        TS="50,50"
        CTX=4096
        CTK="q8_0"
        CTV="turbo3"
        MTP="--spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-n-min 0"
        LABEL="35b-mtp"
        ;;
    *)
        echo "Usage: $0 {9b-mtp|llama8b|35b-mtp}"
        exit 1
        ;;
esac

OUTDIR="${BENCH_DIR}/${LABEL}"
mkdir -p "${OUTDIR}/traces"

echo "=========================================================="
echo "  D4.1 Benchmark: ${LABEL}"
echo "  Model: $(basename ${MODEL})"
echo "  ctx=${CTX} ts=${TS} ctk=${CTK} ctv=${CTV}"
echo "=========================================================="

# Verify RPC
if ! nc -z 127.0.0.1 50051 2>/dev/null; then
    echo "ERROR: RPC server not running on :50051"
    exit 1
fi
echo "RPC server OK on :50051"

# Kill any existing server on our port
PREV_PID=$(lsof -ti:${PORT} 2>/dev/null || true)
[[ -n "$PREV_PID" ]] && kill "$PREV_PID" 2>/dev/null || true
sleep 1

# 1. Run llama-bench
echo ""
echo "--- Step 1: llama-bench ---"
${ROOT}/build-rocm-docker/bin/llama-bench \
    -m "$MODEL" -rpc "$RPC" -ngl 99 -ts "$TS" -sm layer \
    -ctk "$CTK" -ctv "$CTV" -p 512 -n 128 -r 3 -o json \
    2>&1 | tee "${OUTDIR}/llama-bench.json"

echo ""
echo "    llama-bench complete"

# 2. Start llama-server with tracing
echo ""
echo "--- Step 2: Starting llama-server with tracing ---"

export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm-7.2.3/lib"
export HIP_VISIBLE_DEVICES=0
export GGML_PIPELINE_PLUS=1
export GGML_PIPELINE_MULTI_BACKEND_SEQ=1
export GGML_RPC_TRACE=1
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE_FILE="${OUTDIR}/traces/rpc-trace.jsonl"
export GGML_SCHED_TRACE_FILE="${OUTDIR}/traces/sched-trace.jsonl"

SERVER_ARGS=(
    -m "$MODEL" -ngl 99 -c "$CTX" -ctk "$CTK" -ctv "$CTV"
    -fa on --host 127.0.0.1 --port "$PORT"
    --parallel 1 -np 1 --cont-batching
    --split-mode layer -ts "$TS"
    --rpc "$RPC"
    --fit off
    --metrics --slots
    --log-timestamps --log-prefix
    --reasoning off
    --no-webui
)

# Add MTP args if needed
if [[ -n "$MTP" ]]; then
    # shellcheck disable=SC2086
    SERVER_ARGS+=($MTP)
fi

"${SERVER}" "${SERVER_ARGS[@]}" > "${OUTDIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"

# Wait for server ready
WAIT_MAX=180
WAITED=0
READY=0
while [[ "$WAITED" -lt "$WAIT_MAX" ]]; do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        if grep -q "model loaded" "${OUTDIR}/server.log" 2>/dev/null; then
            READY=1
            echo "  Server ready at ${WAITED}s"
            break
        fi
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  ERROR: llama-server exited during load"
        tail -30 "${OUTDIR}/server.log"
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [[ "$READY" -ne 1 ]]; then
    echo "  ERROR: timeout waiting for server (${WAIT_MAX}s)"
    tail -20 "${OUTDIR}/server.log"
    exit 1
fi

# 3. Run inference requests
echo ""
echo "--- Step 3: Trace-capture inference ---"

# 3a. Short prompt
echo "  Run 1: short prompt"
curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}' \
    2>/dev/null > "${OUTDIR}/traces/run1-short.json" || echo "  FAIL: run1"

sleep 2

# 3b. Medium prompt with reasoning
echo "  Run 2: medium prompt (math reasoning)"
curl -s --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Solve this step by step: If you have 3 apples and you buy 5 more, then give away 2, how many do you have left?"}],"max_tokens":128}' \
    2>/dev/null > "${OUTDIR}/traces/run2-math.json" || echo "  FAIL: run2"

sleep 2

# 3c. Longer prompt (for PP trace)
echo "  Run 3: long prompt"
LONG_PROMPT=$(python3 -c "
import random
random.seed(42)
words = ['lorem','ipsum','dolor','sit','amet','consectetur','adipiscing','elit','sed','do','eiusmod','tempor','incididunt','ut','labore','et','dolore','magna','aliqua']
print(' '.join(random.choices(words, k=500)))
")
curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "import json; print(json.dumps({'messages':[{'role':'user','content':'${LONG_PROMPT}'}],'max_tokens':64}))")" \
    2>/dev/null > "${OUTDIR}/traces/run3-long.json" || echo "  FAIL: run3"

sleep 2

# 3d. Multi-turn conversation
echo "  Run 4: multi-turn conversation"
SESSION_ID="conv_$$"
curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is machine learning?"}],"max_tokens":64}' \
    2>/dev/null > "${OUTDIR}/traces/run4-conv1.json" || echo "  FAIL: run4a"

sleep 2

curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"What is machine learning?"},{"role":"assistant","content":"Machine learning is a subset of artificial intelligence..."},{"role":"user","content":"Give me a concrete example in Python code"}],"max_tokens":128}' \
    2>/dev/null > "${OUTDIR}/traces/run4-conv2.json" || echo "  FAIL: run4b"

# Wait for trace flush
sleep 3

# 4. Capture GPU utilization
echo ""
echo "--- Step 4: GPU snapshots ---"
{
    echo "timestamp,gpu,memory_used_mib,util_pct"
    for i in 1 2 3; do
        TS_NOW=$(date +%s)
        if command -v rocm-smi &>/dev/null; then
            ROCM_MEM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[0\]" | head -1 | awk '{print $5/1024/1024}' 2>/dev/null || echo "0")
            echo "${TS_NOW},7900 XTX,${ROCM_MEM},vram"
        fi
        if command -v nvidia-smi &>/dev/null; then
            NVIDIA_STAT=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0,0")
            echo "${TS_NOW},3060 Ti,${NVIDIA_STAT%,*},${NVIDIA_STAT#*,}"
        fi
        sleep 2
    done
} > "${OUTDIR}/gpu-mon.csv"

# 5. Stop server
echo ""
echo "--- Step 5: Cleaning up ---"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "  Server stopped"

# 6. Parse results
echo ""
echo "--- Step 6: Parsing results ---"
python3 << 'PYEOF'
import json, os, sys

label = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('LABEL', 'unknown')
outdir = os.environ.get('OUTDIR', f'/tmp/d41-baseline/{label}')

results = {
    'model': label,
    'config': {
        'ctx': int(os.environ.get('CTX', '8192')),
        'ts': os.environ.get('TS', ''),
        'ctk': os.environ.get('CTK', ''),
        'ctv': os.environ.get('CTV', ''),
        'rpc': os.environ.get('RPC', '')
    }
}

# Parse llama-bench
bench_file = os.path.join(outdir, 'llama-bench.json')
if os.path.exists(bench_file):
    with open(bench_file) as f:
        content = f.read()
    bench_results = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                bench_results.append(json.loads(line))
            except:
                pass
        elif line.startswith('[{') or line.startswith('['):
            # probably a json array
            pass
        elif line.startswith('{'):
            # partial line, try appending
            try:
                bench_results.append(json.loads(line))
            except:
                pass
    # Also try parsing as a whole array
    try:
        arr = json.loads(content.strip())
        if isinstance(arr, list):
            bench_results = arr
    except:
        pass
    results['llama_bench'] = bench_results

# Parse trace metadata
for tname in ['traces/rpc-trace.jsonl', 'traces/sched-trace.jsonl']:
    tf = os.path.join(outdir, tname)
    if os.path.exists(tf):
        sz = os.path.getsize(tf)
        results[tname.replace('/', '_')] = {
            'bytes': sz,
            'lines': sum(1 for _ in open(tf) if _.strip())
        }

# Parse server log for timing
server_log = os.path.join(outdir, 'server.log')
if os.path.exists(server_log):
    with open(server_log) as f:
        content = f.read()
    import re
    pp_matches = re.findall(r'pp\s+(\d+\.?\d*)\s+t/s', content)
    tg_matches = re.findall(r'tg\s+(\d+\.?\d*)\s+t/s', content)
    if pp_matches:
        results['server_pp_t_s'] = [float(x) for x in pp_matches]
    if tg_matches:
        results['server_tg_t_s'] = [float(x) for x in tg_matches]

# Parse run results
for run_file in ['traces/run1-short.json', 'traces/run2-math.json', 'traces/run3-long.json', 'traces/run4-conv1.json', 'traces/run4-conv2.json']:
    rf = os.path.join(outdir, run_file)
    if os.path.exists(rf):
        try:
            with open(rf) as f:
                data = json.load(f)
            run_key = run_file.replace('/', '_').replace('.json', '')
            timings = data.get('timings', {})
            usage = data.get('usage', {})
            results[run_key] = {
                'prompt_tokens': usage.get('prompt_tokens', timings.get('prompt_n', 0)),
                'generated_tokens': usage.get('completion_tokens', timings.get('predicted_n', 0)),
                'prompt_per_second': timings.get('prompt_per_second', 0),
                'predicted_per_second': timings.get('predicted_per_second', 0),
                'prompt_time_ms': timings.get('prompt_ms', timings.get('prompt_time', 0)),
                'predicted_time_ms': timings.get('predicted_ms', timings.get('predicted_time', 0)),
            }
            # Content preview
            choices = data.get('choices', [])
            if choices:
                msg = choices[0].get('message', {})
                content_preview = msg.get('content', '')[:200]
                results[run_key]['content_preview'] = content_preview
                results[run_key]['finish_reason'] = choices[0].get('finish_reason', '')
        except Exception as e:
            results[run_file.replace('/', '_').replace('.json', '') + '_error'] = str(e)
            # Try to get error response
            try:
                with open(rf) as f:
                    raw = f.read()[:500]
                results[run_file.replace('/', '_').replace('.json', '') + '_raw'] = raw
            except:
                pass

# GPU monitoring
gpu_file = os.path.join(outdir, 'gpu-mon.csv')
if os.path.exists(gpu_file):
    with open(gpu_file) as f:
        lines = f.read().strip().split('\n')
    results['gpu_snapshots_csv'] = '\n'.join(lines)

with open(os.path.join(outdir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'Results: {outdir}/results.json')
print(f'  llama_bench: {len(results.get("llama_bench", []))} entries')
for run_key in ['traces_run1-short', 'traces_run2-math', 'traces_run3-long', 'traces_run4-conv1', 'traces_run4-conv2']:
    if run_key in results:
        d = results[run_key]
        print(f'  {run_key}: pp={d.get("prompt_per_second",0):.1f} t/s, tg={d.get("predicted_per_second",0):.1f} t/s, {d.get("generated_tokens",0)} tokens')
PYEOF

echo ""
echo "  Done: ${LABEL}"
