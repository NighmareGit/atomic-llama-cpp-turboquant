#!/usr/bin/env bash
# D4.1 Longer Context Tests
# Runs conversation and complex reasoning tests with tracked context length.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${ROOT}/build-rocm-docker/bin/llama-server"
PORT=8083
RPC="127.0.0.1:50051"
OUTDIR="/tmp/d41-baseline/long-context"
mkdir -p "$OUTDIR"

# Start server with larger context (16K for 9B MTP model)
echo "=== Starting server (16K context) ==="
export LD_LIBRARY_PATH="${ROOT}/build-rocm-docker/bin:/opt/rocm-7.2.3/lib"
export HIP_VISIBLE_DEVICES=0
unset GGML_PIPELINE_PLUS GGML_PIPELINE_MULTI_BACKEND_SEQ

# Kill any existing on our port
PREV=$(lsof -ti:${PORT} 2>/dev/null || true)
[[ -n "$PREV" ]] && kill "$PREV" 2>/dev/null || true
sleep 1

${SERVER} \
  -m /mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf \
  -ngl 99 -c 16384 -ctk q8_0 -ctv turbo3 \
  -fa on --host 127.0.0.1 --port ${PORT} \
  --parallel 1 -np 1 --cont-batching \
  --split-mode layer -ts 45,55 \
  --rpc ${RPC} --fit off \
  --metrics --slots --log-timestamps --log-prefix \
  --reasoning off --no-webui --no-warmup \
  > "$OUTDIR/server.log" 2>&1 &
PID=$!
echo "PID=$PID"

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
    echo "ready at ${i}s"
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "SERVER DIED"
    tail -30 "$OUTDIR/server.log"
    exit 1
  fi
  sleep 2
done

echo ""
echo "=== Test 1: Long conversation (10-turn hypothetical scenario) ==="
# Build a multi-turn conversation about building a Mars colony
CONV_MSGS='[
  {"role":"user","content":"What are the main challenges of establishing a permanent human colony on Mars?"},
  {"role":"assistant","content":"The main challenges include: 1) Radiation exposure due to lack of magnetic field and thin atmosphere, increasing cancer risk. 2) Extreme temperature swings from -140C to 20C. 3) Low gravity (38% of Earth) causing muscle atrophy and bone density loss. 4) Dust storms that can last months and damage equipment. 5) Water extraction from subsurface ice requires significant energy. 6) Food production needs controlled environment agriculture. 7) Psychological isolation from Earth with 3-22 minute communication delay. 8) Reliable life support systems must be completely closed-loop."},
  {"role":"user","content":"How would you design a power system for the first 100 colonists? Focus on reliability and redundancy."},
  {"role":"assistant","content":"A robust Mars colony power system for 100 people would need approximately 2-5 MW. Primary generation would come from compact nuclear fission reactors (kilopower-type) providing 24/7 baseload power regardless of dust storms. Secondary would be large solar arrays on tall poles above dust level, sized at 3x peak demand with battery storage. Redundancy would include three independent reactor units (each capable of 60% load), distributed battery storage at each habitat module, and hydrogen fuel cells as emergency backup. Power distribution would use a microgrid with automatic islanding so failures don\'t cascade. Thermal management would use heat rejection radiators and waste heat for water recycling and greenhouse heating."},
  {"role":"user","content":"What computational infrastructure would be needed? Consider AI assistants, communication, and scientific computing."},
  {"role":"assistant","content":"Mars colony computation would need tiered architecture: 1) Edge AI nodes in each habitat running local llama.cpp instances for latency-sensitive assistant tasks (no 22-min round trip to Earth). 2) Central computing cluster with redundant GPU servers for scientific modeling, life support monitoring, and resource optimization. 3) Communication relay with data compression and store-and-forward buffers handling the Earth-Mars latency. 4) Distributed sensor network for environmental monitoring. 5) Digital twin of the entire colony running simulations. The total compute would be roughly 10-20 petaflops, powered by the nuclear reactors."},
  {"role":"user","content":"Explain how you would implement a closed-loop life support system that recycles water, air, and waste. Include specific technologies."}
]'

curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
msgs = json.loads('$CONV_MSGS')
print(json.dumps({
    'messages': msgs,
    'max_tokens': 256,
}))" 2>/dev/null)" \
  > "$OUTDIR/conv-long.json" 2>&1

python3 -c "
import json
with open('$OUTDIR/conv-long.json') as f:
    data = json.load(f)
t = data.get('timings', {})
u = data.get('usage', {})
print(f'prompt_tokens={u.get(\"prompt_tokens\",t.get(\"prompt_n\",\"?\"))}')
print(f'generated_tokens={u.get(\"completion_tokens\",t.get(\"predicted_n\",\"?\"))}')
print(f'pp={t.get(\"prompt_per_second\",0):.1f} t/s')
print(f'tg={t.get(\"predicted_per_second\",0):.1f} t/s')
c = data.get('choices',[{}])[0].get('message',{}).get('content','')
if c: print(f'reponse ({len(c)} chars): {c[:200]}...')
else: print('EMPTY response - check server log')
print(f'finish_reason={data.get(\"choices\",[{}])[0].get(\"finish_reason\",\"?\")}')
" 2>&1

echo ""
echo "=== Test 2: Complex math problem (multi-step calculus) ==="
curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Solve the following problem step by step: A particle moves along a line such that its velocity at time t is v(t) = 3t^2 - 12t + 9 meters per second. (a) Find the displacement of the particle from t=0 to t=5. (b) Find the total distance traveled from t=0 to t=5. Show all your work."}],
    "max_tokens":512
  }' > "$OUTDIR/math-calc.json" 2>&1

python3 -c "
import json
with open('$OUTDIR/math-calc.json') as f:
    data = json.load(f)
t = data.get('timings', {})
u = data.get('usage', {})
print(f'prompt_tokens={u.get(\"prompt_tokens\",t.get(\"prompt_n\",\"?\"))}')
print(f'generated_tokens={u.get(\"completion_tokens\",t.get(\"predicted_n\",\"?\"))}')
print(f'pp={t.get(\"prompt_per_second\",0):.1f} t/s')
print(f'tg={t.get(\"predicted_per_second\",0):.1f} t/s')
c = data.get('choices',[{}])[0].get('message',{}).get('content','')
if c: print(f'response ({len(c)} chars): {c[:300]}...')
else: print('EMPTY response')
print(f'finish_reason={data.get(\"choices\",[{}])[0].get(\"finish_reason\",\"?\")}')
" 2>&1

echo ""
echo "=== Test 3: Logic puzzle ==="
curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Solve this logic puzzle: There are five houses in a row, each of a different color. Each house is occupied by a person of a different nationality. Each person drinks a different beverage, smokes a different brand of cigar, and keeps a different pet. Use the following clues: 1. The Englishman lives in the red house. 2. The Spaniard owns the dog. 3. Coffee is drunk in the green house. 4. The Ukrainian drinks tea. 5. The green house is immediately to the right of the ivory house. 6. The Old Gold smoker owns snails. 7. Kools are smoked in the yellow house. 8. Milk is drunk in the middle house. 9. The Norwegian lives in the first house. 10. The man who smokes Chesterfields lives in the house next to the man with the fox. 11. Kools are smoked in the house next to the house where the horse is kept. 12. The Lucky Strike smoker drinks orange juice. 13. The Japanese smokes Parliaments. 14. The Norwegian lives next to the blue house. Question: Who drinks water? Who owns the zebra? Show your reasoning."}],
    "max_tokens":512
  }' > "$OUTDIR/logic-zebra.json" 2>&1

python3 -c "
import json
with open('$OUTDIR/logic-zebra.json') as f:
    data = json.load(f)
t = data.get('timings', {})
u = data.get('usage', {})
print(f'prompt_tokens={u.get(\"prompt_tokens\",t.get(\"prompt_n\",\"?\"))}')
print(f'generated_tokens={u.get(\"completion_tokens\",t.get(\"predicted_n\",\"?\"))}')
print(f'pp={t.get(\"prompt_per_second\",0):.1f} t/s')
print(f'tg={t.get(\"predicted_per_second\",0):.1f} t/s')
c = data.get('choices',[{}])[0].get('message',{}).get('content','')
if c: print(f'response ({len(c)} chars): {c[:300]}...')
else: print('EMPTY response')
print(f'finish_reason={data.get(\"choices\",[{}])[0].get(\"finish_reason\",\"?\")}')
" 2>&1

echo ""
echo "=== Test 4: Multi-turn conversation follow-up (deepening) ==="
# First turn
curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain how quantum computing differs from classical computing."}],"max_tokens":128}' \
  > "$OUTDIR/conv-deep1.json" 2>&1

python3 -c "
import json
with open('$OUTDIR/conv-deep1.json') as f:
    data = json.load(f)
t = data.get('timings', {})
u = data.get('usage', {})
c = data.get('choices',[{}])[0].get('message',{}).get('content','')
print(f'Turn 1: prompt={u.get(\"prompt_tokens\",\"?\")} gen={u.get(\"completion_tokens\",\"?\")} pp={t.get(\"prompt_per_second\",0):.1f} tg={t.get(\"predicted_per_second\",0):.1f}')
" 2>&1

# Second turn (uses KV cache from first)
curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain how quantum computing differs from classical computing."},{"role":"assistant","content":"Quantum computing uses qubits that can exist in superposition of states, unlike classical bits that are either 0 or 1. This allows quantum computers to process certain problems exponentially faster through quantum parallelism and entanglement."},{"role":"user","content":"What specific algorithms demonstrate quantum advantage?"}],"max_tokens":128}' \
  > "$OUTDIR/conv-deep2.json" 2>&1

python3 -c "
import json
with open('$OUTDIR/conv-deep2.json') as f:
    data = json.load(f)
t = data.get('timings', {})
u = data.get('usage', {})
c = data.get('choices',[{}])[0].get('message',{}).get('content','')
print(f'Turn 2: prompt={u.get(\"prompt_tokens\",\"?\")} gen={u.get(\"completion_tokens\",\"?\")} pp={t.get(\"prompt_per_second\",0):.1f} tg={t.get(\"predicted_per_second\",0):.1f}')
" 2>&1

# Third turn
curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain how quantum computing differs from classical computing."},{"role":"assistant","content":"Quantum computing uses qubits that can exist in superposition of states..."},{"role":"user","content":"What specific algorithms demonstrate quantum advantage?"},{"role":"assistant","content":"Key algorithms include Shor'"'"'s algorithm for factoring (exponential speedup over classical), Grover'"'"'s algorithm for search (quadratic speedup), and the Quantum Fourier Transform which underlies many quantum algorithms."},{"role":"user","content":"What are the main engineering challenges in building a fault-tolerant quantum computer?"}],"max_tokens":128}' \
  > "$OUTDIR/conv-deep3.json" 2>&1

python3 -c "
import json
with open('$OUTDIR/conv-deep3.json') as f:
    data = json.load(f)
t = data.get('timings', {})
u = data.get('usage', {})
print(f'Turn 3: prompt={u.get(\"prompt_tokens\",\"?\")} gen={u.get(\"completion_tokens\",\"?\")} pp={t.get(\"prompt_per_second\",0):.1f} tg={t.get(\"predicted_per_second\",0):.1f}')
c = data.get('choices',[{}])[0].get('message',{}).get('content','')
if c: print(f'  response ({len(c)} chars): {c[:150]}...')
" 2>&1

echo ""
echo "=== GPU snapshot ==="
{
  echo "timestamp,gpu,memory_used_mib"
  TS=$(date +%s)
  rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[0\]" | head -1 | awk -v ts="$TS" '{print ts",7900 XTX,"($5/1024/1024)}'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -v ts="$TS" '{print ts",3060 Ti,"$1}'
} > "$OUTDIR/gpu-snap.csv"

# Stop server
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo ""
echo "=== LONG CONTEXT TESTS COMPLETE ==="
echo "Results in $OUTDIR/"
