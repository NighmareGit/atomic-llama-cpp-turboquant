#!/bin/bash
# Phase 1 feedback loop: detect garbling in profiler output
# PASS (exit 0) = clean output, FAIL (exit 1) = garbled output
set -euo pipefail

MODEL="/mnt/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
RPC_ENDPOINT="localhost:50056"
PROFILER="${HOME}/projects/path-d-gpipeline-assembly-line/build-hip/bin/llama-gpipe-profiler"
WORKDIR="${1:-/tmp/garble-check-$$}"
RPC_LAYERS="${2:-5}"
ROCm_LAYERS="${3:-15}"

# Total n_gpu_layers = layers before RPC + RPC layers + ROCm layers + output layer
# The "before RPC" part goes to CPU.
# Layers assignment by the system:
#   Layers 0..8 -> CPU (default, before RPC)
#   Layers next RPC_LAYERS -> RPC0
#   Layers next ROCm_LAYERS + output -> ROCm0
N_GPU=$((RPC_LAYERS + ROCm_LAYERS + 1))

mkdir -p "${WORKDIR}"

echo "=== Garble Check ==="
echo "Model: ${MODEL}"
echo "RPC: ${RPC_ENDPOINT}"
echo "RPC layers: ${RPC_LAYERS}"
echo "ROCm layers: ${ROCm_LAYERS}"
echo "n_gpu_layers: ${N_GPU}"
echo ""

# Ensure clean env
unset GGML_SCHED_GPIPE GGML_SCHED_GPIPE_DEPTH GGML_RPC_TRACE GGML_SCHED_TRACE GGML_PIPELINE_TRACE

# Run profiler with RPC output
echo ">>> Running profiler (RPC + ROCm)..."
LLAMA_LOG_LEVEL=3 \
  LD_LIBRARY_PATH="${HOME}/projects/path-d-gpipeline-assembly-line/build-hip/lib" \
  "${PROFILER}" \
    -m "${MODEL}" \
    --rpc "${RPC_ENDPOINT}" \
    --n-gpu-layers "${N_GPU}" \
    --tasks tg \
    --n-gen 64 \
    --sample \
    --repeat 1 \
    --warmup \
    --out-dir "${WORKDIR}" \
    -o "${WORKDIR}/heatmap.json" \
    > "${WORKDIR}/stdout.txt" \
    2>"${WORKDIR}/stderr.txt" || true

# Get the generated text (skip progress dots)
TEXT=$(cat "${WORKDIR}/stdout.txt" | tr -d '.\nc' | head -c 2000)

echo ""
echo "=== OUTPUT ==="
echo "${TEXT}"
echo ""
echo "=== ANALYSIS ==="

python3 -c "
import sys, json, re, math

text = sys.stdin.read()

# Strip leading/trailing whitespace
text = text.strip()

if len(text) == 0:
    print('FAIL: empty output')
    sys.exit(1)

# Heuristic 1: substring repetition (len>=3, repeated >=4 times)
reps_found = False
for l in range(3, min(9, len(text)//4 + 1)):
    for i in range(len(text) - l*4 + 1):
        chunk = text[i:i+l]
        j = i + l
        count = 1
        while j + l <= len(text) and text[j:j+l] == chunk:
            count += 1
            j += l
        if count >= 4 and not chunk.isspace() and chunk.strip():
            reps_found = True
            break
    if reps_found:
        break

# Heuristic 2: very low non-whitespace content
nw_chars = sum(1 for c in text if not c.isspace())
ratio = nw_chars / max(len(text), 1)

# Heuristic 3: look for actual English word-like tokens
words = re.findall(r'[a-zA-Z]{2,}', text.lower())
word_count = len(words)
unique_words = len(set(words))

# Heuristic 4: null/unicode replacement characters
null_chars = text.count('\u0000') + text.count('\ufffd') + text.count('<|endoftext|>')

# Heuristic 5: very short text (likely EOS on first token)
if len(text) < 10:
    print('FAIL: text too short (likely immediate EOS)')
    sys.exit(1)

garbled = False
reasons = []

if reps_found:
    garbled = True
    reasons.append(f'repetition')
if ratio < 0.05:
    garbled = True
    reasons.append(f'low-content(ratio={ratio:.3f})')
if word_count == 0:
    garbled = True
    reasons.append('no-words')
if unique_words <= 2 and word_count >= 8:
    garbled = True
    reasons.append(f'word-repetition(uniq={unique_words})')
if null_chars >= 3:
    garbled = True
    reasons.append(f'null-unicode-chars(count={null_chars})')

verdict = 'GARBLED' if garbled else 'CLEAN'
print(f'{verdict} len={len(text)} nw_chars={nw_chars} ratio={ratio:.3f} words={word_count} uniq_words={unique_words} nulls={null_chars}')
if reasons:
    print(f'  reasons: {', '.join(reasons)}')

sys.exit(1 if garbled else 0)
" <<< "${TEXT}"
rc=$?

echo ""
echo "Output file: ${WORKDIR}/stdout.txt"
echo "Stderr file: ${WORKDIR}/stderr.txt"
echo ""

exit $rc
