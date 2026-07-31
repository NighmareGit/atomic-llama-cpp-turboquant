#!/bin/bash
# Phase 1 feedback loop: detect KV-cache garble under GGML_PIPELINE_PLUS=1
# PASS (exit 0) = clean output, FAIL (exit 1) = garbled output
# usage: loop-check-garble.sh [PORT PROMPT TOKENS SEED EXPECT_SUBSTR]
#   EXPECT_SUBSTR (optional): case-insensitive substring; GREEN requires it
set -euo pipefail

PORT="${1:-8095}"
PROMPT="${2:-What is 2+2?}"
TOKENS="${3:-64}"
SEED="${4:-42}"
EXPECT_SUBSTR="${5:-}"

PAYLOAD=$(python3 -c "
import json
d = {
    'messages': [{'role': 'user', 'content': '''${PROMPT}'''}],
    'max_tokens': ${TOKENS},
    'temperature': 0,
    'seed': ${SEED},
    'stream': False
}
print(json.dumps(d))
")

RESP=$(curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}")

if [ -z "$RESP" ]; then
    echo "FAIL: empty response from server"
    exit 1
fi

EXPECT_SUBSTR="${EXPECT_SUBSTR}" python3 -c "
import sys, json, re, os

raw = sys.stdin.read()
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    print(f'FAIL: JSON parse error: {e}')
    sys.exit(1)

if 'error' in data:
    print(f'FAIL: server error: {data[\"error\"]}')
    sys.exit(1)

msg = data['choices'][0]['message']
content = msg.get('content') or ''
reasoning = msg.get('reasoning_content') or ''
n_tokens = data['usage']['completion_tokens']

# prefer content; fall back to reasoning_content when content blank
if content.strip():
    display = content
    text = content + (' ' + reasoning if reasoning.strip() else '')
else:
    display = reasoning
    text = reasoning
text = text.replace('\n', ' ')

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

garbled = False
reasons = []

if reps_found:
    garbled = True
    reasons.append('repetition')
if ratio < 0.05 and n_tokens >= 16:
    garbled = True
    reasons.append(f'low-content(ratio={ratio:.3f})')
if n_tokens >= 16 and word_count == 0:
    garbled = True
    reasons.append('no-words')
if n_tokens >= 32 and unique_words <= 2 and word_count >= 8:
    garbled = True
    reasons.append(f'word-repetition(uniq={unique_words})')

# Semantic assert: optional EXPECT_SUBSTR must appear (case-insensitive)
if os.environ.get('EXPECT_SUBSTR'):
    if os.environ['EXPECT_SUBSTR'].lower() not in text.lower():
        garbled = True
        reasons.append(f'semantic-miss(expected={os.environ[\"EXPECT_SUBSTR\"]})')

verdict = 'GARBLED' if garbled else 'CLEAN'
print(f'{verdict} tokens={n_tokens} nw_chars={nw_chars} ratio={ratio:.3f} words={word_count} uniq_words={unique_words}')
if reasons:
    print(f'  reasons: {\", \".join(reasons)}')
print('---CONTENT---')
print(display[:300])
print('---END---')

sys.exit(1 if garbled else 0)
" <<< "$RESP"
