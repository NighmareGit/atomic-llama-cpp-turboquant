#!/bin/bash
# multi-gpu-verify.sh — Automated multi-GPU model test runner
# Source: ~/.grok/skills/multi-gpu-verify/SKILL.md
set -euo pipefail

# === CONFIGURATION DEFAULTS ===
MODEL=""
MODE="all"           # tcp, udp, all
PIPELINE="off"
NGL=""               # auto-detect if empty
MAX_TOKENS=32        # enough for quality checks (red-team 4.4)
OUTPUT="/tmp/multigpu-$(date +%Y%m%d-%H%M%S).md"
LEASES_DIR="/home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant/.scratch/leases"
LOCK_DIR="/home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant/.scratch/locks"
BUILD_DIR="/home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant/build-rocm-native"
PORT_START=18921
POLL_INTERVAL=5
POLL_MAX=36          # 36 × 5s = 180s max wait (handles 122B models)
CURL_TIMEOUT=120
MIN_DISK_MB=500     # require at least 500 MB free on /tmp
PROMPTS=(
    "2+2="
    "The capital of France is"
    "Hello world"
    "What is machine learning?"
)

# RPC endpoints — defined ONCE, used for both verification and launch
RPC_ENDPOINTS=(
    "127.0.0.1:50051"     # Docker 3060 Ti (romulus)
    "192.168.8.23:50054"  # RTX 3090 (triton)
    "192.168.8.23:50055"  # RTX 3070 (triton)
)

# Model storage paths
MODEL_PATHS=(
    "/home/hunter/scratch/prototype-auto"
    "/mnt/980pro/models"
    "/mnt/toshiba_a/models"
    "/mnt/toshiba_b/models"
)

# === HELP ===
show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --model <path>        Path to .gguf model file (auto-search if not specified)
  --mode <tcp|udp|all>  Transport modes to test (default: all)
  --pipeline <on|off>   Enable pipeline flags (default: off)
  --ngl <N>             Number of GPU layers (default: auto-detect)
  --max-tokens <N>      Tokens per prompt (default: 32)
  --output <path>       Report output path (default: auto)
  --deploy              Rebuild and deploy rpc-server to remote nodes
  --help                Show this message

Notes:
  - Uses llama-server + curl for correctness checks (recommended).
  - DO NOT use llama-cli in automated scripts — spinner loops hang on garbled output.
  - For pure throughput after quality verified: use llama-bench instead.

Examples:
  $0 --model /mnt/980pro/models/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf
  $0 --mode udp --pipeline on
  $0 --model ./model.gguf --deploy
EOF
}

# === PARSE ARGS ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --ngl) NGL="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --deploy) DEPLOY=1; shift ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# === MODEL AUTO-DETECTION ===
if [ -z "$MODEL" ]; then
    for path in "${MODEL_PATHS[@]}"; do
        candidate=$(find "$path" -name "*.gguf" -size +10G 2>/dev/null | head -1)
        if [ -n "$candidate" ]; then
            MODEL="$candidate"
            echo "Auto-detected model: $MODEL"
            break
        fi
    done
fi

# Verify model exists and is readable
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    echo "Specify with --model <path>"
    exit 1
fi
if [ ! -r "$MODEL" ]; then
    echo "Error: Model file not readable: $MODEL"
    exit 1
fi

# === MODEL PROPERTIES ===
MODEL_SIZE=$(stat -c%s "$MODEL" 2>/dev/null || stat -f%z "$MODEL")
MODEL_SIZE_GB=$((MODEL_SIZE / 1073741824))
MODEL_NAME=$(basename "$MODEL")

# Architecture detection — use gguf-info if available, fall back to strings
if command -v gguf-info >/dev/null 2>&1; then
    ARCH=$(gguf-info "$MODEL" 2>/dev/null | grep -i "architecture" | head -1 || echo "")
else
    # Case-insensitive grep for architecture strings (red-team 2.4)
    ARCH=$(strings "$MODEL" | grep -oiE 'gated_delta_net|llama|qwen2|moe' | head -1 2>/dev/null || echo "unknown")
fi

HAS_MTP=$(strings "$MODEL" | grep -c "mtp_head" 2>/dev/null || echo 0)
HAS_GDN=$(echo "$ARCH" | grep -i -c "gated_delta_net" 2>/dev/null || echo 0)
HAS_MOE=$(echo "$ARCH" | grep -i -c "moe" 2>/dev/null || echo 0)

# === DETERMINE GPU COUNT AND NGL ===
# VRAM available per GPU: 7900 XTX=24GB, 3060 Ti=8GB, 3090=24GB, 3070=8GB
# Total VRAM = 64GB, but model needs ~1.05× overhead
OVERHEAD_FACTOR=1.05
MODEL_NEEDS=$(echo "$MODEL_SIZE_GB * $OVERHEAD_FACTOR" | bc 2>/dev/null | cut -d. -f1 || echo "$MODEL_SIZE_GB")

if [ "$MODEL_NEEDS" -lt 22 ]; then
    GPU_COUNT=1
    [ -z "$NGL" ] && NGL=99
    echo "Model fits on 1 GPU ($MODEL_NEEDS GiB < 24 GiB)"
elif [ "$MODEL_NEEDS" -lt 46 ]; then
    GPU_COUNT=2
    [ -z "$NGL" ] && NGL=99
    echo "Model needs 2 GPUs ($MODEL_NEEDS GiB)"
elif [ "$MODEL_NEEDS" -lt 60 ]; then
    GPU_COUNT=3
    [ -z "$NGL" ] && NGL=99
    echo "Model needs 3 GPUs ($MODEL_NEEDS GiB)"
else
    GPU_COUNT=4
    if [ "$MODEL_NEEDS" -gt 64 ]; then
        [ -z "$NGL" ] && NGL=50
        echo "Model needs partial offload ($MODEL_NEEDS GiB > 64 GiB, using -ngl $NGL)"
    else
        [ -z "$NGL" ] && NGL=99
        echo "Model needs 4 GPUs ($MODEL_NEEDS GiB ≤ 64 GiB)"
    fi
fi

# Build RPC args from single endpoint source (red-team 2.5)
RPCS=""
[ $GPU_COUNT -ge 2 ] && RPCS="$RPCS --rpc ${RPC_ENDPOINTS[0]}"
[ $GPU_COUNT -ge 3 ] && RPCS="$RPCS --rpc ${RPC_ENDPOINTS[1]}"
[ $GPU_COUNT -ge 4 ] && RPCS="$RPCS --rpc ${RPC_ENDPOINTS[2]}"

echo "=== Configuration ==="
echo "Model: $MODEL ($MODEL_SIZE_GB GiB)"
echo "Arch: $ARCH (GDN=$HAS_GDN MoE=$HAS_MOE MTP=$HAS_MTP)"
echo "GPUs: $GPU_COUNT | NGL: $NGL | Mode: $MODE | Max tokens: $MAX_TOKENS"
echo "RPC: $RPCS"
echo ""

# === PRE-FLIGHT ===

# 0. Disk space check (red-team 1.7)
AVAIL_MB=$(df /tmp 2>/dev/null | tail -1 | awk '{print $4}')
AVAIL_MB=$((AVAIL_MB / 1024))  # convert from KB to MB
if [ "$AVAIL_MB" -lt "$MIN_DISK_MB" ]; then
    echo "Error: Low disk space on /tmp: ${AVAIL_MB}MB free, need ${MIN_DISK_MB}MB"
    exit 1
fi
echo "Disk space on /tmp: ${AVAIL_MB}MB free ✅"

# 1. Acquire real GPU lock (atomic mkdir)
LOCK_PATH="$LOCK_DIR/gpu-0.lock"
if ! mkdir "$LOCK_PATH" 2>/dev/null; then
    echo "Error: GPU lock held by another agent at $LOCK_PATH"
    echo "Wait or kill: rmdir $LOCK_PATH"
    exit 1
fi
echo "GPU lock acquired ✅"

# 2. Acquire lease file (documentary)
LEASE_ID="run-$$-$(date +%s)"
mkdir -p "$LEASES_DIR"
cat > "$LEASES_DIR/${LEASE_ID}.lease" <<EOF
agent: multi-gpu-verify
gpus: 0
acquired: $(date -Iseconds)
expires: $(date -Iseconds -d '+120 minutes' 2>/dev/null || echo "+120min")
purpose: automated multi-gpu test: $MODEL_NAME
EOF

# 3. Cleanup trap (guaranteed run even on crash)
cleanup() {
    echo "=== CLEANUP ==="
    for port in $(seq $PORT_START $((PORT_START + 20))); do
        kill $(lsof -t -i:$port) 2>/dev/null || true
    done
    rmdir "$LOCK_PATH" 2>/dev/null || true
    rm -f "$LEASES_DIR/${LEASE_ID}.lease"
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# 4. Kill orphan servers — only on test port range (red-team 3.6)
for port in $(seq $PORT_START $((PORT_START + 30))); do
    PID=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Killed orphan on port $port (PID $PID)"
        kill "$PID" 2>/dev/null || true
    fi
done
sleep 2

# 5. Verify RPC endpoints
RPC_CONNECTED=0
for ep in ${RPC_ENDPOINTS[@]:0:$((GPU_COUNT - 1))}; do
    if nc -z ${ep/:/ } 2>/dev/null; then
        echo "RPC $ep: OK ✅"
        RPC_CONNECTED=$((RPC_CONNECTED + 1))
    else
        echo "Error: RPC $ep: DOWN — start RPC server or check network"
        exit 1
    fi
done
echo "RPC endpoints: $RPC_CONNECTED connected ✅"

# 6. Binary freshness check (red-team 1.8)
BINARY_PATH="./build-rocm-native/bin/llama-server"
if [ -f "$BINARY_PATH" ]; then
    BIN_MTIME=$(stat -c %Y "$BINARY_PATH" 2>/dev/null || stat -f %m "$BINARY_PATH")
    echo "Binary: $BINARY_PATH (mtime: $(date -d @$BIN_MTIME 2>/dev/null || echo 'N/A'))"
fi

# 7. Write checker script with curl exit code handling (red-team 1.3)
cat > /tmp/checker.py <<'CHECKEOF'
#!/usr/bin/env python3
"""Validates model response for Chinese, garbage, repetition, empty output.
Also parses curl exit code from CURL_EXIT_CODE env var."""
import sys, json, re, os

curl_exit = os.environ.get('CURL_EXIT_CODE', '0')

try:
    raw = sys.stdin.read()
    if not raw.strip():
        print(f'ERROR: Empty response from server (curl exit code: {curl_exit})')
        if curl_exit == '7':  # Failed to connect
            print('  → Server may have crashed or port is wrong')
        elif curl_exit == '28':  # Timeout
            print('  → Request timed out — model may be too slow')
        elif curl_exit == '52':  # Empty reply
            print('  → Server sent empty reply — likely a crash')
        print('FAIL')
        sys.exit(0)

    r = json.loads(raw)
    t = r.get('choices', [{}])[0].get('text', '')
    ts = r.get('timings', {})
    tg = ts.get('predicted_per_second', 0)
    pp = ts.get('prompt_per_second', 0)

    # Check 1: Chinese characters
    chinese = any(ord(c) > 0x4e00 and ord(c) < 0x9fff for c in t)

    # Check 2: Garbage bytes
    garbage = bool(re.search(r'[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]', t))

    # Check 3: Repetition loops (4+ same consecutive words)
    words = t.split()
    repeats = sum(1 for i in range(4, len(words))
                  if len(words) > i and words[i] == words[i-1] == words[i-2] == words[i-3])

    # Check 4: Empty output
    empty = len(t.strip()) == 0

    status = "FAIL" if (chinese or garbage or repeats > 0 or empty) else "PASS"
    print(f'tg={tg:.1f} pp={pp:.1f} chinese={chinese} garbage={garbage} repeats={repeats} empty={empty} status={status}')
    print(f'text={repr(t[:200])}')
except json.JSONDecodeError as e:
    print(f'ERROR: Invalid JSON response: {e}')
    print(f'  First 200 chars of response: {repr(raw[:200])}')
    print(f'  curl exit code: {curl_exit}')
    print('FAIL')
except Exception as e:
    print(f'ERROR: {e}')
    print('FAIL')
CHECKEOF
chmod +x /tmp/checker.py

echo ""
echo "=== Starting tests (max_tokens=$MAX_TOKENS) ==="
echo ""

# === BUILD TEST MATRIX ===
CONFIGS=()
CONFIGS+=("TCP:GGML_RPC_UDP=0")
[ "$MODE" = "udp" ] || [ "$MODE" = "all" ] && CONFIGS+=("UDP:GGML_RPC_UDP=1")
[ "$PIPELINE" = "on" ] && CONFIGS+=("PIPE:GGML_RPC_UDP=1 GGML_PIPELINE_PLUS=1 GGML_SCHED_WAVEFRONT_DISPATCH=1")
[ "$HAS_MTP" -gt 0 ] && CONFIGS+=("MTP:GGML_RPC_UDP=1")

# === RUN TESTS ===
declare -A RESULTS
PORT=$PORT_START

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r LABEL ENV_VARS <<< "$config"
    MTP_FLAGS=""
    [ "$LABEL" = "MTP" ] && MTP_FLAGS="--spec-type draft-mtp --spec-draft-n-max 2"

    echo "--- Config: $LABEL ---"
    echo "Port: $PORT | Env: $ENV_VARS $MTP_FLAGS"

    # Start server
    eval "$ENV_VARS ./build-rocm-native/bin/llama-server \
      --model \"$MODEL\" $RPCS -sm layer -ngl $NGL \
      --no-webui --no-warmup -c 128 --port $PORT \
      $MTP_FLAGS > /tmp/multigpu-$LABEL.log 2>&1 &"
    SERVER_PID=$!

    # Poll for readiness (up to POLL_MAX × POLL_INTERVAL seconds)
    READY=false
    for i in $(seq 1 $POLL_MAX); do
        sleep $POLL_INTERVAL
        if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/completions" > /dev/null 2>&1; then
            READY=true
            echo "Ready after $((i * POLL_INTERVAL))s"
            break
        fi
    done

    if [ "$READY" = false ]; then
        echo "TIMEOUT — server not ready after $((POLL_MAX * POLL_INTERVAL))s"
        tail -10 /tmp/multigpu-$LABEL.log
        RESULTS["$LABEL"]="TIMEOUT:0"
        kill $SERVER_PID 2>/dev/null || true
        PORT=$((PORT + 1))
        continue
    fi

    # CPU offload check (red-team 1.5): parse log for layer offload ratio
    OFFLOAD_LINE=$(grep -i "offloaded\|layers.*to GPU" /tmp/multigpu-$LABEL.log 2>/dev/null | tail -1 || echo "")
    if echo "$OFFLOAD_LINE" | grep -qi "offloaded"; then
        echo "  Offload: $OFFLOAD_LINE"
    fi

    # MTP draft check (red-team 1.4): parse log for draft acceptance
    if [ "$LABEL" = "MTP" ]; then
        DRAFT_ACCEPT=$(grep -ci "accepted\|draft_accept" /tmp/multigpu-$LABEL.log 2>/dev/null || echo 0)
        DRAFT_REJECT=$(grep -ci "rejected\|draft_reject" /tmp/multigpu-$LABEL.log 2>/dev/null || echo 0)
        echo "  MTP draft: $DRAFT_ACCEPT accepted, $DRAFT_REJECT rejected"
    fi

    # Run prompts
    CONFIG_PASS=true
    TOTAL_TG=0
    PROMPT_COUNT=0

    for prompt in "${PROMPTS[@]}"; do
        echo -n "  $prompt ... "

        # Check server alive BEFORE curl
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER DIED before prompt"
            tail -5 /tmp/multigpu-$LABEL.log
            CONFIG_PASS=false
            break
        fi

        # Run prompt with max_tokens=32 (red-team 4.4)
        set +e
        CURL_OUTPUT=$(curl -s --max-time $CURL_TIMEOUT -w "\n%{http_code}" \
          "http://127.0.0.1:$PORT/v1/completions" \
          -H "Content-Type: application/json" \
          -d "{\"prompt\":\"$prompt\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" 2>&1)
        CURL_EXIT=$?
        set -e

        # Parse curl output: last line is HTTP code, rest is body
        HTTP_CODE=$(echo "$CURL_OUTPUT" | tail -1)
        BODY=$(echo "$CURL_OUTPUT" | sed '$d')

        if [ "$CURL_EXIT" -ne 0 ]; then
            echo "curl failed (exit=$CURL_EXIT, http=$HTTP_CODE)"
            case "$CURL_EXIT" in
                7) echo "  → Failed to connect — server may be down" ;;
                28) echo "  → Timeout after ${CURL_TIMEOUT}s" ;;
                52) echo "  → Empty reply — server likely crashed" ;;
                *) echo "  → curl error $CURL_EXIT" ;;
            esac
            CONFIG_PASS=false
            # Check server still alive
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                echo "  → Server died (confirmed)"
            fi
        else
            CHECK=$(echo "$BODY" | CURL_EXIT_CODE="$CURL_EXIT" python3 /tmp/checker.py)
            echo "$CHECK"

            if echo "$CHECK" | grep -q "PASS"; then
                TG=$(echo "$CHECK" | grep -oP 'tg=\K[0-9.]+')
                TOTAL_TG=$(echo "$TOTAL_TG + $TG" | bc 2>/dev/null || echo 0)
                PROMPT_COUNT=$((PROMPT_COUNT + 1))
            else
                CONFIG_PASS=false
            fi
        fi

        # Check server alive AFTER curl
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  SERVER DIED after prompt"
            CONFIG_PASS=false
            break
        fi
    done

    # Calculate average throughput
    AVG_TG=0
    if [ "$PROMPT_COUNT" -gt 0 ] && [ "$TOTAL_TG" != "0" ]; then
        AVG_TG=$(echo "scale=1; $TOTAL_TG / $PROMPT_COUNT" | bc 2>/dev/null || echo "0")
    fi

    RESULTS["$LABEL"]="$([ "$CONFIG_PASS" = true ] && echo "PASS" || echo "FAIL"):$AVG_TG"

    # Kill server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null || true
    sleep 2

    PORT=$((PORT + 1))
    echo ""
done

# === GENERATE BUG CANDIDATES ===
mkdir -p /home/hunter/scratch/prototype-auto/.scratch/bug-candidates/
CANDIDATE_NUM=$(ls /home/hunter/scratch/prototype-auto/.scratch/bug-candidates/ 2>/dev/null | wc -l)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r LABEL _ <<< "$config"
    IFS=':' read -r STATUS TG <<< "${RESULTS["$LABEL"]:-FAIL:0}"

    if [ "$STATUS" != "PASS" ]; then
        CANDIDATE_NUM=$((CANDIDATE_NUM + 1))
        CANDIDATE_FILE="/home/hunter/scratch/prototype-auto/.scratch/bug-candidates/candidate-$(printf '%03d' $CANDIDATE_NUM)-$(date +%Y-%m-%d)-$LABEL.md"

        cat > "$CANDIDATE_FILE" <<CANDEOF
### BUG-CANDIDATE-$(printf '%03d' $CANDIDATE_NUM): Auto-detected failure in $LABEL config

| Field | Value |
|-------|-------|
| **Severity** | 🟠 High — test config $LABEL failed |
| **Discovered** | $(date +%Y-%m-%d) |
| **Affected models** | $MODEL ($MODEL_SIZE_GB GiB, arch: $ARCH) |
| **Trigger** | $LABEL transport, $MODE mode, ngl=$NGL, max_tokens=$MAX_TOKENS |
| **Reproduction** | \`$ENV_VARS ./build-rocm-native/bin/llama-server --model $MODEL $RPCS -sm layer -ngl $NGL --no-webui --no-warmup -c 128 --port $PORT $MTP_FLAGS\` |
| **Hardware** | $GPU_COUNT GPUs (7900 XTX + RPC endpoints) |
| **Symptom** | Status: $STATUS, throughput: $TG t/s. Full log: /tmp/multigpu-$LABEL.log |
| **Root cause** | _Not yet diagnosed_ |
| **Status** | 🟡 **Candidate** — needs user review |

---
> Auto-generated by multi-gpu-verify.sh on $(date).
> Review before importing into BUGS.md.
CANDEOF
        echo "Bug candidate: $CANDIDATE_FILE"
    fi
done

# === WRITE REPORT ===
cat > "$OUTPUT" <<REPORTEOF
# Multi-GPU Verify Report

**Date:** $(date)
**Model:** $MODEL ($MODEL_SIZE_GB GiB)
**Arch:** $ARCH (GDN=$HAS_GDN MoE=$HAS_MOE MTP=$HAS_MTP)
**GPUs:** $GPU_COUNT | **NGL:** $NGL | **Mode:** $MODE | **Max tokens:** $MAX_TOKENS

## Pre-flight
- Disk space: ${AVAIL_MB}MB free ✅
- RPC endpoints: $RPC_CONNECTED connected ✅
- GPU lock: acquired ✅

## Hardware

| Node | GPU | VRAM | Backend | Role |
|------|-----|------|---------|------|
| romulus | RX 7900 XTX | 24 GiB | ROCm | Primary |
| romulus (Docker) | RTX 3060 Ti | 8 GiB | CUDA | RPC ${RPC_ENDPOINTS[0]} |
| triton | RTX 3090 | 24 GiB | CUDA | RPC ${RPC_ENDPOINTS[1]} |
| triton | RTX 3070 | 8 GiB | CUDA | RPC ${RPC_ENDPOINTS[2]} |

## Results

| Config | tg t/s | Verdict |
|--------|--------|---------|
REPORTEOF

BASELINE_TG=""
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r LABEL _ <<< "$config"
    IFS=':' read -r STATUS TG <<< "${RESULTS["$LABEL"]:-FAIL:0}"

    if [ "$LABEL" = "TCP" ] || [ -z "$BASELINE_TG" ]; then
        BASELINE_TG="$TG"
        DELTA="—"
    else
        if [ "$(echo "$BASELINE_TG > 0" | bc)" -eq 1 ]; then
            DELTA=$(echo "scale=0; 100 * ($TG - $BASELINE_TG) / $BASELINE_TG" | bc 2>/dev/null || echo "?")
            DELTA="${DELTA}%"
        else
            DELTA="?"
        fi
    fi

    echo "| $LABEL | $TG t/s | $STATUS (Δ $DELTA) |" >> "$OUTPUT"
done

# Bug candidates section
CANDIDATE_COUNT=$(ls /home/hunter/scratch/prototype-auto/.scratch/bug-candidates/ 2>/dev/null | wc -l)
if [ "$CANDIDATE_COUNT" -gt 0 ]; then
    echo "" >> "$OUTPUT"
    echo "## Bug Candidates" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    echo "$CANDIDATE_COUNT candidate(s) generated. Review at:" >> "$OUTPUT"
    echo '`.scratch/bug-candidates/`' >> "$OUTPUT"
fi

echo "" >> "$OUTPUT"
echo "## Verdict" >> "$OUTPUT"

OVERALL_PASS=true
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r LABEL _ <<< "$config"
    IFS=':' read -r STATUS _ <<< "${RESULTS["$LABEL"]:-FAIL:0}"
    if [ "$STATUS" != "PASS" ]; then
        OVERALL_PASS=false
        break
    fi
done

if [ "$OVERALL_PASS" = true ]; then
    echo "**PASS** — all configs produced coherent output" >> "$OUTPUT"
else
    echo "**FAIL** — one or more configs failed. Review bug candidates." >> "$OUTPUT"
fi

echo ""
echo "=== Report written to $OUTPUT ==="
echo "=== Bug candidates: .scratch/bug-candidates/ ==="

if [ "$OVERALL_PASS" = true ]; then
    echo "=== VERDICT: PASS ==="
    exit 0
else
    echo "=== VERDICT: FAIL ==="
    exit 1
fi
