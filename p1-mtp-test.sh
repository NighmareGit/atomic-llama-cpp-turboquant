#!/usr/bin/env bash
# MTP matrix test - self-contained, internal timeouts
set -u
cd <user-home>/projects/path-d-gpipeline-assembly-line

export GGML_PIPELINE_PLUS=1
export LD_LIBRARY_PATH=$PWD/build-hip/bin:/opt/rocm/lib

COMBO="${1:-gpipe0-mtp1}"  # gpipe0-mtp1 | gpipe1-mtp0 | gpipe1-mtp1

case "$COMBO" in
  gpipe0-mtp1) PORT=18090; SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0); unset GGML_SCHED_GPIPE; GPPIPE=0; MTP=1 ;;
  gpipe1-mtp0) PORT=18091; SPEC_ARGS=(); export GGML_SCHED_GPIPE=1; GPPIPE=1; MTP=0 ;;
  gpipe1-mtp1) PORT=18092; SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0); export GGML_SCHED_GPIPE=1; GPPIPE=1; MTP=1 ;;
  *) echo "unknown combo $COMBO"; exit 2 ;;
esac

LOG="p1-${COMBO}.log"
pkill -f "llama-server.*${PORT}" 2>/dev/null; sleep 1
rm -f "$LOG"

echo "=== COMBO $COMBO (GPipe=$GPPIPE MTP=$MTP) port $PORT ==="
echo "=== $(date) starting server ==="
./build-hip/bin/llama-server \
  -m <model-mount>/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf --rpc 127.0.0.1:50051 \
  -ngl 99 -sm layer -ts 2,98 -c 4096 -ctk q8_0 -ctv turbo3 -fa on --jinja \
  "${SPEC_ARGS[@]}" \
  --host 127.0.0.1 --port "$PORT" --metrics -sp \
  > "$LOG" 2>&1 &
BGPID=$!
echo "PID: $BGPID"

# wait for model load (max 90s)
LOADED=0
for i in $(seq 1 30); do
  if ! kill -0 $BGPID 2>/dev/null; then echo "SERVER DIED after $((i*3))s"; break; fi
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q Qwen; then
    echo "READY after $((i*3))s"; LOADED=1; break
  fi
  sleep 3
done

if [ "$LOADED" -eq 1 ]; then
  echo "=== $(date) sending completion (n_predict=32) ==="
  curl -s --max-time 45 "http://127.0.0.1:${PORT}/completion" -H "Content-Type: application/json" \
    -d '{"prompt":"Hello","n_predict":32}' > /tmp/p1-${COMBO}-resp.json 2>&1
  echo "curl exit: $?"
  sleep 1
fi

echo "=== $(date) results ==="
echo "--- accept/draft lines ---"
grep -iE "draft-mtp.*#acc|spec.*accept|acc_rate|n_accepted" "$LOG" 2>/dev/null | tail -8
echo "--- errors/aborts ---"
grep -iE "error|abort|OOM|crash|traceback|assert|fatal" "$LOG" 2>/dev/null | grep -v "running without SSL" | tail -8
echo "--- last 5 log lines ---"
tail -5 "$LOG" 2>/dev/null

echo "=== cleanup ==="
kill -TERM $BGPID 2>/dev/null
sleep 2
kill -9 $BGPID 2>/dev/null
echo "=== COMBO $COMBO DONE $(date) ==="
