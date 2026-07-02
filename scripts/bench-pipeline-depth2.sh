#!/usr/bin/env bash
# T0: A/B bench for LLAMA_PIPELINE_DEPTH2 (Layer A speculative overlap).
# Note: llama_decode_mtp_* and prepare_next are stub (depth-1 sync) when LLAMA_PIPELINE_DEPTH2=1 (Phase R1)
#
# Compares depth-2 ON (unset) vs OFF (=0) against the same running llama-server.
# Does not start a server unless START_SERVER=1 (experimental).
#
# usage:
#   HOST=127.0.0.1 PORT=8080 N_PREDICT=128 RUNS=3 ./scripts/bench-pipeline-depth2.sh
#   ./scripts/bench-pipeline-depth2.sh --help
#
# env:
#   HOST, PORT, N_PREDICT, RUNS, PROMPT
#   START_SERVER=1  -> background scripts/run-gemma4-mtp-server.sh (requires GGUF paths)
#   BENCH_OUT_DIR   -> default benches/path-b-plus/depth-2/<timestamp>-depth2-ab
#   LLAMA_MTP_ACC_TRACE=1|<path>  optional MTP acceptance NDJSON (Layer A)
#   BENCH_SERVER_TRACE=1          GGML_SCHED_TRACE + GGML_RPC_TRACE on server (START_SERVER=1)
#   DRY_RUN=1       -> print plan and exit

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
HP="${HOST}:${PORT}"
N_PREDICT="${N_PREDICT:-128}"
RUNS="${RUNS:-3}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog.}"
START_SERVER="${START_SERVER:-0}"
BENCH_SERVER_TRACE="${BENCH_SERVER_TRACE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${BENCH_OUT_DIR:-${ROOT}/benches/path-b-plus/depth-2/${STAMP}-depth2-ab}"
SERVER_TELEMETRY="${OUT_DIR}/server-telemetry"
MTP_TRACE="${LLAMA_MTP_ACC_TRACE:-}"

usage() {
    cat <<'EOF'
bench-pipeline-depth2.sh -- T0 depth-2 A/B for pipeline Layer A

Requires llama-server with MTP or NextN already listening on HOST:PORT
unless START_SERVER=1 (launches run-gemma4-mtp-server.sh).

Env:
  HOST=127.0.0.1   PORT=8080   N_PREDICT=128   RUNS=3
  DRY_RUN=1        print cells and exit
  LLAMA_MTP_ACC_TRACE=1|<path>  optional acceptance trace
  BENCH_SERVER_TRACE=1          sched/rpc trace under <OUT_DIR>/server-telemetry/

Output:
  <OUT_DIR>/summary.md  <OUT_DIR>/env.txt

See BENCHMARKING.md Matrix A and docs/development/pipeline-depth-2-pure-overlap.md
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

measure_one() {
    local depth2_label="$1"
    local n_predict="$2"
    local run_idx="$3"

    if [[ "$depth2_label" == "off" ]]; then
        export LLAMA_PIPELINE_DEPTH2=0
    else
        unset LLAMA_PIPELINE_DEPTH2 || true
    fi

    PROMPT="${PROMPT}" N_PREDICT="${n_predict}" HP="${HP}" CHAT_MODEL="${CHAT_MODEL:-}" python3 - <<'PY'
import json, os, sys, urllib.request
body = {
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["N_PREDICT"]),
    "temperature": 0,
    "cache_prompt": False,
    "stream": False,
}
m = os.environ.get("CHAT_MODEL", "").strip()
if m:
    body["model"] = m
req = urllib.request.Request(
    f"http://{os.environ['HP']}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
except Exception as e:
    print(f"ERR|0.0|0|{e}")
    sys.exit(0)
t = d.get("timings", {}) or {}
u = d.get("usage", {}) or {}
ct = u.get("completion_tokens", 0) or 0
tps = t.get("predicted_per_second", 0.0)
dn = t.get("draft_n", 0) or 0
da = t.get("draft_n_accepted", 0) or 0
acc = (100.0 * da / dn) if dn else 0.0
print(f"{tps:.2f}|{acc:.1f}|{ct}")
PY
}

mtp_trace_lines() {
    local f="$1"
    if [[ -z "$f" || "$f" == "1" || ! -f "$f" ]]; then
        echo "n/a"
        return
    fi
    wc -l <"$f" 2>/dev/null | tr -d ' ' || echo "n/a"
}

median_tps() {
    python3 - "$@" <<'PY'
import sys, statistics
vals = []
for s in sys.argv[1:]:
    try:
        v = float(s.split("|")[0])
        if v > 0:
            vals.append(v)
    except Exception:
        pass
print(f"{statistics.median(vals):.2f}" if vals else "ERR")
PY
}

wait_for_ready() {
    local max_wait=120
    local elapsed=0
    while (( elapsed < max_wait )); do
        if curl -sf "http://${HP}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "error: server not ready at http://${HP}/health" >&2
    return 1
}

resolve_chat_model() {
    CHAT_MODEL="$(curl -sf "http://${HP}/v1/models" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    data = d.get('data') or []
    print(data[0]['id'] if data else '')
except Exception:
    print('')
" 2>/dev/null || true)"
    export CHAT_MODEL
}

stop_bg_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 1
    fi
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: would bench http://${HP} N_PREDICT=${N_PREDICT} RUNS=${RUNS}"
    echo "  cell depth2-on:  unset LLAMA_PIPELINE_DEPTH2"
    echo "  cell depth2-off: LLAMA_PIPELINE_DEPTH2=0"
    echo "  output: ${OUT_DIR}/summary.md"
    exit 0
fi

mkdir -p "$OUT_DIR"
{
    echo "HOST=${HOST}"
    echo "PORT=${PORT}"
    echo "N_PREDICT=${N_PREDICT}"
    echo "RUNS=${RUNS}"
    echo "LLAMA_MTP_ACC_TRACE=${MTP_TRACE:-<unset>}"
    echo "BENCH_SERVER_TRACE=${BENCH_SERVER_TRACE}"
    echo "GIT_SHA=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${OUT_DIR}/env.txt"

if [[ "$START_SERVER" == "1" ]]; then
    trap stop_bg_server EXIT
    if [[ "$BENCH_SERVER_TRACE" == "1" ]]; then
        mkdir -p "$SERVER_TELEMETRY"
        export GGML_SCHED_TRACE=1
        export GGML_RPC_TRACE=1
        export GGML_PIPELINE_TRACE=1
        export GGML_SCHED_TRACE_FILE="${SERVER_TELEMETRY}/sched-trace.jsonl"
        export GGML_RPC_TRACE_FILE="${SERVER_TELEMETRY}/rpc-trace.jsonl"
        export GGML_PIPELINE_TRACE_FILE="${SERVER_TELEMETRY}/pipeline-trace.jsonl"
    fi
    if [[ -n "${MTP_TRACE:-}" ]]; then
        export LLAMA_MTP_ACC_TRACE="$MTP_TRACE"
    fi
    echo "starting background MTP server via run-gemma4-mtp-server.sh ..."
    bash "${ROOT}/scripts/run-gemma4-mtp-server.sh" >"${OUT_DIR}/server.log" 2>&1 &
    SERVER_PID=$!
    wait_for_ready
fi

if ! wait_for_ready; then
    echo "hint: start llama-server with --spec-type mtp or nextn, or set START_SERVER=1" >&2
    exit 1
fi

resolve_chat_model
echo "bench target: http://${HP} model=${CHAT_MODEL:-<default>}"

declare -a ON_SAMPLES OFF_SAMPLES

for ((r = 1; r <= RUNS; r++)); do
    echo "=== run ${r}/${RUNS} depth2-on ==="
    on="$(measure_one on "$N_PREDICT" "$r")"
    echo "  ${on}"
    ON_SAMPLES+=("$on")
    sleep 1
    echo "=== run ${r}/${RUNS} depth2-off ==="
    off="$(measure_one off "$N_PREDICT" "$r")"
    echo "  ${off}"
    OFF_SAMPLES+=("$off")
    sleep 1
done

ON_MED="$(median_tps "${ON_SAMPLES[@]}")"
OFF_MED="$(median_tps "${OFF_SAMPLES[@]}")"

MTP_LINES="n/a"
if [[ -n "${MTP_TRACE:-}" && "$MTP_TRACE" != "1" && -f "$MTP_TRACE" ]]; then
    MTP_LINES="$(mtp_trace_lines "$MTP_TRACE")"
fi

DELTA="n/a"
if [[ "$ON_MED" != "ERR" && "$OFF_MED" != "ERR" ]]; then
    DELTA="$(ON_MED="$ON_MED" OFF_MED="$OFF_MED" python3 - <<'PY'
import os
try:
    on = float(os.environ["ON_MED"])
    off = float(os.environ["OFF_MED"])
    if off > 0:
        print(f"{(on/off - 1)*100:+.1f}%")
    else:
        print("n/a")
except Exception:
    print("n/a")
PY
)"
fi

cat >"${OUT_DIR}/summary.md" <<EOF
# depth-2 A/B -- ${STAMP}

| Cell | LLAMA_PIPELINE_DEPTH2 | median G (t/s) | runs |
|------|----------------------|----------------|------|
| depth2-on | unset (on) | ${ON_MED} | ${RUNS} |
| depth2-off | 0 | ${OFF_MED} | ${RUNS} |

Delta (on vs off): ${DELTA}

Host: http://${HP}
N_PREDICT: ${N_PREDICT}
LLAMA_MTP_ACC_TRACE: ${MTP_TRACE:-<unset>}
MTP trace lines: ${MTP_LINES}
BENCH_SERVER_TRACE: ${BENCH_SERVER_TRACE}
Server telemetry: $([[ "$BENCH_SERVER_TRACE" == "1" ]] && echo "${SERVER_TELEMETRY}" || echo n/a)
Git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
Layer A: stub implementation (Phase R1) - see PLAN.md 1.4

Raw samples (tps|accept|tokens):

depth2-on:
$(printf '  %s\n' "${ON_SAMPLES[@]}")

depth2-off:
$(printf '  %s\n' "${OFF_SAMPLES[@]}")
EOF

echo "BENCH_DEPTH2_DONE on=${ON_MED} off=${OFF_MED} delta=${DELTA}"
echo "summary -> ${OUT_DIR}/summary.md"