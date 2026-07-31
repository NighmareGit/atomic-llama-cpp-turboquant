#!/usr/bin/env bash
# T1: A/B bench for GGML_PIPELINE_PLUS (Layer B scheduler pipeline).
#
# Default: llama-pipeline-profiler --mode ab-plus (native client).
# Fallback: BENCH_HTTP=1 -> rpc-server-bench.sh (HTTP gate).
#
# usage:
#   BENCH_MODEL=/path/model.gguf BENCH_RPC_ENDPOINT=... BENCH_TS=50,50 \
#     ./scripts/bench-pipeline-plus-ab.sh my-cell

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RPC_PATCH="${ROOT}/rpc-patch"
BENCH_SH="${RPC_PATCH}/scripts/rpc-server-bench.sh"
resolve_profiler_bin() {
    local cand="${1}"
    if [[ -x "$cand" || -f "$cand" ]]; then
        echo "$cand"
        return
    fi
    for alt in \
        "${cand}.exe" \
        "${ROOT}/build/bin/Release/llama-pipeline-profiler" \
        "${ROOT}/build/bin/Release/llama-pipeline-profiler.exe"; do
        if [[ -x "$alt" || -f "$alt" ]]; then
            echo "$alt"
            return
        fi
    done
    echo "$cand"
}

PROFILER="$(resolve_profiler_bin "${PROFILER_BIN:-${ROOT}/build/bin/llama-pipeline-profiler}")"
DIAGNOSE="${ROOT}/scripts/llama-pipeline-diagnose.sh"
REGRESS="${ROOT}/scripts/llama-pipeline-regression.sh"
REGRESSION_FILE="${REGRESSION_FILE:-${ROOT}/benches/path-b-plus/regression.jsonl}"
LABEL_BASE="${1:-}"
DRY_RUN="${DRY_RUN:-0}"
BENCH_HTTP="${BENCH_HTTP:-0}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
PUBLISH_DIR="${BENCH_PUBLISH_DIR:-${ROOT}/benches/path-b-plus/plus-ab/${STAMP}-${LABEL_BASE:-plus-ab}}"
RUN_DIR="${PUBLISH_DIR}/run"

usage() {
    cat <<'EOF'
bench-pipeline-plus-ab.sh -- T1 Path-B Plus A/B for pipeline Layer B

Default (native):
  llama-pipeline-profiler --mode ab-plus

Fallback (HTTP gate):
  BENCH_HTTP=1 -> rpc-server-bench.sh twice

Env:
  BENCH_MODEL, BENCH_RPC_ENDPOINT, BENCH_TS, BENCH_TRACE=1, BENCH_HTTP=1
  PROFILER_BIN  path to llama-pipeline-profiler
EOF
}

if [[ "${LABEL_BASE}" == "--help" || "${LABEL_BASE}" == "-h" || -z "${LABEL_BASE}" ]]; then
    usage
    [[ -z "${LABEL_BASE}" ]] && exit 1 || exit 0
fi

read_diagnose_field() {
    local f="$1"
    local key="$2"
    python3 - "$f" "$key" <<'PY'
import json, sys
p, k = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p, encoding="utf-8"))
    v = d.get(k)
    print(v if v is not None else "n/a")
except Exception:
    print("n/a")
PY
}

read_result_g() {
    local f="$1"
    python3 - "$f" <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p, encoding="utf-8") as fh:
        line = fh.readline().strip()
    if not line:
        print("n/a")
    else:
        print(json.loads(line).get("G_tps", "n/a"))
except Exception:
    print("n/a")
PY
}

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$BENCH_HTTP" == "1" ]]; then
        echo "DRY_RUN: rpc-server-bench HTTP path"
    else
        echo "DRY_RUN: ${PROFILER} -m \${BENCH_MODEL} --mode ab-plus --out-dir ${RUN_DIR}"
    fi
    exit 0
fi

mkdir -p "$PUBLISH_DIR"
{
    echo "LABEL_BASE=${LABEL_BASE}"
    echo "GIT_SHA=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "BENCH_MODEL=${BENCH_MODEL:-<required>}"
    echo "BENCH_RPC_ENDPOINT=${BENCH_RPC_ENDPOINT:-}"
    echo "BENCH_TS=${BENCH_TS:-}"
    echo "BENCH_TRACE=${BENCH_TRACE:-0}"
    echo "BENCH_HTTP=${BENCH_HTTP}"
    echo "client_kind=$([[ "$BENCH_HTTP" == "1" ]] && echo http || echo native)"
} >"${PUBLISH_DIR}/env.txt"

G_ON="n/a"
G_OFF="n/a"
STALL_ON="n/a"
STALL_OFF="n/a"
OVL_ON="n/a"
OVL_OFF="n/a"
GATE_ON="n/a"
GATE_OFF="n/a"
DRAIN_ON="n/a"
DRAIN_OFF="n/a"

if [[ "$BENCH_HTTP" == "1" ]]; then
    LABEL_ON="${LABEL_BASE}-plus-on"
    LABEL_OFF="${LABEL_BASE}-plus-off"
    GGML_PIPELINE_PLUS=1 BENCH_TRACE="${BENCH_TRACE:-0}" bash "$BENCH_SH" pathb "$LABEL_ON"
    GGML_PIPELINE_PLUS=0 BENCH_TRACE="${BENCH_TRACE:-0}" bash "$BENCH_SH" pathb "$LABEL_OFF"
    LOG_DIR="${BENCH_LOG_DIR:-${RPC_PATCH}/patch/bench-results/rpc-server-bench}"
    extract_g() { rg -o 'G=[0-9.]+' "$1" 2>/dev/null | tail -1 | cut -d= -f2 || echo "n/a"; }
    G_ON="$(extract_g "${LOG_DIR}/${LABEL_ON}.result")"
    G_OFF="$(extract_g "${LOG_DIR}/${LABEL_OFF}.result")"
    if [[ "${BENCH_TRACE:-0}" == "1" && -x "$DIAGNOSE" ]]; then
        bash "$DIAGNOSE" "${LOG_DIR}/${LABEL_ON}/telemetry" --gen-only 2>/dev/null || true
        bash "$DIAGNOSE" "${LOG_DIR}/${LABEL_OFF}/telemetry" --gen-only 2>/dev/null || true
        STALL_ON="$(read_diagnose_field "${LOG_DIR}/${LABEL_ON}/telemetry/diagnose.json" stall_ratio)"
        OVL_ON="$(read_diagnose_field "${LOG_DIR}/${LABEL_ON}/telemetry/diagnose.json" overlap_pct)"
        GATE_ON="$(read_diagnose_field "${LOG_DIR}/${LABEL_ON}/telemetry/diagnose.json" gate_b6)"
        DRAIN_ON="$(read_diagnose_field "${LOG_DIR}/${LABEL_ON}/telemetry/diagnose.json" drain_flush_ms)"
        STALL_OFF="$(read_diagnose_field "${LOG_DIR}/${LABEL_OFF}/telemetry/diagnose.json" stall_ratio)"
        OVL_OFF="$(read_diagnose_field "${LOG_DIR}/${LABEL_OFF}/telemetry/diagnose.json" overlap_pct)"
        GATE_OFF="$(read_diagnose_field "${LOG_DIR}/${LABEL_OFF}/telemetry/diagnose.json" gate_b6)"
        DRAIN_OFF="$(read_diagnose_field "${LOG_DIR}/${LABEL_OFF}/telemetry/diagnose.json" drain_flush_ms)"
    fi
else
    if [[ ! -x "$PROFILER" ]] && [[ ! -f "$PROFILER" ]]; then
        echo "error: llama-pipeline-profiler not found at ${PROFILER}" >&2
        echo "hint: cmake --build build --target llama-pipeline-profiler" >&2
        exit 1
    fi
    if [[ -z "${BENCH_MODEL:-}" ]]; then
        echo "error: BENCH_MODEL required for native profiler path" >&2
        exit 1
    fi
    TRACE_ARGS=()
    if [[ "${BENCH_TRACE:-0}" == "1" ]]; then
        TRACE_ARGS=(--trace)
    fi
    RPC_ARGS=()
    [[ -n "${BENCH_RPC_ENDPOINT:-}" ]] && RPC_ARGS=(-rpc "${BENCH_RPC_ENDPOINT}")
    TS_ARGS=()
    [[ -n "${BENCH_TS:-}" ]] && TS_ARGS=(-ts "${BENCH_TS}")
    if [[ -z "${BENCH_RPC_ENDPOINT:-}" ]]; then
        echo "error: BENCH_RPC_ENDPOINT required (no --topology presets)" >&2
        exit 1
    fi
    cd "$ROOT"
    "$PROFILER" -m "${BENCH_MODEL}" -n "${BENCH_GEN_TOKENS:-128}" \
        --mode ab-plus --out-dir "${RUN_DIR}" "${TRACE_ARGS[@]}" "${RPC_ARGS[@]}" "${TS_ARGS[@]}"
    G_ON="$(read_result_g "${RUN_DIR}/plus-on/result.jsonl")"
    G_OFF="$(read_result_g "${RUN_DIR}/plus-off/result.jsonl")"
    if [[ -f "${RUN_DIR}/plus-on/telemetry/diagnose.json" ]]; then
        STALL_ON="$(read_diagnose_field "${RUN_DIR}/plus-on/telemetry/diagnose.json" stall_ratio)"
        OVL_ON="$(read_diagnose_field "${RUN_DIR}/plus-on/telemetry/diagnose.json" overlap_pct)"
        GATE_ON="$(read_diagnose_field "${RUN_DIR}/plus-on/telemetry/diagnose.json" gate_b6)"
        DRAIN_ON="$(read_diagnose_field "${RUN_DIR}/plus-on/telemetry/diagnose.json" drain_flush_ms)"
    fi
    if [[ -f "${RUN_DIR}/plus-off/telemetry/diagnose.json" ]]; then
        STALL_OFF="$(read_diagnose_field "${RUN_DIR}/plus-off/telemetry/diagnose.json" stall_ratio)"
        OVL_OFF="$(read_diagnose_field "${RUN_DIR}/plus-off/telemetry/diagnose.json" overlap_pct)"
        GATE_OFF="$(read_diagnose_field "${RUN_DIR}/plus-off/telemetry/diagnose.json" gate_b6)"
        DRAIN_OFF="$(read_diagnose_field "${RUN_DIR}/plus-off/telemetry/diagnose.json" drain_flush_ms)"
    fi
    ln -sfn "${RUN_DIR}" "${PUBLISH_DIR}/run" 2>/dev/null || cp -a "${RUN_DIR}" "${PUBLISH_DIR}/run" 2>/dev/null || true
    if [[ -f "${RUN_DIR}/plus-on/telemetry/diagnose.json" ]]; then
        bash "$REGRESS" "${RUN_DIR}/plus-on/telemetry" --label "${LABEL_BASE}-plus-on" \
            --regression-file "$REGRESSION_FILE" --out-dir "${RUN_DIR}/plus-on" \
            --mode ab-plus --plus 1 || true
    fi
    if [[ -f "${RUN_DIR}/plus-off/telemetry/diagnose.json" ]]; then
        bash "$REGRESS" "${RUN_DIR}/plus-off/telemetry" --label "${LABEL_BASE}-plus-off" \
            --regression-file "$REGRESSION_FILE" --out-dir "${RUN_DIR}/plus-off" \
            --mode ab-plus --plus 0 || true
    fi
fi

cat >"${PUBLISH_DIR}/summary.md" <<EOF
# Plus A/B -- ${LABEL_BASE} (${STAMP})

| Cell | Plus | G (t/s) | stall_ratio | overlap_pct | gate_b6 | drain_flush_ms |
|------|------|---------|-------------|-------------|---------|----------------|
| plus-on | 1 | ${G_ON} | ${STALL_ON} | ${OVL_ON} | ${GATE_ON} | ${DRAIN_ON} |
| plus-off | 0 | ${G_OFF} | ${STALL_OFF} | ${OVL_OFF} | ${GATE_OFF} | ${DRAIN_OFF} |

Git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
client_kind: $([[ "$BENCH_HTTP" == "1" ]] && echo http || echo native)

Diagnose: \`scripts/llama-pipeline-diagnose.sh <telemetry> --gen-only\`
EOF

echo "BENCH_PLUS_AB_DONE on=${G_ON} off=${G_OFF}"
echo "summary -> ${PUBLISH_DIR}/summary.md"