#!/usr/bin/env bash
# Romulus-local 2-GPU: ROCm llama-server (7900 XTX) + docker rpc-server (3060 Ti).
#
# usage:
#   bash scripts/romulus-local-up.sh --build
#   bash scripts/romulus-local-up.sh                    # default: Qwen3.6 35B MTP GGUF
#   bash scripts/romulus-local-up.sh -c 4096 --ctk q8_0 --ctv turbo3
#   bash scripts/romulus-local-up.sh -m /mnt/models/other.gguf --mtp off
#   bash scripts/romulus-local-up.sh --stop
#
# Common flags:
#   -m, --model PATH     GGUF (default: ROMULUS_DEFAULT_MODEL or *MTP* under /mnt/models)
#   -c, --ctx N           context slots (default 8192)
#   --ctk TYPE            cache type K (default q8_0)
#   --ctv TYPE            cache type V (default turbo3; use q8_0 for non-MTP)
#   --mtp on|off|TYPE     speculative: default on for MTP GGUF, off for others
#   --host ADDR           default 0.0.0.0
#   --port N              default 8080
#   --ts A,B              override tensor split (else VRAM preflight)
#   --build               run romulus-local-build.sh first
#   --no-preflight        use default ts=50,50 ngl=99
#   --rpc-only            start docker RPC only
#   --stop                stop llama-server + rpc container

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRESET="b6-2gpu-romulus-local"
BUILD_DIR="${ROMULUS_BUILD_DIR:-build-rocm-docker}"
COMPOSE="${ROMULUS_PATHB_COMPOSE:-${ROOT}/scripts/romulus-pathb/docker-compose.yml}"
SERVER="${LLAMA_SERVER:-${ROOT}/${BUILD_DIR}/bin/llama-server}"

DEFAULT_MTP_MODEL="${ROMULUS_DEFAULT_MODEL:-/mnt/models/Qwen3.6-35B-A3B-UD-Q4_K_XL_MTP.gguf}"

MODEL=""
CTX=8192
CTK="q8_0"
CTV="turbo3"
HOST="0.0.0.0"
PORT=8080
TS=""
NGL=""
FITT=""
SPEC=""   # empty = auto from model name
DO_BUILD=0
DO_STOP=0
RPC_ONLY=0
SKIP_PREFLIGHT=0
TS_MODE="${ROMULUS_TS_MODE:-equal}"
PHASE="${ROMULUS_VRAM_PHASE:-load}"
EXTRA_ARGS=()

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

resolve_model() {
    local explicit="${1:-}"
    if [[ -n "$explicit" ]]; then
        if [[ -f "$explicit" ]]; then
            printf '%s\n' "$explicit"
            return 0
        fi
        local candidate="/mnt/models/${explicit}"
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
        echo "error: model not found: ${explicit} (also tried ${candidate})" >&2
        return 1
    fi

    if [[ -f "$DEFAULT_MTP_MODEL" ]]; then
        printf '%s\n' "$DEFAULT_MTP_MODEL"
        return 0
    fi

    local f glob
    for glob in \
        /mnt/models/Qwen3.6-35B-A3B*UDT*MTP*.gguf \
        /mnt/models/Qwen3.6-35B-A3B*MTP*.gguf \
        /mnt/models/Qwen3.6-35B*MTP*.gguf; do
        for f in $glob; do
            if [[ -f "$f" ]]; then
                echo "warn: default MTP model missing; using ${f}" >&2
                printf '%s\n' "$f"
                return 0
            fi
        done
    done

    echo "error: no Qwen3.6 35B MTP model found under /mnt/models" >&2
    echo "hint: set -m PATH or ROMULUS_DEFAULT_MODEL=${DEFAULT_MTP_MODEL}" >&2
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="${2:?}"; shift 2 ;;
        -c|--ctx) CTX="${2:?}"; shift 2 ;;
        --ctk) CTK="${2:?}"; shift 2 ;;
        --ctv) CTV="${2:?}"; shift 2 ;;
        --mtp)
            case "${2:-}" in
                on|1|yes) SPEC="draft-mtp" ;;
                off|0|no|none) SPEC="none" ;;
                *) SPEC="${2:?}" ;;
            esac
            shift 2
            ;;
        --host) HOST="${2:?}"; shift 2 ;;
        --port) PORT="${2:?}"; shift 2 ;;
        --ts) TS="${2:?}"; shift 2 ;;
        --build) DO_BUILD=1; shift ;;
        --no-preflight) SKIP_PREFLIGHT=1; shift ;;
        --rpc-only) RPC_ONLY=1; shift ;;
        --stop) DO_STOP=1; shift ;;
        -h|--help) usage 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

stop_all() {
    echo "=== stop romulus-local ==="
    pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
    docker rm -f pathb-rpc-romulus 2>/dev/null || true
    echo "STOP_OK"
}

if [[ "$DO_STOP" -eq 1 ]]; then
    stop_all
    exit 0
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
    bash "${ROOT}/scripts/romulus-local-build.sh"
fi

export LD_LIBRARY_PATH="${ROOT}/${BUILD_DIR}/bin:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GGML_PIPELINE_PLUS="${GGML_PIPELINE_PLUS:-1}"
export GGML_PIPELINE_MULTI_BACKEND_SEQ="${GGML_PIPELINE_MULTI_BACKEND_SEQ:-1}"
export GGML_RPC_DUAL_SOCKET="${GGML_RPC_DUAL_SOCKET:-0}"
export GGML_SCHED_WAVEFRONT_DISPATCH="${GGML_SCHED_WAVEFRONT_DISPATCH:-0}"

start_rpc() {
    echo "=== start rpc-server docker ==="
    docker rm -f pathb-rpc-romulus 2>/dev/null || true
    docker compose -f "$COMPOSE" up -d --force-recreate
    for _ in $(seq 1 30); do
        nc -z 127.0.0.1 50051 2>/dev/null && break
        sleep 1
    done
    nc -zv 127.0.0.1 50051
}

start_rpc

if [[ "$RPC_ONLY" -eq 1 ]]; then
    echo "RPC_ONLY_OK http://${HOST}:${PORT} not started"
    exit 0
fi

MODEL="$(resolve_model "$MODEL")"

if [[ -z "$SPEC" ]]; then
    if [[ "$MODEL" == *MTP* || "$MODEL" == *NextN* || "$MODEL" == *nextn* || "$MODEL" == *UDT* ]]; then
        # Qwen 3.x NextN heads: upstream --spec-type draft-mtp (not legacy "nextn").
        SPEC="draft-mtp"
    else
        SPEC="none"
    fi
fi

RPC="127.0.0.1:50051"

if [[ "$SKIP_PREFLIGHT" -eq 0 && -z "$TS" ]]; then
    echo "=== VRAM preflight (${PRESET}) ==="
    OUT="$(python3 "${ROOT}/rpc-patch/scripts/pathb-rpc-vram-preflight.py" \
        --preset "$PRESET" \
        --gguf "$MODEL" \
        --ts-mode "$TS_MODE" \
        --phase "$PHASE" \
        --ctx "$CTX" \
        --strict 2>&1)" || {
        printf '%s\n' "$OUT"
        echo "FAIL: preflight" >&2
        exit 1
    }
    printf '%s\n' "$OUT"
    TS="$(printf '%s\n' "$OUT" | awk -F= '/^  BENCH_TS=/{print $2; exit}')"
    FITT="$(printf '%s\n' "$OUT" | awk -F= '/^  BENCH_FITT=/{print $2; exit}')"
    # Keep ngl=99 for layer split across RPC+ROCm. Preflight may suggest lower ngl to
    # satisfy static VRAM math, but partial CPU offload with --split-mode layer corrupts RPC.
fi

TS="${TS:-50,50}"
NGL="${NGL:-99}"
FITT="${FITT:-1024,1024}"

if [[ ! -x "$SERVER" ]]; then
    echo "error: llama-server missing: ${SERVER}" >&2
    echo "hint: bash scripts/romulus-local-build.sh --client-only" >&2
    exit 1
fi

if [[ "$SPEC" != "none" && "${VERIFY_NEXTN_GGUF:-1}" != "0" ]]; then
    if [[ -f "${ROOT}/scripts/verify-qwen36-nextn-gguf.py" ]]; then
        python3 "${ROOT}/scripts/verify-qwen36-nextn-gguf.py" "$MODEL" || exit 1
    fi
fi

pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
sleep 1

echo "=== plan ==="
echo "  model=${MODEL}"
echo "  ctx=${CTX} ctk=${CTK} ctv=${CTV} mtp=${SPEC}"
echo "  ts=${TS} ngl=${NGL} rpc=${RPC}"
echo "  listen=${HOST}:${PORT}"
echo "  pipeline_plus=${GGML_PIPELINE_PLUS} multi_backend_seq=${GGML_PIPELINE_MULTI_BACKEND_SEQ}"

ARGS=(
    -m "$MODEL"
    -c "$CTX"
    -ngl "$NGL"
    -ctk "$CTK"
    -ctv "$CTV"
    -fa on
    --host "$HOST"
    --port "$PORT"
    --parallel 1
    -np 1
    --cont-batching
    --split-mode layer
    -ts "$TS"
    --rpc "$RPC"
    --fit off
    --fit-target "$FITT"
    --metrics
    --slots
    --log-timestamps
    --log-prefix
    --reasoning off
)

if [[ "$SPEC" != "none" ]]; then
    ARGS+=(
        --spec-type "$SPEC"
        --spec-draft-n-max "${DRAFT_MAX:-16}"
        --spec-draft-n-min "${DRAFT_MIN:-0}"
    )
    # Same combined *_MTP.gguf: omit -md so server reuses target llama_model (LLAMA_CONTEXT_TYPE_MTP).
    # Pass ROMULUS_DRAFT_MODEL only for a separate draft artifact.
    DRAFT_MODEL="${ROMULUS_DRAFT_MODEL:-}"
    if [[ -n "$DRAFT_MODEL" ]]; then
        ARGS+=(-md "$DRAFT_MODEL")
        if [[ -n "${NGLD:-}" ]]; then
            ARGS+=(--spec-draft-ngl "$NGLD")
        fi
    fi
fi

echo "=== llama-server ==="
echo "  health: http://${HOST}:${PORT}/health"
echo "  chat:   http://${HOST}:${PORT}/v1/chat/completions"
exec "$SERVER" "${ARGS[@]}" "${EXTRA_ARGS[@]}"