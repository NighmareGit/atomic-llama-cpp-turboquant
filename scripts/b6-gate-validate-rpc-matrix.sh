#!/usr/bin/env bash
# PR 8: validate-rpc investigation matrix (client x endpoint).
#
# usage: b6-gate-validate-rpc-matrix.sh [--run]
#   without --run: print planned cells only
#
# env:
#   B6_VALIDATE_OUT_DIR  default benches/path-b-plus/validate-rpc-matrix
#   B6_VALIDATE_TIMEOUT  seconds per cell (default 45)
#   B6_ROMULUS_HOST      default 192.168.8.108
#   B6_ROMULUS_PASS      default 12345

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${B6_VALIDATE_OUT_DIR:-${ROOT}/benches/path-b-plus/validate-rpc-matrix}"
TIMEOUT_SEC="${B6_VALIDATE_TIMEOUT:-45}"
ROMULUS_HOST="${B6_ROMULUS_HOST:-192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
RUN=0
[[ "${1:-}" == "--run" ]] && RUN=1

mkdir -p "$OUT_DIR"
TSV="${OUT_DIR}/results.tsv"
SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf 'cell\tclient\tbinary\tendpoint\tts\tstatus\telapsed_s\tnotes\tsha\tstamp\n' >"$TSV"

record() {
    local cell="$1" client="$2" binary="$3" endpoint="$4" ts="$5" status="$6" elapsed="$7" notes="$8"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cell" "$client" "$binary" "$endpoint" "$ts" "$status" "$elapsed" "$notes" "$SHA" "$STAMP" >>"$TSV"
    echo "  ${cell}: ${status} (${elapsed}s) ${notes}"
}

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=accept-new "hunter@${ROMULUS_HOST}" "$@"
    else
        ssh -o StrictHostKeyChecking=accept-new "hunter@${ROMULUS_HOST}" "$@"
    fi
}

run_docker_cuda_validate() {
    local endpoint="$1" ts="$2" log="$3"
    local build_bin="${ROOT}/build-cuda-b-bin/bin"
    timeout "$TIMEOUT_SEC" docker run --rm --network=host --gpus=all \
        -v "${ROOT}:/src" -w /src \
        -e LD_LIBRARY_PATH=/src/build-cuda-b-bin/bin \
        nvidia/cuda:12.8.0-devel-ubuntu22.04 \
        bash -c "apt-get update -qq && apt-get install -y -qq libgomp1 libopenblas0 >/dev/null && \
            /src/build-cuda-b-bin/bin/llama-pipeline-profiler --validate-rpc -rpc '${endpoint}' -ts '${ts}'" \
        >"$log" 2>&1
}

run_cell() {
    local cell="$1" client="$2" binary="$3" endpoint="$4" ts="$5"
    local log="${OUT_DIR}/${cell}.log"
    local t0 t1 elapsed status notes

    echo "=== cell ${cell} (${client} -> ${endpoint}) ==="
    t0=$(date +%s)
    status=FAIL
    notes=""

    case "$cell" in
        A)
            if ssh_romulus "timeout ${TIMEOUT_SEC} ~/atomic-llama-cpp-turboquant/build-rocm-docker/bin/llama-pipeline-profiler --validate-rpc -rpc '${endpoint}' -ts '${ts}'" >"$log" 2>&1; then
                status=PASS
            fi
            ;;
        B|C)
            if run_docker_cuda_validate "$endpoint" "$ts" "$log"; then
                status=PASS
            fi
            ;;
        D)
            if ssh_romulus "timeout ${TIMEOUT_SEC} ~/atomic-llama-cpp-turboquant/build-rocm-docker/bin/llama-pipeline-profiler --validate-rpc -rpc '${endpoint}' -ts '${ts}'" >"$log" 2>&1; then
                status=PASS
            fi
            notes="4gpu endpoints"
            ;;
        *)
            notes="unknown cell"
            ;;
    esac

    t1=$(date +%s)
    elapsed=$((t1 - t0))
    if [[ "$status" == PASS ]] && ! grep -q '"ok": true' "$log" 2>/dev/null; then
        status=FAIL
        notes="${notes} missing ok:true in report"
    fi
    record "$cell" "$client" "$binary" "$endpoint" "$ts" "$status" "$elapsed" "$notes"
}

cat <<EOF
validate-rpc matrix (PR 8)
out=${OUT_DIR}
sha=${SHA}

Cells:
  A  romulus ROCm  -> triton :50054
  B  remus docker CUDA -> triton :50054
  C  remus docker CUDA -> remus :50051
  D  romulus ROCm  -> 4-GPU endpoints
EOF

if [[ "$RUN" -eq 0 ]]; then
    echo "dry-run only; pass --run to execute"
    exit 0
fi

run_cell A romulus rocm-profiler "192.168.8.23:50054" "50,50"
run_cell B remus-docker cuda-docker-profiler "192.168.8.23:50054" "50,50"
run_cell C remus-docker cuda-docker-profiler "192.168.8.176:50051" "50,50"
run_cell D romulus rocm-profiler "192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053" "25,12,25,38"

echo ""
echo "=== matrix written: ${TSV} ==="
cat "$TSV"