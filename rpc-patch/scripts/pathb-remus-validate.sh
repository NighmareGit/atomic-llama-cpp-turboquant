#!/usr/bin/env bash
# Re-validate remus RPC with correct IPs, --fit off, --verbose, GPU monitoring.
#
# usage: pathb-remus-validate.sh <scenario> [model-preset]
#
# scenarios:
#   remus-standalone     llama-server on remus 5060 Ti (no RPC)
#   romulus+remus        7900 XTX client + remus RPC only (2 devices)
#   romulus+remus+3060   7900 XTX + remus 5060 + local 3060 (3 devices)
#
# presets: 4b | 9b | 27b  (progressive, default runs all for scenario)

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
BENCH="${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh"
MONITOR="${RPC_PATCH_ROOT}/scripts/pathb-gpu-monitor.sh"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
REMUS_SSH="${PATHB_REMUS_SSH:-hunter@${REMUS_IP}}"
MODELS="${MODELS_ROOT:-/mnt/models}"
LOG_DIR="${RPC_PATCH_ROOT}/patch/bench-results/remus-validate"
SCENARIO="${1:?scenario: remus-standalone|romulus+remus|romulus+remus+3060}"
PRESET_FILTER="${2:-all}"

mkdir -p "$LOG_DIR"
export PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
export REMUS_RPC_IP="$REMUS_IP"

declare -A PRESET_MODEL PRESET_CTX PRESET_CTK PRESET_CTV PRESET_NGL PRESET_TS PRESET_FITT
PRESET_MODEL[4b]="${MODELS}/Qwen3.5-4B-Q4_K_M.gguf"
PRESET_CTX[4b]=4096; PRESET_CTK[4b]=q4_0; PRESET_CTV[4b]=q4_0; PRESET_NGL[4b]=99

PRESET_MODEL[9b]="${MODELS}/Qwen3.5-9B-MTP-Q4_K_M.gguf"
PRESET_CTX[9b]=4096; PRESET_CTK[9b]=q8_0; PRESET_CTV[9b]=turbo3; PRESET_NGL[9b]=99

PRESET_MODEL[27b]="${MODELS}/Qwen3.5-27B-Q5_K_M.gguf"
PRESET_CTX[27b]=8192; PRESET_CTK[27b]=q8_0; PRESET_CTV[27b]=turbo3; PRESET_NGL[27b]=99

case "$SCENARIO" in
    remus-standalone)
        run_standalone() {
            local preset="$1"
            local label="remus-standalone-${preset}"
            local model="${PRESET_MODEL[$preset]}"
            [[ -f "$model" ]] || { echo "SKIP missing $model"; return 0; }
            local meta="${LOG_DIR}/${label}.meta"
            local gpu="${LOG_DIR}/${label}.gpu"
            : >"$meta"
            echo "=== $label standalone llama-server on remus ===" | tee -a "$meta"
            ssh_run() {
                if [[ -n "${PATHB_REMUS_SSH_PASS:-}" ]] && command -v sshpass >/dev/null; then
                    sshpass -p "$PATHB_REMUS_SSH_PASS" ssh -o StrictHostKeyChecking=accept-new "$REMUS_SSH" "$@"
                else
                    ssh -o StrictHostKeyChecking=accept-new "$REMUS_SSH" "$@"
                fi
            }
            ssh_run "docker rm -f remus-standalone-bench 2>/dev/null; docker run -d --name remus-standalone-bench \
                --gpus all --network host \
                -v /mnt/models:/mnt/models:ro \
                --entrypoint /usr/local/bin/llama-server \
                atomic-llama-remus-pathb-rpc:latest \
                -m ${model} -ngl 99 -c ${PRESET_CTX[$preset]} \
                -ctk ${PRESET_CTK[$preset]} -ctv ${PRESET_CTV[$preset]} \
                --host 0.0.0.0 --port 8082 --verbose --no-warmup -np 1" >>"$meta" 2>&1 || true
            sleep 5
            ssh_run "nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader" | tee -a "$meta"
            curl -sf "http://${REMUS_IP}:8082/health" && echo "health OK" | tee -a "$meta" || echo "health FAIL" | tee -a "$meta"
            out=$(curl -sf "http://${REMUS_IP}:8082/v1/chat/completions" -H 'Content-Type: application/json' \
                -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' 2>/dev/null) || true
            if [[ -n "$out" ]]; then
                python3 -c "import json,sys; d=json.load(sys.stdin); t=d.get('timings',{}); print(f\"G={t.get('predicted_per_second',0):.1f}\")" <<<"$out" | tee -a "$meta"
            fi
            ssh_run "docker logs remus-standalone-bench 2>&1 | tail -40" >>"${LOG_DIR}/${label}-server.log" 2>&1 || true
            ssh_run "docker rm -f remus-standalone-bench" 2>/dev/null || true
        }
        if [[ "$PRESET_FILTER" == "all" ]]; then
            for p in 4b 9b; do run_standalone "$p"; done
        else
            run_standalone "$PRESET_FILTER"
        fi
        ;;
    romulus+remus|romulus+remus+3060)
        # Device order: --rpc list order = RPC0, RPC1, ... then ROCm0
        # romulus+remus: RPC0=remus 5060 (25%), ROCm0=7900 (75%)
        # romulus+remus+3060: RPC0=3060 local (10%), RPC1=remus 5060 (25%), ROCm0=7900 (65%)
        if [[ "$SCENARIO" == "romulus+remus" ]]; then
            export BENCH_RPC_MODE=remote
            export BENCH_RPC_HOST="$REMUS_IP"
            export BENCH_RPC_ENDPOINT="${REMUS_IP}:50051"
            PRESET_TS[4b]="20,80"
            PRESET_TS[9b]="20,80"
            PRESET_TS[27b]="25,75"
            PRESET_FITT[4b]="900,900"
            PRESET_FITT[9b]="900,900"
            PRESET_FITT[27b]="900,900"
            rpc_prep() {
                "${RPC_PATCH_ROOT}/scripts/pathb-remus-rpc.sh" start
                docker rm -f pathb-rpc 2>/dev/null || true
            }
        else
            export BENCH_RPC_MODE=multi
            # RPC0=local 3060, RPC1=remus 5060
            export BENCH_RPC_ENDPOINT="127.0.0.1:50051,${REMUS_IP}:50051"
            PRESET_TS[4b]="10,20,70"
            PRESET_TS[9b]="10,20,70"
            PRESET_TS[27b]="10,25,65"
            PRESET_FITT[4b]="900,900,900"
            PRESET_FITT[9b]="900,900,900"
            PRESET_FITT[27b]="900,900,900"
            rpc_prep() {
                "${RPC_PATCH_ROOT}/scripts/pathb-remus-rpc.sh" start
                local cuda_bin="${TQ}/build-cuda-b-bin-sync/bin"
                if [[ ! -x "${cuda_bin}/rpc-server" ]]; then
                    cuda_bin="${TQ}/build-cuda-b-bin-rebuild/bin"
                fi
                if [[ ! -x "${cuda_bin}/rpc-server" ]]; then
                    cuda_bin="${TQ}/build-cuda-b-bin/bin"
                fi
                PATHB_BIN_CUDA="$cuda_bin" "${RPC_PATCH_ROOT}/scripts/pathb-start-rpc.sh"
            }
        fi

        run_bench() {
            local preset="$1"
            local label="${SCENARIO}-${preset}"
            local model="${PRESET_MODEL[$preset]}"
            [[ -f "$model" ]] || { echo "SKIP missing $model"; return 0; }

            rpc_prep

            export BENCH_MODEL="$model"
            export BENCH_CTX="${PRESET_CTX[$preset]}"
            export BENCH_CTK="${PRESET_CTK[$preset]}"
            export BENCH_CTV="${PRESET_CTV[$preset]}"
            export BENCH_NGL="${PRESET_NGL[$preset]}"
            export BENCH_TS="${PRESET_TS[$preset]}"
            export BENCH_LOAD_TIMEOUT=600
            export BENCH_GEN_TOKENS=48
            export BENCH_RUNS=3
            export BENCH_NO_WARMUP=1
            export BENCH_NP=1
            export BENCH_LOG_DIR="$LOG_DIR"
            export BENCH_EXTRA="--fit off --fit-target ${PRESET_FITT[$preset]} --verbose"

            local gpu_log="${LOG_DIR}/${label}.gpu"
            "$MONITOR" "$gpu_log" 180 &
            local mon_pid=$!
            trap "kill $mon_pid 2>/dev/null || true" RETURN

            if "$BENCH" pathb "$label"; then
                echo "PASS $label" | tee -a "${LOG_DIR}/summary.txt"
            else
                echo "FAIL $label" | tee -a "${LOG_DIR}/summary.txt"
            fi
            kill "$mon_pid" 2>/dev/null || true

            # verify RPC in server log
            local slog="${LOG_DIR}/${label}-server.log"
            if [[ -f "$slog" ]]; then
                if grep -qE 'RPC0|Cannot resolve|Failed to connect' "$slog"; then
                    grep -E 'device_info|RPC|Cannot resolve|Failed to connect|load_tensors|split' "$slog" | head -20 \
                        | tee -a "${LOG_DIR}/${label}-rpc-check.txt"
                fi
            fi
            # peak power on remus
            if [[ -f "$gpu_log" ]]; then
                python3 - <<'PY' "$gpu_log" | tee -a "${LOG_DIR}/${label}-rpc-check.txt"
import re, sys
t = open(sys.argv[1]).read()
vals = []
for line in t.split('[remus-nvidia]')[1:]:
    row = line.strip().split('\n', 1)[0]
    parts = [p.strip() for p in row.split(',')]
    if len(parts) >= 3:
        m = re.match(r'([\d.]+)', parts[2])
        if m:
            vals.append(float(m.group(1)))
print(f'remus_peak_power_W={max(vals) if vals else 0}')
print(f'remus_power_samples={len(vals)}')
PY
            fi
            docker rm -f pathb-rpc bench-llama 2>/dev/null || true
        }

        if [[ "$PRESET_FILTER" == "all" ]]; then
            for p in 4b 9b 27b; do
                run_bench "$p"
            done
        else
            run_bench "$PRESET_FILTER"
        fi
        ;;
    *)
        echo "unknown scenario: $SCENARIO" >&2
        exit 1
        ;;
esac