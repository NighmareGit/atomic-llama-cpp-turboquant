#!/usr/bin/env bash
# 72B+ phased RPC matrix: Config C remus-first, ts=35,15,50.
#
# usage: pathb-72b-matrix.sh [--phase 1|2|3|4|all] [preset...]
#   presets: llama70b qwen72b kimi72b qwen-next-80b coder-next coder-next-q4
#
# Load strategy (per review):
#   1. --fit off + manual -ngl (preferred; better split than auto-fit)
#   2. On load failure, retry --fit on -ngl 0 with same -ts
#   3. --verbose -lv 4 + BENCH_EXTRACT_VRAM=1 for per-device MiB logs
#   4. try_load: PASS load-only when server log shows gpu_layers>0 but gen fails
#
# Phase 1-4 results: patch/bench-results/72b-matrix/README.md
#
# env: PATHB_REMUS_SSH_PASS, REMUS_RPC_IP, BENCH_MATRIX_SKIP=1

set -euo pipefail

RPC_PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${RPC_PATCH_ROOT}/.." && pwd)"
TQ="${LLAMA_TURBOQUANT_ROOT:-${REPO_ROOT}}"
BENCH="${RPC_PATCH_ROOT}/scripts/rpc-server-bench.sh"
VRAM_CALC="${RPC_PATCH_ROOT}/scripts/pathb-72b-vram-calc.py"
DISK="${RPC_PATCH_ROOT}/scripts/pathb-disk-preflight.sh"
MONITOR="${RPC_PATCH_ROOT}/scripts/pathb-gpu-monitor.sh"
PROMPTS="${RPC_PATCH_ROOT}/bench-prompts/large-model-eval.json"
MODELS="${MODELS_ROOT:-/mnt/models}"
REMUS_IP="${REMUS_RPC_IP:-192.168.8.176}"
LOG_DIR="${RPC_PATCH_ROOT}/patch/bench-results/72b-matrix"
SUMMARY="${LOG_DIR}/matrix-summary.txt"
LOAD_RANK="${LOG_DIR}/load-ranking.txt"
WINNERS="${LOG_DIR}/phase-winners.txt"
SKIP_ON_FAIL="${BENCH_MATRIX_SKIP:-1}"

RPC_ENDPOINT="${REMUS_IP}:50051,127.0.0.1:50051"
TS="35,15,50"
FIT_TARGETS="620,1024,880"
VERBOSE_EXTRA="--verbose -lv 4 --reasoning off"

declare -A P_MODEL P_MOE P_LABEL

P_MODEL[llama70b]="${MODELS}/meta-llama-3-70b-instruct.Q4_K_M.gguf"
P_MOE[llama70b]=0; P_LABEL[llama70b]="llama70b-q4km"

P_MODEL[qwen72b]="${MODELS}/Qwen3-72B-Instruct.IQ4_XS.gguf"
P_MOE[qwen72b]=0; P_LABEL[qwen72b]="qwen72b-iq4xs"

P_MODEL[kimi72b]="${MODELS}/Kimi-Dev-72B-IQ4_XS.gguf"
P_MOE[kimi72b]=0; P_LABEL[kimi72b]="kimi72b-iq4xs"

P_MODEL[qwen-next-80b]="${MODELS}/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf"
P_MOE[qwen-next-80b]=1; P_LABEL[qwen-next-80b]="qwen-next-80b-q5km"

P_MODEL[coder-next]="${MODELS}/Qwen3-Coder-Next-APEX-I-Quality.gguf"
P_MOE[coder-next]=1; P_LABEL[coder-next]="coder-next-apex"

P_MODEL[coder-next-q4]="${MODELS}/Qwen_Qwen3-Coder-Next-Q4_K_M.gguf"
P_MOE[coder-next-q4]=1; P_LABEL[coder-next-q4]="coder-next-q4km"

ALL_PRESETS=(llama70b qwen72b kimi72b qwen-next-80b coder-next coder-next-q4)

PHASE="all"
TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        *) TARGETS+=("$1"); shift ;;
    esac
done
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("${ALL_PRESETS[@]}")

mkdir -p "$LOG_DIR"
export PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}"
export REMUS_RPC_IP="$REMUS_IP"

log_summary() { echo "$*" | tee -a "$SUMMARY"; }

rpc_prep() {
    "${RPC_PATCH_ROOT}/scripts/pathb-remus-rpc.sh" start
    local cuda_bin="${TQ}/build-cuda-b-bin-sync/bin"
    [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin-rebuild/bin"
    [[ -x "${cuda_bin}/rpc-server" ]] || cuda_bin="${TQ}/build-cuda-b-bin/bin"
    PATHB_BIN_CUDA="$cuda_bin" "${RPC_PATCH_ROOT}/scripts/pathb-start-rpc.sh"
}

ngl_list_for_model() {
    local model="$1"
    python3 - "$VRAM_CALC" "$model" <<'PY'
import subprocess, re, sys
calc, path = sys.argv[1], sys.argv[2]
out = subprocess.check_output([calc, "--gguf", path, "--config", "config-c"], text=True, stderr=subprocess.STDOUT)
ngls = [int(m.group(1)) for m in re.finditer(r'^  ngl=\s*(\d+)', out, re.M)]
for n in sorted(set(ngls), reverse=True):
    print(n)
PY
}

coherent_preview() {
    local result="$1"
    python3 - "$result" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read() if __import__('os').path.isfile(path) else ""
previews = re.findall(r"preview='([^']*)'", text)
if not previews:
    print(0); raise SystemExit
bad = 0
for p in previews:
    if not p.strip():
        bad += 1; continue
    non_ascii = sum(1 for ch in p if ord(ch) > 0x2E80)
    if non_ascii > max(3, len(p) * 0.12):
        bad += 1
print(0 if bad < len(previews) else 1)
PY
}

run_bench() {
    local label="$1"
    local model="$2"
    local ngl="$3"
    local ncmoe="$4"
    local ctx="$5"
    local ctk="$6"
    local ctv="$7"
    local extra="$8"
    local gen="$9"
    local runs="${10}"
    local prompts="${11:-}"

    export BENCH_RPC_MODE=multi
    export BENCH_RPC_ENDPOINT="$RPC_ENDPOINT"
    export BENCH_MODEL="$model"
    export BENCH_CTX="$ctx"
    export BENCH_CTK="$ctk"
    export BENCH_CTV="$ctv"
    export BENCH_NGL="$ngl"
    export BENCH_TS="$TS"
    export BENCH_NCMOE="$ncmoe"
    export BENCH_LOAD_TIMEOUT=1800
    export BENCH_GEN_TOKENS="$gen"
    export BENCH_RUNS="$runs"
    export BENCH_NO_WARMUP=1
    export BENCH_NP=1
    export BENCH_LOG_DIR="$LOG_DIR"
    export BENCH_EXTRACT_VRAM=1
    export BENCH_VERBOSE_LV=4
    export BENCH_CURL_TIMEOUT=600
    export BENCH_EXTRA="$extra"
    [[ -n "$prompts" ]] && export BENCH_PROMPTS_FILE="$prompts" || unset BENCH_PROMPTS_FILE

    local gpu_log="${LOG_DIR}/${label}.gpu"
    "$MONITOR" "$gpu_log" 600 &
    local mon_pid=$!
    local rc=0
    "$BENCH" pathb "$label" || rc=$?
    kill "$mon_pid" 2>/dev/null || true
    docker rm -f pathb-rpc bench-llama 2>/dev/null || true
    return "$rc"
}

try_load() {
    local preset="$1"
    local model="$2"
    local ngl="$3"
    local ncmoe="$4"
    local mode="$5"
    local label="${P_LABEL[$preset]}-p1-${mode}-ngl${ngl}"
    [[ -n "$ncmoe" ]] && label="${label}-ncmoe${ncmoe}"

    local extra="$VERBOSE_EXTRA"
    if [[ "$mode" == "fitoff" ]]; then
        extra="--fit off ${extra}"
    else
        extra="--fit on --fit-target ${FIT_TARGETS} ${extra}"
        ngl=0
    fi

    log_summary "--- phase1 ${label} ngl=${ngl} ncmoe=${ncmoe:-none} mode=${mode} ---"
    rpc_prep
    local bench_rc=0
    run_bench "$label" "$model" "$ngl" "$ncmoe" 8192 q4_0 q4_0 "$extra" 16 1 "" || bench_rc=$?

    local result="${LOG_DIR}/${label}.result"
    local slog="${LOG_DIR}/${label}-server.log"
    local gpu_layers="0"
    if [[ -f "$slog" ]]; then
        gpu_layers=$(rg -o 'offloaded [0-9]+' "$slog" 2>/dev/null | tail -1 | sed 's/offloaded //' || echo 0)
    fi

    local gs="0"
    if [[ -f "$result" ]]; then
        gs=$(python3 -c "
import re,sys
gs=[]
for l in open(sys.argv[1]):
 m=re.search(r'G=([\d.]+)',l)
 if m: gs.append(float(m.group(1)))
print(f'{sum(gs)/len(gs):.1f}' if gs else '0')
" "$result")
    fi

    if [[ "$bench_rc" -eq 0 && -f "$result" ]]; then
        if coherent_preview "$result" | grep -q 1; then
            log_summary "FAIL load ${preset} mode=${mode} (garbled output) avg_G=${gs}"
            return 1
        fi
        log_summary "PASS load ${preset} mode=${mode} ngl=${ngl} ncmoe=${ncmoe:-none} gpu_layers=${gpu_layers} avg_G=${gs}"
        echo "${preset} ${mode} ngl=${ngl} ncmoe=${ncmoe:-none} avg_G=${gs}" >>"$LOAD_RANK"
        echo "${preset} ngl=${ngl} ncmoe=${ncmoe:-none} mode=${mode}" >>"$WINNERS"
        return 0
    fi

    if [[ -f "$slog" ]] && [[ "${gpu_layers:-0}" -gt 0 ]]; then
        log_summary "PASS load-only ${preset} mode=${mode} ngl=${ngl} gpu_layers=${gpu_layers} (gen failed/slow; see server log)"
        echo "${preset} ${mode} ngl=${ngl} ncmoe=${ncmoe:-none} avg_G=0 load-only" >>"$LOAD_RANK"
        echo "${preset} ngl=${ngl} ncmoe=${ncmoe:-none} mode=${mode}" >>"$WINNERS"
        return 0
    fi

    log_summary "FAIL load ${preset} mode=${mode} ngl=${ngl} ncmoe=${ncmoe:-none} bench_rc=${bench_rc}"
    return 1
}

phase1() {
    log_summary "=== phase1 load feasibility $(date -u +%Y-%m-%dT%H:%M:%SZ) ts=${TS} endpoint=${RPC_ENDPOINT} ==="
    PATHB_CONFIG=config-c "$DISK" --cleanup
    PATHB_REMUS_SSH_PASS="${PATHB_REMUS_SSH_PASS:-}" PATHB_CONFIG=config-c "$DISK" --remote --min-gb 20

    for preset in "${TARGETS[@]}"; do
        model="${P_MODEL[$preset]:-}"
        [[ -z "$model" ]] && { log_summary "SKIP unknown $preset"; continue; }
        [[ ! -f "$model" ]] && { log_summary "SKIP missing $preset: $model"; continue; }

        python3 "$VRAM_CALC" --gguf "$model" --config config-c --ctx 8192 | tee -a "${LOG_DIR}/${preset}-vram-calc.txt"

        mapfile -t ngls < <(ngl_list_for_model "$model")
        [[ ${#ngls[@]} -eq 0 ]] && ngls=(60 56 52 48)

        loaded=0
        if [[ "${P_MOE[$preset]}" == "1" ]]; then
            ncmoe_vals=(8 12 16 20)
        else
            ncmoe_vals=("")
        fi

        for ngl in "${ngls[@]}"; do
            for ncmoe in "${ncmoe_vals[@]}"; do
                if try_load "$preset" "$model" "$ngl" "$ncmoe" "fitoff"; then
                    loaded=1
                    break 2
                fi
            done
        done

        if [[ "$loaded" -eq 0 ]]; then
            log_summary "RETRY ${preset} with --fit on (same ts=${TS})"
            for ncmoe in "${ncmoe_vals[@]}"; do
                if try_load "$preset" "$model" 0 "$ncmoe" "fiton"; then
                    loaded=1
                    break
                fi
            done
        fi

        [[ "$loaded" -eq 0 && "$SKIP_ON_FAIL" != "1" ]] && exit 1
        sleep 3
    done
}

read_winner() {
    local preset="$1"
    grep "^${preset} " "$WINNERS" 2>/dev/null | tail -1 || true
}

phase2() {
    log_summary "=== phase2 kv-unified x flash-attn $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    [[ ! -f "$WINNERS" ]] && { echo "no phase1 winners; run --phase 1 first" >&2; exit 1; }

    for preset in "${TARGETS[@]}"; do
        win=$(read_winner "$preset")
        [[ -z "$win" ]] && continue
        model="${P_MODEL[$preset]}"
        ngl=$(echo "$win" | sed -n 's/.*ngl=\([^ ]*\).*/\1/p')
        ncmoe=$(echo "$win" | sed -n 's/.*ncmoe=\([^ ]*\).*/\1/p')
        mode=$(echo "$win" | sed -n 's/.*mode=\([^ ]*\).*/\1/p')
        fit_extra="--fit off"
        [[ "$mode" == "fiton" ]] && fit_extra="--fit on --fit-target ${FIT_TARGETS}"
        [[ "$ncmoe" == "none" ]] && ncmoe=""

        for kvu in off on; do
            for fa in on off; do
                kflag="kvuoff"
                [[ "$kvu" == "on" ]] && kflag="kvuon"
                label="${P_LABEL[$preset]}-p2-${kflag}-fa${fa}-ngl${ngl}"
                extra="${fit_extra} -fa ${fa}"
                [[ "$kvu" == "on" ]] && extra="${extra} --kv-unified"
                extra="${extra} ${VERBOSE_EXTRA}"

                log_summary "--- phase2 ${label} ---"
                rpc_prep
                if run_bench "$label" "$model" "$ngl" "$ncmoe" 8192 q4_0 q4_0 "$extra" 32 3 ""; then
                    python3 - <<PY "$LOG_DIR/${label}.result" "$preset" "$kvu" "$fa" | tee -a "$SUMMARY"
import re, sys
path, preset, kvu, fa = sys.argv[1:5]
gs = [float(m.group(1)) for l in open(path) for m in [re.search(r'G=([\d.]+)', l)] if m]
print(f"PASS p2 {preset} kvu={kvu} fa={fa} avg_G={sum(gs)/len(gs):.1f}" if gs else f"PASS p2 {preset} kvu={kvu} fa={fa}")
PY
                else
                    log_summary "FAIL p2 ${preset} kvu=${kvu} fa=${fa}"
                    [[ "$SKIP_ON_FAIL" != "1" ]] || true
                fi
                sleep 2
            done
        done
    done
}

phase3() {
    log_summary "=== phase3 KV/context $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    [[ ! -f "$WINNERS" ]] && { echo "no phase1 winners" >&2; exit 1; }

    declare -a KV_CFGS=(
        "8192:q4_0:q4_0"
        "8192:q8_0:turbo3"
        "32768:q8_0:turbo3"
    )

    for preset in "${TARGETS[@]}"; do
        win=$(read_winner "$preset")
        [[ -z "$win" ]] && continue
        model="${P_MODEL[$preset]}"
        ngl=$(echo "$win" | sed -n 's/.*ngl=\([^ ]*\).*/\1/p')
        ncmoe=$(echo "$win" | sed -n 's/.*ncmoe=\([^ ]*\).*/\1/p')
        mode=$(echo "$win" | sed -n 's/.*mode=\([^ ]*\).*/\1/p')
        fit_extra="--fit off"
        [[ "$mode" == "fiton" ]] && fit_extra="--fit on --fit-target ${FIT_TARGETS}"
        [[ "$ncmoe" == "none" ]] && ncmoe=""

        for cfg in "${KV_CFGS[@]}"; do
            IFS=: read -r ctx ctk ctv <<<"$cfg"
            if [[ "$ctv" == "turbo3" && "$model" == *IQ4_XS* ]]; then
                log_summary "SKIP p3 ${preset} ctx=${ctx} turbo3 (IQ4_XS quality risk)"
                continue
            fi
            label="${P_LABEL[$preset]}-p3-ctx${ctx}-${ctk}-${ctv}-ngl${ngl}"
            extra="${fit_extra} -fa on ${VERBOSE_EXTRA}"
            log_summary "--- phase3 ${label} ---"
            rpc_prep
            if run_bench "$label" "$model" "$ngl" "$ncmoe" "$ctx" "$ctk" "$ctv" "$extra" 32 2 ""; then
                log_summary "PASS p3 ${preset} ctx=${ctx} ctk=${ctk} ctv=${ctv}"
            else
                log_summary "FAIL p3 ${preset} ctx=${ctx} ctk=${ctk} ctv=${ctv}"
            fi
            sleep 2
        done
    done
}

phase4() {
    log_summary "=== phase4 eval prompts $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    [[ ! -f "$WINNERS" ]] && { echo "no phase1 winners" >&2; exit 1; }
    [[ ! -f "$PROMPTS" ]] && { echo "missing $PROMPTS" >&2; exit 1; }

    for preset in "${TARGETS[@]}"; do
        win=$(read_winner "$preset")
        [[ -z "$win" ]] && continue
        model="${P_MODEL[$preset]}"
        ngl=$(echo "$win" | sed -n 's/.*ngl=\([^ ]*\).*/\1/p')
        ncmoe=$(echo "$win" | sed -n 's/.*ncmoe=\([^ ]*\).*/\1/p')
        mode=$(echo "$win" | sed -n 's/.*mode=\([^ ]*\).*/\1/p')
        fit_extra="--fit off -fa on"
        [[ "$mode" == "fiton" ]] && fit_extra="--fit on --fit-target ${FIT_TARGETS} -fa on"
        [[ "$ncmoe" == "none" ]] && ncmoe=""

        label="${P_LABEL[$preset]}-p4-eval-ngl${ngl}"
        extra="${fit_extra} ${VERBOSE_EXTRA}"
        log_summary "--- phase4 ${label} ---"
        rpc_prep
        if run_bench "$label" "$model" "$ngl" "$ncmoe" 8192 q4_0 q4_0 "$extra" 256 1 "$PROMPTS"; then
            log_summary "PASS p4 ${preset} eval prompts"
        else
            log_summary "FAIL p4 ${preset} eval prompts"
        fi
        sleep 2
    done
}

case "$PHASE" in
    1) : >"$SUMMARY"; phase1 ;;
    2) phase2 ;;
    3) phase3 ;;
    4) phase4 ;;
    all) : >"$SUMMARY"; phase1; phase2; phase3; phase4 ;;
    *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
esac

df -h / | tee -a "$SUMMARY"
log_summary "=== done 72b-matrix phase=${PHASE} ==="