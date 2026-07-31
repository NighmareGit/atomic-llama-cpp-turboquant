#!/usr/bin/env bash
# SYNC vs Path-B+ production comparison matrix (profiler decode cells).
#
# Tier 2 (default): separate branch SHAs + two workloads per cell:
#   n384        — gate-depth single-turn (profiler-reasoning-long.txt)
#   n2048-mt    — long multi-turn (profiler-hard-multiturn.txt, 2048 gen)
#
# usage:
#   DRY_RUN=1 bash scripts/b6-gate-sync-vs-plus-comparison.sh
#   B6_COMPARE_MODELS=M1,M6 bash scripts/b6-gate-sync-vs-plus-comparison.sh
#
# env:
#   B6_ROMULUS_HOST / B6_ROMULUS_PASS / B6_ROMULUS_REPO
#   B6_COMPARE_TOPOLOGIES   default b6-3gpu-g-triton,b6-5gpu-g-prod
#   B6_COMPARE_MODELS       default M1,M3,M5,M6,M7,M8
#   B6_COMPARE_ARMS         default sync,plus
#   B6_COMPARE_WORKLOADS    default n384,n2048-mt
#   B6_COMPARE_TIER         2 (branch SHA) or 1 (Plus on/off same SHA)
#   B6_SYNC_GIT_REF         default feature/turboquant-kv-cache
#   B6_SYNC_SHA / B6_PLUS_SHA
#   DRY_RUN=1

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ROMULUS_HOST="${B6_ROMULUS_HOST:-user@192.168.8.108}"
ROMULUS_PASS="${B6_ROMULUS_PASS:-12345}"
ROMULUS_REPO="${B6_ROMULUS_REPO:-/home/hunter/atomic-llama-cpp-turboquant}"
TOPOLOGIES="${B6_COMPARE_TOPOLOGIES:-b6-3gpu-g-triton,b6-5gpu-g-prod}"
MODELS="${B6_COMPARE_MODELS:-M1,M3,M5,M6,M7,M8}"
ARMS="${B6_COMPARE_ARMS:-sync,plus}"
WORKLOADS="${B6_COMPARE_WORKLOADS:-n384,n2048-mt}"
TIER="${B6_COMPARE_TIER:-2}"
SYNC_REF="${B6_SYNC_GIT_REF:-feature/turboquant-kv-cache}"
SYNC_SHA="${B6_SYNC_SHA:-}"
PLUS_SHA="${B6_PLUS_SHA:-}"
DRY_RUN="${DRY_RUN:-0}"
OUT_BASE="${ROMULUS_REPO}/benches/path-b-plus/b6-sync-vs-plus-${STAMP}"
JSONL="${ROMULUS_REPO}/benches/path-b-plus/sync-vs-plus-comparison.jsonl"
PROMPT_LONG="${ROMULUS_REPO}/benches/path-b-plus/prompts/profiler-reasoning-long.txt"
PROMPT_MT="${ROMULUS_REPO}/benches/path-b-plus/prompts/profiler-hard-multiturn.txt"
PLUS_BRANCH="Path-B-Event-Support-Pipeline-Plus"

declare -A MODEL_PATH
MODEL_PATH[M1]="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
MODEL_PATH[M3]="/mnt/models/gemma-4-26B-A4B-APEX-I-Compact.gguf"
MODEL_PATH[M5]="/mnt/models/Qwen3.5-27B-Q5_K_M.gguf"
MODEL_PATH[M6]="/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf"
MODEL_PATH[M7]="/mnt/models/Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf"
MODEL_PATH[M8]="/mnt/models/Kimi-Dev-72B-IQ4_XS.gguf"

declare -A MODEL_TOPO_OK
MODEL_TOPO_OK[M1]="b6-3gpu-g-triton,b6-5gpu-g-prod"
MODEL_TOPO_OK[M3]="b6-3gpu-g-triton,b6-5gpu-g-prod"
MODEL_TOPO_OK[M5]="b6-3gpu-g-triton,b6-5gpu-g-prod"
MODEL_TOPO_OK[M6]="b6-5gpu-g-prod"
MODEL_TOPO_OK[M7]="b6-5gpu-g-prod"
MODEL_TOPO_OK[M8]="b6-5gpu-g-prod"

# workload_id -> n_gen|prompt_path|trace_sample|preflight_phase
declare -A WL_NGEN WL_PROMPT WL_TRACE WL_PREF_PHASE
WL_NGEN[n384]=384
WL_PROMPT[n384]="$PROMPT_LONG"
WL_TRACE[n384]=5
WL_PREF_PHASE[n384]=load

WL_NGEN[n2048-mt]=2048
WL_PROMPT[n2048-mt]="$PROMPT_MT"
WL_TRACE[n2048-mt]=25
WL_PREF_PHASE[n2048-mt]=decode

CURRENT_ARM=""
CURRENT_WL=""

ssh_romulus() {
    if command -v sshpass &>/dev/null && [[ -n "$ROMULUS_PASS" ]]; then
        sshpass -p "$ROMULUS_PASS" ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROMULUS_HOST" "$@"
    fi
}

model_ok_for_topo() {
    local mid="$1"
    local topo="$2"
    [[ "${MODEL_TOPO_OK[$mid]:-}" == *"$topo"* ]]
}

workload_ok() {
    local wl="$1"
    [[ -n "${WL_NGEN[$wl]:-}" ]]
}

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

profiler_bin_for_arm() {
    local arm="$1"
    case "$arm" in
        sync)
            if [[ "$TIER" == "1" ]]; then
                echo "${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler"
            else
                echo "${ROMULUS_REPO}/build-rocm-docker-sync/bin/llama-pipeline-profiler"
            fi
            ;;
        plus)
            echo "${ROMULUS_REPO}/build-rocm-docker/bin/llama-pipeline-profiler"
            ;;
    esac
}

prepare_arm_checkout() {
    local arm="$1"
    local wl="$2"
    if [[ "$arm" == "$CURRENT_ARM" && "$wl" == "$CURRENT_WL" ]]; then
        return 0
    fi
    CURRENT_ARM="$arm"
    CURRENT_WL="$wl"
    echo "=== checkout arm=${arm} tier=${TIER} (workload batch ${wl}) ==="
    case "$arm" in
        sync)
            if [[ "$TIER" == "1" ]]; then
                ssh_romulus "cd ${ROMULUS_REPO} && git checkout ${PLUS_BRANCH}"
            elif [[ -n "$SYNC_SHA" ]]; then
                ssh_romulus "cd ${ROMULUS_REPO} && git fetch origin && git checkout '${SYNC_SHA}'"
            else
                ssh_romulus "cd ${ROMULUS_REPO} && git fetch origin && git checkout '${SYNC_REF}'"
            fi
            ;;
        plus)
            if [[ -n "$PLUS_SHA" ]]; then
                ssh_romulus "cd ${ROMULUS_REPO} && git fetch origin && git checkout '${PLUS_SHA}'"
            else
                ssh_romulus "cd ${ROMULUS_REPO} && git checkout ${PLUS_BRANCH}"
            fi
            ;;
    esac
    local bin
    bin="$(profiler_bin_for_arm "$arm")"
    if ! ssh_romulus "test -x '${bin}'"; then
        echo "WARN: missing profiler ${bin} — build on romulus before continuing" >&2
    fi
}

echo "=== SYNC vs PLUS comparison plan ==="
echo "stamp=${STAMP} tier=${TIER}"
echo "workloads=${WORKLOADS}"
echo "topologies=${TOPOLOGIES}"
echo "models=${MODELS} arms=${ARMS}"
echo "sync_ref=${SYNC_REF} sync_sha=${SYNC_SHA:-HEAD} plus_sha=${PLUS_SHA:-HEAD}"
echo "out_base=${OUT_BASE}"
echo "jsonl=${JSONL}"
echo

plan=()
IFS=',' read -ra WL_ARR <<<"$WORKLOADS"
IFS=',' read -ra TOPO_ARR <<<"$TOPOLOGIES"
IFS=',' read -ra MODEL_ARR <<<"$MODELS"
IFS=',' read -ra ARM_ARR <<<"$ARMS"

# Order: workload -> arm -> topo -> model (minimize git checkouts on tier 2)
for wl in "${WL_ARR[@]}"; do
    workload_ok "$wl" || { echo "skip unknown workload ${wl}"; continue; }
    for arm in "${ARM_ARR[@]}"; do
        for topo in "${TOPO_ARR[@]}"; do
            for mid in "${MODEL_ARR[@]}"; do
                [[ -n "${MODEL_PATH[$mid]:-}" ]] || continue
                model_ok_for_topo "$mid" "$topo" || continue
                label="cmp-${wl}-${topo}-${mid}-${arm}-${STAMP}"
                plan+=("${wl}|${topo}|${mid}|${arm}|${label}")
            done
        done
    done
done

printf '%-4s %-10s %-22s %-4s %-6s %s\n' "idx" "workload" "topology" "model" "arm" "label"
for i in "${!plan[@]}"; do
    IFS='|' read -r wl topo mid arm label <<<"${plan[$i]}"
    printf '%-4s %-10s %-22s %-4s %-6s %s\n' "$((i+1))" "$wl" "$topo" "$mid" "$arm" "$label"
done
echo "total cells: ${#plan[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1 — no remote execution"
    exit 0
fi

gate_label_for() {
    local topo="$1" arm="$2"
    if [[ "$arm" == "sync" ]]; then
        case "$topo" in
            b6-3gpu-g-triton) echo "b6-3gpu-g-triton-plus0" ;;
            b6-5gpu-g-prod) echo "b6-5gpu-g-prod-plus0" ;;
            *) echo "$topo" ;;
        esac
    else
        echo "$topo"
    fi
}

run_cell() {
    local wl="$1" topo="$2" mid="$3" arm="$4" label="$5"
    local path="${MODEL_PATH[$mid]}"
    local out="${OUT_BASE}/${label}"
    local n_gen="${WL_NGEN[$wl]}"
    local prompt_file="${WL_PROMPT[$wl]}"
    local trace_sample="${WL_TRACE[$wl]}"
    local pref_phase="${WL_PREF_PHASE[$wl]}"
    local gate_label profiler_bin env_extra preflight prod_source dual_socket

    gate_label="$(gate_label_for "$topo" "$arm")"

    prepare_arm_checkout "$arm" "$wl"

    if ! ssh_romulus "test -f '${path}'"; then
        echo "SKIP ${label} missing ${path}"
        return 0
    fi

    profiler_bin="$(profiler_bin_for_arm "$arm")"

    case "$arm" in
        sync)
            env_extra=""
            preflight=""
            prod_source=""
            ;;
        plus)
            env_extra=""
            prod_source=""
            if [[ "$topo" == "b6-5gpu-g-prod" ]]; then
                preflight="python3 ${ROMULUS_REPO}/rpc-patch/scripts/pathb-rpc-vram-preflight.py \\
                  --preset b6-5gpu-g-prod --gguf '${path}' --ts-mode equal --phase ${pref_phase} >/dev/null 2>&1 || true; "
            else
                preflight=""
            fi
            ;;
        *)
            echo "unknown arm ${arm}" >&2
            return 1
            ;;
    esac

    dual_socket=1
    case "${topo}" in
        b6-3gpu-g-triton|b6-5gpu-g-prod) dual_socket=1 ;;
    esac

    echo "=== RUN ${label} gate=${gate_label} (n=${n_gen} wl=${wl}) ==="

    set +e
    ssh_romulus "${preflight}cd ${ROMULUS_REPO} && \\
      ${prod_source}\\
      ${env_extra} \\
      BENCH_MODEL='${path}' BENCH_GEN_TOKENS=${n_gen} \\
      BENCH_PROMPT_FILE='${prompt_file}' \\
      BENCH_CTK=\${BENCH_CTK:-q8_0} BENCH_CTV=\${BENCH_CTV:-turbo3} \\
      GGML_RPC_DUAL_SOCKET=\${GGML_RPC_DUAL_SOCKET:-${dual_socket}} \\
      B6_GATE_PRESET='${topo}' PATHB_VRAM_PREFLIGHT=1 PROFILER_OUT_DIR='${out}' PROFILER_LOCAL=1 PROFILER_SKIP_VALIDATE=1 \\
      PROFILER_BIN='${profiler_bin}' \\
      LD_LIBRARY_PATH=\$(dirname '${profiler_bin}'):/opt/rocm/lib \\
      bash scripts/b6-gate-run-remote.sh ${gate_label} --no-warmup --skip-rpc-validate \\
        -ctk q8_0 -ctv turbo3 --trace-sample ${trace_sample}" 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -ne 0 ]] || ! ssh_romulus "test -f '${out}/telemetry/diagnose.json'"; then
        echo "FAIL ${label} rc=${rc}"
        ssh_romulus "python3 -c \"
import json, datetime
from pathlib import Path
row = {
  'ts_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
  'campaign': '${STAMP}',
  'workload': '${wl}',
  'n_gen_tokens': ${n_gen},
  'topology': '${topo}',
  'model_id': '${mid}',
  'arm': '${arm}',
  'tier': ${TIER},
  'label': '${label}',
  'status': 'fail',
  'G_tps': None,
}
Path('${JSONL}').parent.mkdir(parents=True, exist_ok=True)
with open('${JSONL}', 'a') as f:
    f.write(json.dumps(row) + chr(10))
\""
        return 0
    fi

    ssh_romulus "python3 -c \"
import json, datetime, subprocess
from pathlib import Path
out = Path('${out}')
diag = json.loads((out/'telemetry/diagnose.json').read_text())
sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd='${ROMULUS_REPO}').decode().strip()
env = {}
for line in (out/'env.txt').read_text().splitlines():
    if '=' in line:
        k,v=line.split('=',1); env[k]=v
row = {
  'ts_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
  'campaign': '${STAMP}',
  'workload': '${wl}',
  'n_gen_tokens': ${n_gen},
  'prompt_file': '${prompt_file}',
  'topology': '${topo}',
  'model_id': '${mid}',
  'arm': '${arm}',
  'tier': ${TIER},
  'label': '${label}',
  'git_sha': sha,
  'status': 'ok',
  'G_tps': diag.get('G_tps') or diag.get('gen_tps'),
  'overlap_pct': diag.get('overlap_pct'),
  'stall_ratio': diag.get('stall_ratio'),
  'blocking_ms': diag.get('blocking_ms'),
  'GGML_PIPELINE_PLUS': env.get('GGML_PIPELINE_PLUS'),
  'GGML_RPC_DUAL_SOCKET': env.get('GGML_RPC_DUAL_SOCKET'),
  'model_path': '${path}',
  'out_dir': str(out),
}
Path('${JSONL}').parent.mkdir(parents=True, exist_ok=True)
with open('${JSONL}', 'a') as f:
    f.write(json.dumps(row) + chr(10))
print('OK', row.get('G_tps'))
\""
}

for entry in "${plan[@]}"; do
    IFS='|' read -r wl topo mid arm label <<<"$entry"
    run_cell "$wl" "$topo" "$mid" "$arm" "$label"
done

echo "=== done; results -> ${JSONL} ==="
echo "Refresh chart: python3 scripts/b6-gate-sync-vs-plus-comparison.py"