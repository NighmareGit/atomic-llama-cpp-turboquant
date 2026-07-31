#!/usr/bin/env bash
# Blast-radius matrix: Plus=1 TSC across architectures (2-GPU fox, q8_0/q8_0).
#
# usage:
#   pathb-plus1-tsc-blast-radius.sh [phase1|phase2|cell...]
#   pathb-plus1-tsc-blast-radius.sh phase1          # plus1 scan all models
#   pathb-plus1-tsc-blast-radius.sh phase2          # kv_unified x plus on affected
#   pathb-plus1-tsc-blast-radius.sh qwen35moe-apex-plus1-kvu-on
#
# docs: docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-ROOTCAUSE.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

if [[ -f .scratch/cluster-access.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .scratch/cluster-access.env
    set +a
fi

BENCH="${SCRIPT_DIR}/pathb-romulus-2gpu-bench.sh"
LOG_DIR="${BENCH_LOG_DIR:-/tmp/plus1-tsc-blast-radius}"
mkdir -p "$LOG_DIR"

BASE_EXTRA="--fit off --verbose -lv 4 --reasoning off"

declare -A MODEL_PATH EXPECT_ARCH EXPECT_GDN EXPECT_MOE EXPECT_MTP
MODEL_PATH[qwen35moe-apex]="/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf"
EXPECT_ARCH[qwen35moe-apex]="qwen35moe"
EXPECT_GDN[qwen35moe-apex]="yes"
EXPECT_MOE[qwen35moe-apex]="yes"
EXPECT_MTP[qwen35moe-apex]="no"

MODEL_PATH[qwen35moe-mtp]="/mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"
EXPECT_ARCH[qwen35moe-mtp]="qwen35moe"
EXPECT_GDN[qwen35moe-mtp]="yes"
EXPECT_MOE[qwen35moe-mtp]="yes"
EXPECT_MTP[qwen35moe-mtp]="weights"

MODEL_PATH[qwen35-dense]="/mnt/models/Qwen3.5-27B-Q5_K_M.gguf"
EXPECT_ARCH[qwen35-dense]="qwen35"
EXPECT_GDN[qwen35-dense]="yes"
EXPECT_MOE[qwen35-dense]="no"
EXPECT_MTP[qwen35-dense]="no"

MODEL_PATH[qwen-coder-next]="/mnt/models/Qwen3-Coder-Next-APEX-I-Quality.gguf"
EXPECT_ARCH[qwen-coder-next]="?"
EXPECT_GDN[qwen-coder-next]="?"
EXPECT_MOE[qwen-coder-next]="?"
EXPECT_MTP[qwen-coder-next]="no"

MODEL_PATH[gemma4-moe]="/mnt/models/gemma-4-26B-A4B-APEX-I-Compact.gguf"
EXPECT_ARCH[gemma4-moe]="gemma4"
EXPECT_GDN[gemma4-moe]="no"
EXPECT_MOE[gemma4-moe]="yes"
EXPECT_MTP[gemma4-moe]="no"

MODEL_PATH[gemma3-dense]="/mnt/models/gemma-3-12b-it-Q5_K_M.gguf"
EXPECT_ARCH[gemma3-dense]="gemma3"
EXPECT_GDN[gemma3-dense]="no"
EXPECT_MOE[gemma3-dense]="no"
EXPECT_MTP[gemma3-dense]="no"

MODEL_PATH[llama3-8b]="/mnt/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
EXPECT_ARCH[llama3-8b]="llama"
EXPECT_GDN[llama3-8b]="no"
EXPECT_MOE[llama3-8b]="no"
EXPECT_MTP[llama3-8b]="no"

MODEL_PATH[qwen2-7b]="/mnt/toshiba_a/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
EXPECT_ARCH[qwen2-7b]="qwen2"
EXPECT_GDN[qwen2-7b]="no"
EXPECT_MOE[qwen2-7b]="no"
EXPECT_MTP[qwen2-7b]="no"

MODEL_PATH[qwen35moe-9b-mtp]="/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf"
EXPECT_ARCH[qwen35moe-9b-mtp]="qwen35moe"
EXPECT_GDN[qwen35moe-9b-mtp]="yes"
EXPECT_MOE[qwen35moe-9b-mtp]="yes"
EXPECT_MTP[qwen35moe-9b-mtp]="weights"

gate_b() {
    python3 - <<'PY' "$1"
import re, sys
text = open(sys.argv[1]).read()
previews = re.findall(r"preview='([^']*)'", text)
if not previews:
    print("unknown")
    raise SystemExit(0)
bad = 0
for p in previews:
    low = p.lower()
    if re.search(r'(here){4,}', low, re.I):
        bad += 1
        continue
    if re.search(r'(\w{2,})\1\1', low):
        bad += 1
        continue
    if re.search(r'(!){6,}', p):
        bad += 1
        continue
    if 'pangram' in low or 'alphabet' in low or 'typing practice' in low:
        continue
    if len(p.strip()) < 40:
        bad += 1
        continue
    if p.count('Here') >= 6:
        bad += 1
        continue
print("coherent" if bad == 0 else "stutter")
PY
}

extract_meta() {
    local log="$1"
    python3 - <<'PY' "$log"
import re, sys
text = open(sys.argv[1]).read()
arch = re.search(r"arch\s*=\s*(\S+)", text)
nlay = re.search(r"n_layer_all\s*=\s*(\d+)", text)
nextn = re.search(r"n_layer_nextn\s*=\s*(\d+)", text)
rs = "yes" if "llama_memory_recurrent" in text else "no"
pipe = "yes" if "pipeline parallelism enabled" in text else "no"
kv = re.search(r"kv_unified\s*=\s*(\w+)", text)
print("|".join([
    arch.group(1) if arch else "?",
    nlay.group(1) if nlay else "?",
    nextn.group(1) if nextn else "0",
    rs,
    pipe,
    kv.group(1) if kv else "?",
]))
PY
}

run_cell() {
    local id="$1"
    local plus="$2"
    local kvu="$3"
    local path="${MODEL_PATH[$id]:-}"
    if [[ -z "$path" ]]; then
        echo "error: unknown model id '$id'" >&2
        return 1
    fi

    local label="trace-g-2gpu-blast-${id}-p${plus}-kv${kvu}"
    local extra="$BASE_EXTRA"
    case "$kvu" in
        on)  extra="$extra --kv-unified" ;;
        off) extra="$extra --no-kv-unified" ;;
        def) ;;
        *) echo "error: kvu must be on|off|def" >&2; return 1 ;;
    esac

    echo "=== cell=${id} plus=${plus} kv=${kvu} ==="
    echo "  model=${path}"

    BENCH_MODEL="$path" \
    BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 \
    BENCH_GEN_TOKENS=64 BENCH_RUNS=1 BENCH_TRACE=0 \
    BENCH_LOAD_TIMEOUT=1200 BENCH_NCMOE= BENCH_EXTRA="$extra" \
    GGML_PIPELINE_PLUS="$plus" \
        "$BENCH" "$label" 2>&1 | tee "${LOG_DIR}/${id}-p${plus}-kv${kvu}.log" || true

    local gate meta
    gate=$(gate_b "${LOG_DIR}/${id}-p${plus}-kv${kvu}.log")
    meta=$(extract_meta "${LOG_DIR}/${id}-p${plus}-kv${kvu}.log" 2>/dev/null || echo "?|?|?|?|?|?")
    IFS='|' read -r arch nlay nextn rs pipe kv <<<"$meta"
    echo "GATE_B=${gate} id=${id} plus=${plus} kv=${kvu} arch=${arch} n_layer_all=${nlay} n_layer_nextn=${nextn} rs_mem=${rs} kv_unified=${kv}"
    echo "${id}|${plus}|${kvu}|${gate}|${arch}|${nlay}|${nextn}|${rs}|${kv}" >> "${LOG_DIR}/summary.tsv"
}

SUMMARY="${LOG_DIR}/summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
    echo "id|plus|kv|gate_b|arch|n_layer_all|n_layer_nextn|rs_mem|kv_unified" > "$SUMMARY"
fi

phase1() {
    for id in qwen35moe-apex qwen35moe-mtp qwen35-dense qwen-coder-next gemma4-moe gemma3-dense llama3-8b qwen2-7b qwen35moe-9b-mtp; do
        run_cell "$id" 1 def
        run_cell "$id" 0 def
    done
}

phase2() {
    # kv_unified sensitivity on known-affected + clean control
    for id in qwen35moe-apex qwen35-dense llama3-8b; do
        for kv in on off; do
            run_cell "$id" 1 "$kv"
        done
    done
}

if [[ $# -eq 0 ]]; then
    set -- phase1
fi

for arg in "$@"; do
    case "$arg" in
        phase1) phase1 ;;
        phase2) phase2 ;;
        *-plus*-kv*)
            # e.g. qwen35moe-apex-plus1-kvu-on
            id="${arg%-plus*}"
            rest="${arg#*-plus}"
            plus="${rest%%-*}"
            plus="${plus#plus}"
            kv="${arg##*-kvu-}"
            run_cell "$id" "$plus" "$kv"
            ;;
        *)
            echo "error: unknown arg '$arg'" >&2
            exit 1
            ;;
    esac
done

echo "=== blast-radius complete; summary: ${SUMMARY} ==="