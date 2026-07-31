#!/usr/bin/env bash
# Compare bisect diagnose.json against client-matched canonical baseline.
#
# usage: b6-gate-bisect-compare.sh [--allow-cross-client] <bench-label> [more...]
#
# Prints TSV: label client_kind overlap stall delta_overlap delta_stall bisect_verdict

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ALLOW_CROSS=0
LABELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-cross-client) ALLOW_CROSS=1; shift ;;
        -h|--help)
            sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) LABELS+=("$1"); shift ;;
    esac
done

[[ ${#LABELS[@]} -gt 0 ]] || {
    echo "usage: $0 [--allow-cross-client] <bench-label> [more...]" >&2
    exit 1
}

resolve_dir() {
    local arg="$1"
    if [[ -d "$arg" ]]; then
        echo "$(cd "$arg" && pwd)"
        return
    fi
    local cand="${ROOT}/benches/path-b-plus/${arg}"
    [[ -d "$cand" ]] && { echo "$(cd "$cand" && pwd)"; return; }
    echo "error: bench dir not found: $arg" >&2
    return 1
}

resolve_canon_telem() {
    local bench_dir="$1"
    local env="${bench_dir}/env.txt"
    local kind="" out_dir=""
    if [[ -f "$env" ]]; then
        kind="$(grep -E '^client_kind=' "$env" 2>/dev/null | cut -d= -f2- || true)"
        out_dir="$(grep -E '^OUT_DIR=' "$env" 2>/dev/null | cut -d= -f2- || true)"
    fi
    if [[ "$kind" == "native" && "$out_dir" == /src/* ]]; then kind=cuda-docker; fi
    if [[ "$kind" == "native" && "$out_dir" != /src/* && -n "$out_dir" ]]; then kind=rocm-native; fi
    if [[ -z "$kind" && "$out_dir" == /src/* ]]; then kind=cuda-docker; fi
    if [[ -z "$kind" && "$bench_dir" == *romulus-native* ]]; then kind=rocm-native; fi
    if [[ -z "$kind" && "$bench_dir" == *remus-docker* ]]; then kind=cuda-docker; fi
    case "$kind" in
        cuda-docker)  echo "${ROOT}/benches/path-b-plus/b6-2gpu-f-triton-n384-remus-docker/telemetry" ;;
        rocm-native)  echo "${ROOT}/benches/path-b-plus/b6-2gpu-f-triton-n384-romulus-native/telemetry" ;;
        *) echo "UNKNOWN_CLIENT_KIND:${kind}" >&2; return 1 ;;
    esac
}

printf 'label\tclient_kind\toverlap_pct\tstall_ratio\tdelta_overlap\tdelta_stall\tbisect_verdict\tbaseline_path\n'

for arg in "${LABELS[@]}"; do
    dir="$(resolve_dir "$arg")" || continue
    python3 - <<'PY' "$dir" "$ROOT" "$ALLOW_CROSS"
import json, re, sys
from pathlib import Path

bench_dir = Path(sys.argv[1])
root = Path(sys.argv[2])
allow_cross = int(sys.argv[3])

def load_diag(telem: Path) -> dict:
    p = telem / "diagnose.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text())
    return raw.get("diagnose", raw)

def infer_kind(bench_dir: Path) -> str:
    env = bench_dir / "env.txt"
    kind = ""
    out_dir = ""
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("client_kind="):
                kind = line.split("=", 1)[1].strip()
            elif line.startswith("OUT_DIR="):
                out_dir = line.split("=", 1)[1].strip()
    if kind == "native" and out_dir.startswith("/src/"):
        return "cuda-docker"
    if kind == "native" and out_dir and not out_dir.startswith("/src/"):
        return "rocm-native"
    if not kind and out_dir.startswith("/src/"):
        return "cuda-docker"
    name = bench_dir.name
    if "romulus-native" in name:
        return "rocm-native"
    if "remus-docker" in name:
        return "cuda-docker"
    return kind or "unknown"

def canon_telem(kind: str) -> Path:
    if kind == "cuda-docker":
        return root / "benches/path-b-plus/b6-2gpu-f-triton-n384-remus-docker/telemetry"
    if kind == "rocm-native":
        return root / "benches/path-b-plus/b6-2gpu-f-triton-n384-romulus-native/telemetry"
    raise ValueError(f"unknown client_kind={kind}")

def classify(delta_o: float, delta_s: float, overlap: float, stall: float,
             cross: bool) -> str:
    o_band = 0.3 if cross else 0.1
    s_band = 0.05 if cross else 0.02
    if overlap >= 1.0 and stall < 0.80:
        return "M1-CANDIDATE"
    if overlap >= 5.0 and stall < 0.50:
        return "M3-PASS"
    if abs(delta_o) < o_band and abs(delta_s) < s_band:
        return "NULL"
    if delta_o <= -0.3 or (delta_s >= 0.05 and delta_o <= 0):
        return "MITIGATION_HELPS"
    if delta_o >= 0.3 or (delta_s <= -0.05 and delta_o >= 0):
        return "MITIGATION_HURTS"
    return "MIXED"

label = bench_dir.name
telem = bench_dir / "telemetry"
kind = infer_kind(bench_dir)
try:
    canon = canon_telem(kind)
except ValueError:
    print(f"{label}\t{kind}\tNA\tNA\tNA\tNA\tUNKNOWN_KIND\tNA")
    sys.exit(0)

bisect = load_diag(telem)
base = load_diag(canon)
if not bisect or not base:
    print(f"{label}\t{kind}\tNA\tNA\tNA\tNA\tMISSING_DIAG\t{canon}")
    sys.exit(0)

bo = float(bisect.get("overlap_pct", 0) or 0)
bs = float(bisect.get("stall_ratio", 0) or 0)
co = float(base.get("overlap_pct", 0) or 0)
cs = float(base.get("stall_ratio", 0) or 0)
delta_o = round(bo - co, 3)
delta_s = round(bs - cs, 4)
verdict = classify(delta_o, delta_s, bo, bs, allow_cross == 1)
print(f"{label}\t{kind}\t{bo}\t{bs}\t{delta_o}\t{delta_s}\t{verdict}\t{canon}")
PY
done