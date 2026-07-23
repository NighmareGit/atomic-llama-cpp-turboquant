#!/usr/bin/env bash
# Append a profiler/bench run to regression.jsonl from diagnose.json + run metadata.
#
# usage:
#   llama-pipeline-regression.sh <telemetry_dir> --label NAME [options]
#
# options:
#   --regression-file PATH   default: benches/path-b-plus/regression.jsonl
#   --out-dir PATH           parent run dir (result.jsonl, env.txt)
#   --topology NAME
#   --mode NAME
#   --plus 0|1
#   --client-kind native|http

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE_DIR=""
LABEL=""
REGRESSION_FILE="${REGRESSION_FILE:-${ROOT}/benches/path-b-plus/regression.jsonl}"
OUT_DIR=""
TOPOLOGY=""
MODE=""
PLUS=""
CLIENT_KIND="native"

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --label) LABEL="${2:?}"; shift 2 ;;
        --regression-file) REGRESSION_FILE="${2:?}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
        --topology) TOPOLOGY="${2:?}"; shift 2 ;;
        --mode) MODE="${2:?}"; shift 2 ;;
        --plus) PLUS="${2:?}"; shift 2 ;;
        --client-kind) CLIENT_KIND="${2:?}"; shift 2 ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *)
            if [[ -z "$TRACE_DIR" ]]; then
                TRACE_DIR="$1"
            else
                echo "unexpected arg: $1" >&2
                usage
            fi
            shift
            ;;
    esac
done

[[ -n "$TRACE_DIR" && -n "$LABEL" ]] || usage
TRACE_DIR="$(cd "$TRACE_DIR" && pwd)"
DIAGNOSE="${TRACE_DIR}/diagnose.json"
[[ -f "$DIAGNOSE" ]] || { echo "error: missing ${DIAGNOSE}" >&2; exit 1; }

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$(cd "${TRACE_DIR}/.." && pwd)"
fi

PY_BIN="${PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PY_BIN=python3
    else
        PY_BIN=python
    fi
fi

export ROOT TRACE_DIR OUT_DIR LABEL REGRESSION_FILE TOPOLOGY MODE PLUS CLIENT_KIND
exec "$PY_BIN" - <<'PY'
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["ROOT"])
trace_dir = Path(os.environ["TRACE_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
label = os.environ["LABEL"]
regression_file = Path(os.environ["REGRESSION_FILE"])
topology = os.environ.get("TOPOLOGY", "")
mode = os.environ.get("MODE", "")
plus = os.environ.get("PLUS", "")
client_kind = os.environ.get("CLIENT_KIND", "native")

diagnose = json.loads((trace_dir / "diagnose.json").read_text(encoding="utf-8"))

def read_env_txt():
    env_path = out_dir / "env.txt"
    if not env_path.is_file():
        return {}
    out = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def read_result_g():
    result_path = out_dir / "result.jsonl"
    if result_path.is_file():
        try:
            line = result_path.read_text(encoding="utf-8").splitlines()[0].strip()
            if line:
                v = json.loads(line).get("G_tps")
                if v is not None:
                    return v
        except Exception:
            pass
    for name in ("result.meta", "bench.result", "profile-summary.txt"):
        meta_path = out_dir / name
        if not meta_path.is_file():
            continue
        try:
            text = meta_path.read_text(encoding="utf-8", errors="replace")
            vals = [float(x) for x in re.findall(r"G=([\d.]+)", text)]
            if vals:
                return max(vals)
        except Exception:
            pass
    return diagnose.get("G_tps")

def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"

env = read_env_txt()
g_tps = read_result_g()
if g_tps is None:
    g_tps = diagnose.get("G_tps")

record = {
    "version": 1,
    "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "label": label,
    "git_sha": env.get("GIT_SHA") or git_sha(),
    "client_kind": env.get("client_kind", client_kind),
    "trace_dir": str(trace_dir),
    "out_dir": str(out_dir),
    "topology": topology or env.get("TOPOLOGY", ""),
    "mode": mode or env.get("MODE", ""),
    "GGML_PIPELINE_PLUS": int(plus) if plus != "" else int(env.get("GGML_PIPELINE_PLUS", "1") or 1),
    "G_tps": g_tps,
    "gate_s5": diagnose.get("gate_s5"),
    "gate_b6": diagnose.get("gate_b6"),
    "overlap_pct": diagnose.get("overlap_pct"),
    "assembly_overlap_count": diagnose.get("assembly_overlap_count"),
    "stall_ratio": diagnose.get("stall_ratio"),
    "straggler_backend": diagnose.get("straggler_backend"),
    "straggler_ms_per_token": diagnose.get("straggler_ms_per_token"),
    "drain_flush_ms": diagnose.get("drain_flush_ms"),
    "blocking_ms": diagnose.get("blocking_ms"),
    "rpc_rtt_per_token": diagnose.get("rpc_rtt_per_token"),
    "overlap_efficiency": diagnose.get("overlap_efficiency"),
    "gpu_smell_flags": diagnose.get("gpu_smell_flags", []),
    "gen_tokens_est": diagnose.get("gen_tokens_est"),
    "diagnose_version": diagnose.get("version", 1),
}

regression_file.parent.mkdir(parents=True, exist_ok=True)
with regression_file.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, separators=(",", ":")) + "\n")

print(f"regression append -> {regression_file}")
print(f"  label={label} G_tps={g_tps} overlap_pct={record.get('overlap_pct')} gate_b6={record.get('gate_b6')}")
PY