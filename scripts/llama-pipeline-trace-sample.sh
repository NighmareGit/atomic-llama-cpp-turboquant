#!/usr/bin/env bash
# Downsample large trace jsonl files for storage/sharing. Keeps all pipeline barriers.
#
# usage:
#   llama-pipeline-trace-sample.sh <telemetry_dir> [--every N] [--max-lines M]
#
# writes:
#   sched-trace.sample.jsonl
#   rpc-trace.sample.jsonl
#   pipeline-trace.sample.jsonl  (copy; usually small)
#   sample-meta.json

set -euo pipefail

TRACE_DIR=""
EVERY="${TRACE_SAMPLE_EVERY:-10}"
MAX_LINES="${TRACE_SAMPLE_MAX:-0}"

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --every) EVERY="${2:?}"; shift 2 ;;
        --max-lines) MAX_LINES="${2:?}"; shift 2 ;;
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

[[ -n "$TRACE_DIR" ]] || usage
TRACE_DIR="$(cd "$TRACE_DIR" && pwd)"

PY_BIN="${PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PY_BIN=python3
    else
        PY_BIN=python
    fi
fi

export TRACE_DIR EVERY MAX_LINES
exec "$PY_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

trace_dir = Path(os.environ["TRACE_DIR"])
every = max(1, int(os.environ.get("EVERY", "10")))
max_lines = int(os.environ.get("MAX_LINES", "0"))

def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows

def sample_rows(rows: list[dict], *, keep_phase: set[str] | None = None, keep_blocking: bool = False) -> list[dict]:
    if not rows:
        return []
    out: list[dict] = []
    seen_decode: set[int] = set()
    for i, row in enumerate(rows):
        keep = False
        phase = row.get("phase", "")
        if keep_phase and phase in keep_phase:
            keep = True
        if keep_blocking and row.get("blocking") is True:
            keep = True
        decode_id = row.get("decode_id")
        if isinstance(decode_id, int) and decode_id not in seen_decode and decode_id % every == 0:
            keep = True
            seen_decode.add(decode_id)
        if i % every == 0:
            keep = True
        if keep:
            out.append(row)
    if max_lines > 0 and len(out) > max_lines:
        step = max(1, len(out) // max_lines)
        out = out[::step][:max_lines]
    return out

def write_jsonl(path: Path, rows: list[dict]) -> int:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)

meta = {"version": 1, "every": every, "max_lines": max_lines, "files": {}}

sched_src = trace_dir / "sched-trace.jsonl"
rpc_src = trace_dir / "rpc-trace.jsonl"
pipe_src = trace_dir / "pipeline-trace.jsonl"

sched_rows = load_jsonl(sched_src)
rpc_rows = load_jsonl(rpc_src)
pipe_rows = load_jsonl(pipe_src)

sched_sample = sample_rows(sched_rows, keep_phase={"split_total"})
rpc_sample = sample_rows(rpc_rows, keep_blocking=True)

meta["files"]["sched-trace.jsonl"] = {"source_lines": len(sched_rows), "sample_lines": write_jsonl(trace_dir / "sched-trace.sample.jsonl", sched_sample)}
meta["files"]["rpc-trace.jsonl"] = {"source_lines": len(rpc_rows), "sample_lines": write_jsonl(trace_dir / "rpc-trace.sample.jsonl", rpc_sample)}

if pipe_rows:
    meta["files"]["pipeline-trace.jsonl"] = {"source_lines": len(pipe_rows), "sample_lines": write_jsonl(trace_dir / "pipeline-trace.sample.jsonl", pipe_rows)}
elif pipe_src.is_file():
    shutil.copy2(pipe_src, trace_dir / "pipeline-trace.sample.jsonl")
    meta["files"]["pipeline-trace.jsonl"] = {"source_lines": "copied", "sample_lines": "copied"}

(trace_dir / "sample-meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"trace sample -> {trace_dir}")
for name, info in meta["files"].items():
    print(f"  {name}: {info['source_lines']} -> {info['sample_lines']}")
PY