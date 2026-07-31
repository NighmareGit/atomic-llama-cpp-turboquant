#!/bin/bash
# Append a profiler run to the scratch results file and print comparison.
# Usage:
#   ./scripts/profile-scratch.sh <backend_label> <heatmap.json> [model_name]
#   ./scripts/profile-scratch.sh --show   (print current scratch)

SCRATCH="${SCRATCH_FILE:-/tmp/tel-run/scratch-results.jsonl}"

if [ "$1" = "--show" ]; then
    if [ ! -f "$SCRATCH" ]; then
        echo "No scratch file at $SCRATCH"
        exit 1
    fi
    python3 -c "
import json
print(f'{\"Backend\":20s} {\"Model\":24s} {\"tps\":>7s} {\"wall_ms\":>8s} {\"clean\":>6s}')
print('-' * 70)
with open('$SCRATCH') as f:
    for line in f:
        e = json.loads(line)
        print(f'{e[\"backend\"]:20s} {e[\"model\"][:22]:24s} {e[\"tps\"]:>7.1f} {e[\"wall_ms\"]:>8.1f} {e[\"clean\"]:>6s}')
"
    exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <backend_label> <heatmap.json> [model_name]"
    echo "       $0 --show"
    exit 1
fi

BACKEND="$1"
HEATMAP="$2"
MODEL="${3:-unknown}"

mkdir -p "$(dirname "$SCRATCH")"

python3 -c "
import json, time, os, sys

path = '$HEATMAP'
backend = '$BACKEND'
model_name = '$MODEL'

if not os.path.exists(path):
    print(f'ERROR: heatmap not found: {path}', file=sys.stderr)
    sys.exit(1)

with open(path) as f:
    data = json.load(f)

task = data['tasks']['tg']
m = data.get('model', {})
gpus = data.get('gpu_metadata', [])
layers = task.get('layer_rollup', [])
cats = task.get('op_categories', [])

if not model_name or model_name == 'unknown':
    model_name = m.get('path', '?').split('/')[-1]

entry = {
    'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'backend': backend,
    'gpu': gpus[0]['backend'] if gpus else 'CPU',
    'model': model_name,
    'params_b': m.get('param_count_b', 0),
    'n_layers': m.get('n_layers', 0),
    'wall_ms': round(task['wall_ms'], 1),
    'tps': round(task['tps'], 1),
    'n_gen': task.get('n_gen_tokens', '?'),
    'layer_rollup_count': len(layers),
    'layer_us_min': min(l['us'] for l in layers) if layers else 0,
    'layer_us_max': max(l['us'] for l in layers) if layers else 0,
    'op_categories': {c['category']: round(c['ms'], 3) for c in cats},
    'clean': 'YES' if (max(l['us'] for l in layers) if layers else 0) < 1_000_000 else 'GARBAGE',
}

with open('$SCRATCH', 'a') as f:
    f.write(json.dumps(entry) + '\n')

print(f'Appended: {backend} | {model_name[:30]} | tps={entry[\"tps\"]} | clean={entry[\"clean\"]}')
print(f'Scratch: $SCRATCH ({sum(1 for _ in open(\"$SCRATCH\"))} entries)')
"
