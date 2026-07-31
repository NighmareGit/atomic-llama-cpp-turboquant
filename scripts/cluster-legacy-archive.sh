#!/usr/bin/env bash
# Stop romulus-local services and archive legacy checkout paths.
#
# usage (on romulus):
#   bash scripts/cluster-legacy-archive.sh
#   bash scripts/cluster-legacy-archive.sh --no-stop
#
# Archives:
#   ~/atomic-llama-cpp-turboquant
#   ~/projects/atomic-llama-cpp-turboquant  (entire tree including nested junk)

set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
DO_STOP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-stop) DO_STOP=0; shift ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

archive_dir() {
    local src="$1"
    if [[ -d "$src" ]]; then
        local dst="${src}.archived-${STAMP}"
        echo "=== archive ${src} -> ${dst} ==="
        mv "$src" "$dst"
    fi
}

if [[ "$DO_STOP" -eq 1 ]]; then
    echo "=== stop romulus-local services ==="
    pkill -f 'llama-server.*--port' 2>/dev/null || true
    docker rm -f pathb-rpc-romulus 2>/dev/null || true
fi

archive_dir "$HOME/atomic-llama-cpp-turboquant"
archive_dir "$HOME/projects/atomic-llama-cpp-turboquant"

echo "LEGACY_ARCHIVE_OK stamp=${STAMP}"
echo "next: bash scripts/cluster-node-fresh-clone.sh"