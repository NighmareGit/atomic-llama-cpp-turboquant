#!/usr/bin/env bash
# Tarball high-value TSC bench artifacts from legacy checkout before archive.
#
# usage (on romulus):
#   bash scripts/cluster-legacy-salvage.sh
#   LEGACY_HOME=~/atomic-llama-cpp-turboquant bash scripts/cluster-legacy-salvage.sh
#
# output: ~/archive/romulus-legacy-salvage-<timestamp>.tar.gz

set -euo pipefail

LEGACY_HOME="${LEGACY_HOME:-$HOME/atomic-llama-cpp-turboquant}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$HOME/archive}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${ARCHIVE_DIR}/romulus-legacy-salvage-${STAMP}.tar.gz"
BENCH="${LEGACY_HOME}/rpc-patch/patch/bench-results/rpc-server-bench"

if [[ ! -d "$BENCH" ]]; then
    echo "error: bench dir missing: ${BENCH}" >&2
    exit 1
fi

mkdir -p "$ARCHIVE_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

MANIFEST="${TMP}/MANIFEST.txt"
{
    echo "salvage_date=${STAMP}"
    echo "legacy_home=${LEGACY_HOME}"
    if [[ -d "${LEGACY_HOME}/.git" ]]; then
        echo "legacy_sha=$(git -C "$LEGACY_HOME" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    fi
} > "$MANIFEST"

cd "$BENCH"
shopt -s nullglob
FILES=(tsc-* trace-g-*tsc*)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "warn: no TSC artifacts found in ${BENCH}" >&2
    exit 0
fi

echo "=== salvaging ${#FILES[@]} files from ${BENCH} ==="
tar -czf "$OUT" -C "$BENCH" "${FILES[@]}"
cp "$MANIFEST" "${ARCHIVE_DIR}/romulus-legacy-salvage-${STAMP}.manifest"

echo "SALVAGE_OK ${OUT} ($(du -h "$OUT" | awk '{print $1}'))"
echo "manifest: ${ARCHIVE_DIR}/romulus-legacy-salvage-${STAMP}.manifest"