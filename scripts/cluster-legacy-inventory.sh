#!/usr/bin/env bash
# Print inventory of legacy romulus checkouts (run on romulus or via SSH).
#
# usage: bash scripts/cluster-legacy-inventory.sh

set -euo pipefail

LEGACY_HOME="${LEGACY_HOME:-$HOME/atomic-llama-cpp-turboquant}"
LEGACY_PROJECTS="${LEGACY_PROJECTS:-$HOME/projects/atomic-llama-cpp-turboquant}"
CANONICAL="${CLUSTER_REPO:-$HOME/projects/atomic-llama-cpp-turboquant}"

section() {
    echo
    echo "=== $1 ==="
}

path_report() {
    local p="$1"
    if [[ -d "$p" ]]; then
        du -sh "$p" 2>/dev/null || true
        if [[ -d "$p/.git" ]]; then
            (cd "$p" && git rev-parse --short HEAD 2>/dev/null && git status -sb 2>/dev/null | head -1)
        else
            echo "(not a git root)"
        fi
        local untracked
        untracked="$(cd "$p" 2>/dev/null && git status -u --porcelain 2>/dev/null | wc -l || echo 0)"
        [[ "$untracked" -gt 0 ]] && echo "untracked entries: ${untracked}"
    else
        echo "missing"
    fi
}

section "legacy home checkout"
path_report "$LEGACY_HOME"

section "legacy projects checkout"
path_report "$LEGACY_PROJECTS"

section "canonical target"
path_report "$CANONICAL"

if [[ -d "$LEGACY_HOME/rpc-patch/patch/bench-results/rpc-server-bench" ]]; then
    section "TSC bench artifacts (legacy home)"
    echo -n "tsc-* files: "
    find "$LEGACY_HOME/rpc-patch/patch/bench-results/rpc-server-bench" -maxdepth 1 -name 'tsc-*' 2>/dev/null | wc -l
    echo -n "trace-g-*tsc* files: "
    find "$LEGACY_HOME/rpc-patch/patch/bench-results/rpc-server-bench" -maxdepth 1 -name 'trace-g-*tsc*' 2>/dev/null | wc -l
    du -sh "$LEGACY_HOME/rpc-patch/patch/bench-results" 2>/dev/null || true
fi

if [[ -d "$LEGACY_HOME/benches/path-b-plus" ]]; then
    section "benches/path-b-plus (legacy home)"
    echo -n "labels: "
    ls "$LEGACY_HOME/benches/path-b-plus" 2>/dev/null | wc -l
    du -sh "$LEGACY_HOME/benches/path-b-plus" 2>/dev/null || true
fi

section "docker staging"
du -sh "$HOME/docker/Atomic-Llama-Romulus-PathB/staging" 2>/dev/null || echo "missing"

section "today scripts (legacy)"
for f in \
    "$LEGACY_HOME/scripts/b6-gate-romulus-local-mtp-up.sh" \
    "$LEGACY_HOME/scripts/b6-gate-romulus-pathb-deploy.sh"; do
    [[ -f "$f" ]] && echo "present: $f" || echo "absent: $f"
done

echo
echo "See rpc-patch/patch/LEGACY-ROMULUS-REVIEW-2026-07-04.md for salvage vs discard guidance."