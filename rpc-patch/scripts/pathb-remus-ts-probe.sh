#!/usr/bin/env bash
# Tensor-split probe with remote remus RPC (wraps rpc-ts-fit-probe.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="${1:?label required}"

export BENCH_RPC_MODE="${BENCH_RPC_MODE:-remote}"
export BENCH_RPC_HOST="${BENCH_RPC_HOST:-remus.local}"
export PROBE_VARIANT="${PROBE_VARIANT:-pathb}"
export PROBE_TS="${PROBE_TS:-15,85}"

"$SCRIPT_DIR/rpc-ts-fit-probe.sh" "$LABEL"