#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for f in rpc-patch/scripts/pathb-remus-rpc.sh \
         rpc-patch/scripts/pathb-remus-rx6600-rpc.sh \
         rpc-patch/scripts/pathb-remus-multi-rpc-win.sh \
         rpc-patch/scripts/pathb-gpu-monitor-win.sh; do
  sed -i 's/\r$//' "$f"
  chmod +x "$f"
done
echo "fixed crlf"