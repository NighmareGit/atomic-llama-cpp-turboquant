# Path-B Plus (Windows collateral)

Canonical docs: [rpc-patch/docs/rpc-path-b-plus-plan.md](../../rpc-patch/docs/rpc-path-b-plus-plan.md)

## Quick links

- Plan: [rpc-path-b-plus-plan.md](../../rpc-patch/docs/rpc-path-b-plus-plan.md)
- Tracking: [rpc-path-b-plus-tracking.md](../../rpc-patch/docs/rpc-path-b-plus-tracking.md)
- Spikes: [rpc-path-b-plus-spikes.md](../../rpc-patch/docs/rpc-path-b-plus-spikes.md)
- Baseline: [RPC-BUG-HUNT.md](RPC-BUG-HUNT.md)
- Wait sites: [RPC-WAIT-MAP.md](RPC-WAIT-MAP.md)

## Build + trace

```bat
cmd /c "D: && cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant && powershell -NoProfile -File scripts\cuda-windows-5070ti\build.ps1"
set GGML_PIPELINE_PLUS=1
powershell -NoProfile -File scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-3gpu
```