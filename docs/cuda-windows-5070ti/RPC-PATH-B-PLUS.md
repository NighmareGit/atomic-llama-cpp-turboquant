# Path-B Plus (Windows collateral)

Canonical docs live under `rpc-patch/docs/`. Start with the overview.

## Quick links

- Overview: [rpc-path-b-plus-overview.md](../../rpc-patch/docs/rpc-path-b-plus-overview.md)
- Plan: [rpc-path-b-plus-plan.md](../../rpc-patch/docs/rpc-path-b-plus-plan.md)
- Tracking: [rpc-path-b-plus-tracking.md](../../rpc-patch/docs/rpc-path-b-plus-tracking.md)
- Spikes: [rpc-path-b-plus-spikes.md](../../rpc-patch/docs/rpc-path-b-plus-spikes.md)
- Handover: [rpc-path-b-plus-handover.md](../../rpc-patch/docs/rpc-path-b-plus-handover.md)
- Path C (Phase 6): [rpc-path-c-plan.md](../../rpc-patch/docs/rpc-path-c-plan.md)
- Baseline: [RPC-BUG-HUNT.md](RPC-BUG-HUNT.md)
- Wait sites: [RPC-WAIT-MAP.md](RPC-WAIT-MAP.md)

## Status

B+1 **production ready** on 2-device Config F (`ts=50,50`, G ~49 t/s). Phase 5 complete. Phase 6 closed (Path C declined). B+4..B+6 shipped — **rebuild required**. Phase 7-8: Linux trace tooling + Config G cluster scripts shipped. Phase 9: S0-lite 72B benches.

## Production bench (recommended)

```bat
cmd /c "D: && cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant && powershell -NoProfile -File scripts\cuda-windows-5070ti\build.ps1"
set GGML_PIPELINE_PLUS=1
powershell -NoProfile -File scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-2gpu-plus
powershell -NoProfile -File scripts\cuda-windows-5070ti\pathb-rpc-trace-parse.ps1 -TraceDir docs\cuda-windows-5070ti\benchmarks\trace-f-2gpu-plus\telemetry
```

## Comparison matrix (optional)

```bat
powershell -NoProfile -File scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-3gpu-plus,trace-f-2gpu-plus
```

Remus deploy: `./rpc-patch/scripts/pathb-remus-rpc.sh rebuild` (proto 4.3.2). See [MULTI-NODE.md](MULTI-NODE.md).