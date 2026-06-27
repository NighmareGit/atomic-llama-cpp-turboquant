# Path-B Plus Spikes

Validation spikes for Tier 0/1. Tier 0 fixes are **mandatory**; spikes prove correctness and measure overlap.

## S5 assembly-line metric (P3)

Parser reads `sched-trace.jsonl` `split_total` events with `ts_us` and `copy` fields.

**Overlap detected** when for two events A and B:

- `A.backend != B.backend`
- `A.ts_us < B.ts_us < A.ts_us + A.elapsed_us` (concurrent splits on different backends)

Report: `assembly_overlap_count`, `assembly_overlap_pct` of split pairs.

## Spike table

| ID | Purpose | Pass | Status |
|----|---------|------|--------|
| S0 | 5-endpoint baseline | G + trace captured | PENDING |
| S4 | P0 barrier correctness | 4B/36B no corruption | PENDING |
| S5 | P0-P2 overlap | overlap_count > 0 steady gen | PENDING |
| S1 | Cross-port COPY count | gen-only COPY count | PENDING |
| S3 | Drain ratio before/after P2 | drain/cmd down | PENDING |

## Commands

```bat
cmd /c "D: && cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant && powershell -NoProfile -File scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-3gpu"
```

```powershell
.\scripts\cuda-windows-5070ti\pathb-rpc-trace-parse.ps1 -TraceDir docs\cuda-windows-5070ti\benchmarks\trace-f-3gpu\telemetry
```

Env: `GGML_PIPELINE_PLUS=1`, `GGML_SCHED_TRACE=1`, `GGML_RPC_TRACE=1`