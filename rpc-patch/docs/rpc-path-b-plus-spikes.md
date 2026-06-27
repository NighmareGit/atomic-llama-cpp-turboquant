# Path-B Plus Spikes

Validation spikes for Tier 0/1. Tier 0 fixes are **mandatory**; spikes prove correctness and measure overlap.

**Phase 5b status (2026-06-27):** S4, S5, S1, S3 PASS on production topology (`trace-f-2gpu-plus`). S0 deferred.

## S5 assembly-line metric (P3)

Parser reads `sched-trace.jsonl` `split_total` events with `ts_us` and `copy` fields.

**Overlap detected** when for two events A and B:

- `A.backend != B.backend`
- `A.ts_us < B.ts_us < A.ts_us + A.elapsed_us` (concurrent splits on different backends)

Report: `assembly_overlap_count`, `assembly_overlap_pct` of split pairs.

## Spike table

| ID | Purpose | Pass criteria | Status | Evidence |
|----|---------|---------------|--------|----------|
| S0 | 5-endpoint baseline | G + trace captured | **DEFERRED** | 72B+ cluster; not blocking B+1 |
| S4 | P0 barrier correctness | 4B/36B no corruption | **PASS** | `s4-4b-2gpu-plus` G=44.2; `trace-f-2gpu-plus` G=48.9; pipeline + sched copies=4 |
| S5 | P0-P2 overlap | overlap_count > 0 steady gen | **PASS** | `trace-f-2gpu-plus`: count=267, pct=0.5% |
| S1 | Cross-port COPY count | gen COPY budget documented | **PASS** | 2gpu: SET_TENSOR_HASH=168, COPY_TENSOR=0; 3gpu-plus: COPY_TENSOR=144 |
| S3 | Drain ratio before/after P2 | drain down vs legacy | **PASS** | drain_flush 2523 ms (legacy) -> 1734 ms (2gpu-plus), -31% |

## S3 drain comparison (128-token window)

| Run | Plus | Topology | blocking_ms | drain_flush_ms | overlap_count | copy rotation |
|-----|------|----------|-------------|----------------|---------------|---------------|
| trace-f-3gpu-legacy | 0 | 3-device | 32375 | 2523 | 127 | broken (copy3=508) |
| trace-f-3gpu-plus | 1 | 3-device | 31144 | 2599 | 289 | OK (132/132/128/128) |
| trace-f-2gpu-plus | 1 | 2-device | 29812 | **1734** | 267 | OK (99/99/96/96) |

P0 copy rotation fix is the primary win on 3gpu (legacy had 508 splits on copy3 only). P2 scoped drain + 2-device topology reduces drain_flush further.

## S4 smoke runs

| Run | Model | G (t/s) | pipeline | sched copies | Result |
|-----|-------|---------|----------|--------------|--------|
| s4-4b-2gpu-plus | gemma-4-E4B Q4_K_M | 44.2 | enabled | 4 | PASS |
| trace-f-2gpu-plus | Qwen3.6-35B NL | 48.9 | enabled | 4 | PASS |

Artifacts: `docs/cuda-windows-5070ti/benchmarks/s4-4b-2gpu-plus/`, `trace-f-2gpu-plus/`

## Commands

Production topology (recommended):

```powershell
.\scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-2gpu-plus
.\scripts\cuda-windows-5070ti\pathb-rpc-trace-parse.ps1 -TraceDir docs\cuda-windows-5070ti\benchmarks\trace-f-2gpu-plus\telemetry
```

Full matrix (comparison):

```bat
cmd /c "D: && cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant && powershell -NoProfile -File scripts\cuda-windows-5070ti\pathb-trace-runbook.ps1 -Runs trace-f-3gpu-plus,trace-f-2gpu-plus"
```

S4 4B smoke:

```powershell
$env:GGML_PIPELINE_PLUS = "1"
.\scripts\cuda-windows-5070ti\rpc-server-bench.ps1 -Label s4-4b-2gpu-plus -Config config-f `
  -RpcEndpoint 192.168.8.176:50051 -TensorSplit 50,50 `
  -ModelPath D:\models\gemma-4-E4B.i1-Q4_K_M.gguf -Ctk q4_0 -Ctv q4_0 `
  -GenTokens 64 -Runs 1 -EnsurePathbRpc
```

Env: `GGML_PIPELINE_PLUS=1`, `GGML_SCHED_TRACE=1`, `GGML_RPC_TRACE=1` (set by `rpc-server-bench.ps1 -Trace`).