# llama-pipeline-profiler -- implementation plan

Navigation: [OVERVIEW.md](OVERVIEW.md) | [TRACKING.md](TRACKING.md)

## Goal

Ship a generic multi-backend pipeline profiler that answers:

1. Which graph-split hop stalls (stall vs compute)?
2. Is the cross-token pipeline saturated (`assembly_overlap_pct`)?
3. Do Plus A/B and topology changes help (T1 matrix)?
4. Are GPU metrics consistent with scheduler stalls (hybrid smells)?

## Non-goals (v1)

- Replace `llama-server` for T0 depth-2 (HTTP stub remains)
- RX6600-focused work (parked outlier)
- Path C server aggregation (schema only in v1)

## Stub gap fixes

### T1: `bench-pipeline-plus-ab.sh`

- Default: `llama-pipeline-profiler --mode ab-plus`
- Fallback: `BENCH_HTTP=1` -> `rpc-server-bench.sh`
- Summary from `diagnose.json`: stall_ratio, overlap_pct, gates, smells

### T0: `bench-pipeline-depth2.sh`

- `LLAMA_MTP_ACC_TRACE` path in summary
- `BENCH_SERVER_TRACE=1` for server sched/rpc trace files

## Metrics strategy

**External:** `gpu-telemetry-collect.sh` -- NVIDIA CSV + ROCm JSONL, GEN phase only.

**Internal:** `GGML_SCHED_TRACE`, `GGML_RPC_TRACE`, `llama_perf_context`, watchdog idle detection.

**Smells:** power x util mismatch, orchestration stall, idle loop, clock throttle.

## Red-team requirements

- Dual gates: S5 (`overlap > 0`) + B+6 (`overlap >= target`, default 5%)
- `client_kind` in `env.txt`
- Trace observer: compare `throughput` vs `trace` mode
- `diagnose.json` for stubs and future `regression.jsonl`
- Path C reserved fields in TELEMETRY.md

## Atomic implementation phases

| ID | Deliverable |
|----|-------------|
| P0-2 | Doc hub |
| P0-1 | 4-GPU traced baseline (when cluster online) |
| P1-1 | gpu-telemetry-collect.sh |
| P1-2 | llama-pipeline-diagnose.sh |
| P1-3 | Tool README, TELEMETRY, GATES |
| P2-1 | llama-pipeline-profiler binary |
| P2-2 | bench-pipeline-plus-ab.sh |
| P2-3 | bench-pipeline-depth2.sh |
| P2-4 | llama-pipeline-profiler-cluster.sh |
| P3 | decode_id, pipeline_barrier, GGML_PIPELINE_TRACE |
| P4 | regression.jsonl, trace sampling |
| P5 | Cross-links, plus-ab matrix |

Update [TRACKING.md](TRACKING.md) after each row.

## Refactor: llama-bench parity (R1-R5)

Plan v1 shipped scripts + a binary that shells out to bash/Python. v2 makes the
profiler a cross-platform C++ tool like `llama-bench`: generic `-rpc`/`-ts`, no
hardcoded cluster topologies, diagnose/trace/regression in-process.

| ID | Deliverable |
|----|-------------|
| R1-1 | `pipeline-trace-parse` + `pipeline-diagnose` C++ (port diagnose.sh logic) |
| R1-2 | `pipeline-trace-sample` + `pipeline-regression` C++ |
| R1-3 | Profiler calls C++ modules; no `system("bash ...")` on default path |
| R2-1 | Remove `--topology` / `apply_topology()`; require `-rpc`/`-ts` |
| R2-2 | llama-bench-class flags: `-ngl`, `-ctk`, `-ctv`, `-sm`, `-r`, `-p`, `-b`, `-ub` |
| R3-1 | Combined result jsonl to stdout by default (`-o jsonl`) |
| R3-2 | Optional `--out-dir` for artifacts; trace files only when `--trace` |
| R4-1 | Stubs invoke binary with flags only (no `--topology`) |
| R4-2 | Cluster/import scripts remain optional conveniences |
| R5-1 | `--validate-rpc` preflight + cluster/r5-validate scripts; auto-probe before load |

Update [TRACKING.md](TRACKING.md) after each R row.

## Success criteria (v2)

1. TRACKING updated per atomic R step
2. Diagnose + regression work on Windows/Linux without bash/Python
3. No hardcoded RPC IPs or topology presets in the binary
4. Stubs publish trace metrics via native profiler flags
5. Tool README matches llama-bench operator depth

## Success criteria (v1, done)

1. TRACKING updated per atomic step
2. T1/T0 stubs publish trace metrics, not only G=
3. Hybrid metrics + smell detection on diagnose path
4. Tool README matches llama-bench operator depth