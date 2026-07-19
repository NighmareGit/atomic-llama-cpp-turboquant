# llama-pipeline-profiler Tracking

**Overall:** R1-R5 DONE (refactor complete) | **P0-1 live cluster trace rerun:** PENDING
**Branch:** Path-B-Event-Support-Pipeline-Plus

## Refactor checklist (R1-R5)

| Phase | Item | Status | Evidence |
|-------|------|--------|----------|
| R1 | `pipeline-trace-parse` C++ | DONE | `pipeline-trace-parse.cpp` |
| R1 | `pipeline-diagnose` C++ | DONE | `pipeline-diagnose.cpp` |
| R1 | `pipeline-trace-sample` C++ | DONE | `pipeline-trace-sample.cpp` |
| R1 | `pipeline-regression` C++ | DONE | `pipeline-regression.cpp` |
| R1 | Remove `system("bash")` from profiler default path | DONE | diagnose/sample/regression in-process |
| R1 | Offline verify trace-f-2gpu-plus | DONE | stall_ratio=0.6774, overlap_pct=0.6 |
| R2 | Remove `--topology` / hardcoded RPC IPs | DONE | generic `-rpc`/`-ts` only |
| R2 | llama-bench-class CLI flags | DONE | `-ngl`, `-sm`, `-ctk`, `-ctv`, `-r`, `-b`, `-ub`, `-t`, `-o` |
| R3 | stdout jsonl default (`-o jsonl`) | DONE | one line per rep |
| R3 | `--diagnose-only` offline mode | DONE | no model required |
| R4 | Stubs use binary flags only | DONE | `bench-pipeline-plus-ab.sh`, cluster wrapper |
| R4 | Cluster/import scripts optional | DONE | scripts remain as conveniences |
| R5 | `--validate-rpc` + auto preflight | DONE | `pipeline-rpc-validate.cpp`, cluster + r5-validate.sh |
| R5 | `-r` repetitions wired | DONE | trace on last rep; mean G in summary |
| R5 | C++ local GPU telemetry | DONE | `pipeline-gpu-telemetry.cpp` (nvidia-smi) |
| R5 | README + OVERVIEW refresh | DONE | no topology presets in docs |

## Phase checklist (v1)

| Phase | Item | Status | Evidence |
|-------|------|--------|----------|
| P0 | Doc hub (PLAN, TRACKING, OVERVIEW) | DONE | `docs/llama-pipeline-profiler/` |
| P0 | Traced 4-GPU baseline rerun | SEEDED | offline hotpath; live full trace pending |
| P1 | `gpu-telemetry-collect.sh` + schema | DONE | optional remote SSH sampler |
| P1 | `llama-pipeline-diagnose.sh` v1 | DONE | reference; binary uses C++ |
| P1 | Tool README + TELEMETRY + GATES | DONE | README refreshed 2026-06-28 |
| P2 | `llama-pipeline-profiler` binary v1 | DONE | `build/bin/Release/llama-pipeline-profiler.exe` |
| P2 | Fix `bench-pipeline-plus-ab.sh` | DONE | requires `BENCH_RPC_ENDPOINT` |
| P2 | Fix `bench-pipeline-depth2.sh` | DONE | MTP trace + BENCH_SERVER_TRACE |
| P2 | `llama-pipeline-profiler-cluster.sh` | DONE | R5 preflight before run |
| P3 | C++ `decode_id` + `GGML_PIPELINE_TRACE` | DONE | `pipeline-trace.jsonl` |
| P4 | `regression.jsonl`, trace sampling | DONE | C++ in-process |
| P5 | Cross-links + plus-ab matrix rows | DONE | `benches/path-b-plus/` |

## Implementation log

| Date | Atomic step | Action | Artifact |
|------|-------------|--------|----------|
| 2026-07-01 | R5-1b | PR 8 matrix all PASS; `pipeline_rpc_validate_prepare()` RPC-only load | `b6-gate-validate-rpc-matrix.sh` |
| 2026-06-28 | R5-1 | `--validate-rpc` + auto preflight | live OK: 192.168.8.176, 192.168.8.21 |
| 2026-06-28 | R5-2 | `llama-pipeline-r5-validate.sh` + cluster preflight | `scripts/` |
| 2026-06-28 | R5-3 | Wire `-r` repetitions in `run_cell` | `llama-pipeline-profiler.cpp` |
| 2026-06-28 | R5-4 | C++ local GPU telemetry | `pipeline-gpu-telemetry.*` |
| 2026-06-28 | R5-5 | README + OVERVIEW refresh | `tools/llama-pipeline-profiler/README.md` |
| 2026-06-28 | R0-R4 | llama-bench parity refactor | see prior log rows |
| 2026-06-28 | P0-P5 | Path-B profiler v1 | see prior log rows |

## Issues

| Date | Issue | Status |
|------|-------|--------|
| 2026-06-28 | P0-1 live cluster full trace rerun | 2/3 RPC hops OK from Windows; `127.0.0.1:50051` is romulus-local | OPEN |
| 2026-06-28 | Remote remus/rocm GPU telemetry | still in optional `gpu-telemetry-collect.sh` (SSH) | OPEN (optional) |

## Benchmark / gate results

| Label | G | overlap_pct | gate_s5 | gate_b6 | stall_ratio | Notes |
|-------|---|-------------|---------|---------|-------------|-------|
| trace-g-4gpu-primary-trace | ~37 | 0.1% | PASS | FAIL | TBD | hotpath seed in regression.jsonl |
| trace-f-2gpu-plus | n/a | 0.6% | PASS | FAIL | 0.6774 | C++ diagnose verified |

## Next steps

1. When romulus online: `BENCH_RPC_ENDPOINT=... BENCH_TS=... ./scripts/llama-pipeline-profiler-cluster.sh`
2. Replace hotpath-seeded 4-GPU row with full `pipeline-trace.jsonl` + live diagnose
3. Publish paired plus-on/off rows from live `bench-pipeline-plus-ab.sh`