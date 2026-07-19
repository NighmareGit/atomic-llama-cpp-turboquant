# 2026-07 comparison matrix — path-b-plus

Cross-topology summary after Phase 1.1 trace re-parse and B+6 mitigation ladder exhaustion.
Data sources: in-repo `diagnose.json`, `trace-parse-extended.json`, `bench.result`, `b6-diagnosis-matrix.tsv`.

**Scope:** Path-B+ only. Path C is out of scope (reference for ideas, no implementation on this branch).

## Executive summary

| Goal | Best result | Gate |
|------|-------------|------|
| Production throughput (36B NL MoE) | **48.9 t/s** (`trace-f-2gpu-plus`, 2-device) | PASS |
| B+6 overlap (M3 >= 5%) | **0.9%** (triton guard-n128 only) | **FAIL** |
| Mitigation ladder (B+8–B+10, B+7a') | NULL on overlap at n=384 | Ceiling documented |

**Takeaway:** Throughput and overlap are decoupled. Triton swap raises G (128–200 t/s) without moving `overlap_pct` off 0.2% at n=384. Serial RPC critical path + collapsed copy-slot pipelining is structural within Path-B+.

## Topology comparison (canonical runs)

| Label | Devices | RPC workers | Client | n | G (t/s) | overlap % | stall | drain (ms) | straggler (ms/tok) | Verdict |
|-------|---------|-------------|--------|---|---------|-----------|-------|------------|-------------------|---------|
| `trace-f-2gpu-plus` | 2 | remus 5060 | Win 5070 | 130 | **48.9** | 0.6 | 0.68 | 1755 | 20.4 (RPC) | **Production** |
| `trace-f-3gpu-plus` | 3 (+RX6600) | 5060 + 6600 | Win 5070 | 130 | 42.8 | 0.3 | 0.75 | 2599 | 18.2 | Deprecated for 35B+ MoE |
| `b6-2gpu-f-triton-n384-romulus-native` | 2 | triton 3090 | romulus 7900 | 384 | 200.8 | 0.2 | 0.86 | 1752 | 5.9 | STRAGGLER_DOMINANT |
| `b6-2gpu-f-triton-guard-n128` | 2 | triton 3090 | remus docker | 128 | 116.0 | **0.9** | 0.64 | 788 | 13.7 | Best overlap (short n) |
| `b6-4gpu-g-n384-romulus-native` | 4 | 5060+3060+JUPITER | romulus 7900 | 384 | 69.5 | 0.2 | 0.96 | 6362 | 8.4 | M1 FAIL |
| `b6-4gpu-g-triton-n384-romulus-native` | 4 | triton swap | romulus 7900 | 384 | 73.2 | 0.1 | 0.94 | 4632 | — | Drain improved, overlap FAIL |
| `trace-g-4gpu-primary` | 4 (no 6600) | cluster | romulus | 128 | ~43 | 0.1 | — | — | — | June baseline |
| `trace-g-4gpu-primary-retry-M35` | 4 (no 6600) | 5060+3060+5070 | romulus | 128 | **44.9** | — | — | — | — | **2026-07-02 PASS** (-3.2% vs 2G) |
| `trace-g-2gpu-retry-M35` | 2 | remus 5060 | romulus | 128 | **46.4** | — | — | — | — | Config G 2G baseline |
| `trace-g-3gpu-5070-retry-M35` | 3 | 5060+5070 | romulus | 128 | 44.7 | — | — | — | — | 5070 third GPU OK |
| `trace-g-3gpu-3060-retry-M35` | 3 | 5060+3060 | romulus | 128 | 42.8 | — | — | — | — | 3060 third GPU OK |
| `b6-4gpu-g-n384-config-g-retry-M35` | 4 | Config G | romulus | 384 | 76.0 | 0.2 | 0.67 | — | 7.1 (5060) | Profiler (higher orch penalty) |
| `b6-4gpu-g-n128-config-g-retry-M70` | 4 | Config G | romulus | 128 | 16.3 | 0.4 | 0.67 | — | 33.3 (3060) | Llama-70B dense |

## Phase 1.1 hot-path signals (n=384 triton ladder)

Romulus-native canonical (`b6-2gpu-f-triton-n384-romulus-native`):

| Metric | Value | vs compute |
|--------|-------|------------|
| `input_wait_copy_ms` | 1976 | 6.1x `graph_compute_async_ms` (323) |
| `rpc_rtt` total | 1371 ms / 385 tok | 3.6 ms/tok |
| `rpc_rtt_p50` / `p95` / `p99` | 0.35 / 3.68 / 71.55 ms | tail spikes |
| `per_split_rpc_rtt` split=2 backend=1 | 432 events, 142 ms | triton RPC stage |
| `assembly_overlap` | 990 / 444675 pairs = **0.2%** | backends rarely concurrent |

## Mitigation bisect summary (2-GPU triton n=384)

| Bisect | overlap | stall | vs canonical | Verdict |
|--------|---------|-------|--------------|---------|
| canonical | 0.2% | 0.858 | — | baseline |
| B+9 OFF (`no-defer`) | 0.2% | 0.870 | Δoverlap 0 | NULL |
| B+8 OFF (`no-partial`) | 0.2% | 0.904 | Δoverlap 0 | NULL / stall worse |
| B+10 OFF (`no-async-copy`) | 0.2% | 0.906 | Δoverlap 0 | NULL / stall worse |
| B+7a' OFF (4-GPU `no-flush`) | 0.1% | — | Δoverlap -0.1 | NULL / worse |

## validate-rpc (PR 8)

Hang was **configuration-dependent** (remus docker CUDA + triton), not host-permanent. Root cause addressed by `ggml-rpc.cpp` handshake fixes (combined send buffer, init-phase drain skip, post-hello pending reset).

Matrix re-run 2026-07-01: **all cells PASS** — see `benches/path-b-plus/validate-rpc-matrix/results.tsv`.

| Cell | Client | Target | Result |
|------|--------|--------|--------|
| A | romulus ROCm | triton :50054 | PASS |
| B | remus docker CUDA | triton :50054 | PASS |
| C | remus docker CUDA | remus :50051 | PASS |
| D | romulus ROCm | 4-GPU endpoints | PASS |

**Policy:** 4-GPU gate runs may re-enable RPC preflight. Bisect waves may keep `PROFILER_SKIP_VALIDATE=1` for speed.

## Recommendations (path-b-plus scope)

1. **Ship:** `trace-f-2gpu-plus` (48.9 t/s); no RX6600 for 35B+ A3B MoE (enforced in matrix scripts).
2. **Do not pursue:** Path C implementation on this branch.
3. **Parked pending grill:** B+11–B+13 (proto/cross-cutting; explicit scope required).
4. **Hygiene:** PR 8 closed; optional `pipeline_rpc_validate_prepare()` loads RPC-only for `--validate-rpc`.

## SYNC vs Plus deployment matrix (planned)

Apples-to-apples **atomic sync branch** vs **Path-B+ production** on 3-GPU and 5-GPU Linux:

- Chart: [path-b-plus-vs-sync-comparison.md](path-b-plus-vs-sync-comparison.md)
- Orchestrator: `scripts/b6-gate-sync-vs-plus-comparison.sh`
- Results jsonl: `benches/path-b-plus/sync-vs-plus-comparison.jsonl`

## Artifact index

| Path | Contents |
|------|----------|
| `benches/path-b-plus/b6-diagnosis-matrix.tsv` | Gate run matrix |
| `benches/path-b-plus/b6-2gpu-f-triton-n384-*/telemetry/trace-parse-extended.json` | Phase 1.1 |
| `benches/path-b-plus/validate-rpc-matrix/` | PR 8 investigation logs |
| `docs/rpc-multi-backend-pipeline-plus/TRACKING.md` | Structural ceiling verdict |

---

Generated 2026-07-01 after Phase 1.1 re-parse and ladder exhaustion. Refresh when new gate runs land.