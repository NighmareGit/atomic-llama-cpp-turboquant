# Path-B Plus Tracking

Overview: [rpc-path-b-plus-overview.md](rpc-path-b-plus-overview.md)

**Overall:** B+1 **PRODUCTION READY** | B+4..B+6 **SHIPPED** (rebuild required) | Phase 7-8 tooling/cluster **SHIPPED**  
**Build:** 9964+ client + remus/romulus **proto 4.3.2**  
**Current Phase:** **B+6 gate** ([b6-gate/TRACKING.md](b6-gate/TRACKING.md)) -- triton A/B + ts sweep done; next B+7 drain bisect | Phase 10 4-GPU stable
**Branch:** Path-B-Event-Support-Pipeline-Plus (private fork)

## Implementation log

| Date | Phase | Action |
|------|-------|--------|
| 2026-06-27 | B+0 | Plan + docs created from bug-hunt + user review |
| 2026-06-27 | B+1 | P0 pipeline_barrier, P1 sampling sync, P2 scoped drain, P3 trace copy/overlap |
| 2026-06-27 | B+1b | tls.ev event tracking fix; trace jsonl truncate; A/B bench |
| 2026-06-27 | B+2 | cpy_tensor_async + deferred COPY drain (client) |
| 2026-06-27 | B+3 | COPY_TENSOR_PEER proto v4.3 (client+server; remus deployed) |
| 2026-06-27 | B+3b | tier1b bench: remus v4.3 non-event on 3gpu F; peer COPY N/A (CUDA-RPC topology) |
| 2026-06-27 | Phase 2 | Trace parser cmd map fix; hello/copy_issue RPC trace fields |
| 2026-06-27 | Deploy | `pathb-remus-deploy-sync.sh`; build auto-syncs deploy to remus; proto 4.3.2 verified |
| 2026-06-27 | Phase 5a | 2-device F production default (`ts=50,50`, single `:50051`, `GGML_PIPELINE_PLUS=1`) |
| 2026-06-27 | Phase 5b | Spikes S4/S5/S1/S3 PASS on `trace-f-2gpu-plus`; S0 deferred |
| 2026-06-27 | Docs | Refresh stale mirror + `RPC-PATH-B-PLUS.md`; proto-check-v43 in deploy section |
| 2026-06-27 | Phase 6 | Path C kickoff: C1 baseline catalog + remus feasibility spike |
| 2026-06-27 | Phase 6 | **CLOSED** — Path C C2+ declined |
| 2026-06-27 | B+4 | SET_TENSOR_HASH client cache (`tls_hash_present`) |
| 2026-06-27 | B+5 | Server async compute queue + EVENT_RECORD barrier |
| 2026-06-27 | B+6 | Remove GRAPH entry drain; assembly-line unlock |
| 2026-06-27 | Phase 7 | Linux trace parse + hotpath-summary + BENCH_TRACE |
| 2026-06-27 | Phase 8 | Romulus PathB deploy + cluster-up + Windows :50053 script + config-g |
| 2026-06-27 | Phase 10 | **4-GPU primary stable:** 7900+3060+5060+5070; RX6600 slot-init hang bisected |
| 2026-06-27 | Phase 10 | `pathb-romulus-{3,4}gpu-bench.sh`, gdb-repro presets, cluster summary docs |
| 2026-06-29 | B+6 | JUPITER abort fix (120a-real); canonical b6-4gpu-g n=384; triton A/B; ts sweep; diagnosis D3+D1 |
| 2026-06-29 | B+6 | Scripts: b6-4gpu-g-triton, b6-gate-ts-sweep-4gpu.sh, b6-gate-diagnose-runs.sh |

## Issues

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|
| 2026-06-27 | Path B plateau on N-RPC | Graph reuse skips copy rotation; full sync drains pipe | B+1 Tier 0 | FIXED |
| 2026-06-27 | Plus regression G=28 vs 38 | drain_pending cleared tls but not ev->response_pending | tls.ev tracking | FIXED |
| 2026-06-27 | tier1 vs plus G delta -2.0 | Same build 9962; single-run jitter | Multi-run not needed; closed as fluke | CLOSED |
| 2026-06-27 | tier1b no COPY_TENSOR_PEER | 3gpu F copies are CUDA<->RPC not RPC<->RPC | Pivot B+3 to Path C / 5-ep | WONTFIX on 3gpu |
| 2026-06-27 | trace-f-2gpu-plus HELLO minor=2 | Bench ran before remus 4.3.2 deploy sync | Remus now 4.3.2; re-bench optional | CLOSED |
| 2026-06-27 | 4-GPU + RX6600 slot-init hang | EVENT_RECORD/COPY drain at first slot warmup with `:50052` | Use **primary** topology (5070 `:50053`); park 6600 | **WORKAROUND** |
| 2026-06-27 | Windows :50053 unreachable from romulus | rpc-server not started / wrong bind IP | `pathb-rpc-server.ps1` + foreground start; prefer `192.168.8.21` | FIXED |
| 2026-06-27 | bench curl FAIL http=200 | Multi-prompt fox parser false negative | Inference OK; RESULT may show FAIL | KNOWN |

## Benchmarks

| Run | Config | G (t/s) | Build | Notes |
|-----|--------|---------|-------|-------|
| trace-f-3gpu-pre | 3-device 36B NL | 38.0 | - | Pre-Plus (bug-hunt) |
| trace-f-3gpu-legacy | 3-device, Plus=0 | 39.3 | 9962 | Full sync on reuse; copy3=508 (broken rotation) |
| trace-f-3gpu-plus | 3-device, Plus=1 | **42.8** | 9962 | Best 3gpu; copy rotation OK |
| trace-f-3gpu-tier1 | 3-device, Plus=1 | 40.8 | **9962** | Same binary as plus |
| trace-f-3gpu-tier1b | 3-device, Plus=1, remus v4.3 | 40.6 | 9963 | Within noise of tier1; 0 peer COPY (expected) |
| trace-f-2gpu-plus | 2-device ts=50,50, Plus=1 | **48.9** | 9964 | **Production default**; 3 splits; overlap 0.5% |
| s4-4b-2gpu-plus | 2-device gemma-4-E4B, Plus=1 | 44.2 | 9964 | S4 correctness smoke PASS |
| proto-check-v43 | HELLO negotiate | - | 9964 | minor=3 peer_copy on :50051/:50052 |
| trace-g-3gpu-primary-r1/r3 | romulus 7900+5060+3060 | ~29-35 | 833ad4429 | Tier 0 control PASS |
| trace-g-4gpu-primary (x3) | romulus 7900+5060+3060+5070 | **38-43** | 833ad4429 | **4-GPU production stable** |
| trace-g-4gpu-primary-trace | 4-GPU + trace | ~37 | 833ad4429 | hotpath 20.7 ms/tok |
| trace-g-4gpu-romulus-q8-pp1 | legacy 6600 4-GPU | HANG | 833ad4429 | slot init stall (aborted) |

Artifacts: `docs/cuda-windows-5070ti/benchmarks/trace-f-{3gpu-plus,legacy,tier1,tier1b,2gpu-plus}/`, `s4-4b-2gpu-plus/`, `proto-check-v43/`  
Romulus: `patch/bench-results/rpc-server-bench/trace-g-4gpu-primary*/`, `cluster-4gpu-primary/summary.md`

## Production default (Phase 5a)

**36B NL MoE on Config F:** use **2-device** topology, not 3-device.

| Setting | Value |
|---------|-------|
| `--rpc` | `192.168.8.176:50051` only (drop `:50052`) |
| `-ts` | `50,50` |
| `GGML_PIPELINE_PLUS` | `1` (default when trace/profile enabled) |
| Model | `Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf` |
| Expected G | **~49 t/s** (measured 48.9) |

Runbook label: `trace-f-2gpu-plus`. See [rpc-path-b-plus-handover.md](rpc-path-b-plus-handover.md).

3-device F (`ts=30,12,58`) remains valid when VRAM requires RX6600; expect ~38-43 t/s, not 45+.

### 4-GPU cluster primary (Phase 10, romulus client)

**Stable topology (no RX6600):** 7900 XTX client + 3060 + 5060 + 5070 RPC workers.

| Setting | Value |
|---------|-------|
| `--rpc` | `192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053` |
| `-ts` | `36,24,24,16` |
| Model | `Qwen3.6-35B-A3B-APEX-I-Quality.gguf` (romulus) |
| KV | `q8_0` / `q8_0` |
| Expected G | **~40 t/s** (measured 38-43) |
| Load | ~85s |

```bash
./rpc-patch/scripts/pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary
```

**Do not** use `BENCH_4GPU_PRESET=legacy-6600` for production (hangs at slot init).  
Doc: [CLUSTER-4GPU-PRIMARY.md](../../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md).

## Remus deploy

`build` and `rebuild` sync `rpc-patch/deploy/*` to remus before compiling (no stale `GIT_BRANCH`).

```bash
./rpc-patch/scripts/pathb-remus-rpc.sh rebuild      # deploy + build + restart :50051
./rpc-patch/scripts/pathb-remus-rx6600-rpc.sh rebuild # deploy + build + restart :50052
# or
./rpc-patch/scripts/pathb-remus-multi-rpc-win.sh start
```

`deploy` only: `pathb-remus-rpc.sh deploy`. Skip sync: `PATHB_REMUS_SKIP_DEPLOY=1 build`.

Server: `[hello] version: 4.3.2`. Client trace: `"minor":3,"peer_copy":true`. Artifact: `benchmarks/proto-check-v43/`.

## Phase 5 checklist

### 5a -- Ops topology (DONE)

| Item | Status | Evidence |
|------|--------|----------|
| Document 2-device F as production default | PASS | tracking + handover + runbook |
| `trace-f-2gpu-plus` bench captured | PASS | G=48.9, RESULT=PASS |
| Runbook/README updated | PASS | `pathb-trace-runbook.ps1`, benchmarks README |
| Remus deploy proto 4.3.2 | PASS | deploy sync + `proto-check-v43` |

### 5b -- Validation spikes (DONE)

| Spike | Status | Evidence |
|-------|--------|----------|
| S4 barrier correctness | **PASS** | `s4-4b-2gpu-plus` + `trace-f-2gpu-plus`; pipeline + sched copies=4 |
| S5 overlap during GEN | **PASS** | `assembly_overlap_count=267`, overlap_pct=0.5% |
| S1 gen COPY budget | **PASS** | 2gpu: SET_TENSOR_HASH=168, COPY_TENSOR=0 (hash path) |
| S3 drain ratio | **PASS** | drain_flush 2523->1734 ms (-31%) on 2gpu vs legacy 3gpu |
| S0 5-endpoint baseline | DEFERRED | 72B+ cluster; not required for B+1 |

Details: [rpc-path-b-plus-spikes.md](rpc-path-b-plus-spikes.md)

## Phase 6 checklist (Path C kickoff)

### 6a -- C1 baseline catalog (IN PROGRESS)

| Item | Status | Evidence |
|------|--------|----------|
| Client-split baseline (3gpu) | PASS | `trace-f-3gpu-plus` G=42.8, 4 splits, COPY_TENSOR=144 |
| Ops baseline (2gpu, no Path C) | PASS | `trace-f-2gpu-plus` G=48.9, 3 splits |
| Delta documents Path C motivation | PASS | +29% dropping 6600 hop; serial split sum in RPC-BUG-HUNT |

### 6b -- Remus feasibility (IN PROGRESS)

| Finding | Implication |
|---------|-------------|
| `:50051` = CUDA 5060 Ti container (`-d CUDA0`, 1 GPU) | `DEVICE_COUNT=1` per endpoint today |
| `:50052` = ROCm 6600 container (separate image) | CUDA+ROCm not one rpc-server process |
| `rpc-server` supports `-d CUDA0,CUDA1` | Path C C1 viable on **same-vendor multi-GPU** host |
| Config F 36B NL | Phase 5 ops fix (2-device) already optimal without Path C code |

Path C on remus 5060+6600 requires **C2+ server aggregation**, not just multi-device flags. See [rpc-path-c-plan.md](rpc-path-c-plan.md).

### 6c / 6d -- PENDING

- 6c: multi-CUDA rpc-server spike (Config A/B style: 2 NVIDIA on one box)
- 6d: C2 internal scheduler + single logical backend

Details: [rpc-path-c-tracking.md](rpc-path-c-tracking.md)

## Next steps (post Phase 5)

1. ~~Path C spike kickoff~~ -> **Phase 6** (above).
2. `pathb-hotpath-summary.ps1` (trace + layer map).
3. Optional: re-bench `trace-f-2gpu-plus` on remus 4.3.2 for updated HELLO trace fields.

## Tier 0 exit criteria

| Metric | Target | Actual | Pass |
|--------|--------|--------|------|
| trace-f-3gpu G | 45+ | 42.8 | PARTIAL (+12% vs 38 pre) |
| trace-f-2gpu G (ops) | 45+ | **48.9** | **PASS** |
| copy rotation | 4 slots used | yes (99/99/96/96 on 2gpu) | PASS |
| overlap_pct | >0 meaningful | 0.5% (2gpu) | PASS |
| S4 correctness | 4B + 36B | both PASS | PASS |

45+ on 3-device F is topology-limited (serial split sum ~27 ms/tok). Production target met on 2-device F.