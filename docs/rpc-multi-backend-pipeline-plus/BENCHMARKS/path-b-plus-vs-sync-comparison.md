# Path-B+ vs atomic sync baseline — apples-to-apples comparison

**Status:** Matrix defined; cells **TBD** until `b6-gate-sync-vs-plus-comparison.sh` completes.  
**Date:** 2026-07-01  
**Navigation:** [BENCHMARKING.md](../../../BENCHMARKING.md) | [2026-07-comparison-matrix.md](2026-07-comparison-matrix.md) | [TRACKING.md](../TRACKING.md)

---

## 1. Purpose

Publish a **direct deployment comparison** between:

| Arm | Name | What it represents |
|-----|------|-------------------|
| **SYNC** | Atomic sync baseline | Fork branch **without** Path-B+ pipeline stack (sync RPC, no `GGML_PIPELINE_PLUS`, no B+11–B+15 prod tuning) |
| **PLUS** | Path-B+ production | `Path-B-Event-Support-Pipeline-Plus` @ deploy tag, `b6-*-g-prod` env, equal-safe preflight where required |

**Topologies:** 3-GPU (`b6-3gpu-g-triton`) and 5-GPU (`b6-5gpu-g-prod`).

**Goal:** One chart file operators can cite for "what did Path-B+ buy us on real cluster models?"

---

## 2. Arm definitions (locked for fair compare)

### 2.1 SYNC arm (baseline deploy)

| Setting | Value |
|---------|-------|
| Git ref | `B6_SYNC_GIT_REF` default **`feature/turboquant-kv-cache`** @ pinned SHA (record in each row) |
| Client build | `build-rocm-docker-sync/` on romulus (separate from Plus build) |
| RPC workers | rpc-server built from **same SYNC SHA** on remus + triton + romulus docker |
| `GGML_PIPELINE_PLUS` | **0** (or N/A if branch predates flag) |
| `GGML_RPC_DUAL_SOCKET` | **0** |
| RPC events / pipeline | **Off** on branches before Path-B; on turboquant fork = synchronous RPC path |
| KV default | `-ctk q8_0 -ctv turbo3` (match Plus arm) |
| Preflight | `PATHB_VRAM_PREFLIGHT=1` when load fails (same policy as Plus) |

**Note:** If SYNC SHA still has Path-B events but not Plus, document actual caps in `env.txt` per run. Prefer a SHA **before** `Path-B-Event-Support` merge for true pre-pipeline baseline; use in-branch `GGML_PIPELINE_PLUS=0` as **Tier 1** quick check (see s2.3).

### 2.2 PLUS arm (production)

| Setting | Value |
|---------|-------|
| Git ref | `Path-B-Event-Support-Pipeline-Plus` @ **`path-b-plus-deploy-*`** tag or `9121d16d4+` |
| Client build | `build-rocm-docker/` |
| RPC workers | Path-B+ rpc-server (proto 4.4.x), prod matrix validated |
| Env | `source scripts/b6-gate-5gpu-production-env.sh` for 5-GPU; 3-GPU per `b6-gate-run-remote.sh` |
| `GGML_PIPELINE_PLUS` | **1** |
| `GGML_RPC_DUAL_SOCKET` | **1** on 5-GPU prod label ( **0** on 3-GPU triton unless bisect says otherwise) |
| Wavefront / hash defer | **OFF** (prod defaults) |
| Preflight | `--ts-mode equal --phase load` for 70B+ / kimi on 5-GPU |

### 2.3 Tier 1 vs Tier 2 (optional fast path)

| Tier | Compare | When to use |
|------|---------|-------------|
| **T1** | Same Plus branch SHA: `GGML_PIPELINE_PLUS=0` vs prod env | Isolates Plus layer only; 1 binary |
| **T2** | SYNC branch SHA vs Plus branch SHA | **Primary** for "deployment" story |

This matrix uses **T2** unless `B6_COMPARE_TIER=1`.

---

## 3. Model roster (this campaign)

Subset from Tier A archetype matrix ([`b6-gate-model-archetype-matrix.sh`](../../../scripts/b6-gate-model-archetype-matrix.sh)).

| ID | Class | Model (romulus path) | Speculative | 3-GPU | 5-GPU |
|----|-------|----------------------|-------------|-------|-------|
| **M1** | Qwen3.6 MoE ~35B | `Qwen3.6-35B-A3B-APEX-I-Quality.gguf` | NextN optional | yes | yes |
| **M2** | Qwen3.6 + NextN | same + `--spec-type nextn` (server) | **NextN on** | yes | yes |
| **M3** | Gemma 4 MoE | `gemma-4-26B-A4B-APEX-I-Compact.gguf` | MTP optional | yes | yes |
| **M4** | Gemma 4 + MTP | same + MTP draft (server) | **MTP on** | yes | yes |
| **M5** | ~30B dense | `Qwen3.5-27B-Q5_K_M.gguf` | off | yes | optional |
| **M6** | Llama 70B | `meta-llama-3-70b-instruct.Q4_K_M.gguf` | off | optional | yes |
| **M7** | Qwen-Next 80B | `Qwen3-Next-80B-A3B-Instruct-Q5_K_M.gguf` | NextN | optional | yes |
| **M8** | Kimi 72B | `Kimi-Dev-72B-IQ4_XS.gguf` | off | optional | yes |

**Profiler path (default):** M1, M3, M5, M6, M7, M8 — target decode, no spec draft.  
**Server path (MTP/NextN rows):** M2, M4 — `llama-server` + HTTP bench; see [BENCHMARKING.md](../../../BENCHMARKING.md) T0/T2.

---

## 4. Workloads (both required for Tier 2)

| Workload ID | `BENCH_GEN_TOKENS` | Prompt file | Purpose | Trace sample |
|-------------|-------------------|-------------|---------|--------------|
| **n384** | 384 | `prompts/profiler-reasoning-long.txt` | Canonical B+6 gate depth | 5 |
| **n2048-mt** | 2048 | `prompts/profiler-hard-multiturn.txt` | Long multi-turn steady-state (6 user turns in prompt) | 25 |

Pattern matches `b6-gate-retest-matrix.sh` (`R3` n384, `R4` n2048 multiturn on 5-GPU prod).

**5-GPU preflight phase:** `load` for n384; **`decode`** for n2048-mt (post-prefill VRAM).  
**Profiler `n_ctx`:** auto-sized from prompt + gen in `llama-pipeline-profiler`.

Default orchestrator workloads: `B6_COMPARE_WORKLOADS=n384,n2048-mt`.

**Cell count (profiler, default models):** 3-GPU (M1,M3,M5) + 5-GPU (M1,M3,M5,M6,M7,M8) x 2 arms x 2 workloads = **36 cells** (`DRY_RUN=1` confirms).

---

## 5. Run contract (invariant across arms)

| Field | Value |
|-------|-------|
| Harness | `llama-pipeline-profiler` via `b6-gate-run-remote.sh` (decode); server for spec rows |
| Client | romulus native ROCm (`build-rocm-docker` + `build-rocm-docker-sync`) |
| Workloads | **n384** + **n2048-mt** (both) |
| Runs | 1 profiler run per cell |
| Metrics | **G (t/s)** primary; `stall_ratio`, `overlap_pct`, `blocking_ms`, load PASS/FAIL |
| Metadata | `git_sha`, `arm`, `topology`, `model_id`, `workload`, `env.txt` per out dir |
| Git checkout order | workload -> **arm** -> topo -> model (minimize branch switches) |

---

## 6. Master comparison chart

**Legend:** G = generation t/s (profiler steady-state unless noted). `TBD` = not run yet.  
**Delta%** = `(PLUS - SYNC) / SYNC * 100`.  
Generated tables: [path-b-plus-vs-sync-results.md](path-b-plus-vs-sync-results.md) (after jsonl fill).

### 6.1 Workload `n384` — gate depth

#### 3-GPU `b6-3gpu-g-triton`

| Model ID | Model | SYNC G | PLUS G | Delta% | SYNC load | PLUS load |
|----------|-------|--------|--------|--------|-----------|-----------|
| M1 | Qwen3.6-35B-A3B | TBD | TBD | TBD | TBD | TBD |
| M3 | Gemma-4-26B-A4B | TBD | TBD | TBD | TBD | TBD |
| M5 | Qwen3.5-27B | TBD | TBD | TBD | TBD | TBD |

#### 5-GPU `b6-5gpu-g-prod`

| Model ID | Model | SYNC G | PLUS G | Delta% | SYNC load | PLUS load | Notes |
|----------|-------|--------|--------|--------|-----------|-----------|-------|
| M1 | Qwen3.6-35B-A3B | TBD | ~59.7* | TBD | TBD | PASS | *B+14 n384 ref |
| M3 | Gemma-4-26B-A4B | TBD | TBD | TBD | TBD | TBD | |
| M5 | Qwen3.5-27B | TBD | TBD | TBD | TBD | TBD | |
| M6 | Llama-3-70B Q4 | TBD | TBD | TBD | TBD | PASS* | *L4 equal-safe |
| M7 | Qwen3-Next-80B | TBD | TBD | TBD | TBD | TBD | |
| M8 | Kimi-Dev-72B | TBD | TBD | TBD | TBD | PASS* | *L4 equal-safe |

### 6.2 Workload `n2048-mt` — long multi-turn

#### 3-GPU `b6-3gpu-g-triton`

| Model ID | Model | SYNC G | PLUS G | Delta% | Notes |
|----------|-------|--------|--------|--------|-------|
| M1 | Qwen3.6-35B-A3B | TBD | TBD | TBD | 6-turn hard prompt |
| M3 | Gemma-4-26B-A4B | TBD | TBD | TBD | |
| M5 | Qwen3.5-27B | TBD | TBD | TBD | |

#### 5-GPU `b6-5gpu-g-prod`

| Model ID | Model | SYNC G | PLUS G | Delta% | Notes |
|----------|-------|--------|--------|--------|-------|
| M1 | Qwen3.6-35B-A3B | TBD | TBD | TBD | ref: retest R4 pattern |
| M3 | Gemma-4-26B-A4B | TBD | TBD | TBD | |
| M5 | Qwen3.5-27B | TBD | TBD | TBD | |
| M6 | Llama-3-70B Q4 | TBD | TBD | TBD | preflight `--phase decode` |
| M7 | Qwen3-Next-80B | TBD | TBD | TBD | |
| M8 | Kimi-Dev-72B | TBD | TBD | TBD | |

### 6.3 Server + speculative (MTP / NextN) — separate campaign

Same topologies; **HTTP** `rpc-server-bench.sh` or gemma/qwen server scripts. Depth-2 on unless `LLAMA_PIPELINE_DEPTH2=0` bisect row needed.

| Model ID | Spec | Topology | SYNC G | PLUS G | Delta% | Harness |
|----------|------|----------|--------|--------|--------|---------|
| M2 | NextN on | 3-GPU | TBD | TBD | TBD | qwen NextN server |
| M2 | NextN on | 5-GPU | TBD | TBD | TBD | |
| M4 | MTP on | 3-GPU | TBD | TBD | TBD | `run-gemma4-mtp-server.sh` |
| M4 | MTP on | 5-GPU | TBD | TBD | TBD | |

### 6.4 Summary rollup (fill after campaign)

| Workload | Topology | Cells | SYNC wins | PLUS wins | Median Delta% |
|----------|----------|-------|-----------|-----------|---------------|
| n384 | 3-GPU | 3 | TBD | TBD | TBD |
| n384 | 5-GPU | 6 | TBD | TBD | TBD |
| n2048-mt | 3-GPU | 3 | TBD | TBD | TBD |
| n2048-mt | 5-GPU | 6 | TBD | TBD | TBD |
| spec server | both | 4 | TBD | TBD | TBD |

**Runtime note:** n2048-mt cells are **~5x wall** vs n384 per model; run n384 matrix first, then n2048-mt overnight batch.

---

## 7. How to run

### 7.1 One-shot orchestrator (profiler cells)

```bash
# Dry-run matrix (36 cells: 2 workloads x sync/plus x topo x models)
DRY_RUN=1 bash scripts/b6-gate-sync-vs-plus-comparison.sh

# Full Tier 2 campaign (both workloads)
B6_COMPARE_TIER=2 \
B6_COMPARE_WORKLOADS=n384,n2048-mt \
B6_SYNC_GIT_REF=feature/turboquant-kv-cache \
B6_SYNC_SHA=<pin> \
B6_PLUS_SHA=9121d16d4 \
bash scripts/b6-gate-sync-vs-plus-comparison.sh

# n384 only (faster first pass)
B6_COMPARE_WORKLOADS=n384 bash scripts/b6-gate-sync-vs-plus-comparison.sh

# Long multiturn only
B6_COMPARE_WORKLOADS=n2048-mt bash scripts/b6-gate-sync-vs-plus-comparison.sh

# Subset
B6_COMPARE_MODELS=M1,M6,M8 \
B6_COMPARE_TOPOLOGIES=b6-5gpu-g-prod \
bash scripts/b6-gate-sync-vs-plus-comparison.sh
```

### 7.2 Manual single cell

```bash
# PLUS
source scripts/b6-gate-5gpu-production-env.sh
BENCH_MODEL=/mnt/models/meta-llama-3-70b-instruct.Q4_K_M.gguf \
  bash scripts/b6-gate-run-remote.sh b6-5gpu-g-prod --no-warmup

# SYNC (in-branch T1 quick)
GGML_PIPELINE_PLUS=0 GGML_RPC_DUAL_SOCKET=0 B6_PERF_AUTO=0 \
BENCH_MODEL=/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf \
  bash scripts/b6-gate-run-remote.sh b6-3gpu-g-triton --no-warmup
```

### 7.3 Publish results

```bash
# n2048 multiturn manual (5-GPU prod)
source scripts/b6-gate-5gpu-production-env.sh && b6_5gpu_production_env
BENCH_GEN_TOKENS=2048 \
BENCH_PROMPT_FILE=benches/path-b-plus/prompts/profiler-hard-multiturn.txt \
BENCH_MODEL=/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf \
  bash scripts/b6-gate-run-remote.sh b6-5gpu-g-prod --no-warmup --trace-sample 25
```

### 7.4 Regenerate markdown from jsonl

After runs, regenerate tables:

```bash
python3 scripts/b6-gate-sync-vs-plus-comparison.py \
  --jsonl benches/path-b-plus/sync-vs-plus-comparison.jsonl \
  --out docs/rpc-multi-backend-pipeline-plus/BENCHMARKS/path-b-plus-vs-sync-comparison.md
```

Or append rows to `sync-vs-plus-comparison.jsonl` and update s5 manually.

---

## 8. Preconditions checklist

- [ ] SYNC SHA pinned; rpc-servers on remus/triton/romulus docker rebuilt from that SHA
- [ ] PLUS at deploy tag; validate-rpc matrix PASS
- [ ] Model files present on romulus `/mnt/models/`
- [ ] 5-GPU: equal-safe preflight run for M6/M7/M8 before PLUS arm
- [ ] Two build dirs on romulus: `build-rocm-docker` (plus), `build-rocm-docker-sync` (sync)
- [ ] Record `B6_SYNC_SHA` and `B6_PLUS_SHA` in jsonl for every row

---

## 9. Artifact index

| Path | Contents |
|------|----------|
| `benches/path-b-plus/sync-vs-plus-comparison.jsonl` | One JSON row per cell |
| `benches/path-b-plus/cmp-*` | Per-cell profiler out dirs |
| `scripts/b6-gate-sync-vs-plus-comparison.sh` | Orchestrator |
| `scripts/b6-gate-sync-vs-plus-comparison.py` | Table refresh from jsonl |

---

## 10. References (existing Plus-only data)

| Label | Topology | G (t/s) | Notes |
|-------|----------|---------|-------|
| `trace-f-2gpu-plus` | 2-GPU Win | 48.9 | Production champion (not 3/5 GPU) |
| `b6-2gpu-f-triton-n384-romulus-native` | 2-GPU | 200.8 | Plus only |
| B+14 5-GPU A1 n=384 | 5-GPU | ~59.7 | Plus only |
| `trace-f-3gpu-legacy` vs `plus` | 3-GPU Win | 39.3 vs 42.8 | Plus A/B same SHA (T1) |

---

**Last updated:** 2026-07-01 — matrix template; campaign not executed.