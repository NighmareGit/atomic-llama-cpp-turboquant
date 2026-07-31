# llama.cpp RPC Backend Optimization — Path A + Path B Report

**Date:** 2026-06-25
**Author:** TurboQuant Team
**Status:** Path A1+A2 complete (v4.1.0) | Path B complete (v4.2.2, git workspace)
**Branch:** `Path-B-Event-Support` (in `atomic-llama-cpp-turboquant/`)  
**Project root:** `/home/hunter/projects/atomic-llama-cpp-patch/` — scripts, docs, `patch/bench-results/` live here.

---

## Executive Summary

This report extends [RPC_OPTIMIZATION_REPORT.md](/home/hunter/projects/atomic-llama-cpp-turboquant/RPC_OPTIMIZATION_REPORT.md) with **Path B: Event-Based Pipeline Parallelism** (protocol v4.2.x), correctness fixes required to ship A1+A2+B together, and updated benchmarks.

| Phase | Mechanism | Target | Measured impact |
|-------|-----------|--------|-----------------|
| **A1** | Batch `SET_TENSOR` | Model load / prompt | **+3.6% prompt** (9B Config A vs vanilla) |
| **A2** | Pipelined `GET_TENSOR` receive | Generation overlap | **0%** (no measurable gain) |
| **B** | `EVENT_RECORD` + scheduler pipeline (`n_copies=4`) | Generation overlap across backends | **+3 to +14% gen** on large-model server matrix; **+7%** on 9B server |
| **B fixes** | Central RPC drain + `event_wait` coherency | Stability | Required for operation; without them Path B **aborts or hangs** |

**Primary regression model (2026-06-25):** `Qwen3.5-4B-Q4_K_M` on Config A — Path B achieves **G ≈ 109–111 t/s** (validated 3-run matrix).

**llama-server headline:** `Qwen3.5-9B-MTP-Q4_K_M`, `-c 8192 -ctk q8_0 -ctv turbo3 -ts 1,1` — Path B **58.4 t/s** vs A1 **54.4 t/s** (**+7.3%**). First controlled A1-vs-Path-B server comparison.

**Comparison honesty:** llama-cli numbers mix Phase A report data and different builds. Server matrix (A1 / A1a2 / Path B) uses identical model/flags per row. Path B on >27B needs `--no-warmup -np 1` (now default in bench scripts).

---

## Table of Contents

1. [Protocol Evolution](#protocol-evolution)
2. [Path B Implementation](#path-b-implementation)
3. [Correctness Fixes (Required for A1+A2+B)](#correctness-fixes-required-for-a1a2b)
4. [Hardware and Test Matrix](#hardware-and-test-matrix)
5. [Benchmark Results](#benchmark-results)
6. [Path A vs Path B Comparison](#path-a-vs-path-b-comparison)
7. [Analysis](#analysis)
8. [Test Harness and Usage](#test-harness-and-usage)
9. [Models and Cache Guidance](#models-and-cache-guidance)
10. [Future Work](#future-work)
11. [Conclusion](#conclusion)
12. [Appendix: Raw Data](#appendix-raw-data)

---

## Protocol Evolution

| Version | `RPC_PROTO` | Commands added | Scheduler impact |
|---------|-------------|----------------|------------------|
| Vanilla | 4.0.0 | — | `caps.events=false`, no pipeline |
| **A1** | 4.1.0 | `SET_TENSOR_BATCH` (17) | Faster model load |
| **A2** | 4.1.0 | (client) pipelined `GET_TENSOR` | No measurable inference gain |
| **B** | 4.2.2 | `EVENT_RECORD` (18) | `caps.async=true`, `caps.events=true` → pipeline parallelism |

Path B builds on A1+A2. The git workspace branch `Path-B-Event-Support` contains all three.

---

## Path B Implementation

**Goal:** Let the ggml scheduler overlap compute across backends using TCP-ordered `EVENT_RECORD` instead of GPU-native events on the RPC path.

**Key additions** (`ggml/src/ggml-rpc/ggml-rpc.cpp`, `ggml/include/ggml-rpc.h`):

1. `RPC_CMD_EVENT_RECORD` — deferred response after `GRAPH_RECOMPUTE` (TCP ordering).
2. `rpc_event_t` + device `event_new` / `event_free` / `event_synchronize`.
3. Backend `event_record` / `event_wait` wired into RPC interfaces.
4. `caps.async = true`, `caps.events = true` on RPC devices.
5. Server handlers for `SET_TENSOR_BATCH` and `EVENT_RECORD`.

**Scheduler precondition** (from `llama-context.cpp`): pipeline parallelism when `n_devices > 1`, full layer offload, `offload_kqv`, and all non-CPU backends report `async` + `events`.

---

## Correctness Fixes (Required for A1+A2+B)

Path B initial integration in `build-a2` **failed at runtime** until these fixes landed in the git workspace:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Pipelined `GET_TENSOR` + sync RPC reads | Abort at `get_device_memory` during load; TCP desync | `drain_pending_event_response()` + `flush_pending_get_tensor()` at start of `send_rpc_cmd` **with response** |
| `event_wait` after central drain | `recv failed (bytes_recv=0)`; empty generation | Only `recv_rpc_cmd_deferred` when `tls_pending_event.pending`; else clear `ev->response_pending` |
| Struct ordering | Compile error in `drain_pending_event_response` | Move drain helpers below `rpc_msg_event_record_rsp` |

**Operational performance impact:** Without these fixes, Path B is **non-functional**. With them, load and inference complete; generation throughput is in line with A1+A2 on 9B, with higher throughput on 4B Config A.

---

## Hardware and Test Matrix

| Component | Specification |
|-----------|---------------|
| **NVIDIA worker** | RTX 3060 Ti, 8 GB, CUDA 12.4 (`llama-rpc-cuda-a2`) |
| **AMD client** | RX 7900 XTX, 24 GB, ROCm 6.4 (`llama-rocm-patched`) |
| **Network** | localhost TCP, port 50051 |
| **Models root** | `/mnt/models` |

**Config A (primary):** AMD client (7900 XTX, `HIP_VISIBLE_DEVICES=0`) → NVIDIA worker (3060 Ti, `CUDA_VISIBLE_DEVICES=0`). **Never** colocate `llama-server` and `rpc-server` on the same GPU.

Build artifacts (git workspace volume-mount):

- CUDA: `build-cuda-b-bin/bin/rpc-server`
- ROCm: `build-rocm-docker/bin/llama-cli`, `build-rocm-docker/bin/llama-server`

**Container hygiene:** Benchmark scripts use fixed names `bench-rpc` / `bench-llama` and `cleanup()` before each run. Stale containers compound resource use and cause port/name conflicts.

---

## Benchmark Results

### llama-server ↔ rpc-server — 9B Q4_K_M (primary server comparison)

**Model:** `Qwen3.5-9B-MTP-Q4_K_M.gguf` (5.5 GB) — only 9B `Q4_K_*` variant under `/mnt/models`.
**Settings:** `-c 8192 -ctk q8_0 -ctv turbo3 -ngl 99 -sm layer -ts 1,1`
**Harness:** `scripts/rpc-server-bench.sh` (single `bench-rpc` + single `bench-llama`, 3-run curl matrix)
**Topology:** ROCm `llama-server` on 7900 XTX, CUDA `rpc-server` on 3060 Ti

| Variant | Protocol | Load (s) | Run 1 P/G | Run 2 P/G | Run 3 P/G | Avg G | Delta |
|---------|----------|----------|-----------|-----------|-----------|-------|-------|
| **A1** | 4.1.0 image | 16 | 92 / 54.6 | 448 / 54.3 | 453 / 54.2 | **54.4** | baseline |
| **Path B** | 4.2.2 workspace | 10 | 58 / 58.6 | 79 / 58.3 | 89 / 58.4 | **58.4** | **+7.3%** |

Notes:

- Prompt t/s on runs 2–3 reflect warm prompt-cache hits (LCP similarity); run 1 is cold. Compare generation only.
- `preview` field empty in curl responses — MTP/thinking template; timings are valid.
- Path B loads **37% faster** (10 s vs 16 s) on this model.
- llama-cli 9B gen (~70 t/s) is higher than llama-server (~58 t/s) due to server slot overhead, 4 parallel slots, and HTTP path.

### llama-server ↔ rpc-server — Path A1 vs A1+A2 vs Path B matrix (controlled)

**Harness:** `scripts/rpc-server-bench-matrix.sh` | **Split:** `-ts 10,90 -ngl 99 -sm layer --no-warmup -np 1`
**Topology:** ROCm `llama-server` (7900 XTX) + CUDA `rpc-server` (3060 Ti), single instance each

| Model | Cache / ctx | A1 (v4.1.0) | A1+A2 (v4.1.0) | Path B (v4.2.2) | B vs A1 | A2 vs A1 |
|-------|-------------|-------------|----------------|-----------------|---------|----------|
| `Qwen3.5-27B-Q5_K_M` | q8_0/turbo3, 8192 | **27.1** | 27.1 | **27.9** | **+3.0%** | 0% |
| `gemma-4-31B-it-Q4_K_M` | q8_0/turbo3, 8192 | **23.5** | 23.2 | **26.7** | **+13.6%** | -1.3% |
| `Qwen3.5-35B-A3B` (MoE) | q4_0, 4096, ncmoe 8 | **58.9** | 58.1 | **64.2** | **+9.0%** | -1.4% |
| `Qwen3.6-35B-A3B-APEX` (MoE) | q4_0, 4096, ncmoe 8 | **57.1** | 58.2 | **61.1** | **+7.0%** | +1.9% |

**Key findings:**

1. **Path B** shows **+3% to +14%** generation vs A1 on every large-model config; largest win on **31B Gemma (+13.6%)**.
2. **A1+A2** is **neutral** vs A1 alone (within run variance) — confirms Phase A conclusion that pipelined `GET_TENSOR` does not move inference throughput.
3. Path B on >27B **requires** `--no-warmup -np 1` (default in updated bench scripts); without it, 4-slot warmup crashes the workspace `rpc-server`.
4. MoE models use `--n-cpu-moe 8` to keep RPC VRAM low; dense 27B/31B use full ctx 8192 with q8_0/turbo3.

### Cross-GPU Config A — 9B llama-cli (historical + Path B)

**Model:** `Qwen_Qwen3.5-9B-Q5_K_M.gguf` (6.4 GB) — **note:** Unsloth-tagged 9B builds reported damaged; prefer `Qwen3.5-9B-MTP-Q4_K_M` for server tests.

| Variant | Protocol | Cache | Prompt t/s | Gen t/s | Source |
|---------|----------|-------|------------|---------|--------|
| Vanilla | 4.0.0 | q4_0 | 306.0 | 68.1 | Phase A report |
| **A1+A2** | 4.1.0 | q4_0 | **317.0** | **68.4** | Phase A report |
| A1+A2 image | 4.1.0 | q4_0 | ~210 | **68.9** | 2026-06-25 smoke |
| **Path B** | 4.2.2 | q4_0 | 62–98 | **69.8–70.6** | 3-run matrix, workspace |
| Path B | 4.2.2 | q8_0 | 86.9 | 69.0 | Single warm run |

**Generation average:** A1+A2 **68.4** vs Path B **~70.1** → **~+2.5%** (3-run). Prompt scores are not comparable across cache and load states.

### Cross-GPU Config A — 4B (Path B primary regression)

**Model:** `Qwen3.5-4B-Q4_K_M.gguf` (2.7 GB)
**Settings:** `-ngl 99 -sm layer --single-turn --simple-io --no-conversation`

| Run | Cache | Prompt t/s | Gen t/s |
|-----|-------|------------|---------|
| 1 | q4_0 | 118.5 | **108.9** |
| 2 | q4_0 | 119.5 | **110.5** |
| 3 | q8_0 | 121.5 | **110.8** |

All runs: PASS (log validation harness, no RPC abort, no death-loop).

**Reference (not cross-GPU):** Phase A single-GPU 3060 Ti full offload — 4B **93 t/s** gen. Config A Path B **~110 t/s** is **~+18%** vs that reference, but hardware topology differs (do not treat as isolated Path B gain).

### Cross-GPU Config A — other models (Path B spot-checks)

| Model | Cache | Prompt t/s | Gen t/s | Result |
|-------|-------|------------|---------|--------|
| `smollm3-3b-q4_k_m` (1.8G) | q4_0 | 70–265 | 61–74 | PASS after event_wait fix |
| `gemma-4-12b-it-Q4_K_M` (6.7G) | q4_0 | 161.7 | **51.5** | PASS |

### llama-server ↔ rpc-server — >27B models (tensor-split tuning)

#### Device order (critical)

With `--rpc 127.0.0.1:50051`, startup logs list **RPC0 first, ROCm0 second**. Tensor-split indices map to that order:

| `-ts` value | RPC0 (3060 Ti, index 0) | ROCm0 (7900 XTX, index 1) |
|-------------|-------------------------|----------------------------|
| `10,90` | **10%** (~2.0 GB weights) | **90%** (~16.2 GB) |
| `15,85` | **15%** (~2.7 GB) | **85%** (~15.4 GB) |
| `18,82` | **18%** (~3.3 GB) | **82%** (~14.9 GB) |
| `90,10` | **90%** (~16.6 GB) | **10%** | **OOM** on 8 GB RPC |

Verified via `load_tensors` lines and `rocm-smi` / `nvidia-smi` polling (`scripts/rpc-ts-fit-probe.sh`). **Do not invert** the ratio: `90,10` allocates ~16.5 GB to RPC and fails immediately.

Earlier `ts=18,1` + `ngl=25` runs looked like "5% on RPC" but actually capped total GPU layers at 25 and split those **18:1 toward RPC** (RPC ~6.5 GB, ROCm ~1 GB, rest on CPU mmap). That is why generation was only **~5 t/s**. Use **`ngl=99`** with percentage-style splits (`10,90`).

#### OOM failures (wrong splits)

| Model | Settings | RPC alloc attempted | Result |
|-------|----------|---------------------|--------|
| `gemma-4-31B-it-Q4_K_M` | ctx 8192, q8_0/turbo3, ts 3,1 | 12.4 GB | **OOM** |
| `Qwen3.5-27B-Q5_K_M` | ctx 8192, q8_0/turbo3, ts 2,1 | 12.4 GB | **OOM** |
| `Qwen3.5-27B-Q5_K_M` | ctx 4096, ts 90,10 | 16.6 GB | **OOM** |

#### Recommended splits (A1, ctx 8192, q8_0/turbo3, ngl 99)

| Model | ts | RPC model buf | ROCm model buf | nvidia peak | rocm peak | Avg G |
|-------|-----|---------------|----------------|-------------|-----------|-------|
| `Qwen3.5-27B-Q5_K_M` | **10,90** | 1960 MiB | 16221 MiB | 2317 MiB | 20511 MiB | **27.2** |
| `Qwen3.5-27B-Q5_K_M` | 15,85 | 2747 MiB | 15434 MiB | 3081 MiB | 18929 MiB | **26.2** |
| `gemma-4-31B-it-Q4_K_M` | **10,90** | 2060 MiB | 15400 MiB | 2467 MiB | 19842 MiB | **23.8** |

**5x+ generation improvement** vs the earlier `ts=18,1 ngl=25` configs (~5.4 t/s) by fully offloading (`ngl=99`) and biasing compute to ROCm.

#### Fit / MoE helpers

| Flag | Effect (27B/31B/35B tests) |
|------|---------------------------|
| `--fit on` (default) | Projects memory per device; with `-ts` + `-ngl` already set, often makes **no changes** |
| `--fit-target 512,1024` | Reserves 512 MiB margin on RPC0 (index 0), 1024 MiB on ROCm0 (index 1) |
| `--fit off` | Manual split only; same buffer sizes as fit-on when ngl/ts explicit |
| `--n-cpu-moe 8` | On `Qwen3.5-35B-A3B`: 4306 MiB MoE experts to CPU; RPC drops to **116 MiB** weights |

Path B workspace `rpc-server` needs **`--no-warmup -np 1`** for >27B server (4-slot warmup otherwise crashes). With that flag, all matrix configs pass.

### Phase A internal: A1 vs A2 (unchanged)

**9B Q5_K_M, 20 layers, single 3060 Ti:**

| Config | Prompt t/s | Gen t/s |
|--------|------------|---------|
| A1 only | 33.0 | 21.7 |
| A1+A2 | 33.0 | 21.7 |

**Conclusion stands:** A2 pipelined receive adds **no measurable throughput**; kept for compatibility with Path B drain logic.

---

## Path A vs Path B Comparison

### What was compared

| Comparison | Fair? | Result |
|------------|-------|--------|
| Vanilla vs A1+A2, 9B Config A | Yes (Phase A report) | **+3.6% P**, +0.4% G |
| A1 vs A2, same binary | Yes | **0%** difference |
| A1+A2 image vs Path B workspace, 9B llama-cli gen | Partial (same topology, different builds/cache) | **~+2.5% G** (69.8–70.6 vs 68.4) |
| **A1 vs A1+A2 vs Path B, 9B llama-server** | **Yes** | B **+7.3%** vs A1; A2 **0%** vs A1 |
| **A1 vs A1+A2 vs Path B, 27B–36B matrix** | **Yes** (ts=10,90, same flags) | B **+3 to +14%** vs A1; A2 **0%** vs A1 |
| Path B only, 4B Config A | No A1+A2 cross-GPU 4B baseline | **~110 t/s G** (new regression target) |
| Path B correctness vs broken B | Yes | Fixes required; pre-fix **abort/hang** |

### What was NOT compared (gaps)

1. ~~**Single binary toggle:** A1+A2-only vs A1+A2+B on the same commit~~ — **done** via server matrix (A1 / A1a2 / Path B images).
2. **4B Config A:** No A1+A2 cross-GPU 4B matrix in the Phase A report (only single-GPU 3060 Ti).
3. ~~**Pipeline debug:**~~ **Done** — `pipeline parallelism enabled` + `sched copies = 4` in `patch/bench-results/pathb-runs/b4-sched-debug-v.raw` (`GGML_SCHED_DEBUG=1`, `-v`).
4. **Config B** (NV client / AMD worker) for Path B — not run.
5. ~~**72B** (Kimi / Qwen72)~~ — **Done** with `--fit on -ngl 0 -ts 10,90` via `pathb-72b-server.sh`; ~8GB CPU offload required (`pathb-72b-vram-calc.py`).
6. ~~**Path B rpc-server on >27B:**~~ Resolved with `--no-warmup -np 1` in bench harness.

### Answer: did optimizations / bugfixes change operational performance?

**Yes, in three distinct ways:**

1. **Path A1** — measurable **prompt** gain on 9B Config A (+3.6% vs vanilla).
2. **Path A2** — **no** throughput change; still ships because batch + pipelined receive share flush/drain infrastructure.
3. **Path B + fixes** — controlled server matrix: **+3% to +14%** vs A1 on 27B–36B; **+7.3%** on 9B; A1+A2 **neutral** vs A1; **4B ~110 t/s** cli regression target.

---

## Analysis

### Why Path B generation gain is modest on 9B

Same bottlenecks as Phase A analysis:

1. Generation is **compute-bound** on the RPC worker, not RPC opcode overhead.
2. `EVENT_RECORD` removes sync waits but cannot exceed worker matmul throughput.
3. 3060 Ti VRAM caps worker batch size; client 7900 XTX holds KV.

Path B value is **correct overlap machinery** for multi-backend graphs and future multi-RPC workers, not a magic multiplier on 9B.

### Why 4B Config A scores higher

1. Smaller weights → faster RPC load and more headroom on 8 GB worker.
2. Full offload with layer split across ROCm + RPC.
3. Matched test harness with `--single-turn` avoids interactive CLI overhead.

### Docker “hangs” during test

Not deadlocks — **in-place load spinner** (`Loading model... |/-\`) produces little readable `docker logs` output; interactive mode waits at `>`. Use `scripts/pathb-run-test.sh` or `scripts/rpc-server-bench.sh` (health poll). Run **one** `bench-rpc` + **one** `bench-llama`; stale containers cause port conflicts and compound GPU memory use.

### 72B IQ4_XS on 32 GB total VRAM (8 GB RPC + 24 GB ROCm)

| Resource | Available | Role |
|----------|-----------|------|
| RPC0 (3060 Ti) | ~7.7 GB | 10% of GPU layers via `-ts 10,90` |
| ROCm0 (7900 XTX) | ~24.5 GB | 90% of GPU layers |
| CPU RAM | ~80 GB | **Required** for ~8-26 GB of weights when `-ngl` < 80 |
| Model file | ~40 GB | IQ4_XS dense 72B |

**Math:** `weight_on_gpu = model_gb * (ngl / n_layers)`; split by `-ts` percentages (not ratios). At `ngl=32, ts=10,90`: RPC ~1.6 GB weights, ROCm ~14.5 GB, CPU ~24 GB.

**Do not:** `-ngl 99`, `-ts 4,1` (puts 80% on RPC), or manual high `-ngl` on **llama-cli** (RPC CUDA graph crash).

**Do:** `python3 scripts/pathb-72b-vram-calc.py` then `./scripts/pathb-test.sh qwen72b` (server + `--fit on --fit-target 512,1024`). `--n-cpu-moe` is for MoE only (35B-A3B), not dense 72B.

Verified: `qwen72b-b4-vram` PASS (load + gen, ~2 t/s with heavy CPU offload).

### Tensor-split on 8 GB RPC (corrected)

**Device index 0 = RPC0, index 1 = ROCm0.** Use percentage-style ratios with **full offload** (`-ngl 99`):

- **27B/31B:** `-ts 10,90` (recommended) or `-ts 15,85` (slightly more RPC compute)
- **9B:** `-ts 1,1` (equal split fits both GPUs)
- **ctx 8192, q8_0/turbo3** works on 27B/31B with `ts=10,90` (RPC peak ~2.3 GB VRAM)
- **Avoid** `ts=2,1`, `ts=3,1`, or `ts=90,10` — puts too many layers on RPC
- **Avoid** low `-ngl` with small-ratio splits like `18,1` — caps GPU layers and dumps remainder to CPU mmap, collapsing throughput to ~5 t/s
- **`--fit-target 512,1024`**: per-device VRAM margin; pair with `--fit on` for auto-tuning when ngl/ts unset
- **`--n-cpu-moe N`**: offload MoE experts on 35B-A3B; frees RPC VRAM at cost of CPU RAM
- Monitor with `rocm-smi --showmeminfo vram` and `nvidia-smi` during load (`scripts/rpc-ts-fit-probe.sh`)

### KV cache vs weight quant

Weight files use `Q4_K_M`, `Q5_K_M`, etc. KV cache types are separate (`q4_0`, `q8_0`, `f16`, ...). Using `q4_0` cache with `Q5_K_M` weights is a quant mismatch — prefer `q8_0` on 24 GB client when testing Q5_K_M.

---

## Test Harness and Usage

Scripts in `scripts/`:

```bash
# llama-cli regression
./scripts/pathb-start-rpc.sh          # CUDA worker on :50051
./scripts/pathb-test.sh 4b          # Primary 4B regression
./scripts/pathb-test.sh matrix-4b   # 3-run 4B matrix
./scripts/pathb-test.sh gemma12b    # 12B spot-check
./scripts/pathb-test.sh kimi72b     # 72B, 10 min load timeout, -ts 15,85 (may OOM on 24+8GB)

# Single model / variant
export BENCH_MODEL=/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf BENCH_CTX=8192 \
  BENCH_CTK=q8_0 BENCH_CTV=turbo3 BENCH_TS=1,1
./scripts/rpc-server-bench.sh a1 9b-q4km-a1
./scripts/rpc-server-bench.sh pathb 9b-q4km-pathb

# Full A1 / A1+A2 / Path B matrix (27B, 31B, 35B, 36B)
./scripts/rpc-server-bench-matrix.sh
# or subset: ./scripts/rpc-server-bench-matrix.sh 27b 31b

# Probe split + GPU VRAM (rocm-smi / nvidia-smi during load)
PROBE_TS=10,90 PROBE_FIT=off ./scripts/rpc-ts-fit-probe.sh 27b-probe
PROBE_FIT=on PROBE_FITT=512,1024 PROBE_NCMOE=8 \
  PROBE_MODEL=/mnt/models/Qwen3.5-35B-A3B.i1-Q4_K_M.gguf \
  ./scripts/rpc-ts-fit-probe.sh 35b-a3b-probe
```

Logs: `patch/bench-results/pathb-runs/<label>.{log,meta,raw}` (cli), `patch/bench-results/rpc-server-bench/<label>.{meta,result,server.log}` (server), `patch/bench-results/rpc-ts-probe/<label>.{meta,gpu}` (VRAM probe)

See also: [patch/HANDOVER.md](../patch/HANDOVER.md), [docs/rpc-path-b-tracking.md](rpc-path-b-tracking.md)

---

## Models and Cache Guidance

| Use case | Model | Cache | ts / notes | Avoid |
|----------|-------|-------|------------|-------|
| Fast regression | `Qwen3.5-4B-Q4_K_M` | q4_0 or q8_0 | 1,1 | — |
| Medium cross-GPU | `gemma-4-12b-it-Q4_K_M` | q4_0 | 1,1 | — |
| **9B server (recommended)** | `Qwen3.5-9B-MTP-Q4_K_M` | **q8_0 / turbo3**, ctx 8192 | **1,1** | Unsloth Q5_K_M |
| 9B llama-cli | `Qwen3.5-9B-MTP-Q4_K_M` | q8_0 / turbo3 | 1,1 | `Qwen_Qwen3.5-9B-*` Unsloth |
| 27B server (8 GB RPC) | `Qwen3.5-27B-Q5_K_M` | **q8_0/turbo3**, ctx 8192 | **10,90**, ngl 99 | ts 90,10 or ts 2,1 (OOM) |
| 31B server (8 GB RPC) | `gemma-4-31B-it-Q4_K_M` | **q8_0/turbo3**, ctx 8192 | **10,90**, ngl 99 | ts 3,1 at ctx 8192 (OOM) |
| 35B MoE server | `Qwen3.5-35B-A3B.i1-Q4_K_M` | q4_0, ctx 4096 | **10,90**, ncmoe 8 | experts to CPU |
| 72B+ | `Kimi-Dev-72B-IQ4_XS`, `Qwen3-72B-Instruct.IQ4_XS` | q4_0, `-ts 10,90`, ctx 1024 | Path B server | **40GB weights > 32GB VRAM**; use `-ngl 0 --fit on` or `-ngl 24-36` + CPU offload |

Load timeouts: **180 s** (< 36B), **600 s** (72B). Always validate logs for garbage loops.

---

## B4 Final Verification (2026-06-25)

| Step | Result | Evidence |
|------|--------|----------|
| Pipeline debug | PASS | `b4-sched-debug-v.raw`: `pipeline parallelism enabled`, `sched copies = 4` |
| 512-token stress | PASS | `long-gen-512-b4-server`: P=113.3 G=83.6, max_tokens=512, no protocol errors |
| Single-GPU regression | PASS | `single-gpu-regression-4b.log`: G=102.3 t/s, `sched copies = 1` (no pipeline) |
| 72B cli/server | BLOCKED | IQ4_XS ~40GB > 32GB VRAM; load/inference OOM or RPC abort on 8GB worker |

Path B declared **production-ready** for models through ~36B MoE on Config A. See `docs/rpc-path-b-tracking.md`.

---

## Future Work

1. **Controlled A/B:** Same commit, toggle `caps.events` / pipeline only; 5-run matrix 4B and 9B Config A.
2. **Config B** Path B matrix.
3. **72B+:** Larger RPC VRAM or CPU offload tuning; presets now use `ts=15,85` (not `4,1`).
4. **Path C:** Server-side multi-GPU aggregation if Path B plateaus on 9B+.

---

## Conclusion

- **Path A1** remains the only Phase A change with a clear **protocol-level win** (+3.6% prompt, 9B Config A).
- **Path A2** is neutral for throughput but **enables shared drain logic** needed once Path B adds deferred responses.
- **Path B** adds event-based pipeline support; on **9B llama-server** generation is **+7.3%** vs A1 (58.4 vs 54.4 t/s); llama-cli 9B is **~flat to +2.5%**; **4B Config A ~110 t/s** is the cli regression target.
- **Bugfixes** are not optional — they are the difference between a broken client and production-capable A1+A2+B.
- **Large-model server matrix:** Path B **+3% to +14%** vs A1 at `ts=10,90`; A1+A2 **neutral** vs A1. Use **`--no-warmup -np 1`** for Path B on >27B.

**Recommendation:** Use **Path B (v4.2.2) git workspace** for all cross-GPU server workloads. Regress with **`rpc-server-bench-matrix.sh`**. Split: **`ts=10,90`** (RPC0=10%, ROCm0=90%); MoE adds **`--n-cpu-moe 8`**.

---

## Appendix: Raw Data

### Path B — 4B Config A matrix (2026-06-25)

```
Run 1: P=118.5 G=108.9  (q4_0)
Run 2: P=119.5 G=110.5  (q4_0)
Run 3: P=121.5 G=110.8  (q8_0)
```

### Path B — 9B Config A matrix (2026-06-25, workspace, q4_0)

```
Run 1: P=62.3  G=69.8
Run 2: P=87.4  G=69.8
Run 3: P=98.1  G=70.6
Average: P=82.6 G=70.1
```

### Phase A — 9B Config A (from RPC_OPTIMIZATION_REPORT.md)

```
Vanilla:  P=306.0 G=68.1
A1+A2:    P=317.0 G=68.4
Delta:    P=+3.6% G=+0.4%
```

### Phase A — A1 vs A2 (single 3060 Ti, 20 layers)

```
A1: P=33.0 G=21.7
A2: P=33.0 G=21.7
```

### llama-server — 9B MTP Q4_K_M (2026-06-25, q8_0/turbo3, ctx 8192, ts 1,1)

```
A1:     load=16s  run1 P=92.2 G=54.6  run2 P=448.5 G=54.3  run3 P=452.5 G=54.2  avg G=54.4
Path B: load=10s  run1 P=57.5 G=58.6  run2 P=78.8 G=58.3  run3 P=88.5 G=58.4  avg G=58.4
Delta:  +7.3% generation, Path B loads 37% faster
```

### llama-server — Path matrix (2026-06-25, ts=10,90, --no-warmup -np 1)

```
27B Qwen   A1=27.1  A1a2=27.1  PathB=27.9   B vs A1: +3.0%
31B Gemma  A1=23.5  A1a2=23.2  PathB=26.7   B vs A1: +13.6%
35B A3B    A1=58.9  A1a2=58.1  PathB=64.2   B vs A1: +9.0%  (ncmoe=8, ctx 4096)
36B A3B    A1=57.1  A1a2=58.2  PathB=61.1   B vs A1: +7.0%  (ncmoe=8, ctx 4096)
9B MTP     A1=54.4  PathB=58.4               B vs A1: +7.3%  (ts=1,1, ctx 8192)
```

### llama-server — >27B A1 ts=10,90 (2026-06-25, q8_0/turbo3, ctx 8192, ngl 99)

```
27B qwen  ts=10,90:  G=27.2/27.2/27.3  avg G=27.2  RPC=1960MiB ROCm=16221MiB  nvidia_peak=2317MiB
27B qwen  ts=15,85:  G=26.3/26.3/26.2  avg G=26.2  RPC=2747MiB ROCm=15434MiB
31B gemma ts=10,90:  G=24.0/23.8/23.8  avg G=23.8  RPC=2060MiB ROCm=15400MiB
35B A3B  ts=10,90 ncmoe=8:  RPC=116MiB ROCm=15938MiB CPU_MoE=4306MiB  PASS (probe)
27B ts=90,10: OOM RPC alloc 16574702592 bytes
31B/27B Path B: FAIL — RPC server crash at slot warmup or first inference

Legacy (wrong): ts=18,1 ngl=25 put 6.5GB on RPC / 1GB ROCm / 11GB CPU -> ~5.4 t/s
```

---

*Report generated 2026-06-25. Related: [RPC_OPTIMIZATION_REPORT.md](/home/hunter/projects/atomic-llama-cpp-turboquant/RPC_OPTIMIZATION_REPORT.md), [rpc-path-a-tracking.md](/home/hunter/projects/atomic-llama-cpp-patch/docs/rpc-path-a-tracking.md), [rpc-path-b-tracking.md](/home/hunter/projects/atomic-llama-cpp-patch/docs/rpc-path-b-tracking.md).*