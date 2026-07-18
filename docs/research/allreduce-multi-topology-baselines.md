# Multi-topology TG/PP baselines (both rails)

**Date:** 2026-07-17  
**Host:** Romulus (this session)  
**Wayfinder ticket:** [Capture equal-weight multi-topology TG/PP baselines](../../.scratch/allreduce-timing/issues/07-capture-multi-topology-baselines.md)  
**Build:** `build/` commit `71eb7a016` (10145) — **debug + asserts ON** (absolute t/s depressed vs release Path-D numbers; use for structure and relative idle/compute, not fleet peak claims)  
**Artifacts:** `benches/allreduce-baselines/2026-07-17-romulus/`

---

## Scope completed vs deferred

| Topology | Planned | This session |
|----------|---------|--------------|
| **T0** single 7900 XTX | Required | **Done** (Llama-8B, Gemma-4-12B) |
| **T1** 7900 + 3060 Ti RPC layer | Required | **Done** + `split_timing` traces |
| **T1 tensor** | N/A specialized AR | Not run (by design) |
| **T2** 3090+3070 both rails + AR matrix | Required for AR science | **Deferred** — only 1x NVIDIA (3060 Ti) on Romulus; no 3090/3070 here |
| **T3** multi-node Path-D | Required | **Done** (2026-07-17 evening) — see T3 section |
| **T4** hybrid | Later | Deferred |

Instrumentation validated: **528** `event=split_timing` lines on T1 Llama run; AR trace not exercised (no dual-CUDA tensor rail).

---

## Hardware / software

| Item | Value |
|------|--------|
| ROCm | Radeon RX 7900 XTX 24 GB |
| CUDA (RPC) | RTX 3060 Ti via `pathb-rpc-romulus` `127.0.0.1:50051` (healthy) |
| Client | `./build/bin/llama-bench` + `LD_LIBRARY_PATH=build/bin` |
| Bench | `-p 128 -n 64 -r 2`, `-fa on`, `-ctk f16 -ctv f16`, `-ngl 99` |
| Traces | `GGML_SCHED_TRACE=1` + `GGML_SCHED_TRACE_FILE=...` |
| Plus | T1 used `GGML_PIPELINE_PLUS=1` |

---

## Throughput summary (debug build)

### Llama-3.1-8B-Instruct Q5_K_M

| Row | Rail | Config | PP128 t/s | TG64 t/s |
|-----|------|--------|-----------|----------|
| T0 | none | ROCm only | **286.6 ± 2.4** | **105.2 ± 0.1** |
| T1 | layer | ROCm+RPC `-ts 40` | 2096 ± 56 | **65.5 ± 0.6** |
| T1 | layer | ROCm+RPC `-ts 60` | 2102 ± 40 | **64.5 ± 0.6** |

### Gemma-4-12B Q4_K_M (map target arch `gemma4`)

| Row | Rail | Config | PP128 t/s | TG64 t/s |
|-----|------|--------|-----------|----------|
| T0 | none | ROCm only | **171.4 ± 1.1** | **54.5 ± 0.1** |
| T1 | layer | ROCm+RPC `-ts 45` | 1383 ± 46 | **44.5 ± 0.5** |
| T1 | layer | ROCm+RPC `-ts 55` | 1336 ± 96 | **44.3 ± 0.5** |

### Cross-rail (layer vs single) on T1 host

| Model | TG T0 (single) | TG T1 best layer | TG delta |
|-------|----------------|------------------|----------|
| Llama-8B | 105.2 | 65.5 | **-38%** (layer multi-device slower TG on this small model / debug) |
| Gemma-12B | 54.5 | 44.5 | **-18%** |

PP on T1 is much higher than T0 (pipeline/batch effect under dual backends) — not a free lunch for decode.

**Caveat:** Prior Path-D release runs reported ~143 t/s TG on Qwen 35B dual-GPU; these debug absolute numbers are **not** comparable to that fleet baseline. Relative stall structure still useful.

---

## Split timing structure (T1 Llama, 528 events)

Backend IDs from this run (typical order: ROCm=0, CPU=1, RPC=2 — confirm via phase lines with backend names).

From `t1-sched.jsonl` aggregates:

| backend id | idle_us p50 / p90 | compute_us p50 / p90 | n |
|------------|-------------------|----------------------|---|
| 0 | 272 / 696 | **14376 / 14452** | 264 |
| 2 | 15 / 23 | 23 / 28 | 264 |

Sample decode pair (trace_id=1):

| split | backend | idle_us | compute_us |
|-------|---------|---------|------------|
| 0 | 2 | 53 | 2208 |
| 1 | 0 | 664 | 57477 |

**Interpretation (layer rail residual stall):**

- One backend does almost all compute wall time in this instrumented view (async graph_compute host region).  
- Idle before compute is small vs compute on the heavy split (hundreds of us vs tens of ms).  
- Residual TG gap vs single-GPU is consistent with **cross-backend orchestration / straggler / RPC path**, not a missing AllReduce (layer rail has no specialized AR).

Gemma T1 (`t1-gemma12-sched.jsonl`): heavy backend compute p50 ~20 ms; idle p50 ~0.9 ms — same qualitative story.

---

## Tensor rail / AR matrix — T2 triton (2026-07-17)

**Host:** triton `192.168.8.23` — RTX **3090** + RTX **3070**  
**Build:** `build-cuda-t2` Release, CUDA 12.8, arch 86; **NCCL not installed** (`Could NOT find NCCL`)  
**Model:** `Qwen3.5-0.8B.Q4_K_M` (smoke / AR matrix; map-target 35B not transferred this session)  
**Artifacts:** `benches/allreduce-baselines/t2-triton-20260717/`  
**Script:** `scripts/allreduce-t2-baseline.sh` (use `-dev CUDA0/CUDA1` slash form)

### Throughput

| Config | PP128 t/s | TG64 t/s |
|--------|-----------|----------|
| T0 CUDA0 (3090) only | 10051 ± 1745 | **456.8 ± 1.8** |
| T0 CUDA1 (3070) only | 7361 ± 929 | **317.4 ± 1.8** |
| Layer dual `-ts 75,25` | 9004 ± 1507 | **437.2 ± 21** |
| Tensor + `GGML_CUDA_ALLREDUCE=nccl` | 6464 ± 50 | **277.3 ± 6.6** |
| Tensor + `internal` | 6454 ± 49 | **277.0 ± 6.4** |
| Tensor + `none` (butterfly) | 3924 ± 533 | **225.0 ± 1.9** |

### AR timing (6336 calls each; host `duration_us`)

| Env request | Actual provider (first log) | path | duration p50 / mean | nbytes p50 |
|-------------|----------------------------|------|---------------------|------------|
| nccl | **internal** (NCCL missing → fallback) | specialized | **38 / 51** | 4096 |
| internal | internal | specialized | **38 / 51** | 4096 |
| none | butterfly | fallback | **71 / 76** | 4096 |

### Cross-rail conclusions (this model/host)

1. **Best TG is single 3090** (456.8). Dual **layer** is close (437). Dual **tensor** is slower (277 internal, 225 butterfly).  
2. **Internal AR ≈ “nccl” request** here because NCCL is not installed; both log `provider=internal`.  
3. **Butterfly is ~1.9× slower AR p50** than internal (71 vs 38 us) and loses ~19% TG vs internal tensor.  
4. For this small model, **tensor does not beat layer or single-GPU** — AR science still valid; production preference on T2 for 0.8B is layer or single 3090.  
5. Install NCCL on triton and re-run before ranking NCCL vs internal. **Done below.**

### T2 NCCL re-run (same host/model, NCCL 2.30.7 linked)

**Date:** 2026-07-17 (same day, after `apt install libnccl2 libnccl-dev`)  
**Artifacts:** `benches/allreduce-baselines/t2-triton-20260717-nccl/`  
**Build:** `build-cuda-t2` reconfigured; cmake reports `Found NCCL: /usr/lib/x86_64-linux-gnu/libnccl.so`; `ldd libggml-cuda.so` → `libnccl.so.2`

| Config | PP128 t/s | TG64 t/s | Actual AR provider | AR duration p50 / mean (host us) |
|--------|-----------|----------|--------------------|----------------------------------|
| T0 CUDA0 3090 | 10122 ± 1410 | **455.4 ± 2.8** | n/a | n/a |
| Layer dual | 10326 ± 1441* | **454.9 ± 2.9*** | n/a | n/a |
| Tensor **nccl** | 6541 ± 68 | **222.1 ± 2.6** | **nccl** specialized | **5 / 10.1** (n=9456) |
| Tensor **internal** | 6482 ± 42 | **277.6 ± 4.3** | internal specialized | **38 / 51.0** (n=9456) |
| Tensor **butterfly** | 4050 ± 461 | **226.2 ± 1.0** | butterfly fallback | **71 / 74.6** (n=9456) |

\* Layer dual aborted mid-run after first TG table (CUDA error in mmq on second rep); numbers from first successful table.

#### Ranking (TG primary, this model/host)

| Rank | Provider | TG | Notes |
|------|----------|-----|--------|
| 1 | **internal** | 277.6 | Best tensor TG |
| 2 | butterfly | 226.2 | ~19% slower TG than internal |
| 3 | **nccl** | 222.1 | Worst TG here despite lowest host AR us |

**Important measurement caveat:** NCCL `duration_us` is host wall around the enqueue path; NCCL work is stream-async, so **p50=5 us understates GPU AR cost**. Prefer **end-to-end TG/PP** for provider ranking, not raw AR host timers alone. Internal path may include more host-visible wait.

**Implication for plan:** On consumer dual-GPU PCIe (3090+3070) with this small decode workload, **prefer internal AR over NCCL** for TG; do not use butterfly. Single-GPU 3090 still wins overall TG. Re-validate on larger models (35B) when available.

### Runbook (repeat / larger models)

```bash
# on dual-CUDA host
export LLAMA_BENCH=.../build-cuda-t2/bin/llama-bench
export MODEL=.../gemma-4-12b-or-qwen35.gguf
export DEV=CUDA0/CUDA1
bash scripts/allreduce-t2-baseline.sh
```

---

## T3 multi-node Path-D layer (2026-07-17)

**Client:** Romulus debug `build/bin/llama-bench` + `GGML_PIPELINE_PLUS=1` + `GGML_SCHED_TRACE`  
**Workers:** remus `192.168.8.176:50051` pathb-rpc-remus (RTX **5060 Ti** 16 GB); local `127.0.0.1:50051` pathb-rpc-romulus (3060 Ti)  
**Artifacts:** `benches/allreduce-baselines/t3-multinode-20260717/`  
**Tensor:** N/A end-to-end (no cross-RPC AR)

### Throughput (debug build; compare within session only)

| Row | Topology | Model | PP128 | TG64 |
|-----|----------|-------|-------|------|
| T0 ref | ROCm only | Llama-8B | 286.6 | **105.2** |
| T1 ref | ROCm + local 3060 RPC | Llama-8B | ~2100 | **~65.5** |
| **T3a** | ROCm + **remus** 5060 Ti | Llama-8B | ~2050 | **~56.8** |
| **T3b** | ROCm + local 3060 + remus | Llama-8B | ~2080–2240 | **~65.4** |
| T0 ref | ROCm only | Gemma-12B | 171.4 | **54.5** |
| T1 ref | ROCm + local 3060 | Gemma-12B | ~1380 | **~44.5** |
| **T3a** | ROCm + remus | Gemma-12B | ~1410 | **~32.8** |

### Split timing highlights

| Run | Heavy backend idle p50 | Heavy compute p50 | Notes |
|-----|------------------------|-------------------|--------|
| T3a Llama | ~980 us | ~13.9 ms | LAN RPC: higher idle than T1 |
| T3b Llama | ~186 us | ~14.3 ms | More backends; TG similar to T1 local dual |
| T3a Gemma | ~5.4 ms | ~19.9 ms | Idle more visible vs compute |

### Conclusions for plan (08)

1. Multi-node **layer** works (proto 4.4 to remus; telemetry=1 on remus).  
2. **TG: single ROCm > local dual (T1/T3b) > remus-only dual (T3a)** for these small models under debug client — LAN hop hurts more than co-located RPC.  
3. Residual multi-node gap is **orchestration / RPC RTT / straggler**, not AllReduce.  
4. Adding remus to an already dual local setup (T3b) did **not** improve TG vs T1 for Llama-8B (~65 both).  
5. Absolute t/s remain **debug-depressed**; structure is what matters for ranking.

## Map-target models not fully covered

| Model | Status |
|-------|--------|
| Gemma-4-12B | T0+T1 layer done |
| Qwen3.6-35B | **Not run** this session (time/VRAM; use for T1/T2 follow-up on release build) |
| Qwen3-Next | **Not run** |

---

## Implications for ticket 08 (plan lock)

1. On **T1 layer**, instrumentation shows **compute-dominated** splits with modest idle_us; TG still loses to single-GPU for small dense models under dual ROCm+RPC in this debug build — residual gap is **orchestration / split balance / RPC**, not AllReduce.  
2. **AR ranking is blocked** until T2 provider matrix exists; do not rank NCCL vs internal from Romulus-only data.  
3. Hybrid (T4) / RPC TP-unit remains placement-map + future; not contradicted by T1 data.  
4. Re-run T0/T1 on **release** build before citing absolute TG against Path-D 143 t/s lore.  
5. Qwen 35B production model should be added in a follow-up baseline pass on T1 + T2.

---

## Artifact index

| File | Content |
|------|---------|
| `benches/allreduce-baselines/2026-07-17-romulus/t0-llama8b.log` | T0 Llama TG/PP |
| `.../t0-sched.jsonl` | T0 split_timing |
| `.../t1-llama8b-layer.log` | T1 Llama layer |
| `.../t1-sched.jsonl` | T1 Llama traces (528 split_timing) |
| `.../t0-gemma12.log` / `t1-gemma12-*.log` + sched jsonl | Gemma-4-12B |

---

## Follow-up checklist (complete equal-weight claim)

- [x] T2 dual-CUDA host: layer + tensor x {nccl,internal,none} + AR traces  
  - Script: `scripts/allreduce-t2-baseline.sh`  
  - 2026-07-17: **done** on triton 3090+3070 with Qwen3.5-0.8B  
- [x] T2 NCCL install + re-run: libnccl2/dev 2.30.7; true `provider=nccl`; TG ranking internal > butterfly ≈ nccl  
- [ ] T2 follow-up: Gemma-4 / Qwen3.6-35B on dual-CUDA  


- [x] T3 multi-node Path-D best config + split_timing (see T3 section)  

- [ ] Qwen3.6-35B on T1 (and T2 if VRAM allows) release build  
- [ ] Re-baseline T0/T1 release build for absolute TG  
- [ ] Optional: kernel profiler samples on T2 tensor  
