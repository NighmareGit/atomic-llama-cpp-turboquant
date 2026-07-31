# D7 apples-to-apples profiler results

**Date:** 2026-07-19  
**Artifact dir:** `/tmp/d7-a2a-profiler-20260719162451`  
**Binary:** `build-hip` version `10167 (802ccb4c6)` (heatmap embeds `git_sha` `251f1a157` - profiler stamp)  
**Model:** `/mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf`  
**RPC:** `127.0.0.1:50051`  
**Env:** `GGML_PIPELINE_PLUS=1`, `GGML_PIPELINE_MULTI_BACKEND_SEQ=1`, `GGML_CUDA_GRAPHS=0`  
**Common CLI:** `-ngl 99 -sm layer --ctx-size 4096 -ctk q8_0 -ctv q8_0 -n 128 -r 5 --warmup`

## Results

| Arm | Config | TG t/s | PP t/s | exit | Notes |
|-----|--------|-------:|-------:|------|-------|
| **P1** | **D7 shape:** `--gpipe-stages 3` + `draft-mtp` n_max=2 n_min=1, tasks pp+tg | **74.53** | **2509** | 0 | GPipe on, MTP draft ctx created; default device order RPC then ROCm (layers 0-10 RPC, 11-39 ROCm) |
| P3 | No GPipe + draft-mtp n_max=2 | **76.88** | - | 0 | Slightly faster without GPipe |
| P4 | GPipe stages=3, **no** MTP | **74.42** | - | 0 | GPipe alone ~same as P1 |
| P2 | n_copies=4 force | - | - | skip | `GGML_SCHED_FORCE_N_COPIES` not present in this tree |

## Historical targets (same class of workload)

| Source | TG t/s | Tool |
|--------|-------:|------|
| D7.1 RPC + n_max=2 + GPipe | **131-133** | `llama-gpipe-profiler` |
| D7.3 + FA | **143** | profiler |
| Peer 2026-07-14 MTP | **~139** | profiler |
| Peer no-MTP | **~146** | profiler |
| Today server chat (prior matrix) | **74-79** | `llama-server` |

## Verdict

1. **Harness is not the cliff.** Same `llama-gpipe-profiler` + GPipe-3 + draft-mtp n_max=2 + same GGUF lands at **~74.5 t/s**, not 133.
2. **Regression is real vs D7/peer:** ~**44% of** 133, ~**51% of** 146.
3. **GPipe does not help today** (P1 74.5 vs P3 76.9); historically it was part of the 133 stack.
4. **MTP does not help today** on profiler (P1 ~ P4); historically n_max=2 was a major lever.
5. **Server chat ~75-79 matches profiler ~75** - earlier server numbers were not "under-measuring."
6. **GPipe+MTP works in profiler** (exit 0) but **aborted on llama-server** earlier - separate server/init bug, not the TG cliff.

## Observed topology (P1)

- Devices registered: ROCm0 + RPC0 + CPU  
- Default layer assign: **RPC0 layers 0-10**, **ROCm0 layers 11-39** (RPC-first enum order)  
- Log: `GPipe enabled with n_stages=3 (n_backends=3)`  
- Log: `>>> MTP: creating draft context (n_max=2, n_min=1)`  
- TG wall ~1.70-1.72 s / 128 tokens  

## Files

| Path | Content |
|------|---------|
| `runs/P1-d7-gpipe3-mtp2/` | cmdline, heatmap, snips |
| `runs/P3-nogpipe-mtp2/` | control no GPipe |
| `runs/P4-gpipe3-nomtp/` | control no MTP |
| `ASSESSMENT.md` | code-level where-we-lost analysis |
| `/tmp/d7-a2a-profiler-20260719162451/` | full stderr logs |

## Next

- Bisect profiler TG between peer-era tip (`f2d627a6a` / mid-July D7) and `802ccb4c6` if restore is the goal.  
- Or profile one decode (`GGML_SCHED_TRACE`) vs D7 traces for wait/compute split.  
- Server GPipe+MTP abort is still a separate fix from the ~75 t/s ceiling.

---

## Historical tip validation (`b145d6fce`) — 2026-07-19

**Worktree:** `/tmp/llama-d7-133-ref`  
**Build:** `build-hip` Release, `GGML_HIP_ROCWMMA_FATTN=ON`, version `10143 (b145d6fce)`  
**Artifacts:** `/tmp/d7-133-ref-validate-20260719163550`, `runs/ref-b145d6fce/`  
**Same flags as current P1** (GPipe3, MTP n_max=2, ctk/ctv q8_0, n=128, r=5).  
**RPC:** current docker `pathb-rpc-romulus` (not rebuilt at `b145d6fce`).

| Tip | Binary | TG t/s (heatmap) | PP t/s | Notes |
|-----|--------|-----------------:|-------:|-------|
| Current | 802ccb4c6 | **74.5** | 2509 | P1 above |
| **Ref D7-era** | **b145d6fce** | **88.9** | **2497** | MTP+GPipe on; one outlier rep ~23 t/s, others ~88-89 |
| Historical claim | mid-July docs | **131-133** | - | **not reproduced** on this box with current RPC |

**Per-rep TG (ref, n_reused=127):** ~88.4, 89.3, 88.9, **23.5**, 89.0 — median ~**89**, mean heatmap **88.9**.

### Quality / MTP on ref tip

- MTP **activates** (`>>> MTP: creating draft context (n_max=2, n_min=1)`).
- Profiler `--sample` quality run: text **degrades into garble** (`<|im_end|>` spam after a short coherent multi-choice fragment). **Not a clean 133-era success story** under this harness.
- Caveat: that quality run log showed **only ROCm+CPU** devices (RPC not in device list) — treat garble as **suspect until re-run with confirmed dual-GPU RPC**.

### Interpretation

1. Plan is right: validate tip first — **ref is faster than current (~+19% TG)** but **not 133**.
2. Gap **133 vs 89** may be: unmatched RPC binary, different ts/env, measurement noise, or tip not exactly the 133 build.
3. Gap **89 vs 75** is a **real client-side code regression** between `b145d6fce` and `802ccb4c6` under the same profiler flags (same host RPC).
4. **ctk/ctv q8_0** already used on both; not the differentiator for this pair.
5. Next: dual **sched-trace / rocprofv3** on ref vs current; optional **matched RPC** rebuild at `b145d6fce`; then **bisect** `b145d6fce..802ccb4c6` on profiler TG.
