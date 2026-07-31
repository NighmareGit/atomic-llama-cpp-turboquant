# Tip hunt: find the performance peak again

## Docs / artifact anchors

| Source | SHA | TG | Notes |
|--------|-----|---:|-------|
| Peer summary + heatmap | `f2d627a6a` | **~148** no-MTP, ~117 MTP heatmap | **ts 24,76**, Plus=1, q8_0; peer "MTP on" was **nextn extract only**, not full draft-accept loop |
| FA enable heatmap | `f2c0e1192` | **~149** (n=50) | 2026-07-16; model path `/mnt/980pro/...` same GGUF |
| D7.1 table | ~same day as `b145d6fce` | **133** n_max=2 GPipe | docs claim; we got **~89** at `b145d6fce` with GPipe+draft-mtp |
| D7.3 FA | after FA ON | **143** | docs claim |
| Current | `802ccb4c6` | **~74.5** | D7-shaped profiler |

## Candidates to build/test (order)

1. `f2d627a6a` - peer peak  
2. `f2c0e1192` - FA + 149 heatmap  
3. `19db22abb` - newest 2026-07-16 code (after D6.10.1)  
4. `251f1a157` - mid July 19 pre-garble-fix (if still slow, window is earlier)

## Standard test matrix (each tip)

**A. Peer-like (chase 148):**  
`-rpc 127.0.0.1:50051 -ngl 99 -ts 24,76 -ctk q8_0 -ctv q8_0 --tasks tg -n 128 -r 5`  
no `--spec-type`, no `--gpipe-stages` (or GPipe=0)

**B. D7-like (chase 133):**  
+ `--gpipe-stages 3 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1`

Env: `GGML_PIPELINE_PLUS=1`, `GGML_CUDA_GRAPHS=0`

## Current session data inventory

| Data | Status |
|------|--------|
| Current tip profiler D7-shape | **HAVE** `/tmp/d7-a2a-profiler-20260719162451` (~74.5) |
| Current tip rocprofv3 kernel | **DO NOT HAVE** this session |
| Ref `b145d6fce` profiler | **HAVE** ~88.9 |
| Historical heatmaps in worktree profiler-out | **HAVE** (148/149 archived) |

---

## Live tip-hunt (2026-07-19) — donkey builds + sequential profiler

**Builds (parallel subagents):** all OK

| Tip | Worktree |
|-----|----------|
| `f2d627a6a` | `/tmp/llama-tip-f2d627a6a` |
| `f2c0e1192` | `/tmp/llama-tip-f2c0e1192` |
| `19db22abb` | `/tmp/llama-tip-19db22abb` |

**Artifacts:** `/tmp/tip-hunt-20260719164652`, `runs/tip-hunt/SUMMARY.md`

| Run | TG t/s | Notes |
|-----|-------:|-------|
| f2d627a6a peer (ts 24,76, no draft) | **88.1** | dual-GPU RPC layers 0-10 |
| f2d627a6a d7 | **FAIL** | old profiler CLI has no `--spec-type draft-mtp` |
| f2c0e1192 peer | **86.9** | |
| f2c0e1192 d7 (gpipe3+mtp) | **87.8** | |
| **19db22abb peer** | **87.1** | newest Jul-16 code |
| **19db22abb d7** | **89.1** | best live retest |
| 802ccb4c6 peer | **77.4** | current; same peer flags |
| 802ccb4c6 d7 (prior) | **74.5** | |
| b145d6fce d7 (prior) | **88.9** | |

### Finding

1. **No tip re-hit 133-148** on this host with current docker RPC.
2. **D7-era tips cluster ~87-89 t/s**; current is **~74-77** (~15-18% slower).
3. **ts 24,76** still places ~layers 0-10 on RPC (same ballpark as default) — not a magic 148 switch alone.
4. Peer docs: historical **148** used Plus+q8_0+ts24,76; "MTP on 147.9" was **nextn extract without draft-accept loop** — not the same as `--spec-type draft-mtp`.
5. Best live tip for dual traces: **`19db22abb` (~89)** vs **`802ccb4c6` (~75-77)**.

### Next

- rocprofv3 + sched-trace on **19db22abb** vs **802ccb4c6** (still missing kernel data on current).
- Optional: matched RPC rebuild at `19db22abb` if chasing 148.
- Bisect `19db22abb..802ccb4c6` for the ~89→75 drop (not for 148).
