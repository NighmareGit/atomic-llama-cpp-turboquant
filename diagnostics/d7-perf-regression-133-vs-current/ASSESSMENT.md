# Assessment: D7/peer 133-148 t/s vs current ~75-79 t/s

**Date:** 2026-07-19  
**Binary under test:** `build-hip` `802ccb4c6` (+ garble fix `f68e17b9b`)  
**Hardware:** romulus dual-GPU (7900 XTX ROCm + 3060 Ti CUDA RPC `:50051`)  
**Related:** [pipeline-plus-dual-gpu-garble/](../pipeline-plus-dual-gpu-garble/), [D8 double-buffered decode](../../docs/wayfinder/D8-DOUBLE-BUFFERED-DECODE-PIPELINE.md)

---

## 1. Summary

| Claim | Verdict |
|-------|---------|
| Sustained dual-GPU TG is far below D7/peer healthy band | **Yes** (~75-79 t/s server chat vs historical **131-148** profiler) |
| PPLUS multi-GPU **correctness** (garble) is fixed | **Yes** (`f68e17b9b`; draft accept ~60% on long gen) |
| Full Path-D **GPipe + draft-mtp** is runnable today | **No** - RPC abort on MTP init when `GGML_SCHED_GPIPE=1` |
| Missing FA build explains the cliff | **No** - `GGML_HIP_ROCWMMA_FATTN=ON`, WMMA symbol present |
| Missing `MULTI_BACKEND_SEQ` explains the cliff | **No** - SEQ adds ~5% (73 -> 77), not 2x |
| Wrong quant file (no pure Q6 GGUF) | **Docs alias** `APEX-MTP-I-Q6_K` -> `...APEX-MTP-I-Quality.gguf` (same file) |

---

## 2. Historical anchors (good numbers)

| Era | Tool | Config gist | TG (t/s) |
|-----|------|-------------|---------|
| Peer 2026-07-14 (`f2d627a6a`) | `llama-gpipe-profiler` | Plus=1, APEX-MTP I-Quality, n_gen=128 | **~139 MTP / ~146 no-MTP** |
| D7.1 (~2026-07-16) | profiler + GPipe | 2-GPU, **gpipe-stages=3**, n_max=2 | **131-133** |
| D7.3 FA | same + rocWMMA FA | n_gen=128, repeat=5 | **143** |
| Today server a2a | `llama-server` chat | Plus, no GPipe, ts 74,26, n_gen=128 | **74-79** |
| Today pure Q6 AgentWorld | server, full Path-D env | no MTP | **75.6** |

D7.1 explicitly: **n_copies is noise on GPipe**; lever is **dual-GPU + n_max=2**. GPipe uses `gpipe_wait_seq` / `gpipe_record_seq`, **not** `pipeline_barrier` rotation.

---

## 3. Code comparison glance (peer/D7 era -> current)

Hot-file delta since peer `f2d627a6a`: large changes in `ggml-backend.cpp`, `ggml-rpc.cpp`, `llama-context.cpp`.

### 3.1 Where buggy / fix code lives

```
TARGET decode (hot TG)
        |
   +----+----+
   |         |
GPipe ON   GPipe OFF (today healthy matrix)
(D7 133)   (server ~77)
   |         |
gpipe_wait  graph_compute + pipeline_barrier
record_seq  *** f68e17b9b: no rotate on reuse ***
   |         |
   +----+----+
        |
  compute_splits (cur_copy, events, RPC)
        |
   +----+----+
   |         |
prev_copy   RPC serialize / flush
87357519e   72c2aa3cd, always-flush downloads
        |
  MTP draft ctx (second scheduler)
  470798451: pipeline_parallel forced off for MTP
  GPipe+MTP today: RPC crash (open)
```

| Issue | Location | Commit | Intent | Perf side effect |
|-------|----------|--------|--------|------------------|
| Multi-GPU PPLUS garble | `ggml_backend_sched_pipeline_barrier` | `f68e17b9b` | Stop rotate when graph `node->src` frozen on reuse | Lose double-buffer decode (D8); TG ~Plus=0 on non-GPipe |
| MTP draft garble | `llama_context::sched_reserve` PP flag | `470798451` | Draft sched must not rotate vs target on shared KV | Draft loses PPLUS slots (tiny graphs) |
| Multi-GPU KV stale | `prev_copy` + RPC download flush | `87357519e` | Correct stateful tensor source | Extra sync/flush possible |
| RPC garble weak symbol | `serialize_tensor` / is_rpc | `72c2aa3cd` | Correct RPC buffer detection | Correctness |
| GPipe + MTP crash | dual ctx + RPC stage path | **open** | - | **Blocks D7-class stack** |

### 3.2 Why `f68e17b9b` alone is unlikely the full 2x cliff

- D7 measured that **n_copies / barrier rotation ~0%** on the **GPipe** path.
- D8 on 27B: post-fix Plus=1 TG **parity with Plus=0** (pipelining benefit gone, not half speed).
- So rotation disable hurts **non-GPipe PPLUS**, but D7 **133** was GPipe - need GPipe working to compare.

### 3.3 Other confounds vs historical numbers

1. **Harness:** historical 133-148 mostly **profiler**; today matrix was **llama-server** `/v1/chat/completions`.
2. **GPipe:** D7 on; today off (crash with MTP).
3. **Tensor split:** docs/hot paths **30,70** or peer **24,76**; fit path used **74,26**.
4. **Draft accept:** server long-gen ~60%; D7 n_max=2 was a net win (+30% vs 1-GPU).

---

## 4. Server matrix snapshot (2026-07-19)

Artifacts:

- `/tmp/pplus-35b-a2a-bench-20260719160007` - first n_gen=128 a2a
- `/tmp/pplus-35b-fullpathd-matrix-20260719160808` - full Path-D + Q6
- `/tmp/pplus-35b-mtp-flags-matrix-20260719161128` - MTP flag matrix

| Arm | TG mean | Notes |
|-----|--------:|-------|
| PLUS-only + MTP | 73.4 | draft accept 58% |
| PLUS+SEQ+blessed + MTP, GPipe=0 | **77.1** | best healthy MTP |
| Full Path-D GPipe=1 + MTP | **FAIL** | RPC abort |
| APEX no-MTP full Path-D | 78.1 | |
| AgentWorld Q6_K full Path-D | 75.6 | pure Q6 |

---

## 5. Ranked "where we lost perf" (hypothesis)

| Rank | Hypothesis | Expected delta if true |
|------|------------|------------------------|
| 1 | Not on D7 path (GPipe+MTP broken / not used) | Large (cannot hit 133 without it) |
| 2 | Harness mismatch (server chat vs profiler) | Medium-large |
| 3 | ts 74,26 vs 30,70 / default split | Medium |
| 4 | `f68e17b9b` no rotation on non-GPipe B+ | Small-medium (D8: few %) |
| 5 | `87357519e` RPC flush / prev_copy sync | Small-medium (latency composition) |
| 6 | FA not compiled | Ruled out for build-hip |

---

## 6. Apples-to-apples follow-up (this session) - DONE

Re-ran **D7-shaped `llama-gpipe-profiler`** on current binary. Full table: **[RESULTS.md](RESULTS.md)**.

| Arm | TG t/s | vs D7 133 |
|-----|-------:|----------:|
| P1 GPipe3 + MTP n_max=2 (D7 shape) | **74.5** | **-44%** |
| P3 no GPipe + MTP | **76.9** | -42% |
| P4 GPipe3 no MTP | **74.4** | -44% |
| Server chat (prior) | 74-79 | matches profiler |

**Conclusion:** Not a harness artifact. Same profiler + GPipe + MTP + same GGUF is stuck at **~75 t/s**. Real dual-GPU TG regression vs D7/peer **133-148**.

Note: GPipe+MTP **succeeds** in profiler; earlier **llama-server** GPipe+MTP abort is a separate issue.

Artifacts: `/tmp/d7-a2a-profiler-20260719162451`, copies under `runs/`.

---

## 7. Next actions (ordered)

1. **Bisect** profiler TG: mid-July D7 tip (or `f2d627a6a`) vs `802ccb4c6` with fixed P1 cmdline.
2. **Sched/rpc trace** one decode vs archived D7 traces (wait vs compute).
3. Fix **server** GPipe+MTP abort (orthogonal to 75 t/s ceiling).
4. **D8** only after reclaim path is understood (few % expected on non-GPipe).
