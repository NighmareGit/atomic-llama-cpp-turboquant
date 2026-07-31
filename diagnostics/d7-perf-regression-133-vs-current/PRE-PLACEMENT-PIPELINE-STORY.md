# Pre-placement pipeline story (user memory vs measurements)

**Date:** 2026-07-19

## What you remember (and it fits the evidence)

| Memory | Fit |
|--------|-----|
| Good multi-GPU perf **before placement-control plane + allreduce** | **Yes** — placement starts `863c10e39` (2026-07-17 evening); allreduce `614928615` (2026-07-18). Live tip-hunt best dual is still **`19db22abb` (2026-07-16)** ~87–98 t/s depending on split. |
| **100+ t/s multi-GPU** with pipeline “full throttle” | **Plausible historically** (docs/archived 133–148); **not re-hit today**. Best dual live: **~98** (ts 2,98 on 19db22abb). Best 1-GPU: **~104**. |
| **7900 ~100% util, 3060 ~40%+** = both ends of pipeline busy | That is the signature of **working layer-pipeline overlap** (local compute + remote compute overlapping), not straggler-idle RPC. We are **not** seeing that balance now in the sense of recovering those TG numbers. |
| Kernel hacks in D7.x may have hurt GPipe | D7 kernel prototypes exist; **WMMA vec_dot OFF**, **D7.13 nwarps OFF**. **LDS MMVQ prototype is ON** in current `build-hip` only. 1-GPU kernels match across tips — dual gap is not “matvec broke.” GPipe+server MTP still fragile. |

## Timeline (relevant)

```
2026-07-14  f2d627a6a     peer archived ~148 (qwen no-mtp heatmap)
2026-07-16  f2c0e1192     FA ON; heatmap ~149 (n=50)
2026-07-16  b145d6fce     D6.10.1 event skip
2026-07-16  19db22abb     WMMA vec_dot prototype (OFF)  << live dual best tip
2026-07-17  71eb7a016     D7.2-D7.8 docs / LDS notes
2026-07-17  863c10e39     placement P0  << "placement foolery" starts
...         94d462..5af38  placement P1-P3, attn-local packer
2026-07-18  614928615     allreduce plumbing
2026-07-19  5415d1e96..   rpc fixes, weak symbols, serialize overflow
2026-07-19  f68e17b9b     PPLUS garble fix (no rotate on reuse)
2026-07-19  802ccb4c6     current binary under test  << dual ~75-84
```

**Bisect window for “pipeline full throttle” loss (dual TG):**  
prefer **`19db22abb` (or `71eb7a016`) .. `802ccb4c6`**, with emphasis on:

1. **placement** `863c10e39`–`5af38bf52` (especially **attn-local** `5af38bf52` — forces attention/MTP on client, SSM on RPC; can **destroy** the old util balance if ever auto-applied)
2. **allreduce** `614928615` (even if “plumbing only,” confirm no default path change)
3. **rpc** `5415d1e96`, `251f1a157`, `72c2aa3cd`, always-flush / serialize behavior
4. **PPLUS** `f68e17b9b` (correctness; kills copy-slot double-buffer on non-GPipe path)

## What “full throttle” means technically

| Healthy pipeline (your memory) | What we measure now |
|--------------------------------|---------------------|
| 7900 saturated (~100%) | 1-GPU ROCm ~104 t/s — local compute fine |
| 3060 meaningfully busy (~40%+) | Dual still has RPC layers, but TG **below** 1-GPU (98 vs 104 best; 84–88 typical) |
| Overlap hides RPC latency | Overlap weak: dual often **loses** to single-GPU → hop/sync dominates |
| GPipe stages + layer split | GPipe+MTP OK in profiler; server GPipe+MTP aborted earlier |

**If dual < single-GPU**, the pipeline is **not** full throttle: the second device is not buying more tokens/s than it costs in sync.

## Kernel hacks checklist (current build-hip)

| Flag | Current | Notes |
|------|---------|-------|
| `GGML_HIP_ROCWMMA_FATTN` | ON | FA kernels present (`flash_attn_ext_vec` in rocprof) |
| `GGML_HIP_WMMA_VECDOT_EXPERIMENTAL` | OFF | good (was numerically wrong in docs) |
| `GGML_HIP_D713_NWARPS8_Q4K` | OFF | good |
| `GGML_HIP_MMVQ_LDS_PROTOTYPE` | **ON** | D7.8 LDS; docs said ~no TG win; rebuild OFF if suspect |
| Placement plan apply | opt-in paths | **attn-local packer** can rewrite layer map if enabled |

rocprof 1-GPU: **fast tip ≈ current** matvec times → dual regression is **not** those kernel prototypes on the local card alone.

## Recommended next (aligned with your story)

1. **Confirm placement is not active** in profiler/server runs (`--placement` / env / plan JSON).  
2. **Rebuild current with `GGML_HIP_MMVQ_LDS_PROTOTYPE=OFF`** A/B (rule out LDS left on).  
3. **Bisect dual TG** on fixed cmdline (`-ts 2,98` or `5,95`, Plus=1, q8_0, no placement):  
   `19db22abb` → `863c10e39^` → `5af38bf52` → `614928615` → `251f1a157` → `f68e17b9b` → `802ccb4c6`.  
4. **Util sampling** (`rocm-smi` + `nvidia-smi` during TG) on best tip vs current — restore the **100% / 40%+** signature as a success criterion, not only t/s.  
5. Treat **148** as optional secondary (may need matched old RPC image); primary restore target is **multi-GPU > 1-GPU** with both GPUs busy.
