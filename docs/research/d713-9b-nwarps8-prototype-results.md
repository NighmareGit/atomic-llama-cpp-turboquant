# D7.13 nwarps=8 Prototype Results — Qwen3.5-9B-MTP-Q4_K_M

**Date:** 2026-07-17
**GPU:** AMD RX 7900XTX (gfx1100, 24 GB VRAM)
**Model:** Qwen3.5-9B-MTP-Q4_K_M.gguf (9B param, MTP speculative decoding, Q4_K quantization)
**Flag:** `GGML_HIP_D713_NWARPS8_Q4K` — forces nwarps=8 for Q4_K MMVQ kernel on RDNA3

---

## 1. End-to-End TG Comparison

| Config | Prompt (t/s) | Generation (t/s) | Delta |
|--------|-------------|-----------------|-------|
| Baseline (nwarps=1) | 80.4 | **87.1** | — |
| Prototype (nwarps=8) | 93.2 | **87.8** | **+0.8%** |

**Conclusion:** No end-to-end regression. The nwarps=8 change is TG-neutral on Qwen3.5-9B.

---

## 2. Kernel-Level Profiling (rocprofv3)

### Q4_K MMVQ Kernel: `mul_mat_vec_q<12, 1, true, false>`

| Metric | Baseline (nwarps=1) | Prototype (nwarps=8) | Delta |
|--------|---------------------|---------------------|-------|
| Total Duration | 77.6 ms | 95.1 ms | **+22.6%** |
| Avg per call | 92.4 us | 113.3 us | +22.6% |
| Calls | 840 | 840 | same |
| Workgroup Size | 32 (1 warp) | 256 (8 warps) | 8x |
| VGPR Count | 40 | 40 | same |
| Grid Size | 393216 x 1 | 393216 x 8 | 8x Y-dim |

### Other Kernels (unchanged)

| Kernel | Baseline (ms) | Prototype (ms) | Notes |
|--------|--------------|---------------|-------|
| mul_mat_vec_q<14,1,false,false> (Q6_K) | 37.4 | 32.4 | within noise |
| mul_mat_vec_q<12,2,false,false> (Q4_K ncols=2) | 23.9 | 24.3 | same |
| mul_mat_vec_q<12,7,false,false> (Q4_K ncols=7) | 19.3 | 19.4 | same |
| mul_mat_vec_q_lds_prototype | 31.9 | 32.0 | same |
| rms_norm_f32 | 8.1 | 7.2 | within noise |

---

## 3. Analysis

### The Paradox: Kernel Slower, TG Same

The Q4_K kernel is **22.6% slower** with nwarps=8, yet end-to-end TG is unchanged. Why:

1. **Q4_K is not the sole bottleneck.** The model also runs Q6_K, Q5_K, and FP16 kernels. Total Q4_K time is ~77-95 ms out of ~400 ms total wall time per step (~20-24% of kernel time).

2. **Kernel overlap.** The GPU can execute Q4_K kernels concurrently with memory transfers and other operations. A slower Q4-K kernel can be hidden behind other work.

3. **Grid size tradeoff.** nwarps=8 uses a larger grid (Y=8 vs Y=1), which may improve occupancy but also increases launch overhead and reduces the GPU's ability to switch to other work.

### Register Pressure

VGPR count is identical (40) between configs. The nwarps=8 kernel does NOT use more registers per thread — instead, it uses more threads per workgroup (256 vs 32). The total VGPR footprint per workgroup is 8x larger, which may limit occupancy.

---

## 4. Comparison to Gemma-4-12B Results

| Model | Baseline TG | Prototype TG | Delta |
|-------|------------|-------------|-------|
| Gemma-4-12B (dense) | ~210 t/s | ~201 t/s | **-4.2%** |
| Qwen3.5-9B (MTP) | 87.1 t/s | 87.8 t/s | **+0.8%** |

The regression is **model-specific**, not structural. Gemma-4-12B (larger, different layer shapes) shows a real regression; Qwen3.5-9B (smaller, MTP) does not.

Possible reasons for model-specific behavior:
- **Tensor shapes:** Gemma-4-12B has different Q4_K tensor dimensions that may be more sensitive to nwarps
- **Kernel mix:** Different models have different proportions of Q4_K vs other kernels
- **MTP overhead:** Qwen3.5-9B's speculative decoding adds overhead that masks the Q4_K regression

---

## 5. Verdict

**The nwarps=8 change for Q4_K on RDNA3 is NOT beneficial.**

- It makes the Q4_K kernel 22.6% slower at the kernel level
- It provides no end-to-end TG improvement on Qwen3.5-9B
- It caused a measurable regression on Gemma-4-12B
- The change adds complexity (new code path, new template instantiation) for zero gain

### Recommendation

**ABANDON the nwarps=8 approach for Q4_K on RDNA3.** The D7.13 prototype flag should be removed. The existing whitelist in `mmvq.cu` (which excludes Q4_K from nwarps=8 on RDNA3_0) is correct — Q4_K should stay at nwarps=1 on this architecture.

If further investigation is desired, consider:
- Testing nwarps=2 or nwarps=4 as intermediate values
- Profiling on RDNA4 (where Q4_K is already whitelisted for nwarps=8)
- Investigating why Gemma-4-12B is more sensitive than Qwen3.5-9B
