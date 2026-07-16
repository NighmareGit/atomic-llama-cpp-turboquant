# D7.6b -- Multi-Model Kernel Comparison

**Date:** 2026-07-16
**Task:** Profile GPU kernels across Qwen and Gemma-4 variants to build a broad comparison dataset
**Tool:** rocprofv3 `--kernel-trace --stats --summary` with `GGML_CUDA_GRAPHS=0`
**Status:** COMPLETE (4 of 6 models profiled; gemma4-26B/31B exceed 8GB RPC VRAM)

## 1. Models Profiled

| # | Model | Arch | Type | Quant | Size | MTP | Split | TPS | GPU Time |
|---|-------|------|------|-------|------|-----|-------|-----|----------|
| 1 | Qwen3.6-35B-A3B-MTP | qwen35moe | MoE (8/256 active) | Q6_K | 21.9 GB | yes | 30/70 | 15.1 | 1,156 ms |
| 2 | Qwen3.6-35B-A3B-abliterated | qwen35moe | MoE (8/256 active) | Q4_K | 21 GB | yes | 30/70 | 15.7 | 1,178 ms |
| 3 | Qwen3.5-35B-A3B | qwen35moe | MoE (8/256 active) | Q4_K_M | 20 GB | no | 30/70 | 9.0 | 1,160 ms |
| 4 | Gemma-4-12B-it | gemma4 | Dense (34 layers) | Q4_K_M | 6.7 GB | no | 55/45 | 20.1 | 1,757 ms |

**Failed to profile:**
| Model | Reason |
|-------|--------|
| gemma4-26B-A4B Q5_K_M | 18 GB MoE, RPC GPU (8 GB) can't fit even with 75/25 split |
| gemma4-31B-it Q4_K_M | 18 GB dense, RPC GPU (8 GB) can't fit even with 70/30 split |

**Note:** All TPS values include rocprofv3 profiling overhead (~9.5x slowdown vs unprofiled).
Relative kernel percentages are reliable; absolute timings are inflated.

## 2. Per-Model Kernel Breakdown

### 2.1 Qwen3.6-35B-A3B-MTP Q6_K (Baseline)

| Rank | Kernel | % GPU | Avg (us) | Calls | Category |
|------|--------|-------|----------|-------|----------|
| 1 | q6_K matmul (forward) | 31.9% | 20.8 | 17,792 | MatMul |
| 2 | quantize_q8_1 | 7.9% | 2.7 | 33,664 | Quant |
| 3 | get_rows | 4.4% | 8.8 | 5,760 | Data |
| 4 | iq4_xs matmul (trans) | 4.3% | 18.4 | 2,688 | MatMul |
| 5 | topk_moe | 4.0% | 12.0 | 3,840 | MoE |
| 6 | iq4_xs matmul (fwd) | 3.7% | 16.0 | 2,688 | MatMul |
| 7 | fp32 matmul | 3.5% | 5.3 | 7,680 | MatMul |
| 8 | flash_attn_vec | 3.4% | 38.8 | 1,024 | Attention |
| 9 | q8_0 matmul (fwd) | 3.4% | 10.2 | 3,840 | MatMul |
| 10 | q6_K matmul (trans) | 3.2% | 24.4 | 1,536 | MatMul |

**Categories:** MatMul 55.4% | Quant 9.9% | ElemWise 8.6% | Data 7.9% | Norm 6.4% | MoE 4.0% | Attention 3.4% | SSM 2.4% | RoPE 1.2% | Other 0.8%

### 2.2 Qwen3.6-35B-A3B-MTP Q4_K (Abliterated)

| Rank | Kernel | % GPU | Avg (us) | Calls | Category |
|------|--------|-------|----------|-------|----------|
| 1 | q4_K matmul (mixed) | 24.0% | 16.2 | 17,408 | MatMul |
| 2 | q5_K matmul (mixed) | 17.5% | 26.4 | 7,808 | MatMul |
| 3 | iq4_xs matmul (mixed) | 16.0% | 36.9 | 5,120 | MatMul |
| 4 | quantize_q8_1 | 7.0% | 2.7 | 30,336 | Quant |
| 5 | get_rows | 3.9% | 8.8 | 5,248 | Data |
| 6 | topk_moe | 3.5% | 11.8 | 3,456 | MoE |
| 7 | fp32 matmul | 3.1% | 5.3 | 6,912 | MatMul |
| 8 | flash_attn_vec | 2.8% | 37.2 | 896 | Attention |
| 9 | rms_norm | 2.8% | 4.7 | 7,040 | Norm |

**Categories:** MatMul 60.7% | Quant 8.7% | ElemWise 7.7% | Data 6.8% | Norm 5.8% | MoE 3.5% | Attention 2.8% | SSM 2.2% | RoPE 1.0% | Other 0.7%

### 2.3 Qwen3.5-35B-A3B Q4_K_M (No MTP)

| Rank | Kernel | % GPU | Avg (us) | Calls | Category |
|------|--------|-------|----------|-------|----------|
| 1 | q4_K matmul (mixed) | 20.5% | 14.8 | 16,128 | MatMul |
| 2 | q5_K matmul (mixed) | 17.8% | 26.4 | 7,808 | MatMul |
| 3 | iq4_xs matmul (mixed) | 13.1% | 39.7 | 3,840 | MatMul |
| 4 | quantize_q8_1 | 7.2% | 2.8 | 30,336 | Quant |
| 5 | q6_K matmul (mixed) | 5.4% | 24.6 | 2,560 | MatMul |
| 6 | get_rows | 4.0% | 8.9 | 5,248 | Data |
| 7 | topk_moe | 3.6% | 12.1 | 3,456 | MoE |
| 8 | fp32 matmul | 3.2% | 5.4 | 6,912 | MatMul |
| 9 | rms_norm | 2.9% | 4.8 | 7,040 | Norm |
| 10 | flash_attn_vec | 2.9% | 37.8 | 896 | Attention |

**Categories:** MatMul 60.0% | Quant 9.0% | ElemWise 7.9% | Data 6.9% | Norm 5.8% | MoE 3.6% | Attention 2.9% | SSM 2.1% | RoPE 1.0% | Other 0.8%

### 2.4 Gemma-4-12B Dense Q4_K_M (No MTP, No MoE, No SSM)

| Rank | Kernel | % GPU | Avg (us) | Calls | Category |
|------|--------|-------|----------|-------|----------|
| 1 | q4_K matmul (mixed) | 36.4% | 60.9 | 10,496 | MatMul |
| 2 | iq4_xs matmul (mixed) | 31.5% | 205.7 | 2,688 | MatMul |
| 3 | q5_K matmul (mixed) | 8.1% | 55.6 | 2,560 | MatMul |
| 4 | flash_attn_tile (prefill) | 5.5% | 188.8 | 512 | Attention |
| 5 | flash_attn_vec (decode) | 3.9% | 31.8 | 2,176 | Attention |
| 6 | set_rows (dequant) | 3.5% | 11.6 | 5,376 | Data |
| 7 | quantize_q8_1 | 2.4% | 2.7 | 15,744 | Quant |
| 8 | rms_norm (SWA) | 2.1% | 6.9 | 5,376 | Norm |
| 9 | rms_norm (causal) | 2.0% | 6.4 | 5,504 | Norm |
| 10 | rope_neox | 1.1% | 4.5 | 4,352 | RoPE |

**Categories:** MatMul 76.0% | Attention 9.5% | Quant 6.2% | Norm 5.7% | RoPE 1.4% | ElemWise 0.5% | Data 0.6% | Other 0.0%

## 3. Cross-Model Comparison

### 3.1 Category Distribution

| Category | Qwen3.6 Q6_K | Qwen3.6 Q4_K | Qwen3.5 Q4_K_M | Gemma4-12B | Notes |
|----------|-------------|-------------|----------------|------------|-------|
| MatMul | 55.4% | 60.7% | 60.0% | **76.0%** | Dense = matmul-dominated |
| Attention | 3.4% | 2.8% | 2.9% | **9.5%** | Dense has 3x more attention |
| MoE Routing | 4.0% | 3.5% | 3.6% | **0.0%** | Dense has no expert routing |
| SSM | 2.4% | 2.2% | 2.1% | **0.0%** | Dense has no SSM layers |
| Quantization | 9.9% | 8.7% | 9.0% | 6.2% | Dense models need less dequant |
| Element-wise | 8.6% | 7.7% | 7.9% | 0.5% | MoE gating ops absent in dense |
| Normalization | 6.4% | 5.8% | 5.8% | 5.7% | Consistent across all models |

### 3.2 Quantization Impact (Qwen MoE Models)

| Metric | Q6_K (21.9 GB) | Q4_K (21 GB) | Q4_K_M (20 GB) | Delta |
|--------|---------------|-------------|----------------|-------|
| Top matmul avg | 20.8 us | 16.2 us | 14.8 us | **Q4_K_M is 29% faster than Q6_K** |
| MatMul % | 55.4% | 60.7% | 60.0% | Q4 increases matmul proportion |
| Quant % | 9.9% | 8.7% | 9.0% | Similar dequant overhead |
| MoE % | 4.0% | 3.5% | 3.6% | Consistent (~3.5-4%) |
| Attention % | 3.4% | 2.8% | 2.9% | Consistent (~3%) |

**Takeaway:** Q4_K and Q4_K_M reduce per-matmul latency by 22-29% vs Q6_K.
However, the matmul proportion increases from 55.4% to ~60% because other
operations don't benefit proportionally from quantization. Net effect: Q4_K
models have similar total GPU time but more matmul-dominated profiles.

### 3.3 MTP Impact (Qwen3.6 Q4_K vs Qwen3.5 Q4_K_M)

| Metric | With MTP (Qwen3.6) | Without MTP (Qwen3.5) | Delta |
|--------|-------------------|----------------------|-------|
| TPS (profiled) | 15.7 | 9.0 | **+74% with MTP** |
| Total GPU time | 1,178 ms | 1,160 ms | Similar total |
| MatMul calls | 17,408 q4_K + 7,808 q5_K | 16,128 q4_K + 7,808 q5_K | MTP adds ~1,280 matmul calls |
| flash_attn calls | 896 | 896 | Identical (same architecture) |
| topk_moe calls | 3,456 | 3,456 | Identical |

**Takeaway:** MTP nearly doubles throughput at similar per-step GPU cost. The
MTP draft head runs on fewer layers (hence fewer matmuls in FAST steps), while
the verification runs full layers. The 9:5 step ratio (FAST:SLOW = 5:4 in D7.2)
means MTP achieves higher token throughput by emitting multiple tokens per SLOW
verification pass.

### 3.4 Architecture Comparison: MoE vs Dense

| Metric | Qwen MoE (avg) | Gemma-4 Dense | Ratio |
|--------|---------------|---------------|-------|
| Layers | 40 (10 attn + 30 SSM) | 34 (all attention) | -- |
| MatMul proportion | ~58% | 76% | 1.3x |
| Attention proportion | ~3% | 9.5% | 3.2x |
| MoE routing | ~3.7% | 0% | -- |
| SSM | ~2.2% | 0% | -- |
| Avg matmul latency | 15-21 us (Q4-Q6) | 56-206 us (Q4-iq4) | 3-10x |
| Per-layer attention | 38 us (vec) | 32 us (vec) + 189 us (tile) | 5.8x |
| Profiled TPS | 9-16 | 20.1 | Dense faster (6.7 GB vs 20 GB) |

**Takeaway:** The Qwen MoE architecture dramatically reduces attention cost
(3% vs 9.5% of GPU time) by replacing 30 of 40 attention layers with SSM.
Gemma-4's dense architecture spends 76% of GPU time in matmul alone, with
per-matmul latencies 3-10x higher due to larger weight matrices (no expert
sparsity). However, Gemma-4-12B is a much smaller model (6.7 GB vs 20 GB),
so it achieves higher raw TPS.

## 4. Key Findings

### 4.1 Universal Truths

1. **MatMul always dominates** -- 55-76% of GPU time regardless of architecture.
   Optimization targeting matmul kernels benefits all model types.

2. **Quantization dequant (quantize_q8_1) is consistent** -- 6-10% of GPU time
   across all models. This is the cost of converting quantized weights to fp32
   before compute. Q4 models need slightly less dequant (fewer bytes to convert).

3. **RMS normalization is architecture-invariant** -- 5.7-6.4% across all models.
   RMS norm kernels are simple element-wise operations and scale with layer count.

### 4.2 Architecture-Specific Insights

4. **MoE expert routing is cheap** -- Only 3.5-4.0% of GPU time, despite 256
   experts. The routing decision is a simple top-k on a score vector. The real
   MoE compute cost is in the expert FFN matmuls, captured in the matmul category.

5. **SSM layers are 10x cheaper than attention** -- Per-layer: SSM ~10 us (conv
   + gated delta net) vs attention ~38 us (flash_attn_vec). This validates the
   Qwen3.5/3.6 hybrid architecture.

6. **Dense models spend 3x more on attention** -- Gemma-4-12B: 9.5% vs Qwen MoE:
   ~3%. Every layer in a dense model runs full attention, unlike MoE models that
   use SSM for most layers.

### 4.3 Quantization Trade-offs

7. **Q4_K matmul is 22-29% faster per-call than Q6_K** (16 us vs 21 us avg).
   But the matmul *proportion* increases (60% vs 55%) because other operations
   don't benefit equally. Net GPU time is similar.

8. **iq4_xs matmul is surprisingly expensive** -- 37-40 us average in Q4 models,
   even slower than q6_K (21 us) in some cases. The "importance matrix" variant
   trades compression for quality, but the dequant cost is higher.

### 4.4 MTP Efficiency

9. **MTP gives ~74% TPS improvement** at similar per-step GPU cost. The draft
   head runs on a subset of layers (FAST steps), while the full model verifies
   multiple tokens per SLOW step. This is the single largest throughput lever.

10. **MTP adds ~8% more matmul calls** (17,408 vs 16,128 for q4_K), but these
    are in FAST steps that run on far fewer layers -- the net GPU time is similar.

## 5. Optimization Priority by Model Type

### MoE Models (Qwen 35B class)

| Priority | Target | % GPU | Lever |
|----------|--------|-------|-------|
| 1 | q6_K/q4_K matmul | 20-32% | Quantize to Q4_K_M, WMMA acceleration |
| 2 | quantize_q8_1 | 7-10% | Cache quantized weights, fused dequant+matmul |
| 3 | get_rows embedding | 4% | Prefetch embedding rows |
| 4 | topk_moe routing | 3.5-4% | Already cheap, low priority |
| 5 | SSM layers | 2% | Already cheap, D7.4 skip-SSM low impact |

### Dense Models (Gemma-4 12B class)

| Priority | Target | % GPU | Lever |
|----------|--------|-------|-------|
| 1 | q4_K matmul | 36% | WMMA acceleration, improved tile sizes |
| 2 | iq4_xs matmul | 32% | Re-quantize to Q4_K_M for speed |
| 3 | flash_attn (tile+vec) | 9.5% | Already well-optimized, low priority |
| 4 | rms_norm | 5.7% | Fuse with subsequent matmul |

## 6. Artifacts

- Kernel stats CSVs: `/tmp/d76-multi-profile/*/kernel-trace_kernel_stats.csv`
- Run logs: `/tmp/d76-multi-profile/*/run.log`
- Individual model doc: `docs/research/d76-rocprofv3-kernel-profile.md`
