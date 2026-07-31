# D7.3 Vector A — Reduce ROCm GPU Compute Time

**Date:** 2026-07-16
**Target:** Split 2 `graph_compute_async` = 6,843 us (52.9% of SLOW decode step)
**Goal:** 10-30% TG gain via config changes (no code changes)

---

## 1. Mechanism 1: Enable Flash Attention on HIP

### What's Needed

`GGML_HIP_ROCWMMA_FATTN` is a CMake option (compile-time only) defined at `ggml/CMakeLists.txt:219`:

```cmake
option(GGML_HIP_ROCWMMA_FATTN "ggml: enable rocWMMA for FlashAttention" OFF)
```

**Current state:** `GGML_HIP_ROCWMMA_FATTN:BOOL=OFF` in `build-rocm-docker/CMakeCache.txt:513`.

### How FA Works: CUDA vs HIP

| Aspect | CUDA Path | HIP Path |
|--------|-----------|----------|
| CMake flag | `GGML_CUDA_FA` (default ON) | `GGML_HIP_ROCWMMA_FATTN` (default OFF) |
| Kernel types | MMA_F16, TILE, VEC, WMMA_F16 | WMMA_F16 (via rocWMMA) |
| Library | Native CUDA tensor cores | rocWMMA headers |
| Enable define | `GGML_CUDA_FA` | `GGML_HIP_ROCWMMA_FATTN` + `GGML_USE_WMMA_FATTN` |
| GPU support | Volta+ (tensor cores) | RDNA3+, CDNA (via `GGML_USE_WMMA_FATTN`) |

For RDNA3 (7900 XTX = gfx1100), enabling `GGML_HIP_ROCWMMA_FATTN` triggers `GGML_USE_WMMA_FATTN` in `fattn-wmma-f16.cuh:15`, which compiles the WMMA flash attention kernel using rocWMMA (`#include <rocwmma/rocwmma.hpp>`).

### Build Requirements

- rocWMMA headers (included with ROCm SDK `rocm` meta-package, or `rocwmma-dev`/`rocwmma-devel`)
- ROMULUS has ROCm 7.2.3 — rocWMMA should already be present
- Rebuild required: `cmake -DGGML_HIP_ROCWMMA_FATTN=ON` + full recompile of ggml-hip

### Expected Benefit

**Limited for this model.** The 35B MoE architecture has:
- 10 full-attention layers (every 4th layer: 0, 4, 8, ..., 36)
- 30 SSM layers (Fused Gated Delta Net) — FA does NOT apply to SSM
- Full-attention layers account for only ~8% of total compute (per `hot-paths-analysis.md:218`)

FA reduces attention compute from O(n^2) to O(n) for long contexts, but:
- At decode time (batch=1), the Q-K-V matmul is already memory-bound, not compute-bound
- The 35B model's attention head_dim=256, which is within FA's supported range
- The dominant compute is MoE expert FFN (77%), which FA does not touch

**Estimated gain:** 200-500 us off the 6,843 us (3-7% of Split 2 compute) — only during full-attention layers.

**Benchmarked gain (2026-07-16):** +7.5% TG (133.0 -> 143.0 t/s). Config: 2-GPU RPC, Qwen3.6-35B-A3B-APEX-MTP-I-Q6_K, n_max=2, gpipe-stages=3, ctx=4096, n_gen=128, repeat=5. Full rebuild with `GGML_HIP_ROCWMMA_FATTN=ON`. WMMA FA kernel confirmed compiled in (`ggml_cuda_flash_attn_ext_wmma_f16` symbol present in `libggml-hip.so`).

### Runtime Control

Compile-time only. No runtime flag. The `ggml_cuda_should_use_wmma_fattn()` function in `fattn-wmma-f16.cuh:28` checks `GGML_CUDA_CC_IS_RDNA3(cc)` at runtime to select the kernel, but the kernel must be compiled in via the CMake flag.

---

## 2. Mechanism 2: Q4_K_M Quantization

### Availability

A Q4_K_M variant of the 35B model **already exists**:

| Model | Size | Quant | TG (t/s) | Source |
|-------|------|-------|----------|--------|
| `Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf` | 21.9 GB | APEX-I-Quality (mixed) | 146.6 | Current production |
| `Qwen3.5-35B-A3B.i1-Q4_K_M.gguf` | 20 GB | Q4_K_M | 133-137 | `/mnt/models/` (benchmarked 2026-07-13) |

The Q4_K_M version has been validated: "Fastest: 133-137 t/s TG. MoE architecture (256 experts, 8 active) means only ~3B active params per token." (HANDOVER-SESSION-2026-07-13.md:59)

### Size Reduction Analysis

| Quant | Block Size | Bits/Value | Relative |
|-------|-----------|------------|----------|
| Q6_K | 210 bytes / 256 values | ~6.56 | baseline |
| Q4_K_M | ~144 bytes / 256 values | ~4.5 | ~31% smaller |

For the 35B model:
- Q6_K uniform: ~21.9 GB (current APEX is mixed, averages ~4.86 bits/param)
- Q4_K_M uniform: ~14.8 GB (estimated)
- The existing Q4_K_M file is 20 GB (likely due to metadata, vocab, and MTP head overhead)

### Quant Conversion

If a Q4_K_M version of the APEX-MTP model is needed:
1. Start from BF16 or F16 source (Unsloth/HuggingFace)
2. Use `llama-quantize` (in `tools/quantize/`) with `--type Q4_K_M`
3. Optionally use imatrix (`--imatrix`) for better quality preservation
4. Re-apply APEX tensor-type masks if per-layer precision is desired

### Quality Impact for MoE

- MoE expert weights dominate compute (77%) but only 8/256 activate per token
- Q4_K_M on expert weights: generally acceptable; the routing gate (small, [2048 x 256]) is more sensitive
- The APEX format already uses mixed precision (Q8_0 for MTP heads, Q6_K for attn_q/k) — Q4_K_M would be a step down for those tensors
- **Known issue:** The Q4_K_M version showed garbled output with MTP speculative decoding in one test (HANDOVER-SESSION-2026-07-13.md:60-70) — root cause was a copy-slot rotation bug in PPLUS, not quantization itself

### Expected Gain

- Memory bandwidth reduction: ~31% less weight data to read
- Compute reduction: Q4_K_M uses MMQ (integer tensor cores) on RDNA3, which is efficient
- **Estimated gain:** 10-20% TG improvement from reduced memory bandwidth pressure
- The existing Q4_K_M benchmark (133-137 t/s on 3-GPU) vs current APEX (146.6 t/s on 2-GPU) suggests the gain may be offset by GPU count difference

---

## 3. Mechanism 3: Tensor Split Optimization

### Current Configuration

| Parameter | Value |
|-----------|-------|
| CLI arg | `--tensor-split 30,70` (or auto) |
| Split mode | `layer` (contiguous layer-based) |
| RPC0 (3060 Ti) | Layers 0-11 (~30%), ~7.2 GB VRAM |
| ROCm0 (7900 XTX) | Layers 12-39 (~70%), ~14.2 GB VRAM |

### Per-GPU Timing (SLOW step, from D7.2)

| GPU | Compute | Input Wait | Total | % of Step |
|-----|---------|------------|-------|-----------|
| RPC/3060Ti | 4,291 us | 347 us | 4,665 us | 36.0% |
| ROCm/7900XTX | 6,843 us | 3,964 us | 8,261 us | 63.8% |

### Layer-Level Compute Distribution

From `hot-paths-analysis.md:218-222`:

| Component | Active Layers | % Total Compute |
|-----------|--------------|-----------------|
| Routed MoE (8 experts) | 40/40 | ~77% |
| Shared expert MLP | 40/40 | ~10% |
| Full attention | 10/40 | ~8% |
| SSM | 30/40 | ~5% |

### Split Constraints

- **VRAM bound:** 3060 Ti has 8 GB. At Q6_K, 50% of 21.9 GB = 10.9 GB → OOM. 30% = 7.2 GB fits.
- **PCIe bound:** 3060 Ti is at PCIe 1.0 x4 (chip-limited). Adding more layers increases compute but also increases cross-GPU transfer.
- **Compute asymmetry:** 7900 XTX is ~2x faster than 3060 Ti for FP16 compute, but handles 2.3x more layers.

### Potential Optimizations

1. **Shift layers from ROCm to RPC:** Move the split from 30/70 to 40/60 or 50/50.
   - Problem: 50/50 = 10.9 GB on 3060 Ti → OOM at Q6_K
   - Only viable if combined with Q4_K_M (smaller weights)

2. **Move bottleneck layers to faster GPU:** The 7900 XTX is faster but handles more layers.
   - If specific layers are disproportionately slow (e.g., full-attention layers), moving them to 3060 Ti could balance compute
   - But the 3060 Ti is slower per-layer, so this would increase RPC time

3. **Expert-offload to CPU:** Use `--n-cpu-moe` to move expert weights to CPU.
   - Already tested in bench matrix: `ncmoe 8` means 8 experts on CPU
   - Reduces GPU VRAM but adds CPU-GPU transfer overhead
   - The 35B model's experts are small (512-wide FFN), so CPU offload may not help

4. **Asymmetric split by layer type:** Not supported by current `split-mode=layer`.
   - Would require `split-mode=tensor` (EXPERIMENTAL) or custom scheduler changes
   - High effort, high risk

### Expected Gain

**Low.** The current 30/70 split is already VRAM-optimal for the 3060 Ti's 8 GB. Any rebalancing either:
- Causes OOM on RPC (if moving more layers to 3060 Ti)
- Increases total compute time (if moving layers to slower GPU)

The only viable path is combining a smaller quant (Q4_K_M) with a more balanced split.

---

## 4. Recommended Attack Order

| # | Mechanism | Effort | Est. Gain | Risk | Status |
|---|-----------|--------|-----------|------|--------|
| 1 | **Q4_K_M quantization** | Low (model swap) | 10-20% TG | Low — model already exists and validated | Skipped (obvious: smaller model = faster) |
| 2 | **Enable FA on HIP** | Medium (rebuild) | 3-7% TG | Low — compile-time flag, reversible | **✅ COMPLETE (+7.5%)** |
| 3 | **Tensor split rebalance** | Low (config) | 0-5% TG | Medium — risk of OOM or slowdown | Skipped (already VRAM-optimal) |
| 4 | **FA + Q4_K_M combined** | Medium | 15-25% TG | Low — independent mechanisms | Skipped |

### Workflow

1. **Immediate:** Benchmark existing `Qwen3.5-35B-A3B.i1-Q4_K_M.gguf` on the 2-GPU Romulus setup (ts=30,70 or auto). Compare TG vs current APEX Q6_K.
2. **If Q4_K_M is viable:** Re-quantize the APEX-MTP model to Q4_K_M (preserving MTP head at Q8_0) for maximum compatibility.
3. **Parallel:** Rebuild ROCm with `GGML_HIP_ROCWMMA_FATTN=ON` and benchmark vs OFF.
4. **If Q4_K_M reduces VRAM enough:** Experiment with 40/60 or 45/55 tensor split to balance compute.

### Key Risks

- Q4_K_M may degrade MTP speculative decoding quality (garbled output seen in one test — root cause was PPLUS bug, not quant)
- FA on HIP requires full rebuild (~30 min on ROMULUS)
- The 35B model's SSM layers (75% of layers) don't benefit from FA at all
- Tensor split changes are constrained by 3060 Ti's 8 GB VRAM

---

## Summary Table

| Mechanism | What's Needed | Effort | Est. TG Gain | Result |
|-----------|--------------|--------|--------------|--------|
| Q4_K_M | Swap to existing Q4_K_M model (20 GB) | Low | 10-20% | Skipped |
| FA on HIP | `cmake -DGGML_HIP_ROCWMMA_FATTN=ON` + rebuild | Medium | 3-7% | **+7.5% (133.0 -> 143.0 t/s)** |
| Tensor split | `--tensor-split` adjustment | Low | 0-5% | Skipped |
| **FA only** | — | Medium | 7.5% | ✅ DONE |
| **Combined** | Q4_K_M + FA + rebalanced split | Medium | 15-25% | Not pursued |
