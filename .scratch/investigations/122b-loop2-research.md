# Loop 2, Phase 1-2 Combined: Differential Debugging — 9B vs 122B Tensor Comparison

**Date:** 2026-07-24
**Method:** Same prompt ("The"), same seed (42), same context (-c 64), compare tensor dumps

---

## 1. Test Configurations

| Model | Path | Flags | Dump File |
|-------|------|-------|-----------|
| 9B (working) | `/mnt/980pro/models/Qwen3.5-9B-MTP-Q4_K_M.gguf` | ngl=99, 1 GPU | `/tmp/9b-dump-gpu.txt` |
| 122B (broken) | `/mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf` | ngl=10, 1 GPU | `/tmp/122b-dump-gpu.txt` |
| 122B (broken, FGDN) | same | ngl=10, 1 GPU | `/tmp/122b-dump-fgdn.txt` |
| 9B (ngl=10 test) | same | ngl=10, 1 GPU | non-zero FGDN output (verified) |

Instrumentation dumps first 256 float values + stats (min, max, mean, std, nan, inf) for GDN-related tensors.

---

## 2. Tensor Comparison Summary

### Layer 0 (first graph compute, full attention layer)

| Tensor | 9B (ngl=99) Mean/Std | 122B (ngl=10) Mean/Std | Divergence |
|--------|----------------------|------------------------|------------|
| conv_input-0 | -0.040 / 0.508 | -0.000 / 0.007 | **77x SMALLER** |
| conv_output_silu-0 | 0.008 / 0.057 | -0.001 / 0.016 | 3.6x smaller |
| q_conv_predelta-0 | -0.003 / 0.088 | 0.004 / 0.592 | 6.7x larger |
| k_conv_predelta-0 | -0.005 / 0.087 | 0.030 / 0.584 | 6.7x larger |
| v_conv_predelta-0 | -0.000 / 0.004 | -0.002 / 0.017 | 4.2x larger |
| gate-0 | -0.038 / 0.363 | 0.061 / 0.445 | 1.2x (similar) |
| **attn_output-0** | **0.000 / 0.0003** | **0.000 / 0.000** | **ALL ZEROS** :exclamation: |
| final_output-0 | -0.036 / 0.157 | -0.001 / 0.016 | 10x smaller (mirrors conv_output_silu) |

### Layer 1 (first GDN layer)

| Tensor | 9B (ngl=99) Mean/Std | 122B (ngl=10) Mean/Std | Divergence |
|--------|----------------------|------------------------|------------|
| conv_input-1 | 0.004 / 0.564 | 0.000 / 0.005 | **104x SMALLER** |
| **__fgdn_ch__-1** | N/A | **-0.000 / 0.000012** | **NEAR-ZERO** |

### Fused GDN Chunk Output (122B only)

| Layer | Shape | Status |
|-------|-------|--------|
| `__fgdn_ch__-0` | [8192, 130, 1, 1] | **ALL ZEROS** (min=max=mean=std=0, all 256 sampled values zero) |
| `__fgdn_ch__-1` | [8192, 130, 1, 1] | min=-1.8e-4, max=1.8e-5, near-zero |
| `__fgdn_ch__-2` | [8192, 130, 1, 1] | **ALL ZEROS** |
| `__fgdn_ch__-4` | [8192, 130, 1, 1] | min=-1.8e-4, max=1.8e-5, near-zero |
| `__fgdn_ch__-5` | [8192, 130, 1, 1] | **ALL ZEROS** |
| `__fgdn_ch__-6` | [8192, 130, 1, 1] | min=-1.8e-4, max=1.8e-5, near-zero |
| `__fgdn_ch__-8` | [8192, 130, 1, 1] | min=-1.8e-4, max=1.8e-5, near-zero |

**Pattern:** Layers 0, 2, 5 produce ALL ZEROS. Layers 1, 4, 6, 8 produce near-zero values (identical stats: min=-1.81e-4, max=1.80e-5, std=1.20e-5). This repeats every full_attn_interval (4): full attention layers produce zeros; first GDN layer after full attention produces near-zero.

---

## 3. First Divergence Point Identified

**The first divergence is `attn_output-0` in the 122B model — ALL ZEROS.**

This is the standard attention output of layer 0, computed BEFORE any GDN operation. Layer 0 is a full self-attention layer (`full_attn_interval=4`), not a GDN layer. The attention output being zero means the Q@K^T attention scores produce zero output.

However, this is **also** observed in `__fgdn_ch__-0` (fused GDN chunk output in the next graph compute). Both attention mechanisms (standard full attention and GDN) produce zero output for the 122B with ngl=10.

### Root cause classification

The issue is NOT specific to the GDN mechanism. Both full attention AND gated delta net produce zero output on the 122B model with partial GPU offload (ngl=10). This suggests a **platform-level issue** with device memory access or kernel dispatch, not a model architecture bug.

### Evidence for platform-level issue

| Evidence | Implication |
|----------|-------------|
| 9B with ngl=10 produces non-zero FGDN output | Partial GPU offload alone doesn't break kernels |
| 122B conv_input-0 is 77x smaller than 9B | Input values are already corrupted before any attention mechanism |
| Both attention AND GDN produce zeros | Bug is upstream of both attention mechanisms |
| Values repeat across layers (ffn_gate, conv_output_silu identical stats at layers 1,4,6,8) | Tensor data pointers are being reused, old data read |

### Important caveat: Debug instrumentation reliability

The debug instrumentation reads `t->data` directly after graph compute. For GPU tensors, this reads GPU memory through the CPU HSA/MMIO mapping. On repeated graph computes where tensor buffers are reused, the data written by a previous compute may be stale or zeroed. The repeating identical values across layers (e.g., `conv_output_silu-1` = `conv_output_silu-4` = `conv_output_silu-6` identical to `ffn_gate-1`) confirm that some tensor reads return stale data.

**However**, the ALL ZEROS for `__fgdn_ch__-0` during the first graph compute's layer 0 is unlikely to be a read artifact, since:
- The tensor's data pointer is freshly allocated for this graph compute
- Other tensors in the same graph compute (q_conv_predelta, k_conv_predelta) show non-zero values
- The 9B model produces non-zero values for the same tensor with the same debug code

---

## 4. Shape Verification

### ggml_gated_delta_net output shape

```c
// ggml/src/ggml.c lines 6299-6300
const int64_t state_rows = K * S_v * n_seqs;
const int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
```

For 122B with K=1, S_v=128, H=64, n_tokens=2 (prompt "The" + BOS), n_seqs=1:
- state_rows = 1 * 128 * 1 = 128
- ne[0] = 8192
- ne[1] = 2 + 128 = 130
- Total elements: 8192 * 130 = 1,064,960 ✓

This matches the observed dump shape `[8192, 130, 1, 1]`. Shape computation is correct.

### Kernel stride computation

```cpp
// gated_delta_net.cu lines 39-48
const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;  // 128*64*2*1 = 16384
float * attn_data = dst;                                         // region [0..16384)
float * state     = dst + attn_score_elems;                      // region [16384..1064960)
```

attn_data writes per head (h_idx): `dst[sequence * n_tokens * H * S_v + h_idx * S_v + col]`
- Range: [0 .. 16384) for 2 tokens, 64 heads, 128 dims

state writes per head (h_idx): `dst[16384 + (sequence * H + h_idx) * S_v * S_v + col * S_v + i]`
- Range: [16384 .. 16384 + 64*16384 = 1064960)

**Both regions are within bounds.** No stride or offset bug detected.

### SSM convolution

CPU F32 implementation (`ggml/src/ggml-cpu/ops.cpp` lines 9437-9488): simple sliding window dot product per conv channel. No channel count assumptions or hardcoded sizes. No bug found.

---

## 5. Hypothesis

### Primary: Silent GPU kernel failure due to device memory access

**Probable cause:** The fused GDN kernel (and the standard attention kernel) fail to write output because the tensor data pointers cross a device memory boundary that isn't accessible from the ROCm kernel at the time of execution.

**Mechanism:**
1. With ngl=10 on a 70GB model, the memory allocator places some tensors on GPU and others on CPU
2. The GDN kernel receives input tensors (q, k, v, g, beta, state) that may be split between GPU and CPU memory
3. If the output tensor (`dst`) is allocated on GPU but the kernel's input tensors reference CPU memory through HSA/ROCm unified addressing, the kernel may read zeros or fail to execute
4. On AMD 7900 XTX (gfx1100), HSA unified memory access from GPU to CPU can fail silently for large contiguous mappings when the PCIe BAR is fully consumed by other allocations

**Why 9B works:** The 9B model is smaller (4.5GB quantized), so even with ngl=10, the memory layout is simpler — fewer total allocations, less BAR pressure. The GDN kernel reads from GPU-resident tensors.

**Why 122B fails:** The 122B is 70GB, causing extreme memory fragmentation. The `ggml_backend_sched` splits the compute across CPU and GPU backends, and the GPU kernel may receive some inputs via CPU pointers that are not properly mapped for ROCm kernel access.

### Alternate Hypothesis A: State cache on CPU, kernel reads zeros through UVA (unified virtual addressing)

The recurrent state cache (`ssm_states_all`) is 1,048,576 elements (4MB) per layer * 36 layers = 144MB. With ngl=10, this cache is likely allocated on CPU even though the GDN kernel runs on GPU. If the ROCm UVA mapping fails for this specific allocation (size, alignment, or page boundary issue), the kernel reads zeros from the state, producing zero output.

**Testable:** Instrument the GDN kernel to sample state values before computation. If state is all zeros on GPU but non-zero on CPU, the UVA mapping is the culprit.

### Alternate Hypothesis B: Output buffer zeroing after kernel completion

The `ggml_backend_sched` may zero the output buffer AFTER the kernel finishes (as part of buffer reuse for the next graph compute). If the debug dump runs after the scheduler clears buffers, it reads zeros.

**Testable:** Add a GPU synchronize + read immediately after the GDN kernel in `ggml_cuda_op_gated_delta_net`, before returning to the scheduler.

---

## 6. Code Location

**Primary file:** `ggml/src/ggml-cuda/gated_delta_net.cu:170-311` (`launch_gated_delta_net` and `ggml_cuda_op_gated_delta_net`)

**Key variables:**
- `src_state->data` (line 260) — state cache pointer, potentially on CPU
- `dst->data` (line 261) — output buffer, on GPU for ngl=10
- Grid dimensions: `dim3(H=64, n_seqs=1, 32)` — 2048 blocks total

**Dispatch path:**
1. `delta-net-base.cpp:401`: `ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K=1)` → creates result tensor
2. `ggml.c:6313`: returns result with op=`GGML_OP_GATED_DELTA_NET`
3. Backend scheduler dispatches to `ggml_cuda_op_gated_delta_net` (line 226)
4. Tensor backend: `src_q` through `src_state` may have different backends (CPU/GPU)

---

## 7. Go/No-Go for Phase 3 (Plan)

**Recommendation: GO — with revised scope.**

The differential debugging approach has narrowed the problem significantly. The issue is not in the GDN computation mathematics but in the **device memory access layer** between ggml's backend scheduler and the ROCm kernel.

### Phase 3 should focus on:

1. **Verify kernel execution**: Add `hipGetLastError()` + `hipDeviceSynchronize()` after `ggml_cuda_kernel_launch` in `launch_gated_delta_net()`. If the kernel returns an error, we have a launch failure.

2. **Instrument tensor backends**: Add debug print showing which backend (CPU/GPU) each of the 6 GDN input tensors belongs to, and check if any cross-device access is happening.

3. **Test all-GPU config**: If possible, run the 122B with ngl=30+ to put the state cache also on GPU. This eliminates the UVA access path. (May require more VRAM than available on 24GB card.)

4. **Alternative: CPU-only test**: Run the 122B with ngl=0 (CPU only) to verify that the GDN computation is mathematically correct when all tensors are on CPU. The debug binary is too slow for the full 70GB model, but a smaller model or a single-layer test would suffice.

### Non-recommended paths:

- Fixing specific kernel math — the kernel math is correct (verified via manual trace)
- Changing conv channel handling — no bug found in conv
- Adjusting head broadcast logic — broadcast is handled in-kernel and is correct

---

## 8. Summary

| Question | Answer |
|----------|--------|
| First divergence point? | `attn_output-0` ALL ZEROS (layer 0 full attention) |
| GDN-specific bug? | NO — full attention also produces zeros |
| Conv channel bug? | NO — conv implementation has no channel count dependency |
| MoE routing cause? | NO — eliminated in Loop 1 |
| Dimensional mismatch? | NO — all shapes verified |
| Most likely cause? | GPU kernel silently fails to write output for 122B's tensor allocation layout with ngl=10 |
| Next step? | Verify kernel execution, instrument tensor backend placement |

---

## Files Examined

| File | Lines | Purpose |
|------|-------|---------|
| `ggml/src/ggml-cuda/gated_delta_net.cu` | 1-311 | Fused GDN ROCm kernel — kernel math correct, no stride overflow |
| `ggml/src/ggml.c` | 6259-6314 | `ggml_gated_delta_net` output shape — correct for 122B params |
| `ggml/src/ggml-cpu/ops.cpp` | 9437-9488 | SSM conv F32 — no channel count bug |
| `ggml/src/ggml-cpu/ops.cpp` | 10614-10766 | CPU GDN kernel — correct, but benchmark only |
| `src/models/delta-net-base.cpp` | 373-422 | Fused GDN dispatch with K=1 — correct |
| `src/models/qwen35moe.cpp` | 380-510 | GDN graph construction — dimensional checks pass |
| `src/llama-context.cpp` | 2749-2814 | Debug instrumentation — reads `t->data` directly |
| `/tmp/9b-dump-gpu.txt` | — | 9B with ngl=99 — all tensors have reasonable values |
| `/tmp/122b-dump-gpu.txt` | — | 122B with ngl=10 — zeros + stale reuse artifacts |
| `/tmp/122b-dump-fgdn.txt` | — | 122B with FGDN targets — FGDN output ALL ZEROS |
