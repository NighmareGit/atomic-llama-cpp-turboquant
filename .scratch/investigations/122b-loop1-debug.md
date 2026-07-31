# Loop 1, Phase 2 Debug: Qwen3.5-122B Gated Delta Net Instrumentation

**Date:** 2026-07-24

---

## 1. Instrumentation Added

Modified: `/home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant/src/llama-context.cpp`

Added post-compute tensor dump in `graph_compute()` at line ~2750. After `ggml_backend_sched_graph_compute_async` completes, walks all graph nodes and dumps first 256 float values + stats (min, max, mean, std, nan/inf count) for tensors matching these names:
- `conv_output_silu`, `q_conv_predelta`, `k_conv_predelta`, `v_conv_predelta`
- `attn_output`, `final_output`, `linear_attn_out`
- `dnet_add_ar_state`, `q_in`, `k_in`, `v_in`, `g_in`, `b_in`
- `gate`, `beta_sigmoid`

All dumps tagged with `[DEBUG-L1]` prefix. Only dumps for first 4 graph computes (call_count <= 4).

Build: `build-debug/` — CPU-only Release build, no CUDA/ROCm/RPC.

---

## 2. Working 9B Model Baseline (Qwen3.5-9B-MTP-Q4_K_M)

Model: `/mnt/980pro/models/Qwen3.5-9B-MTP-Q4_K_M.gguf`
Command: `-ngl 0 -c 64 -p "The" -n 4 -t 4 --seed 42`
Result: Coherent output ("The capital of France is")

### Tensor values (layer 1, first token):

| Tensor | Shape | Min | Max | Mean | Std | NaN | Inf |
|--------|-------|-----|-----|------|-----|-----|-----|
| conv_input-0 | [5,8192,1,1] | -6.21 | 12.27 | 0.18 | 1.89 | 0 | 0 |
| conv_output_silu-0 | [8192,2,1,1] | -0.28 | 2.88 | -0.018 | 0.224 | 0 | 0 |
| q_conv_predelta-0 | [128,16,2,1] | -1.31 | 2.45 | 0.045 | 0.315 | 0 | 0 |
| k_conv_predelta-0 | [128,16,2,1] | -0.32 | 0.52 | -0.005 | 0.088 | 0 | 0 |
| v_conv_predelta-0 | [128,32,2,1] | — | — | ~0 | ~0.04 | 0 | 0 |
| gate-1 | [32,2,1,1] | -3.22 | 4.05 | -0.083 | 1.43 | 0 | 0 |
| beta_sigmoid-1 | [1,32,2,1] | -5.78 | 2.21 | -0.49 | 1.26 | 0 | 0 |
| attn_output-1 | [128,32,2,1] | -0.003 | 0.002 | ~0 | 0.0004 | 0 | 0 |
| final_output-1 | [4096,2,1,1] | -4.89 | 3.07 | -1.13 | 1.26 | 0 | 0 |
| linear_attn_out-1 | [4096,2,1,1] | -4.26 | 14.16 | 0.34 | 2.07 | 0 | 0 |

### Key observations:
- conv_output_silu: values mostly in [-0.28, 2.88], centered near 0
- q/k after L2 norm: reasonable range, k has smaller variance than q
- gate values include both negative (decay) and positive (expansion) — expected for per-head gating
- attn_output is very small (~1e-5 magnitude) — first token S=0, so output ≈ scale * (k@(v*beta))^T @ q
- final_output has reasonable range after gated norm and output projection
- No NaN or Inf anywhere

---

## 3. 122B Model Test (Qwen3.5-122B-A10B-Q4_K_S)

Model: `/mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf` (70GB)

### Test 1: ROCm GPU, single GPU, ngl=10, -c 64
Command: `HIP_VISIBLE_DEVICES=0 llama-cli -m ... -ngl 10 -c 64 --seed 42 -p "The" -n 8`
Result: **GARBLED**. Output includes Chinese characters and garbage text.
Sample: `IP.收藏本站in` mixed with garbage bytes.

### Test 2: ROCm GPU, single GPU, -c 512 (multi-token)
Result: **GARBLED** (same pattern, confirmed from prior investigation).

### Test 3: CPU-only, ngl=0
Result: Could not complete within practical timeout (70GB model too slow on CPU).

---

## 4. Code Path Analysis

### 4.1 Fused vs non-fused paths
By default, `cparams.fused_gdn_ar = true` and `cparams.fused_gdn_ch = true` (llama-context.cpp:395).

Both fused and non-fused paths use the SAME `GGML_OP_GATED_DELTA_NET` operation, which has both CUDA and CPU backends. The "non-fused" code paths in `delta-net-base.cpp` (build_delta_net_autoregressive and build_delta_net_chunking) are NEVER called unless fused is explicitly disabled or device mismatch detected.

**Key finding**: There is no separate "fused" vs "non-fused" path for GDN computation. ALL paths use the fused GGML_OP_GATED_DELTA_NET kernel, even on CPU. The non-fused paths in delta-net-base.cpp are legacy code.

### 4.2 Files examined for potential bugs

| File | Content | Potential Issue |
|------|---------|-----------------|
| `ggml/src/ggml-cpu/ops.cpp:10614` | CPU GDN kernel | Hand-traced: looks correct. iq1 = iv1 % H_k handles broadcast. |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | CUDA GDN kernel | Hand-traced: same broadcast as CPU. State layout correct. |
| `src/models/qwen35moe.cpp:60-96` | Weight loading shapes | All dims check out: wqkv [3072,12288], conv [4,12288], ssm_out [8192,3072] |
| `src/models/qwen35moe.cpp:380-510` | GDN graph construction | Q/K/V extraction offsets correct, gate/beta shapes correct |
| `ggml/src/ggml-cpu/ops.cpp:9437` | SSM conv F32 kernel | Sliding window correct. Simple dot product per conv channel. |
| `src/models/delta-net-base.cpp` | Non-fused AR/chunking paths | Legacy code, not exercised in default config |

### 4.3 Dimensional consistency check

| Parameter | 122B | 9B | Check |
|-----------|------|-----|-------|
| conv_channels | 12288 | 8192 | = d_inner + 2*group*state ✓ |
| qkv_dim | 12288 | 8192 | = h_k_dim*k_heads*2 + h_v_dim*v_heads ✓ |
| head_v_dim | 128 | 128 | = d_inner / dt_rank ✓ |
| h_v_dim * dt_rank | 8192 | 4096 | = ssm_d_inner ✓ |
| Q offset | 0 | 0 | ✓ |
| K offset | 2048 elems | 2048 elems | = h_k_dim * k_heads ✓ |
| V offset | 4096 elems | 2048 elems | = 2 * h_k_dim * k_heads ✓ |
| n_embd_s() | 1,048,576 | 524,288 | = ssm_d_state * ssm_d_inner ✓ |

**All dimensional checks pass. No shape mismatch found.**

---

## 5. Hypotheses After Systematic Debug

### Hypothesis 1 (STRONGEST — Conv bug at channel=12288)
The SSM convolution (`ggml_ssm_conv`) produces wrong output for channel dimension 12288.

**Evidence:**
- 9B (8192 channels): works
- 122B (12288 channels): fails
- Conv is the first non-trivial operation on the QKV mixed data
- If conv output is wrong, Q/K/V extraction is wrong, hence ALL downstream GDN is wrong

**To test:** Compare conv_output_silu values between 122B and 9B for the same prompt+seed. Need to run 122B with debug dump, or use a Python reference implementation.

### Hypothesis 2 (MEDIUM — MoE FFN corruption)
The MoE routing in the FFN produces garbage that cascades into subsequent layers' GDN.

**Evidence:**
- The 122B is a MoE model (122B total, 10B active)
- If MoE routing is wrong, FFN output is wrong, which corrupts the hidden state
- The GDN itself may be correct, but receiving corrupted input from prior MoE layers

**To test:** Check if GDN output on layer ~1 (before any MoE FFN) is correct. If yes, the issue is downstream.

### Hypothesis 3 (MEDIUM — Quantization issues with Q4_K_S)
The Q4_K_S dequantization for the 122B model's specific weight shapes produces incorrect values.

**Evidence:**
- Both working 9B and failing 122B are tested with Q4_K_M and Q4_K_S respectively
- Different quantization formats may have edge cases at certain dimension sizes
- The working 9B uses Q4_K_M, the failing 122B uses Q4_K_S

**To test:** Try running the 122B with a different quantization (if available), or compare dequantized weight values with Python reference.

### Hypothesis 4 (LOW — Device memory issues on GPU)
The 70GB model barely fits in 7900 XTX (24GB) with ngl=10. Memory corruption from out-of-bounds access.

**Evidence:**
- ngl=10 puts ~10 layers on GPU, the rest on CPU. Model size is 70GB.
- Memory allocation may fail or be incorrect for such a large model
- But CPU-only testing (if it could complete) would rule this out

### Hypothesis 5 (LOW — State initialization)
The recurrent state cache is not properly zeroed for this model's specific state dimensions.

**Evidence:**
- n_embd_s() = 1,048,576 for 122B (vs 524,288 for 9B)
- If state zeroing has an off-by-one, the first token uses garbage state
- Research confirmed rs_zero logic looks correct, but worth verifying

---

## 6. Recommended Next Steps (Phase 3 Plan)

### 6.1 Priority: Test Hypothesis 1 (Conv bug)
Build `llama-cli` with ROCm and debug dump enabled. Run 122B with ngl=10, -c 64, --seed 42, -p "The", -n 1. Dump conv_output_silu for layer 0 and compare with 9B reference.

If 122B shows extreme values or NaN in conv output → confirm conv bug.
If 122B conv output looks normal → move to Hypothesis 2.

### 6.2 Test Hypothesis 2 (MoE FFN)
If conv output is correct, check the output of the first GDN layer (before any MoE FFN). The first recurrent layer (layer 1, since layer 0 is full attention) should produce correct GDN output if GDN is fine but MoE is broken.

### 6.3 Test Hypothesis 3 (Quantization)
Try running the split 122B model (Qwen_Qwen3.5-122B-A10B-Q4_K_S-00001-of-00002.gguf and -00002-of-00002.gguf) with `-m -00001-of-00002.gguf`. Or try the bartowski IQ4_XS version at `/mnt/980pro/models/bartowski/Qwen_Qwen3.5-122B-A10B-IQ4_XS/`.

### 6.4 Alternative: Python reference implementation
Write a minimal Python script that:
1. Loads a single GDN layer's weights from the GGUF file
2. Runs the GDN computation manually
3. Compares with llama.cpp output tensor-by-tensor

This would definitively identify which operation diverges.

---

## 7. Build Artifact

Debug build at: `/home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant/build-debug/`
Instrumented file: `/home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant/src/llama-context.cpp`
All debug output tagged with `[DEBUG-L1]`.

---

## 8. Go/No-Go for Phase 3

**GO** for Phase 3 planning. The bug is reproducible and consistent. Multiple hypotheses remain viable but can be disambiguated with targeted tests. The strongest hypothesis (conv at channel=12288) can be tested with relatively small code changes.

**Risk**: If the 122B model is too large for practical debug testing (70GB model, 24GB GPU, CPU inference too slow), the debug cycle may be impractically slow. Consider building a smaller test harness that only exercises a single GDN layer.
