# Loop 1, Phase 1 Research: Qwen3.5-122B Gated Delta Net Investigation

**Date:** 2026-07-24

---

## 1. GDN Code Path Summary

### Architecture

The Qwen3.5-122B model uses "Gated Delta Net" (GDN) — a non-KDA (non-key-dimension-aligned) delta net variant. K heads (16) are broadcast to V heads (64) with expansion factor 4.

### Hyperparameters (from GGUF metadata)

| Parameter | 122B (broken) | 9B (working) |
|-----------|---------------|--------------|
| Model arch | `qwen35moe` | `qwen35` |
| n_block | 48+1 MTP | 32+1 MTP |
| n_embd | 3072 | 4096 |
| n_head | 32 | 16 |
| n_head_kv | 2 | 4 |
| ssm_d_state | 128 | 128 |
| ssm_d_inner | 8192 | 4096 |
| ssm_n_group | 16 | 16 |
| ssm_dt_rank | 64 | 32 |
| ssm_d_conv | 4 | 4 |
| full_attn_interval | 4 | 4 |

**Key check: `ssm_d_inner == ssm_d_state * ssm_dt_rank`?**
- 122B: 8192 == 128 * 64 -> **YES**
- 9B: 4096 == 128 * 32 -> **YES**

This means `head_v_dim` is consistent whether computed from `ssm_d_state` (load path) or `ssm_d_inner / ssm_dt_rank` (build path). Both give 128.

### Tensor Flow Through GDN (per recurrent layer)

```
1. INPUT: normed hidden state [n_embd=3072, n_tokens, n_seqs]

2. PROJECT: 
   wqkv [3072, 12288] -> qkv_mixed [12288, n_tokens, n_seqs]
     Q: first 2048 (=128*16)
     K: next  2048 (=128*16)
     V: final 8192 (=128*64)
   wqkv_gate -> z [64, n_tokens, n_seqs]
   
3. GATE/BETA:
   alpha [64, n_tokens, n_seqs] + ssm_dt[64] -> softplus -> * ssm_a[64] -> gate [64, n_tokens, n_seqs]
   beta [64, n_tokens, n_seqs] -> sigmoid

4. CONV: concat(conv_state[3, 12288, n_seqs], transpose(qkv_mixed)) -> ssm_conv -> SiLU

5. QKV EXTRACT (from conv output [12288, n_tokens, n_seqs]):
   q_conv: offset 0,     view [128, 16, n_tokens, n_seqs]
   k_conv: offset 2048,  view [128, 16, n_tokens, n_seqs]
   v_conv: offset 4096,  view [128, 64, n_tokens, n_seqs]

6. NORM: q_conv/k_conv -> L2 normalize

7. HEAD BROADCAST (only if fused GDN disabled on both ar+ch):
   repeat q_conv, k_conv from [128, 16, ...] to [128, 64, ...]
   (repeats each K-head 4x to match V-head count)

8. FUSED or NON-FUSED GDN:
   FUSED:   ggml_gated_delta_net(q, k, v, gate, beta, state) -> ROCm kernel
   NON-FUSED AR (n_tokens=1): build_delta_net_autoregressive -> ggml ops
   NON-FUSED CH (n_tokens>1): build_delta_net_chunking -> ggml ops

9. STATE WRITE-BACK: new_state [128, 128, 64, n_seqs] -> ssm_states_all cache

10. GATED NORM: output [128, 64, n_tokens, n_seqs] * norm(z, ssm_norm)
    
11. OUTPUT PROJ: flatten [8192, n_tokens, n_seqs] -> ssm_out [8192, 3072] -> [3072, n_tokens*n_seqs]
```

### Files involved

| File | Role |
|------|------|
| `src/models/qwen35moe.cpp` | Loads hparams, tensors; builds GDN graph via `build_layer_attn_linear` (line 380) |
| `src/models/delta-net-base.cpp` | Non-fused GDN (AR+chunking), fused dispatch, conv state, recurrent attn |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | Fused ROCm/CUDA kernel (non-KDA path) |
| `src/llama-hparams.cpp` | `n_embd_s()` returns `ssm_d_state * ssm_d_inner` (line 201-218) |
| `src/llama-graph.cpp` | `build_rs()`: state fetch from recurrent cache (line 2883) |

---

## 2. Detailed Dimensional Analysis

### 2.1 Tensor shapes are self-consistent

All dimensional checks pass for the 122B model:

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| head_v_dim (load) == head_v_dim (build) | 128 == 128 | ssm_d_state=128, d_inner/dt_rank=8192/64=128 | OK |
| conv_dim == conv_channels | 12288 == 12288 | load: 2*2048+8192=12288, build: 8192+2*16*128=12288 | OK |
| QKV view offsets correct | Q@0, K@2048, V@4096 | Verified byte offset calculations | OK |
| ssm_d_state * ssm_dt_rank == ssm_d_inner | 128*64=8192 | 8192 == 8192 | OK |
| n_embd_s() matches state shape | 128*8192=1048576 | 128*128*64=1048576 | OK |
| n_embd_r() matches conv state | 3*12288=36864 | reshape [3, 12288, n_seqs] | OK |

### 2.2 Head broadcast mechanism (non-KDA, H_k=16, H_v=64)

In the fused kernel (`gated_delta_net.cu` line 36-37):
```cpp
const uint32_t iq1 = fastmodulo(h_idx, neqk1);  // h_idx % 16
```

The grid has H=64 blocks. For h_idx 0-15, iq1 = 0-15. For h_idx 16-31, iq1 = 0-15 (repeated). This correctly broadcasts 16 K-heads to 64 V-heads.

In the non-fused AR path (`delta-net-base.cpp` line 359):
```cpp
k = ggml_repeat(ctx0, k, s);  // ggml broadcast from H_k=16 to H_v=64
```

Both mechanisms are equivalent.

### 2.3 Fused vs non-fused logic equivalence

Both paths compute the same delta net update:

**Fused kernel (line 88-109):**
```
kv = S_old^T @ k
delta = (v - g * kv) * beta
S_new = g * S_old + k @ delta^T
attn = S_new^T @ q
```

**Non-fused AR (`delta-net-base.cpp` lines 339-368):**
```
S_decay = g * S_old        (multiply first)
kv = S_decay^T @ k         (use decayed S)
delta = (v - kv) * beta
S_new = S_decay + k @ delta^T
attn = S_new^T @ q
```

Since g is scalar per head (non-KDA): `g * (S^T @ k) == (g * S)^T @ k`. Both paths are equivalent.

### 2.4 Key negative finding: NO obvious dimensional mismatch

After exhaustive analysis across all 11 files, no dimensional inconsistency was found for the 122B model parameters. The wqkv projection, conv channels, QKV extraction offsets, head broadcast, state sizes, and output projection all line up.

---

## 3. Comparison: 122B (broken) vs 9B (working)

| Aspect | 122B | 9B |
|--------|------|-----|
| Architecture | `qwen35moe` (Mixture of Experts) | `qwen35` (Dense FFN) |
| Conv channels | 12288 | 8192 |
| K heads | 16 | 16 |
| V heads | 64 | 32 |
| Head broadcast factor | 4x | 2x |
| State size | 1,048,576 | 524,288 |
| Recurrent layers | 36 | 24 |
| Tokenizer | Same vocab | Same vocab |

Both models use identical code paths. The main difference is the MoE FFN (routed + shared experts) vs dense FFN, but this should not affect GDN output.

---

## 4. Hypotheses (ranked by likelihood)

### Hypothesis A (HIGH): Incorrect tensor shapes for post-GDN output projection or gated norm

The `build_layer_attn_linear` function (qwen35moe.cpp:492-510) combines GDN output with z-gating and projects back. If any of `z`, `ssm_norm`, or `ssm_out` have unexpected dimensions from the GGUF file, the output would be garbled.

**Specifically:**
- `z` from `wqkv_gate` [n_embd=3072, num_v_heads=64] -> after projection: [64, n_tokens, n_seqs]
- But in `build_qkvz`, z is returned as-is, NOT reshaped
- In `build_layer_attn_linear` (line 494): `z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, ...)`
- This assumes z's ne[0] == head_v_dim == 128 and ne[1] == num_v_heads == 64
- BUT z's actual shape from the projection is [64, n_tokens*n_seqs]
- **This reshape would fail or produce garbage!**

Let me verify: `build_qkvz` returns z with shape [64, n_seq_tokens*n_seqs] (result of `build_lora_mm(model.layers[il].wqkv_gate, input)` where wqkv_gate is [3072, 64]).

Then in build_layer_attn_linear:
```cpp
ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
```
This reshapes [64, n_seq_tokens*n_seqs] to [128, 64, n_seq_tokens, n_seqs]. BUT the total elements: 64*n_tokens*n_seqs != 128*64*n_tokens*n_seqs? No that's wrong...

Wait: z has shape [64, n_seq_tokens*n_seqs]. For n_seq_tokens=1, n_seqs=1: z = [64, 1] = 64 elements.
reshape to [128, 64, 1, 1] = 8192 elements. **THIS IS A MASSIVE MISMATCH!**

z's elements: 64 * n_seq_tokens * n_seqs
Target shape elements: 128 * 64 * n_seq_tokens * n_seqs = 8192 * n_seq_tokens * n_seqs

The number of elements differs by factor 128! This would read garbage beyond the tensor boundary.

**Wait, let me double-check.** `wqkv_gate` is loaded as:
```cpp
layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
```
value_dim = head_v_dim * n_v_heads = 128 * 64 = 8192. So wqkv_gate is [3072, 8192].

Then `build_lora_mm(model.layers[il].wqkv_gate, input)` projects [3072, 8192] @ [3072, n_tokens*n_seqs] -> [8192, n_tokens*n_seqs].

So z has shape [8192, n_tokens*n_seqs]. Then reshape to [128, 64, n_tokens, n_seqs] with elements 8192*n_tokens*n_seqs. Both sides match!

I was wrong — the value_dim is 128*64=8192, not just num_v_heads=64. Let me re-check...

Actually: `layer.wqkv_gate` has shape `[n_embd, value_dim]` where `value_dim = head_v_dim * n_v_heads = ssm_d_state * ssm_dt_rank = 128 * 64 = 8192`.

But in `load_arch_tensors`, `head_v_dim = hparams.ssm_d_state = 128`. And `n_v_heads = hparams.ssm_dt_rank = 64`. So `value_dim = 128 * 64 = 8192`. Correct — wqkv_gate is [3072, 8192].

So z after projection is [8192, n_tokens*n_seqs]. Reshaping to [128, 64, n_tokens, n_seqs] has 128*64*n_tokens*n_seqs = 8192*n_tokens*n_seqs elements. Match!

**Phew, this is correct. I withdraw Hypothesis A.**

### Hypothesis A-REVISED (HIGH): State dimension `n_embd_s()` mismatch with per-model head dimension

For the 122B model, `n_embd_s()` returns `ssm_d_state * ssm_d_inner = 128 * 8192 = 1,048,576`. This is the state size used by the recurrent cache allocator.

But for the 9B model, `n_embd_s() = 128 * 4096 = 524,288`.

The state is reshaped to [head_v_dim=128, head_v_dim=128, num_v_heads, n_seqs] = [128, 128, 64, 1]. Elements: 128*128*64 = 1,048,576. Match.

**HOWEVER**: The `n_embd_s()` function checks `n_embd_head_kda` FIRST (line 209 of llama-hparams.cpp). If this is non-zero, it uses `n_embd_head_kda * n_embd_head_kda * n_head()` as the state size. For the 9B model: 128*128*16=262144 (not the ssm_d_state*ssm_d_inner=524288).

Wait, `n_embd_head_kda` is only set by kimi-linear models. Both qwen35 and qwen35moe leave it at 0. So they both use `ssm_d_state * ssm_d_inner`. OK, nothing wrong here.

### Hypothesis B (MEDIUM): Initial recurrent state is not zeroed correctly

If the recurrent state cache starts with garbage values (because `build_rs` with `rs_zero` parameter doesn't clear the correct region for this model), the GDN computation would start with wrong initial state. This would produce wrong output on the FIRST token even with correct computation.

The `build_rs` function (llama-graph.cpp:2883) uses:
```cpp
ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
```

If `rs_zero` is negative, the view has size 0 and nothing is zeroed. If `rs_zero >= 0`, one state (at index `rs_zero`) is zeroed. This single zeroed state is then copied to other cleared states via `state_copy_main`.

**Test**: If the output is garbled even on the very first generated token (after the prompt), then the initial state is likely correct (zeroed) but the computation itself produces wrong state. The ticket shows garbled from the first generated token.

### Hypothesis C (MEDIUM): `ggml_ssm_conv` produces wrong output for 12288 channels with kernel_size=4

The SSM convolution operates on [3+n_tokens, 12288, n_seqs] with kernel [4, 12288]. If the convolution implementation has a bug specific to 12288 channels (e.g., incorrect padding, stride calculations for this channel count), it would produce wrong Q/K/V values.

**Evidence**: The 9B model uses 8192 channels and works. The 122B model uses 12288 and fails. The difference in channel count (50% larger) could trigger edge cases in the convolution implementation.

### Hypothesis D (MEDIUM): Q4_K_S quantization of conv or ssm kernels produces wrong values

If the Q4_K_S dequantization of the `ssm_conv1d` weights (shape [4, 12288]) or `ssm_out` weights (shape [8192, 3072]) produces incorrect values due to edge cases in the dequantization routine for these specific shapes.

**Test**: Try Q5_K_M or Q8_0 quantization of the same model.

---

## 5. Most Likely Root Cause

**Primary suspect: `ggml_ssm_conv` implementation for channel dimension 12288.**

The 9B model works with 8192 channels. The 122B model fails with 12288 channels. This is the single most significant difference in the GDN code path (50% larger conv channels). The conv operates on a 2D tensor where one dimension (channels) is a multiple of 128 but not a power of 2 (12288 = 96 * 128). This could expose bugs in:
- Stride calculations
- SIMD inner loop alignment
- Kernel launch parameter computation

**Secondary suspect: Head broadcast interaction in non-fused path.** When fused GDN is disabled, q and k are explicitly repeated to match V heads via `ggml_repeat_4d`. If the repeat operation has a bug for the specific broadcast factor (4x for 122B vs 2x for 9B), this would produce wrong output.

---

## 6. Recommendations for Phase 2 Instrumentation

### 6.1 Tensors to dump (in order of diagnostic value)

1. **conv_output after SiLU** (`conv_output_silu` at qwen35moe.cpp:441):
   - Shape: [12288, n_tokens, n_seqs]
   - Dump first 64 values from Q, K, V regions to check if conv produces sane values
   - Compare with 9B model (different scale but similar statistical properties)

2. **q_conv, k_conv, v_conv after L2 norm** (lines 473-474):
   - Verify L2 norm ~ 1.0 for each
   - Check if values are in reasonable range (-3 to 3)

3. **GDN output before gated norm** (`output` at delta-net-base.cpp:549):
   - Shape: [128, 64, n_tokens, n_seqs]
   - Compare with known-good computation (manually verified with small Python reference)

4. **gate and beta tensors** (lines 418, 420):
   - Check: gate values should be negative (pre-exp), exp(gate) in (0, 1]
   - Check: beta values after sigmoid should be in (0, 1)

5. **final_output before ssm_out projection** (line 500):
   - Shape: [8192, n_tokens, n_seqs]
   - Check for NaN, Inf, or unreasonable magnitudes

6. **Recurrent state before/after GDN** (state at line 434, output at line 492):
   - Verify state is correctly read from and written to the cache

7. **First-layer logits after all 48 trunk layers**:
   - Compare argmax with known-good token IDs for the prompt
   - This isolates whether the problem is in GDN or downstream (MoE, output proj)

### 6.2 Where to set breakpoints / dump hooks

| Location | File:Line | What to dump |
|----------|-----------|-------------|
| After conv+SiLU | qwen35moe.cpp:441 | `conv_output_silu` first 256 values |
| After L2 norm | qwen35moe.cpp:473-474 | q_conv/k_conv norms |
| Before GDN | qwen35moe.cpp:492 | shapes of q,k,v,gate,beta,state |
| After GDN (fused) | delta-net-base.cpp:403 | output shape + first 128 values |
| After GDN (non-fused AR) | delta-net-base.cpp:368 | o shape |
| After gated norm | qwen35moe.cpp:498 | `attn_out_norm` first 128 values |
| After final proj | qwen35moe.cpp:507 | `cur` first 64 values |

### 6.3 Single-GPU deterministic test harness

```bash
# Fused GDN path (-c 64 forces n_tokens=1 in prompt, enabling AR path)
HIP_VISIBLE_DEVICES=0 ./build-rocm-rpc-split/bin/llama-cli \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 10 -c 64 --seed 42 \
  -p "Once upon a time," -n 8 -t 4 2>&1 | tee /tmp/122b-gdn-test.log

# Non-fused chunked path (-c 512 with multi-token prompt)
HIP_VISIBLE_DEVICES=0 ./build-rocm-rpc-split/bin/llama-cli \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 10 -c 512 --seed 42 \
  -p "Once upon a time, there lived a wise old owl" -n 8 -t 4 2>&1 | tee /tmp/122b-gdn-chunk-test.log
```

### 6.4 Comparative test (working 9B model)

```bash
# Same seed, same prompt, working model
HIP_VISIBLE_DEVICES=0 ./build-rocm-rpc-split/bin/llama-cli \
  -m /home/hunter/scratch/prototype-auto/Qwen3.5-9B-MTP-Q4_K_M.gguf \
  -ngl 99 -c 64 --seed 42 \
  -p "Once upon a time," -n 8 -t 4 2>&1 | tee /tmp/9b-gdn-test.log
```

---

## 7. Files Examined

| File | Lines | Key Findings |
|------|-------|-------------|
| `src/models/qwen35moe.cpp` | 1-760 | GDN graph construction; dims self-consistent |
| `src/models/qwen35.cpp` | 1-642 | Identical GDN path for dense models; works for 9B |
| `src/models/qwen3next.cpp` | 1-500 | Different beta/alpha tensor merge but same GDN flow |
| `src/models/delta-net-base.cpp` | 1-607 | Non-fused AR/chunking + fused dispatch; AR path premultiplies S before kv but is equivalent |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | 1-311 | Fused kernel; S_v={16,32,64,128} switch; broadcast via h_idx%H_k |
| `src/llama-hparams.cpp` | 179-218 | n_embd_r(), n_embd_s() compute recurrent cache dimensions |
| `src/llama-graph.cpp` | 2883-2960 | build_rs() fetches state from cache; rs_zero clears initial state |
| `src/models/kimi-linear.cpp` | 230-360 | KDA variant; separate Q/K/V conv, different structure |

---

## 8. Summary

The Qwen3.5-122B GDN code paths are **dimensionally self-consistent** — all tensor shapes, strides, offsets, and broadcast factors check out for the model's hyperparameters (ssm_d_state=128, ssm_d_inner=8192, ssm_n_group=16, ssm_dt_rank=64).

The most notable difference from the working 9B model is the 50% larger conv channel dimension (12288 vs 8192) and the 4x head broadcast factor (vs 2x for 9B).

**Strongest actionable recommendation**: Instrument the `ggml_ssm_conv` output in Phase 2 to check if the convolution produces correct values for 12288 channels. If the conv output is correct, then instrument the GDN output tensor before the output projection to narrow down the error location.
