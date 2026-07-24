# Loop 1 Fix Plan: Qwen3.5-122B GDN — MoE Routing Diagnostic

**Date:** 2026-07-24
**Hypothesis:** MoE routing produces corrupted expert weights that cascade into subsequent GDN layers, causing garbled output. The 9B model is dense (no MoE), which is why it works.

## Rationale for This Hypothesis (Highest ROI)

| Criterion | Assessment |
|-----------|-----------|
| Likelihood | MEDIUM — MoE is unique to 122B (vs 9B dense); routing logits pass through softmax over 128 experts, each with per-expert bias terms |
| Cost to test | LOWEST — one-line change, no new files, 30-second test run |
| Diagnostic power | HIGH — definitively rules IN or OUT the entire MoE subsystem |
| Risk to working models | NONE — guarded by `#if 0`, manually toggled for test only |

If uniform routing fixes the garbled output, we know the bug is in MoE router logits/softmax/top-k. If it doesn't fix it, we've eliminated MoE and can confidently focus on conv/GDN kernel in Loop 2.

## The Change

**File:** `src/llama-graph.cpp`
**Line:** ~1494 (after `logits = build_lora_mm(gate_inp, cur)`)

### Before (line 1492-1494):

```cpp
    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
```

### After:

```cpp
    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]

#if 1  // LOOP-1 DIAGNOSTIC: force uniform MoE routing for 122B GDN debug
        if (arch == LLM_ARCH_QWEN35MOE) {
            logits = ggml_scale(ctx0, logits, 0.0f);
        }
#endif

        cb(logits, "ffn_moe_logits", il);
```

### What this does

- Zeroes router logits for qwen35moe models → softmax produces uniform probabilities (1/N per expert)
- `ggml_argsort_top_k` selects first `n_expert_used` experts deterministically (indices 0..k-1)
- All selected experts get equal weight → FFN output is the average of k expert outputs
- Bypasses the actual learned routing entirely

## Build

```bash
cd /home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant
cmake --build build-rocm-rpc-split --target llama-cli -j$(nproc)
```

Use the existing `build-rocm-rpc-split/` directory (ROCm, single-GPU). No cmake reconfiguration needed — this is a `src/` change.

**If build fails** (e.g., `arch` or `LLM_ARCH_QWEN35MOE` not in scope at that line), add `(void)arch;` before the `#if` block, or check with:

```bash
grep -n "const.*arch.*=" src/llama-graph.cpp | head -5
```

## Test Procedure

### Test 1: 122B with uniform routing (fused GDN, single GPU)

```bash
HIP_VISIBLE_DEVICES=0 timeout 120 ./build-rocm-rpc-split/bin/llama-cli \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 10 -c 64 --seed 42 \
  -p "The capital of France is" -n 16 -t 4 2>&1 | tee /tmp/122b-loop1-test.log
```

### Test 2: 122B with uniform routing (non-fused path, multi-token)

```bash
HIP_VISIBLE_DEVICES=0 timeout 120 ./build-rocm-rpc-split/bin/llama-cli \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 10 -c 512 --seed 42 \
  -p "Once upon a time, there lived a wise old owl" -n 16 -t 4 2>&1 | tee /tmp/122b-loop1-chunk.log
```

### Test 3: 35B regression (with uniform routing — output may be lower quality but must be coherent)

```bash
HIP_VISIBLE_DEVICES=0 timeout 60 ./build-rocm-rpc-split/bin/llama-cli \
  -m /home/hunter/scratch/prototype-auto/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  -ngl 99 -c 64 --seed 42 \
  -p "The capital of France is" -n 16 -t 4 2>&1 | tee /tmp/35b-loop1-regression.log
```

## Success Signals

### 122B fix confirmed (MoE routing IS the bug)

- Output contains actual English words (not garbled unicode)
- Output makes basic sense (e.g., "Paris" or "Paris is the capital" appears)
- Same improvement on both `-c 64` and `-c 512` tests

Example of what "garbled" looks like (from prior investigation):
```
IP.收藏本站in    ← Chinese + garbage
///////...       ← repeated slashes (zeroed state)
```
Example of what "coherent" looks like:
```
The capital of France is Paris. It is a major European city known for its
```

### 122B fix NOT confirmed (MoE is NOT the bug)

- Output remains garbled in the same pattern as before
- No change in output quality or token distribution

### 35B regression

- Output must be coherent English (NOT garbled), even if quality is lower than normal due to uniform routing
- "The capital of France is" should complete with something sensible

## Rollback

If uniform routing does NOT fix the garbled output (MoE is not the bug):

1. Revert the change — change `#if 1` to `#if 0` in the `#if 0` / `#if 1` block
2. Rebuild: `cmake --build build-rocm-rpc-split --target llama-cli -j$(nproc)`
3. Proceed to Loop 2 with Hypothesis 1 (conv at 12288 channels) as the #1 suspect

If uniform routing DOES fix the garbled output (MoE IS the bug):

1. Revert the diagnostic change
2. Proceed to investigate MoE router specifically:
   - Check `exp_probs_b` (expert selection bias) tensor loading — wrong shape or dtype?
   - Check softmax numerical stability with 128 experts and Q4_K_S router weights
   - Check `expert_weights_scale` from GGUF metadata
   - Compare router logits with Python reference

## Notes

- This is a **diagnostic change**, not a final fix. It forces all experts equally weighted, which degrades output quality but eliminates the routing as a variable
- The `#if 1` / `#if 0` guard makes this trivially reversible
- The arch guard (`LLM_ARCH_QWEN35MOE`) limits impact to qwen35moe models only
- Testing takes ~2 minutes total (3 tests, model loading included)
- If the first token is still garbled with uniform routing, the bug is upstream of MoE FFN (in GDN computation itself)

### Critical Files for Implementation
- src/llama-graph.cpp - Insert the one-line diagnostic at line ~1494 (build_moe_ffn routing)
- src/models/qwen35moe.cpp - Model graph construction; reference for understanding MoE call site (line 514-533)
- build-rocm-rpc-split/ - Build target directory for ROCm single-GPU binary
- .scratch/investigations/122b-loop1-debug.md - Phase 2 debug findings with tensor reference values
- /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf - Test model
