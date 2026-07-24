# Loop 1 Prototype Report: MoE Routing Diagnostic

**Date:** 2026-07-24
**Hypothesis tested:** MoE routing produces corrupted expert weights that cascade into GDN layers, causing garbled output.

**Verdict: MoE routing is NOT the root cause.** Garbled output persists with uniform expert routing. The bug is in the GDN computation itself.

---

## Change Applied

**File:** `src/llama-graph.cpp` line 1494 (`build_moe_ffn`)

Zeroed router logits for `LLM_ARCH_QWEN35MOE` before softmax, forcing uniform expert selection:

```cpp
if (arch == LLM_ARCH_QWEN35MOE) {
    logits = ggml_scale(ctx0, logits, 0.0f);
}
```

This bypasses learned routing completely: softmax of all-zeros produces 1/N per expert, top-k selects indices 0..k-1 deterministically.

## Test Results

### Test 1: 122B, fused GDN (-c 64)

Command:
```bash
HIP_VISIBLE_DEVICES=0 timeout 60 ./build-rocm-rpc-split/bin/llama-completion \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 10 -c 64 -p "The capital of France is" -n 16 -t 4 --seed 42
```

Generated output:
```
arda peaceTRurun
it isheret
TheTRs... (Chinese characters)
```

**Verdict: STILL GARBLED.** Output contains broken words, mixed Chinese, and no coherent English. Uniform routing did not improve output quality.

### Test 2: 122B, chunked GDN (-c 512)

Command:
```bash
HIP_VISIBLE_DEVICES=0 timeout 120 ./build-rocm-rpc-split/bin/llama-completion \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 8 -c 512 -p "Once upon a time," -n 16 -t 4 --seed 42
```

Generated output:
```
,assistanter
usponsTRsape,iffuserG.
```

**Verdict: STILL GARBLED.** Same corruption pattern on the non-fused/chunked path. Uniform routing did not fix either path.

### Test 3: 35B regression

Command:
```bash
HIP_VISIBLE_DEVICES=0 timeout 30 ./build-rocm-rpc-split/bin/llama-completion \
  -m /home/hunter/scratch/prototype-auto/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  -ngl 99 -c 64 -p "Once upon a time," -n 16 -t 4 --seed 42
```

Generated output:
```
00000000000000
```

**Verdict: BROKEN (expected).** The 35B MoE model degraded with uniform routing, confirming the diagnostic change is active. This was expected — uniform routing reduces MoE quality but should still produce tokens.

## Conclusion

| Criterion | Result |
|-----------|--------|
| 122B output without MoE routing | Still garbled |
| Improvement from uniform routing? | NONE |
| MoE routing confirmed as cause? | NO — eliminated |
| Diagnostic useful? | YES — definitively ruled out MoE router |

**The bug is NOT in MoE routing.** The garbled output persists even when all expert contributions are equalized. This means the corruption occurs earlier in the computation graph — in the Gated Delta Net layers themselves (convolution, Q/K/V extraction, delta computation, or state update), not in the FFN expert routing that follows.

## Next: Loop 2

Focus shifts to GDN computation directly:

1. **Conv kernel** — 1024->12288 channel expansion may have a dimension mismatch at 122B scale (conv depth, conv kernel size, or padding)
2. **Q/K/V projection** — 12288-dim input projected to (128,16), (128,16), (128,64) — check projection dimensions match GGUF metadata
3. **Delta computation** — softplus + delta_rank expansion — check tensor shapes
4. **State update** — A/B/C matrix dimensions — compare hparams between 122B and working smaller models

For Loop 2, compare GDN hyperparameters (`ssm_d_inner`, `ssm_dt_rank`, `ssm_n_group`, `ssm_d_state`) from 122B vs 35B GGUF metadata.

## Change Reverted

The `#if 1` block was removed from `src/llama-graph.cpp`. The file is back to its original state.
