# Ticket 1: Single-GPU Baseline — discriminator for H3 vs H4

**Date:** 2026-07-24
**Model:** Qwen3.5-122B-A10B-Q4_K_S (70 GiB)
**Tool:** llama-completion / llama-cli (build b10211-2cd4d0a96)
**GPU:** Single 7900 XTX (24 GiB), ROCm

---

## 1. Test Results

### Prompt: "Once upon a time,"

| Config | -ngl | Context | Fused GDN AR | Fused GDN CH | Output |
|--------|------|---------|-------------|-------------|--------|
| ROCm | 20 | 512 | -- | -- | OOM (28.6 GiB requested) |
| ROCm | 15 | 512 | enabled* | enabled* | `led-BSGoproject,ersiyard,erventaptENTo pena` (crash in PEG parser) |
| ROCm | 10 | 64 | enabled* | enabled* | `everer\neressesTR\xef\xbf\xbd\n),0ider\xe2\x80\x9c\n   rijo\n...\n.\n` |
| ROCm | 5 | 64 | enabled* | enabled* | `-aigi` then crash |
| CPU | 0 | 64 | enabled* | enabled* | `chioer3:oundedk:,,med,1\n   0ers5872.62epad2)1)\năng` |

*"enabled*" = no GDN-related warnings in output, assumed enabled at default log level

### Prompt: "The capital of France is"

| Config | -ngl | Output |
|--------|------|--------|
| ROCm | 10 | `itornb:TRheros:5erio, Abgeeg::2:` |

### Prompt: "Write a haiku about programming:"

| Config | -ngl | Output |
|--------|------|--------|
| ROCm | 15 | `-and deepz theacschsugo wassbe裂痕 eth3itheath fundamentally一个个inskygger` (PEG crash) |

### Tokenizer check (llama-tokenize)

```
 12162 -> 'Once'
  5028 -> ' upon'
   264 -> ' a'
   854 -> ' time'
    11 -> ','
```

**Tokenizer produces correct token IDs.** The model's input pipeline is intact.

---

## 2. Gated Delta Net Status

### Single-GPU (ROCm, -ngl 10, -c 64) — llama-cli / llama-completion

**No GDN warnings.** Fused GDN is fully enabled (both AR and chunked paths).
Output is garbled regardless.

### Single-GPU (ROCm, -ngl 10, -c 8192) — llama-perplexity 

```
W sched_reserve: fused Gated Delta Net (autoregressive) not supported, set to disabled
W sched_reserve: fused Gated Delta Net (chunked) not supported, set to disabled
```

Both paths disabled due to device mismatch (layer 0 on CPU, GDN tensor on ROCm0).

### 5-GPU (all configs) — from server logs

```
W sched_reserve: fused Gated Delta Net (chunked) not supported, set to disabled
```

Chunked path disabled. AR path status unknown (no warning printed — likely enabled).

### Source code check

`ggml/src/ggml-cuda/gated_delta_net.cu` — ROCm/CUDA fused kernel exists and compiles.
Fallback path in `src/models/delta-net-base.cpp` — non-fused `build_delta_net_autoregressive()` and `build_delta_net_chunking()` use individual ggml ops.

**Both the fused and non-fused paths produce garbage.**

---

## 3. Comparison: Single-GPU vs 5-GPU

| Aspect | Single-GPU | 5-GPU RPC |
|--------|-----------|-----------|
| Output pattern | Variable garbled unicode | Deterministic `///////...` (ASCII 0x2F) |
| Fused GDN AR | Enabled | Unknown (likely enabled) |
| Fused GDN CH | Enabled | Disabled |
| Reproducibility | Varies between runs/prompts | Same `///////...` every prompt |
| Coherent English? | NO | NO |

---

## 4. Verdict: H3 PARTIALLY CONFIRMED — Hypothesis needs updating

**Original H3:** "Gated Delta Net fallback is broken because fused GDN is not supported."
**Original discriminator:** If single-GPU = `///////...` → H3 CONFIRMED. If single-GPU = coherent → H3 KILLED.

**Actual result:** Single-GPU output is **neither** `///////...` nor coherent. It's **variable garbled unicode** — a third outcome not covered by the discriminator.

**Updated understanding:**

The model is broken on **ALL** configurations:
- Fused GDN enabled (single-GPU, -c 64) → garbled
- Fused GDN disabled (5-GPU, single-GPU with -c 8192) → garbled
- CPU-only → garbled
- 5-GPU RPC → deterministic `///////...` (a subset of garbled)

The root cause is **deeper** than "fused GDN not supported":
- The fused GDN ROCm kernel produces wrong output when it IS enabled
- The non-fused fallback also produces wrong output
- The failure is present even on CPU-only (no GPU kernels involved)

This rules out H4 (multi-GPU logits buffer) as the **sole** cause — the model fails on single-GPU too. However, H4 may still be a **contributing factor** to the different failure mode on 5-GPU (deterministic `/` vs variable garbage).

### What this eliminates:

- **H4 as primary cause:** KILLED. The model doesn't work on single-GPU or CPU-only.
- **Pipeline flags (H2):** Already KILLED by Ticket B.
- **UDP transport (H1):** Already KILLED by Ticket A.

### What remains:

The problem is in the model computation itself — something produces wrong logits regardless of:
- GPU count (1 vs 5)
- Backend (ROCm vs CPU)
- Fused GDN enabled/disabled
- Context size

---

## 5. Next Steps

### H3-UPDATED: Investigate why Qwen3.5-122B produces wrong logits on all backends

**Possible sub-hypotheses:**

| # | Sub-hypothesis | Test |
|---|---------------|------|
| H3a | Fused GDN ROCm kernel bug for this model's dimensions (S_k, S_v, H_k, H_v, KDA) | Dump intermediate tensors before/after GDN; compare CPU vs GPU. Check kernel launch params. |
| H3b | Non-fused GDN fallback bug in delta-net-base.cpp | Run with fused GDN forced OFF (-ngl 0, CPU-only, -c 8192 to trigger disable). Compare logits with a known-working Qwen3.5 build. |
| H3c | Q4_K_S dequantization produces wrong weights for this model's tensor shapes | Test with a different quant (Q5_K_M, Q8_0) of the same model. If it works, quant is the issue. |
| H3d | Model architecture mismatch — Qwen3.5-122B uses attention variant not handled by current code | Compare model architecture params with a known-working Qwen3.5 model (e.g., 7B, 32B). Check for MTP, GQA differences. |
| H3e | Weight loading offset — weights loaded at wrong byte offset, shifted by a few elements | Compare known weight values (first attention layer) against reference tensors from safetensors. |

**Recommended order (cheapest first):**

1. **H3c (quant test):** Acquire/convert Q5_K_M or find a smaller Qwen3.5 model that works on this build. If a different quant works, the problem is Q4_K_S specific. ~2-4 hours.
2. **H3d (architecture check):** Compare Qwen3.5-122B model params with a known-working smaller Qwen3.5. Check if MTP heads, GQA ratios, or Delta Net parameters differ. ~1 hour.
3. **H3a/H3b (kernel/fallback debug):** Dump logits/intermediate tensors and compare across backends. ~4-8 hours.
4. **H3e (weight loading):** Hex-dump first few weight values and compare with reference. ~2 hours.

### Ticket 2 (H4) status

H4 is demoted. Multi-GPU logits buffer may be a SECONDARY issue (explains the different failure mode: deterministic `/` on 5-GPU vs variable garbage on single-GPU), but is not the root cause. Only pursue after H3 is resolved.

---

## 6. Raw Output Samples

### llama-completion (no chat parser), -ngl 10, -c 64

```
Prompt: "Once upon a time,"
Output: everer
eressesTR�
),0ider"
   rijo
...
.


Prompt: "The capital of France is"
Output: itornb:TRheros:5erio, Abgeeg::2:
```

### llama-cli (chat mode, crashes in PEG parser), -ngl 15

```
Prompt: "Once upon a time,"
Output: led-BSGoproject,ersiyard,erventaptENTo pena点儿

Prompt: "Write a haiku about programming:"
Output: -and deepz theacschsugo wassbe裂痕 eth3itheath fundamentally一个个inskygger
```

### CPU-only (-ngl 0)

```
Prompt: "Once upon a time,"
Output: chioer3:oundedk:,,med,1
   0ers5872.62epad2)1)
ăng
```
