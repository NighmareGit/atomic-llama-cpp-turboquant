# D7.4 — Reduce MTP Verification Cost

**Date:** 2026-07-16
**Target:** Split 2 `graph_compute_async` = 6,843 us in SLOW steps (MTP verification)
**Goal:** Understand the 185x FAST/SLOW asymmetry and identify attack vectors

---

## 1. MTP Decode Architecture

### 1.1 Model Structure (Qwen3.6-35B-A3B-APEX-MTP)

| Component | Count | Layer Indices | Type |
|-----------|-------|---------------|------|
| Main trunk | 40 | 0-39 | Hybrid (attention + recurrent) |
| Full-attention layers | 10 | 3,7,11,15,19,23,27,31,35,39 | Dense attention (every 4th) |
| Recurrent (SSM) layers | 30 | remainder | Gated Delta Net (linear attn) |
| MTP head | 1 | 40 (n_layer) | Dense attention only |

The MTP head (`n_layer_nextn=1`) is a single dense-attention decoder block at index `il = n_layer() = 40`. It shares the same attention structure as the main trunk's full-attention layers but has its own projection weights (`nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`).

### 1.2 Draft vs Verification Graph

**Draft graph (`LLM_GRAPH_TYPE_DECODER_MTP`)** — built in `qwen35moe.cpp:550`:
- Inputs: token ID + hidden state `h` (from target's `embd_nextn`)
- Operations: RMS norm -> concat -> `eh_proj` -> attention (QKV + RoPE + KV cache) -> MoE FFN -> output norm -> shared head
- **1 layer equivalent** — runs on a single decode step

**Verification graph (`LLM_GRAPH_TYPE_DECODER`)** — the full model:
- 40 layers (10 attention + 30 SSM) + output head
- Evaluated during the target model's decode step

### 1.3 Data Flow

```
Target decode (SLOW):  token -> [40 layers] -> logits -> sample -> hidden_state
                         |
                         v
Draft (FAST):          hidden_state + token -> [MTP head] -> draft_logits -> draft_token
                         |
                         v
Verify (SLOW):         draft_tokens -> [40 layers] -> verify_logits -> accept/reject
```

The target model produces `embd_nextn` (hidden state before final output norm) which feeds the MTP head. The MTP head then drafts tokens autoregressively using its own KV cache.

---

## 2. The 5:4 FAST:SLOW Pattern

### 2.1 Observed Pattern (from D7.2 trace)

| Step Type | Count | Split 2 Compute | Total Step Time |
|-----------|-------|-----------------|-----------------|
| FAST (draft) | 20 | 37 us | 3,229 us |
| SLOW (verify) | 16 | 6,843 us | 12,946 us |

Pattern: **5 FAST + 4 SLOW** repeating.

### 2.2 Why 5:4 Instead of 1:1?

With `n_max=2`, each verify cycle should produce 2 draft tokens (2 FAST) + 1 verify (1 SLOW) = 2:1 ratio. The observed 5:4 ratio (1.25:1) indicates:

1. **Not all draft tokens are accepted.** The server uses `common_sampler_sample_and_accept_n()` which rejects low-probability drafts. If only ~80% of drafts are accepted, the effective draft-per-verify ratio drops.

2. **The verify step is batched.** The target model evaluates all draft tokens in a single decode call (the verify batch contains all `n_draft+1` tokens). So 1 verify step covers multiple draft tokens.

3. **The draft loop runs multiple iterations.** Looking at `speculative.cpp:1101`, the draft loop continues while `n_drafting > 0`. With `n_max=2`, it can produce up to 2 tokens per sequence, but the loop runs until either:
   - `n_max` tokens are drafted
   - Draft confidence falls below `p_min` (default 0.0, so disabled)
   - All sequences stop drafting

4. **The 5:4 ratio is consistent with ~80% acceptance rate.** If the MTP head produces 5 draft tokens across multiple cycles and 4 verify steps confirm them, this implies the verify step is processing ~1.25 draft tokens per verify on average.

### 2.3 Key Insight: Verification Dominates

The SLOW step's 6,843 us compute is **185x** the FAST step's 37 us. This means:
- The MTP head (1 layer) is extremely cheap to evaluate
- The full model (40 layers) is expensive
- **Verification is the bottleneck, not drafting**

---

## 3. Verification Cost Breakdown

### 3.1 What Happens During Verification

The verify step runs the full 40-layer model on the draft tokens. From `server-context.cpp:3630`:

```cpp
auto accepted = common_sampler_sample_and_accept_n(
    slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
```

This calls `llama_decode(ctx_tgt, batch)` where the batch contains:
- 1 seed token (last accepted token)
- N draft tokens

The full model evaluates all 40 layers on all tokens in the batch.

### 3.2 Layer-Level Cost Estimate

| Layer Type | Count | % of Total Compute | Est. Compute (us) |
|------------|-------|-------------------|-------------------|
| Full-attention | 10 | ~25% | ~1,700 |
| SSM (recurrent) | 30 | ~15% | ~1,000 |
| MoE FFN | 40 | ~50% | ~3,400 |
| Embeddings/output | - | ~10% | ~700 |
| **Total** | **40** | **100%** | **~6,843** |

Note: MoE only activates 3B params out of 35B, but the expert routing + sparse matmuls still dominate. SSM layers are cheaper than attention layers (linear vs quadratic in sequence length, but at decode time both are O(1) per token).

### 3.3 Why FAST Steps Are So Cheap

The MTP head is a single attention layer. It has:
- 1 attention block (QKV + RoPE + KV cache lookup)
- 1 MoE FFN block
- No SSM layers

At decode time (batch=1, single token), the MTP head compute is dominated by memory bandwidth (weight loading), not compute. The 37 us is consistent with a single-layer forward pass on the 7900XTX.

---

## 4. Attack Strategies

### 4.1 Strategy A: Reduce `n_max` (Config Change)

**Mechanism:** Fewer draft tokens per verify cycle = fewer tokens in the verify batch.

| n_max | Draft Tokens/Verify | Verify Batch Size | Est. Verify Compute |
|-------|---------------------|-------------------|---------------------|
| 2 (current) | up to 2 | 3 tokens | 6,843 us |
| 1 | 1 | 2 tokens | ~5,500 us |

**Feasibility:** Trivial config change. `--spec-draft-n-max 1`.

**Estimated gain:** ~20% reduction in verify compute (fewer tokens in batch). But: fewer drafts/cycle means fewer tokens accepted/cycle, potentially reducing TG.

**Risk:** Lower acceptance throughput. With n_max=1, every token requires a verify step. The 5:4 pattern becomes 1:1.

### 4.2 Strategy B: Skip SSM Layers During Verification (Code Change)

**Mechanism:** The 30 SSM (recurrent) layers are cheaper than attention layers but still contribute ~1,000 us to the verify step. If verification could skip SSM layers and only evaluate full-attention layers + output head:

- 10 attention layers + output: ~2,400 us (35% of current 6,843 us)
- **Potential savings: ~4,400 us per verify step (64%)**

**Feasibility:** LOW. This requires architectural changes:
1. The verify step must produce the same logits as the full model for acceptance to work
2. Skipping layers changes the logits, making acceptance decisions unreliable
3. Would need a "partial decode" mode that evaluates only selected layers

**Estimated gain:** High (64% reduction) but risky — acceptance rate may drop.

### 4.3 Strategy C: Confidence-Gated Verification (Code Change)

**Mechanism:** Skip verification entirely if the MTP head's draft confidence is high. The MTP head produces logits; if the top-1 probability exceeds a threshold, accept without verification.

**Current state:** `p_min=0.0` (disabled). Setting `--spec-draft-p-min 0.9` would skip low-confidence drafts but does NOT skip verification.

**Feasibility:** MEDIUM. Requires:
1. A new mechanism to compare MTP confidence against a threshold
2. If threshold exceeded, accept draft without running the full model
3. This is essentially "trust the draft" mode

**Risk:** Quality degradation. The MTP head is 1 layer vs 40 — its confidence is poorly calibrated against the full model.

**Estimated gain:** If 50% of drafts are accepted without verification, verify compute drops by 50%.

### 4.4 Strategy C2: Increase `p_min` to Reduce Draft Count (Config Change)

**Mechanism:** The draft loop in `speculative.cpp:1130` stops drafting when `cur_p->data[0].p < params.p_min`. Setting `p_min > 0` reduces the number of draft tokens per cycle.

| p_min | Expected Drafts/Cycle | Verify Batch Size | Est. Verify Compute |
|-------|----------------------|-------------------|---------------------|
| 0.0 (current) | up to 2 | 3 tokens | 6,843 us |
| 0.5 | ~1.5 | ~2.5 tokens | ~6,200 us |
| 0.9 | ~1.1 | ~2.1 tokens | ~5,800 us |

**Feasibility:** Trivial config change. `--spec-draft-p-min 0.9`.

**Estimated gain:** ~10-15% reduction in verify compute. But: fewer drafts means fewer tokens accepted per cycle.

### 4.5 Strategy D: Async MTP Pipeline (Code Change — Future)

**Mechanism:** The `llama_decode_mtp_async()` API (currently a stub at `llama-context.cpp:4389`) is designed to overlap MTP drafting with target verification. When implemented:

- While target verifies batch N, MTP drafts batch N+1
- Verification and drafting happen in parallel on different streams

**Feasibility:** LOW. The stub is explicit: "depth-1 sync, LLAMA_PIPELINE_DEPTH2=1". Full implementation requires:
1. Dual-stream scheduling on the GPU
2. Shared KV cache with lock-free access
3. Hidden state transfer between target and draft

**Estimated gain:** Up to 2x throughput (verify and draft overlap).

### 4.6 Strategy E: Reduce `n_min` (Config Change)

**Mechanism:** `--spec-draft-n-min` controls the minimum number of draft tokens required. With `n_min=1` (current), even 1 draft token triggers verification. Setting `n_min=0` would allow skipping verification when no drafts are produced.

**Feasibility:** Trivial. But: current config already has `n_min=1`, so this has no effect unless drafts fail.

**Estimated gain:** Minimal.

---

## 5. Recommended Next Step

### Short-term (Config Change): Test `--spec-draft-p-min 0.75 --spec-draft-n-max 1`

This combination:
1. Reduces draft count per cycle (higher `p_min`)
2. Reduces verify batch size (lower `n_max`)
3. Expected: ~25-30% reduction in verify compute
4. Trade-off: lower acceptance rate, potentially lower TG

**Test matrix:**

| Config | p_min | n_max | Est. Verify Compute | Est. TG |
|--------|-------|-------|---------------------|---------|
| Baseline | 0.0 | 2 | 6,843 us | 143.0 t/s |
| Test 1 | 0.9 | 2 | ~6,200 us | ~140 t/s |
| Test 2 | 0.0 | 1 | ~5,500 us | ~135 t/s |
| Test 3 | 0.75 | 1 | ~5,200 us | ~130 t/s |

### Medium-term (Code Change): Partial Verification

If config changes hurt TG too much, investigate "partial verification" — running only the last K layers of the target model during verification. This requires:
1. A new graph type `LLM_GRAPH_TYPE_DECODER_VERIFY` that evaluates layers [40-K, 40)
2. Comparing partial logits against full logits for acceptance
3. Calibration to ensure acceptance quality

### Long-term: Async MTP Pipeline

The `llama_decode_mtp_async()` stub is the intended path for overlapping draft and verify. Implementation is blocked on dual-stream GPU scheduling.

---

## 6. Summary Table

| Strategy | Type | Feasibility | Est. Verify Savings | Est. TG Impact | Risk |
|----------|------|-------------|---------------------|----------------|------|
| A: Reduce n_max to 1 | Config | Trivial | ~20% | -5 to -8 t/s | Low |
| B: Skip SSM in verify | Code | Low | ~64% | Unknown | High |
| C: Confidence-gated verify | Code | Medium | ~50% | Unknown | Medium |
| C2: Increase p_min | Config | Trivial | ~10-15% | -3 to -5 t/s | Low |
| D: Async MTP pipeline | Code | Low | Up to 2x | +50-100% | High |
| E: Reduce n_min | Config | Trivial | ~0% | ~0 | None |

**Recommended:** Start with Strategy C2 (increase `p_min` to 0.75) as a zero-risk config test. If TG holds, combine with Strategy A (n_max=1) for maximum verify compute reduction. If TG drops, the cost is a single config flag change.
