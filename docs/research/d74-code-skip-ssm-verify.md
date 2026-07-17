# D7.4 Code Strategy: Skip SSM Layers During MTP Verification

**Date:** 2026-07-16
**Target:** MTP verify step = 6,843 us (SLOW decode, 40 layers)
**Goal:** Design a mechanism to skip the 30 SSM layers during verification, saving ~1,000 us + SSM portion of MoE FFN

---

## 1. Architecture: How the Full Decode Graph Iterates Layers

### 1.1 Graph Type Enum

`src/llama-graph.h:33-36`:
```cpp
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
    LLM_GRAPH_TYPE_DECODER_MTP,
};
```

Only 4 graph types exist. There is no `DECODER_VERIFY` or "partial decode" type.

### 1.2 Graph Type Dispatch

`src/llama-context.cpp:26-31` — context type to graph type:
```cpp
static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}
```

The verify step uses `ctx_tgt` (target context, `LLAMA_CONTEXT_TYPE_DEFAULT`), which maps to `LLM_GRAPH_TYPE_DECODER`. The draft uses `ctx_dft` (MTP context, `LLAMA_CONTEXT_TYPE_MTP`), which maps to `LLM_GRAPH_TYPE_DECODER_MTP`.

### 1.3 Layer Loop in Qwen3.5 MoE

`src/models/qwen35moe.cpp:188-228` — the core decode graph:
```cpp
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;
    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
    // ...
    if (hparams.is_recr(il)) {
        // Linear attention layer (gated delta net)
        cur = build_layer_attn_linear(inp->get_recr(), cur, il);
    } else {
        // Full attention layer
        cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
    }
    // ... residual + MoE FFN ...
    inpL = cur;
}
```

The `hparams.is_recr(il)` check (`src/llama-hparams.h:301`) is the sole discriminator. It is set at load time in `src/models/qwen35moe.cpp:25-30`:
```cpp
uint32_t full_attn_interval = 4;
ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
    hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
}
```

Layers where `(i+1) % 4 == 0` (i.e., 3, 7, 11, ..., 39) are full-attention. The rest are SSM.

### 1.4 Graph Build Dispatch

`src/models/qwen35moe.cpp:153-157`:
```cpp
std::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}
```

The `gtype` field in `llm_graph_params` (`src/llama-context.cpp:2690`) determines which graph class is instantiated. Adding a new graph type requires:
1. New enum value in `llm_graph_type`
2. New `graph_verify` class (or conditional in `graph`)
3. Dispatch logic in `build_arch_graph`
4. A way to select it at runtime (new `ctx_type` or a flag)

---

## 2. Skip Mechanism Design

### 2.1 Approach A: Graph-Level Skip (New Graph Type)

Create `LLM_GRAPH_TYPE_DECODER_VERIFY` that builds a graph iterating only attention layers. The output hidden state from the last attention layer (layer 39) feeds the output head.

**Pros:**
- Clean separation: verify graph is a distinct subgraph
- No runtime branch in the layer loop
- Can be optimized independently (e.g., pre-allocate only attention-layer KV/cache)
- Easy to A/B test (toggle via ctx_type)

**Cons:**
- Requires touching 6+ files for the new enum + dispatch + ctx_type
- The output head expects post-layer-39 hidden state. Skipping SSM layers means the hidden state is "stale" (only went through attention layers, not SSM). The logits will differ from full-model logits.
- Acceptance decisions may degrade because the verify logits no longer match the target distribution.

### 2.2 Approach B: Layer-Level Conditional (Identity Skip)

Reuse the full `LLM_GRAPH_TYPE_DECODER` graph but add a runtime flag `skip_ssm` that makes SSM layers pass-through (identity). The recurrent state (S buffer) is NOT updated.

**Implementation:** In the layer loop, add:
```cpp
if (hparams.is_recr(il) && skip_ssm) {
    // Identity: cur = inpL (skip attn_norm, linear attn, post_norm, MoE FFN)
    // OR: skip only the linear attn but keep MoE FFN
    continue;
}
```

**Pros:**
- Minimal code change (single conditional in the loop)
- Can selectively skip only the SSM attention (keep MoE FFN) for a smaller quality hit
- Easy to toggle via environment variable or cparam

**Cons:**
- The recurrent state (S buffer) is not updated during verify. This means the next target decode will see stale recurrent state for skipped layers. **This is a critical problem** — the SSM state must be consistent.
- MoE FFN still runs on all 40 layers (the 3,400 us cost remains). Only the ~1,000 us SSM attention is saved.
- The "identity" skip breaks the residual stream semantics — the hidden state diverges from what the full model would produce.

### 2.3 Approach C: Partial Last-K Layers (Verify Only Recent Layers)

Run only the last K layers (e.g., layers 35-39) during verify. This is a variant of Approach A but preserves more context.

**Pros:**
- The last layers (especially layer 39, the last full-attention layer) have the most refined representations
- Less quality degradation than skipping all SSM layers
- Recurrent state only needs updating for the last few SSM layers

**Cons:**
- Still requires a new graph type or conditional
- The "which K layers" question is empirical
- KV cache for earlier layers is already computed, but recurrent state for SSM layers must still be maintained

---

## 3. Entry Points: Where Verify Calls Decode

### 3.1 Server-Side Speculative Loop

`tools/server/server-context.cpp:3630-3640` — the verify step:
```cpp
auto accepted = common_sampler_sample_and_accept_n(
    slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
```

This calls `llama_decode(slot.ctx_tgt, batch)` internally (via the sampler). The `ctx_tgt` is the target context with `ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT`.

### 3.2 Speculative Implementation

`common/speculative.cpp:971-1060` — `common_speculative_impl_draft_mtp::process()`:
- Receives the verify batch (all draft tokens + seed token)
- Calls `llama_decode(ctx_tgt, batch)` at line 1039
- Extracts `h_nextn` embeddings for each accepted token
- Stores them in `verify_h` for the accept() step

The `process()` method is the target-side verify. It always uses the full model graph.

### 3.3 Graph Selection Flow

```
llama_decode(ctx_tgt, batch)
  -> llama_context::decode()           [llama-context.cpp:1960]
  -> llama_context::sched_reserve()    [llama-context.cpp:2659]
     gtype = ctx_type_to_graph_type(cparams.ctx_type)  // = LLM_GRAPH_TYPE_DECODER
  -> model.build_graph(gparams)        [llama-model.cpp:2236]
  -> build_arch_graph(params)          [qwen35moe.cpp:153]
  -> graph::graph()                    [qwen35moe.cpp:155]  — full 40-layer loop
```

The graph type is determined solely by `cparams.ctx_type`. There is no per-batch graph type selection.

### 3.4 Where to Inject the Skip

Two injection points:
1. **At the context level:** Add a `LLAMA_CONTEXT_TYPE_VERIFY` context type that maps to a new graph type. Requires creating a separate context for verify.
2. **At the cparams level:** Add a `bool skip_ssm_verify` flag to `llama_cparams`. The graph build checks this flag to decide whether to skip SSM layers.

Option 2 is simpler but requires the flag to be accessible in the graph build (it already is via `cparams`).

---

## 4. Risk Assessment

### 4.1 Acceptance Rate Drop

**Severity: HIGH**

The verify step compares draft tokens against target logits. If the target logits come from a truncated model (10 layers instead of 40), the distribution will differ. The acceptance rate depends on how well the truncated distribution correlates with the full distribution.

**Mitigation:** Measure acceptance rate with and without SSM skip. If the drop is <5%, the trade-off may be worth it.

### 4.2 Recurrent State Corruption

**Severity: CRITICAL**

SSM layers maintain a recurrent state (S buffer) that is updated every decode step. If SSM layers are skipped during verify, the S buffer for those layers is not updated. The next target decode will see stale S state, causing **silent quality degradation** for all subsequent tokens.

**Mitigation:** The SSM state must be rolled back after verify (similar to how the KV cache is rolled back for rejected drafts). The `n_rs_seq` rollback mechanism (`cparams.n_rs_seq`) already exists for this purpose. But: if SSM layers are skipped, there's no state to roll back to — the state was never updated.

**This makes Approach B (identity skip) unsafe without additional state management.**

### 4.3 KV Cache Corruption

**Severity: LOW**

KV cache is only updated in attention layers (every 4th layer). Skipping SSM layers does not affect KV cache. The existing rollback mechanism (`ckpt.load_tgt` in `server-context.cpp:3660`) handles KV cache for rejected drafts.

### 4.4 Hidden State Divergence

**Severity: MEDIUM**

The `h_nextn` embedding (used to seed the MTP head) comes from layer 39's output. If SSM layers are skipped, `h_nextn` reflects only attention layers. The MTP head is trained to expect `h_nextn` from the full 40-layer model. This could degrade draft quality in subsequent cycles.

**Mitigation:** The MTP head could be retrained (not feasible here) or the verify skip could be disabled every N cycles to recalibrate.

---

## 5. Prototype Plan

### Recommended Approach: Graph-Level Skip with State Freeze

Build a `LLM_GRAPH_TYPE_DECODER_VERIFY` graph that:
1. Iterates only the 10 full-attention layers (il = 3, 7, 11, ..., 39)
2. Includes MoE FFN on those layers only
3. Feeds the output head from layer 39's output
4. Does NOT update recurrent state (S buffer) — the SSM state is frozen during verify

The key insight: **the verify step does not need to update any state**. It only needs logits for acceptance. The state update happens during the target decode (which runs the full 40 layers). So skipping SSM during verify is safe IF the verify step is treated as a "read-only" operation that doesn't modify SSM state.

### Implementation Steps

| Step | File | Change | Risk |
|------|------|--------|------|
| 1 | `src/llama-graph.h:33-36` | Add `LLM_GRAPH_TYPE_DECODER_VERIFY` to enum | LOW |
| 2 | `src/llama-hparams.h` | Add helper `is_full_attn(uint32_t il)` for clarity | LOW |
| 3 | `src/models/qwen35moe.cpp` | Add `graph_verify` class that loops only attention layers | MEDIUM |
| 4 | `src/models/qwen35moe.cpp:153` | Add dispatch for new graph type in `build_arch_graph` | LOW |
| 5 | `src/llama-context.cpp:26-31` | Add mapping for new ctx_type (or reuse a flag) | MEDIUM |
| 6 | `src/llama-cparams.h` | Add `bool decoder_verify_skip_ssm` flag | LOW |
| 7 | `tools/server/server-context.cpp:3630` | Use verify graph for the verify step (new ctx_type or flag) | HIGH |
| 8 | `common/speculative.cpp:971` | Ensure verify batch uses the verify graph | HIGH |
| 9 | Benchmark | Measure acceptance rate, TG, quality degradation | — |

### Critical Detail: Recurrent State Handling

The verify graph must NOT call `build_rs()` or `build_recurrent_attn()` for SSM layers. The recurrent state (S buffer) is only read/written by `build_layer_attn_linear()` (`qwen35moe.cpp:366-490`). If the verify graph never enters that function, the S buffer is never touched. The `llm_graph_input_rs` input (`inp->get_recr()`) can be null for the verify graph.

The `build_inp_mem_hybrid()` call at `qwen35moe.cpp:176` creates both `inp_attn` (KV cache) and `inp_rs` (recurrent state) inputs. For the verify graph, only `inp_attn` is needed.

### Estimated Savings

| Component | Current (us) | After Skip (us) | Savings |
|-----------|-------------|-----------------|---------|
| SSM attention (30 layers) | ~1,000 | 0 | ~1,000 |
| MoE FFN (30 SSM layers) | ~2,550 | 0 | ~2,550 |
| MoE FFN (10 attn layers) | ~850 | ~850 | 0 |
| Full attention (10 layers) | ~1,700 | ~1,700 | 0 |
| Output/embeddings | ~700 | ~700 | 0 |
| **Total** | **~6,843** | **~3,250** | **~3,593 (52%)** |

Note: The MoE FFN savings depend on whether we skip FFN for SSM layers too. If we skip the entire SSM layer (attn + FFN), savings are ~3,593 us. If we skip only SSM attention (keep FFN), savings are ~1,000 us.

### First Experiment: Skip SSM Attention Only (Keep FFN)

The safest first experiment:
- Skip only the `build_layer_attn_linear()` call for SSM layers
- Keep the MoE FFN for all 40 layers
- This saves ~1,000 us (15% of verify compute)
- The hidden state still flows through FFN, reducing distribution shift
- Recurrent state is NOT updated for SSM layers (must verify this is safe)

This is the minimal viable change to validate the approach before attempting the full skip.

---

## 6. Prototype Implementation (DONE)

### 6.1 Implementation Summary

**Approach used:** Option 2 (cparams flag), Phase 1 (skip SSM attention only, keep MoE FFN).

**Files changed:**
| File | Change |
|------|--------|
| `src/llama-cparams.h` | Added `bool skip_ssm_verify` field |
| `src/llama-context.cpp` | Env var `LLAMA_SKIP_SSM_VERIFY=1` sets the flag |
| `src/models/qwen35moe.cpp` | Skip `build_layer_attn_linear()` when flag is set; ensure SSM state input buffers are allocated |

**How it works:**
- When `LLAMA_SKIP_SSM_VERIFY=1`, the qwen35moe `graph` constructor skips the 30 `build_layer_attn_linear()` calls for SSM layers
- `cur` (the attn_norm output) passes through to the residual: `cur = ggml_add(ctx0, attn_norm(inpL), inpL)`
- The MoE FFN still runs on all 40 layers, preserving hidden state flow through FFN
- SSM recurrent state (S buffer) is NOT updated during the skip (no `build_rs` call)
- The SSM state input tensors are still expanded into the graph to ensure buffer allocation (prevents null-buffer crash in `set_input`)

### 6.2 Benchmark Results

All tests on Romulus (7900XTX single GPU), Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf, batch=1 ubatch=1 threads=4, q8_0 KV cache.

**TG-only (64 tokens):**

| Config | wall_ms | tps | Compute Buffer |
|--------|---------|-----|---------------|
| Baseline | 612.46 | 104.50 | 4.2975 MiB |
| Skip-SSM | 350.64 | 182.52 | 0.9629 MiB |
| **Delta** | **-42.7%** | **+74.7%** | **-77.6%** |

**TG-only + GPipe stages=3 (20 tokens):**

| Config | wall_ms | tps | Compute Buffer |
|--------|---------|-----|---------------|
| Baseline | 206.94 | 96.65 | 4.2975 MiB |
| Skip-SSM | 116.70 | 171.38 | 0.9629 MiB |
| **Delta** | **-43.6%** | **+77.4%** | **-77.6%** |

**PP+TG (256 prompt + 20 gen, TG phase only):**

| Config | wall_ms | tps | Compute Buffer |
|--------|---------|-----|---------------|
| Baseline | 203.05 | 98.50 | 4.2975 MiB |
| Skip-SSM | 115.41 | 173.30 | 0.9629 MiB |
| **Delta** | **-43.2%** | **+75.9%** | **-77.6%** |

### 6.3 Key Findings

1. **Compute buffer reduction is 77.6%** — SSM attention dominates the graph memory footprint
2. **TG throughput improves 75-77%** consistently across all configurations
3. **The improvement does NOT depend on GPipe** — it's purely from removing SSM attention compute
4. **This is Phase 1 only** (skip SSM attention, keep MoE FFN). The predicted 52% savings from full skip would be even larger

### 6.4 Next Steps: Server Integration

For production use in the speculative decoding pipeline, we need:

1. Create a separate `ctx_verify` context with `LLAMA_CONTEXT_TYPE_VERIFY` (maps to a new graph type that skips SSM)
2. Share KV cache between `ctx_tgt` (authoritative, full model) and `ctx_verify` (fast, skip-SSM)
3. During MTP verify, use `ctx_verify` for logits (fast, approximate acceptance check)
4. `ctx_tgt` still runs the full model for state updates and generation
5. After verify, roll back `ctx_verify`'s KV cache for rejected positions

The key architectural insight: **the verify step does not need to update any state**. It only needs logits for acceptance decisions. The state update (KV cache + SSM state) happens during the target decode, which uses the full 40-layer model.

---

## 7. Quality Verification: Output Garbling Test

### 7.1 Test Setup

Ran `llama-gpipe-profiler --sample` with a 3,500-token prompt (~13K chars covering quantum computing, CRISPR, Byzantine Empire, and deep learning). Generated 100 tokens with temperature 0.7, top-k 40, top-p 0.95.

### 7.2 Results

**Baseline (no skip):** Coherent output — Chinese tutorial text about Spring Framework's RestTemplate, followed by English StackOverflow-style Q&A about PDF/image saving and function outputs. Well-formed sentences, proper grammar, multi-paragraph structure.

**Skip-SSM (LLAMA_SKIP_SSM_VERIFY=1):** COMPLETE GARBAGE. Random Unicode fragments, repeated tokens ("熟悉熟悉熟悉...", "ArrArrArr...", "Pr Pr Pr..."), mixed-language nonsensical output, no coherent sentences. **Output quality collapses entirely.**

### 7.3 Root Cause Analysis

The identity skip (`attn_norm(inpL) + inpL` passthrough) compounds over 30 SSM layers:
1. Each skipped layer replaces SSM attention output with a simple normalized residual
2. The hidden state diverges from the full-model distribution at each layer
3. The FFN processes the degraded signal but cannot compensate for 30 layers of divergence
4. The output head sees a severely corrupted representation → collapses into repetitive patterns

The synthetic token cycling in `run_gen` (cycling through token IDs 0, 1, 2, ...) hid this quality issue. The profiler measured compute time accurately but did not test output correctness.

### 7.4 Updated Risk Assessment

| Risk | Severity | Status |
|------|----------|--------|
| Output garbling | **CRITICAL** | **CONFIRMED** — output collapses completely |
| Acceptance rate drop | HIGH | Not yet measured (requires MTP pipeline test) |
| Recurrent state corruption | CRITICAL | Not triggered in this test (state not updated) |

### 7.5 The Upper Bound: +75% TG Throughput (REVISED by D7.6)

> **D7.6 revision:** The +75% TG figure was measured in an isolated `run_gen` synthetic token cycle that only exercised the skip-SSM path. D7.6's per-kernel profiling (`docs/research/d76-rocprofv3-kernel-profile.md`) shows SSM layers consume only **2.4% of real GPU kernel time** (~10 µs/layer vs 38.8 µs for attention). The +75% gain is an artifact of the synthetic benchmark, not realizable in production. **Revised upper bound: ~5% TG** (and only if quality can be preserved). The refinement catalog below (R1-R5) is archived pending a realistic MTP acceptance benchmark.

The Phase 1 prototype establishes a hard upper bound for SSM attention elimination:

| Metric | Value | Interpretation |
|--------|-------|---------------|
| TG throughput improvement | **+75% to +77%** | Maximum possible gain from eliminating SSM compute |
| Compute buffer reduction | **-77.6%** | SSM attention consumes >3/4 of graph memory |
| Quality impact | **Catastrophic** | Output collapses into repetitive garbage |

This upper bound is useful: any refined approach that preserves quality must fall between 0% (baseline) and +75% (theoretical max). The question becomes: how much of the 75% can we capture while keeping output coherent?

### 7.6 Refinement Catalog (Archived — D7.6 Revised Upper Bound)

> **Status:** Archived. D7.6 profiling showed SSM = 2.4% of GPU time, revising the skip-SSM upper bound from +75% to ~5%. The approaches below are preserved for reference but are not actionable without (a) a realistic MTP acceptance benchmark and (b) a revision of the SSM cost model.

Each approach below reduces the quality cost at the expense of giving up some of the 75% speed gain. Listed in order of estimated effort from lowest to highest.

#### Approach R1: MTP Verify Only (no generation)

**Concept:** Use skip-SSM only for the MTP acceptance check. The verify step needs logits to compare against draft tokens — it does NOT need to produce coherent text. If the skip-SSM logit distribution correlates with the full-model distribution, acceptance decisions remain valid even though the logits are degraded.

**Key question:** Does a degraded model still accept the same draft tokens? This requires running the full speculative decoding pipeline and measuring acceptance rate correlation, not output quality.

**Estimated savings:** +75% on the verify step (same as prototype). Net pipeline impact depends on verify's share of total decode time (~53% of the SLOW 12,946 µs step = ~6,843 µs savings).

**Risk:** Acceptance rate could drop if the distributions diverge too far. An acceptance drop >10% would wipe out the speed gain (fewer accepted tokens = more full decodes needed).

**Implementation:** Dual-context setup (ctx_tgt + ctx_verify) with shared KV cache. ctx_verify has `LLAMA_CONTEXT_TYPE_VERIFY` mapping to a skip-SSM graph. ctx_tgt remains the authoritative model for generation and state updates.

#### Approach R2: Partial Skip (every Nth SSM layer)

**Concept:** Skip only a fraction of SSM attention layers. For example:
- Skip 50% (15/30 layers): alternate skip/keep → ~37% speed gain
- Skip 33% (10/30 layers): skip every 3rd → ~25% speed gain
- Skip 20% (6/30 layers): skip every 5th → ~15% speed gain

The keep:skip ratio controls the quality-speed trade-off. More kept layers → better signal preservation → less garbling.

**Implementation:** Extend `cparams.skip_ssm_verify` to a float `[0.0, 1.0]` representing the fraction of SSM layers to skip. Use a deterministic pattern (e.g., skip first N%, keep rest) to avoid per-token branching overhead.

**Estimated savings:** Proportional to skip fraction (~3.3% per skipped SSM layer out of 30).

**Risks:**
- Still no recurrent state update for skipped layers → state drift over multiple steps
- Pattern selection (which layers to skip) is empirical — early layers vs late layers may have different impact

#### Approach R3: Simplified SSM (lightweight approximation)

**Concept:** Replace the full gated delta net with a cheaper approximation instead of identity passthrough:
1. **Linear projection only:** `cur = W * attn_norm(inpL)` — a single matmul instead of conv + scan + gate + norm + output projection. ~80% compute reduction per SSM layer.
2. **Conv only (skip scan):** Keep the 1D convolution but skip the expensive SSM scan. ~50% compute reduction per SSM layer.
3. **Reduced-rank SSM:** Use a smaller inner dimension (d_inner/2) for the SSM state. ~40% compute reduction.

**Key insight:** The identity skip fails because the SSM output is fundamentally different from the normalized input. Even a crude linear approximation may preserve enough signal to prevent total collapse.

**Implementation:** Requires per-layer weights for the approximation (trained or derived from existing weights). A linear projection could reuse existing `ssm_out` weights but applied directly to the norm input rather than after the full SSM pipeline.

**Estimated savings:** 40-80% per SSM layer depending on approximation fidelity.

**Risks:**
- Requires training or deriving approximation weights (not a pure inference-time change)
- Approximation fidelity vs compute savings trade-off is unknown without experimentation

#### Approach R4: KV-Only Skip (keep recurrent state)

**Concept:** Compute the SSM recurrent state update (conv + scan → write S buffer) but skip the output projection and gated normalization. This preserves the state dynamics (no state drift) while saving the output-side compute.

The SSM layer breakdown:
- Conv + scan (state update): ~40% of SSM compute
- Output projection + gated norm: ~60% of SSM compute

Skipping only the output side saves ~60% of SSM attention compute while keeping the recurrent state consistent. Total savings: ~60% of 1,000 µs SSM attention = ~600 µs per decode step.

**Implementation:** Call `build_rs()` + scan but skip `build_norm_gated()` + `ssm_out` projection. Use the conv output or the state tensor directly as the layer output.

**Estimated savings:** ~9% overall TG improvement (60% of the 15% that SSM attention represents).

**Risks:**
- The output-side compute IS the majority of SSM attention. Only ~9% speed gain may not justify the complexity.
- The conv output is not semantically equivalent to the full SSM output — quality impact unknown.

#### Approach R5: Recurrent State Checkpoint/Rollback

**Concept:** Combine any skip approach (R1-R4) with proper recurrent state management. Before the skip decode, checkpoint the SSM state. After the skip decode (which doesn't update state), roll back the state to the checkpoint. This prevents state drift across multiple steps.

**Implementation:** Extend the existing `n_rs_seq` rollback mechanism to support per-step checkpointing. Before each verify batch, snapshot SSM state. After verify, restore.

**Estimated savings:** N/A — this is an enabler for other approaches, not a standalone speedup.

#### Decision Matrix

| Approach | Est. Speed Gain | Quality Risk | Implementation Effort | Prerequisite |
|----------|----------------|-------------|----------------------|-------------|
| R1 (MTP verify only) | +75% on verify step | Medium (acceptance correlation unknown) | High (dual-context + server wiring) | None |
| R2 (partial skip) | +15-37% | Medium-High (state drift) | Low (extend existing flag) | None |
| R3 (simplified SSM) | +30-60% per SSM layer | Low-Medium | High (needs weights) | Training or derivation |
| R4 (KV-only skip) | ~9% overall | Low | Medium | None |
| R5 (state checkpoint) | Enabler only | — | Medium | Any skip approach |

### 7.7 When to Revisit

> **D7.6 update:** The original revisit triggers below assumed SSM was a major bottleneck. D7.6 showed it is only 2.4% of GPU time. Skip-SSM is no longer a priority vector for Slice 7.

Revisit the skip-SSM concept when:
1. **D7.5 and D7.6 are exhausted** and latency remains above target
2. **A realistic MTP acceptance benchmark exists** (to test R1 acceptance correlation)
3. **Training infrastructure is available** (for R3 weight derivation)
4. **The SLOW verify step (6,843 µs) is confirmed as the dominant bottleneck** after other vectors are optimized

### 7.8 Recommendation

Proceed to Vector B2 (D7.5, overlap RPC download with GPU compute) and Vector C (D7.6, rocprofv3 profiling). These attack different bottlenecks (RPC transfer time and per-kernel GPU utilization) and are lower-risk than any skip-SSM refinement.

The skip-SSM prototype and benchmark infrastructure remain in the codebase behind `LLAMA_SKIP_SSM_VERIFY=1` for quick re-testing if any refinement approach is attempted in the future.
