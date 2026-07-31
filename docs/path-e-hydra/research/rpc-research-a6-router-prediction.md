# A6 — Engine-Layer / MoE-Router Prediction ("MTP-Style Engine Prediction")

**Research agent — READ-ONLY**  
**Date:** 2026-07-31  
**Vector:** A6 — Use MTP draft hidden states to predict MoE router decisions for the upcoming verify batch, then pre-dispatch expert compute to the GPUs holding those weights. Second-order speculation: MTP predicts tokens (level 1), this predicts *compute* (level 2).

---

## 1. Prior-Art Survey

### 1.1 Expert-Prediction / Expert-Prefetch Systems

| System | Venue | Mechanism | Accuracy | Speedup | Relevance |
|--------|-------|-----------|----------|---------|-----------|
| **ExpertFlow** | arxiv 2410.17954 (2024-10) | Transformer-based routing-path predictor estimates expert usage across ALL MoE layers in a single forward pass; token scheduler groups tokens by predicted route; predictive expert cache loads only required experts | Not reported as %; 93.72% GPU memory reduction | 10× throughput vs. offloading baselines | **High** — proves per-layer prediction is feasible; targets offloading |
| **Speculating Experts (Yalis)** | arxiv 2603.19289 (2026-03) | Leverages currently-computed internal representations to speculate future experts; executes speculated experts and maintains task accuracy; lightweight estimators for low-accuracy cases | "Reliably predicted" (no exact %); hit rates high enough to maintain accuracy | 14% TPOT reduction vs. on-demand CPU loading | **High** — closest to A6's "predict compute" idea |
| **MoE-SpeQ** | arxiv 2511.14102 (2025-11) | Small on-device draft model predicts the *sequence* of required experts for future tokens; runtime orchestrator prefetches from host memory during compute; adaptive governor via Amortization Roofline Model | Not reported as % | 2.34× speedup (Phi-MoE, CPU-GPU offload) | **High** — draft-predicts-experts is exactly A6's level-2 speculation |
| **DraftExpert** | arxiv 2607.24434 (2026-07) | Expansion-aware self-speculative decoding; trains one lightweight accelerator-resident draft expert per layer via self-distillation (residual, logit, router-agreement signals); confidence-expansion truncation + target-expert prefetching | 84–87% draft acceptance; **86–88% prefetch hit rate** | 1.45× decode throughput (DeepSeek-V2-Lite, Moonlight-16B-A3B) | **Very High** — self-speculation + per-layer draft experts + prefetch |
| **Pre-Attention Expert Prediction** | arxiv 2511.10676 (2025-11) | Uses activations BEFORE the attention block (same layer) with 2 linear functions + ranking-aware loss to predict expert selection; exploits ranking-preserving property of LLM functions | **93.03%** (DeepSeek V2 Lite), **94.69%** (Qwen3-30B), **97.62%** (Phi-mini-MoE) | ~15% absolute accuracy gain over SOTA predictors | **Very High** — proves pre-attention activations are a strong router predictor |
| **SpecPrefetch** | arxiv 2607.24787 (2026-06) | Shared lightweight adapter predicts next-layer expert candidates for async transfer; frozen native router still determines final execution; window-aware scheduler | Best average expert recall in 9/10 model-benchmark settings | 20% throughput improvement (Snapdragon 8 Elite) | **High** — parameter-efficient, transfer-vs-execution separation |
| **SP-MoE** | arxiv 2510.10302 (2025-10) | SD-aware expert-offloading + compute-communication pipelining; speculative expert prefetching exploits structural correspondence between draft and target models; cutoff-layer policy bounds prefetch depth | Not reported as % | 1.07–3.5× TPOT speedup | **Medium** — targets offloading, not in-VRAM dispatch |
| **SpecMoEOff** | arxiv 2508.21706 (2025-08) | Speculative decoding enlarges per-expert workload; CPU chunked-attention verify kernel; auto-tuned hyperparameters | Not reported | 2.5× decode throughput vs. SOTA offloading | **Medium** — complementary angle |

### 1.2 Routing-Analysis / Routing-Predictability Studies

| Finding | Source | Implication for A6 |
|---------|--------|--------------------|
| Top-weighted expert's output approximates full ensemble; cosine similarity up to 0.95; perplexity +5% with single expert | MoE Lens (arxiv 2603.05806) | Expert prediction waste is LOW — getting the top expert right captures most of the signal |
| Routing is task-conditioned: within-category routing similarity 0.84 vs. across-category 0.62; logistic regression on routing signatures achieves 92.5% task classification | Task-Conditioned Routing Signatures (arxiv 2603.11114) | Router decisions are highly structured and predictable from context |
| Gating residuals (pathway from previous layer) improve routing | MoE++ (arxiv 2410.07348) | Layer-L router is correlated with layer-(L-1) output — cross-layer prediction window exists |

### 1.3 Waste-Threshold Economics

The key economic question: *if router prediction accuracy is p, expert compute cost is C_expert, and mispredict penalty is P_mispredict, when does prediction help?*

From the literature:
- **ExpertFlow**: 93.72% memory reduction → prediction is good enough to cache only 6.28% of experts
- **Pre-Attention**: 93–97% accuracy with JUST two linear functions → prediction is cheap and accurate
- **DraftExpert**: 86–88% prefetch hit rate → even self-distilled single-expert-per-layer drafters are highly accurate
- **MoE Lens**: single-top-expert captures 95% of ensemble output → mispredict waste is bounded

**Waste threshold**: For the 80B-A3B model (3 active experts out of 128 total), expert compute is only ~3/128 ≈ 2.3% of FFN compute, and FFN is ~1/3 of total layer compute. So expert compute is **<1% of total decode compute**. Even with p=1.0 perfect prediction, the maximum speedup from overlapping expert compute is **<1%** — UNLESS the prediction hides a latency cost larger than compute (e.g., RPC network transfer, CPU↔GPU offload).

**Critical insight**: All prior-art systems target **expert offloading** (CPU↔GPU or disk↔GPU), where the cost of a wrong prediction is a PCIe/NVMe transfer (milliseconds), and the benefit of a correct prediction is hiding that transfer behind compute. In the fork's **layer-split** architecture, all experts are already in GPU VRAM — there is NO transfer cost to hide. This fundamentally changes the economics (see §4).

---

## 2. Fork MoE Internals — Dispatch Map

### 2.1 MoE Graph Construction

**File:** `src/llama-graph.cpp`  
**Entry point:** `llm_graph_context::build_moe_ffn()` (line 1520)

```
build_moe_ffn(cur, gate_inp, up_exps, gate_exps, down_exps, exp_probs_b,
              n_expert, n_expert_used, type_op, norm_w, w_scale,
              gating_op, il, probs_in, gate_up_exps, ...)
```

**Data flow (Qwen3Next 80B-A3B, `src/models/qwen3next.cpp:538-580`):**

```
attn_post_norm  ←── norm(attention_output)     [n_embd, n_tokens]
       │
       ▼
logits = build_lora_mm(gate_inp, cur)          [n_expert, n_tokens]     ←── ggml.c:1551
       │
       ▼
probs = ggml_soft_max(logits)                  [n_expert, n_tokens]     ←── ggml.c:1566
       │
       ▼
selected_experts = ggml_argsort_top_k(probs, n_expert_used)             ←── ggml.c:1626
       │                                          [n_expert_used, n_tokens]
       ▼
weights = ggml_get_rows(probs, selected_experts) [1, n_expert_used, n_tokens]
       │
       ▼
up = build_lora_mm_id(up_exps, cur, selected_experts)                   ←── ggml.c:1711
       │                                 GGML_OP_MUL_MAT_ID
       ▼
cur = ggml_swiglu_split(cur, up)
       │
       ▼
experts = build_lora_mm_id(down_exps, cur, selected_experts)            ←── ggml.c:1790+
```

### 2.2 Router Input — THE KEY FACT

**The router input is `cur` = `attn_post_norm` = the output of the CURRENT layer's attention block, post-norm.**

This is established by the Qwen3Next layer loop (`qwen3next.cpp:123-169`):

```cpp
for (int il = 0; il < n_layer; ++il) {           // qwen3next.cpp:123
    cur = build_norm(inpL, attn_norm, ...);       // pre-attention norm
    cur = build_layer_attn(...);                  // attention
    cur = ggml_add(cur, inpSA);                   // residual
    ffn_residual = cur;
    attn_post_norm = build_norm(cur, attn_post_norm, ...);  // ← ROUTER INPUT
    cur = build_layer_ffn(attn_post_norm, il);    // MoE uses attn_post_norm
    cur = ggml_add(cur, ffn_residual);            // residual
}
```

**Consequence:** The router for layer L is computed from `attn_post_norm(L)`, which is the output of layer L's attention. The router input is NOT available until layer L's attention completes. There is **no within-layer prediction window** — the router input and the router are produced by the same sequential computation.

### 2.3 Expert Placement in Layer-Split

In the fork's layer-split architecture:
- Layer L (including its attention, router, and ALL n_expert=128 expert tensors) lives entirely on ONE GPU — the GPU assigned to layer L.
- Expert tensors `ffn_gate_inp`, `ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps` are created per-layer in `qwen3next.cpp:94-96` and placed with that layer's device assignment.
- **There is no cross-GPU expert dispatch.** All experts for a layer are local to that layer's GPU.

### 2.4 MUL_MAT_ID Kernel Dispatch

**CUDA:** `ggml_cuda_mul_mat_id()` at `ggml-cuda.cu:2945`  
- Selects among mmq (quantized), mmf (float), mmvq (quantized vec) paths based on weight type and batch size
- **Assertion at line 2952:** `GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft) && "mul_mat_id does not support split buffers")` — expert weights cannot be in split buffers
- Requires stream synchronization for the sort/reorder step (line ~3010-3030) — **cannot use CUDA graphs for variable-expert dispatch**

**CPU:** `ggml_compute_forward_mul_mat_id()` at `ggml-cpu.c:1573`  
- Sorts tokens by expert, gathers into contiguous batches, dispatches GEMM per expert

### 2.5 80B-A3B Model Parameters

- `n_layer = 48` (qwen3next.cpp:26: `case 48: type = LLM_TYPE_80B_A3B`)
- `n_expert = 128` (typical for Qwen3-Next; loaded from GGUF `expert_count` KV)
- `n_expert_used = 3` (inferred from "A3B" = active 3B params; loaded from `expert_used_count` KV)
- `n_embd = 20480` (80B backbone hidden size)
- `n_ff_exp` = per-expert feed-forward length
- **Expert compute fraction:** 3/128 × (FFN fraction of layer) ≈ **<1% of total decode compute**

---

## 3. Prediction-Window Analysis

### 3.1 Within-Layer Window — DOES NOT EXIST

The router input (`attn_post_norm`) is the attention output. The router (softmax + topk) and the expert compute (MUL_MAT_ID) are sequential in the graph:

```
attention → post_norm → [ROUTER: softmax+topk] → [EXPERT: mul_mat_id]
```

The router input is only available AFTER attention completes. The router itself is microseconds (softmax over 128 logits + top-3 selection). There is no window to predict the router BEFORE attention, because the router's input IS the attention output.

### 3.2 Cross-Layer Window — NEGLIGIBLE

Layer L+1's router input depends on layer L's FULL output (attn + MoE + residual). You cannot predict layer L+1's router from layer L's input without simulating layer L's attention, which is as expensive as computing it.

### 3.3 Cross-Token (Batch) Window — THEORETICAL BUT SMALL

In a batched decode of K tokens, all K attention outputs for layer L are computed (batched), then all K routers, then all K expert computes. A predictor could pre-compute all K routers from the batched attention inputs, but:
- The router is already batched and fast (microseconds for K tokens)
- The expert compute is the bottleneck, and it requires the router output (expert selection)
- No RPC cost to hide (experts are local to the layer's GPU)

### 3.4 MTP Draft-to-Target Window — THE PROPOSED WINDOW

In MTP self-speculation:
1. Target verifies a batch, producing final hidden states h_t
2. Draft runs ahead, producing K draft tokens with hidden states h_{t+1}...h_{t+K} via `llama_get_embeddings_nextn_ith(ctx_dft, i)` (speculative.cpp:1114)
3. Target verifies the draft batch in one batched decode

**The proposed prediction:** Use the draft's K hidden states to predict the target's K router decisions for the verify batch, THEN pre-dispatch expert compute.

**Problem 1 — Representation mismatch:** The draft's hidden state is the output of the LAST draft layer (after all 48 layers). The target's layer-L router input is the output of layer L-1's attention. These are very different representations. Predicting a per-layer router from a final-layer hidden state requires learning a many-to-many mapping across 48 layers.

**Problem 2 — No RPC cost to hide:** In layer-split, the experts for layer L are on the SAME GPU as layer L's attention and router. Pre-dispatching expert compute does not save any RPC transfer because there is no transfer — the expert weights are already in the GPU's VRAM. The GET_TENSOR cost documented in AGENTS.md is for cross-layer transfers (layer L's output → layer L+1's input), not for expert dispatch.

**Problem 3 — Expert compute is <1% of total:** Even with perfect prediction, overlapping expert compute saves <1% of decode time.

### 3.5 The Pre-Attention Window (from prior art)

The Pre-Attention Expert Prediction paper (arxiv 2511.10676) shows that activations BEFORE the attention block in the SAME layer predict expert selection with 93-97% accuracy. In the fork's graph:

```
inpL (layer input) → attn_norm → attention → post_norm → [ROUTER]
```

The "pre-attention activation" is `attn_norm(inpL)` or just `inpL` (the output of the previous layer). This IS available before attention runs. However:
- Using it requires running a SEPARATE predictor (2 linear functions) per layer
- The prediction would need to run DURING attention to be useful (overlapping predictor+attention, then expert)
- The expert still needs the attention output as its input (the `cur` in `mul_mat_id(up_exps, cur, selected_experts)`), so expert compute cannot start until attention completes
- The only overlap is: predict router during attention, then start expert immediately after attention (saving the microseconds of router compute)

---

## 4. RPC / Scheduler Interaction

### 4.1 Current RPC Graph Dispatch

**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:2764` (`ggml_backend_rpc_graph_compute`)

Five paths with different blocking semantics:
- `GRAPH_COMPUTE_STAGE` (GPipe) — BLOCKS
- `GRAPH_RECOMPUTE_ALL` (multi-device reuse) — async (fire-and-forget)
- `GRAPH_COMPUTE_ALL` (multi-device first-time) — BLOCKS
- `GRAPH_RECOMPUTE` (single-device reuse) — async
- `GRAPH_COMPUTE` (single-device first-time) — BLOCKS

The dominant cost is GET_TENSOR (25.3% of blocking RPC time, ~10 calls/token) for cross-layer tensor transfers — NOT expert dispatch.

### 4.2 What Pre-Dispatch Would Require at RPC Level

To pre-dispatch expert compute before the router decision is known, you would need:

**(a) A "speculative expert compute" RPC command** — `RPC_CMD_GRAPH_COMPUTE_SPEC_EXPERT` that takes a PREDICTED expert selection and runs MUL_MAT_ID on the predicted experts. If the prediction is wrong, the result is discarded. This requires:
- New RPC protocol message type
- Server-side support for speculative execution with rollback
- Client-side mechanism to compare predicted vs. actual router output and discard stale results

**(b) Graph splitting** — Split the MoE graph into "router subgraph" and "expert subgraph", run the expert subgraph speculatively on predicted experts while the router subgraph runs for real. This requires:
- Scheduler support for speculative graph execution
- Dependency tracking between speculative and real results
- Rollback/cancel mechanism for wrong predictions

**(c) Both require the expert weights to be REMOTE** — i.e., the model must be in an expert-offloaded or expert-distributed configuration where expert compute happens on a different device than the router. In the fork's layer-split, router and experts are co-located, so there is no device boundary to overlap across.

### 4.3 Interaction with A2′/A3 Infrastructure

A2′ (graph-uid reuse) and A3 (async GRAPH_COMPUTE) are about eliminating per-token RPC stalls. A6 (router prediction) is about overlapping expert compute with router compute. These are orthogonal:
- A2′/A3 reduce the cost of GET_TENSOR and graph submission (the documented 71% + 25% of blocking RPC)
- A6 would reduce the cost of expert dispatch (which is <1% of compute and 0% of RPC cost in layer-split)

A6 does NOT require A2′/A3 infrastructure, but A2′/A3 do not enable A6 either.

---

## 5. Feasibility Verdict

### 5.1 Viability Assessment

| Approach | Viable? | Why |
|----------|---------|-----|
| **(a) Scheduler-side graph reordering** | **NO** | Router input (`attn_post_norm`) is not available earlier in the graph. Cannot reorder expert ops before router without predicting the router input itself. |
| **(b) RPC-protocol speculative pre-dispatch** | **NO** | In layer-split, experts are co-located with their router on the same GPU. No RPC transfer to hide. Pre-dispatch saves zero RPC cost. |
| **(c) After A2′/A3 infrastructure** | **NO** | A2′/A3 address RPC stalls, not expert dispatch. They don't create a prediction window that doesn't exist. |

### 5.2 Expected Gain for 80B-A3B MoE

- **Expert compute fraction:** 3/128 × (FFN ≈ 1/3 of layer) ≈ **<1% of total decode**
- **RPC cost of expert dispatch in layer-split:** **0%** (experts are local to layer GPU)
- **Maximum theoretical speedup (perfect prediction):** **<1%** (overlapping expert compute with router compute, saving microseconds of router latency)
- **Realistic speedup (93% prediction accuracy, accounting for predictor overhead):** **<0.5%**

### 5.3 Kill Criteria — MET

| Criterion | Threshold | Actual | Met? |
|-----------|-----------|--------|------|
| Router prediction accuracy | >90% for net positive | 93-97% achievable (Pre-Attention paper) | Accuracy is NOT the problem |
| Prediction window size | >1ms to amortize dispatch overhead | **0ms within-layer** (router input = attention output); **<0.01ms cross-layer** | **MET — window too small** |
| Expert compute fraction | >5% of total to matter | **<1%** (3/128 active) | **MET — too small** |
| RPC cost to hide | Expert dispatch must incur RPC | **0% RPC cost** in layer-split (co-located) | **MET — nothing to hide** |
| MTP draft hidden state → per-layer router mapping | Must be learnable | Final-layer hidden state is far from per-layer router input; 48 layers × 128 experts = very high-dimensional target | **MET — representation gap** |

### 5.4 Root Cause of Non-Viability

**The fork's layer-split architecture eliminates the very cost that router prediction is designed to hide.** All prior-art systems (ExpertFlow, MoE-SpeQ, DraftExpert, SpecPrefetch) target expert OFFLOADING, where experts live in CPU/disk and must be fetched over PCIe/NVMe — a millisecond-scale cost that prediction can hide behind compute. In layer-split, all 128 experts per layer are already in the GPU's VRAM, local to the router. There is no transfer to hide, and the expert compute itself is <1% of total.

### 5.5 What WOULD Make A6 Viable

A6 would become viable if the fork adopted an **expert-offloading** or **expert-distributed** architecture:
1. **Expert offloading:** Store inactive experts in CPU RAM, prefetch predicted experts during attention (like ExpertFlow/MoE-SpeQ). Requires CPU RAM ≥ expert pool size (128 × 3/4 of FFN ≈ 42 GiB for 80B).
2. **Expert-distributed (expert-parallel):** Place different experts on different GPUs, requiring all-to-all dispatch (like DeepSeek-V3 production). The router prediction would then hide the all-to-all transfer latency.
3. **Sub-layer splitting:** Split individual expert FFNs across GPUs (not the fork's current layer-split). Prediction could then hide the intra-expert transfer.

None of these are the fork's current architecture. The fork's layer-split is optimal for VRAM-constrained fitting, not for expert-dispersed parallelism.

---

## 6. Recommendations

### 6.1 Kill A6 — Do Not Pursue

The vector is **not viable** for the fork's current layer-split architecture. The kill criteria are met on three independent axes: (1) no prediction window exists within the graph, (2) expert compute is <1% of total, (3) there is no RPC cost associated with expert dispatch to hide.

### 6.2 Capture the Learning

The prior art is valuable for a different deployment scenario:
- If the fork ever moves to **expert offloading** (to fit larger models), ExpertFlow/MoE-SpeQ/DraftExpert provide proven blueprints
- The Pre-Attention prediction technique (2 linear functions, 93-97% accuracy) is lightweight enough to be a building block for future work
- The 80B-A3B model's concentrated expertise (MoE Lens finding) suggests that even coarse prediction (top-1 expert) captures most of the signal

### 6.3 Alternative Attack Vectors for the Same Goal

If the goal is "break past the 7900 XTX's 10.2 ms/token floor," A6 is the wrong lever. Better vectors:
- **V4 (pipeline parallelism):** Overlap compute across tokens via wavefront dispatch — amortizes the per-token floor
- **V5 (MTP self-speculation):** Already proven to generate 2-3 tokens per decode, directly multiplying throughput
- **Sub-layer expert distribution:** Redesign layer-split to distribute experts across GPUs (requires new placement-plan infrastructure)

---

## 7. Key References

| Paper | arxiv ID | URL |
|-------|----------|-----|
| ExpertFlow | 2410.17954 | https://arxiv.org/abs/2410.17954 |
| Speculating Experts (Yalis) | 2603.19289 | https://arxiv.org/pdf/2603.19289.pdf |
| MoE-SpeQ | 2511.14102 | https://arxiv.org/abs/2511.14102 |
| DraftExpert | 2607.24434 | https://arxiv.org/pdf/2607.24434.pdf |
| Pre-Attention Expert Prediction | 2511.10676 | https://arxiv.org/abs/2511.10676 |
| SpecPrefetch | 2607.24787 | https://arxiv.org/abs/2607.24787 |
| SP-MoE | 2510.10302 | https://arxiv.org/abs/2510.10302 |
| SpecMoEOff | 2508.21706 | https://arxiv.org/abs/2508.21706 |
| MoE Lens | 2603.05806 | https://arxiv.org/pdf/2603.05806.pdf |
| Task-Conditioned Routing Signatures | 2603.11114 | https://arxiv.org/abs/2603.11114 |
| MoE++ | 2410.07348 | https://arxiv.org/abs/2410.07348 |
| Self-Speculative MoE (SS-MoE) | ACM 3792218 | https://dl.acm.org/citation.cfm?id=3792218 |
| Cascade (utility-driven speculation for MoE) | — | vLLM-based, truncated result |

---

## Appendix A — Key Fork Symbols

| Symbol | File:Line | Role |
|--------|-----------|------|
| `build_moe_ffn()` | `src/llama-graph.cpp:1520` | MoE graph builder |
| `ggml_mul_mat_id()` | `ggml/src/ggml.c:3335` | MUL_MAT_ID op definition |
| `ggml_cuda_mul_mat_id()` | `ggml/src/ggml-cuda/ggml-cuda.cu:2945` | CUDA MoE kernel |
| `ggml_compute_forward_mul_mat_id()` | `ggml/src/ggml-cpu/ggml-cpu.c:1573` | CPU MoE kernel |
| `build_layer_ffn()` | `src/models/qwen3next.cpp:538` | Qwen3Next MoE/dense dispatch |
| `llm_build_decode` loop | `src/models/qwen3next.cpp:123-169` | Layer iteration (attn→MoE) |
| `ggml_backend_rpc_graph_compute()` | `ggml/src/ggml-rpc/ggml-rpc.cpp:2764` | RPC graph dispatch |
| `common_speculative_impl_draft_mtp` | `common/speculative.cpp:832` | MTP self-speculation impl |
| `llama_get_embeddings_nextn_ith()` | `src/llama-ext.h:121` | Draft hidden state extraction |
| `RPC_CMD_GRAPH_COMPUTE_ALL` | `ggml/src/ggml-rpc/ggml-rpc.cpp:192` | Multi-device graph submit |
| `mul_mat_id does not support split buffers` | `ggml-cuda.cu:2952` | Expert weight placement constraint |
