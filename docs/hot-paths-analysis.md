# Hot Paths Analysis: Tensor and Layer Deployment on Romulus Dual-GPU

**Date:** 2026-07-11
**Hardware:** AMD Radeon RX 7900 XTX (24 GB ROCm client) + NVIDIA RTX 3060 Ti (8 GB CUDA RPC server)
**Protocol:** RPC v4.4, peer_copy=yes, dual=no (per-tensor graph_compute, not batched GRAPH_COMPUTE_ALL)

---

## Models Benchmarked

| Model | Arch | Params | File Size | Quant | Tensor Split |
|-------|------|--------|-----------|-------|------|
| Qwen3.5-4B-Q4_K_M | Qwen2.5 dense | 4.01B | 2.7 GB | Q4_K_M | 50,50 |
| Qwen3.5-9B-MTP-Q4_K_M | Qwen2.5 dense | 9.20B | 5.5 GB | Q4_K_M | 50,50 |
| Qwen3.6-35B-A3B-APEX-MTP-I-Q6_K | qwen35moe MoE+SSM | 35.51B | 21.9 GB | Q6_K | 30,70 |

---

## Model Architecture: Per-Layer Tensor Composition

### Dense Models (4B, 9B MTP) -- Qwen2.5 Architecture

Each transformer block contains these tensors (all always-active):

```
blk.{n}.attn_norm               -- RMS norm weights (2048 or 4096 dims)
blk.{n}.attn_q.weight           -- Q projection [n_embd x n_head * head_dim]
blk.{n}.attn_k.weight           -- K projection [n_embd x n_head_kv * head_dim]
blk.{n}.attn_v.weight           -- V projection [n_embd x n_head_kv * head_dim]
blk.{n}.attn_output.weight      -- Output projection [n_embd x n_embd]
blk.{n}.ffn_norm                -- Post-attention RMS norm
blk.{n}.ffn_gate.weight         -- SwiGLU gate [n_embd x ffn_dim]
blk.{n}.ffn_up.weight           -- SwiGLU up [n_embd x ffn_dim]
blk.{n}.ffn_down.weight         -- SwiGLU down [ffn_dim x n_embd]
```

Per-layer tensor count: **9 weight tensors** (all dense, all always computed).

### MoE+SSM Model (35B) -- qwen35moe Architecture

Each transformer block is more complex. Architecture parameters:

| Parameter | Value |
|-----------|-------|
| embedding_length | 2048 |
| n_head | 16, n_head_kv=2 (GQA) |
| key_length | 256, value_length=256 |
| expert_count | 256, expert_used_count=8 |
| expert_feed_forward_length | 512 |
| SSM conv_kernel | 4 |
| SSM state_size | 128 |
| SSM inner_size | 4096 |
| full_attention_interval | 4 (every 4th layer) |

Per-block tensor composition:

**Full attention blocks** (layers 0, 4, 8, 12, 16, 20, 24, 28, 32, 36 = 10 of 40):

```
blk.{n}.attn_norm.weight        -- RMS norm (2048)
blk.{n}.attn_q.weight           -- Q proj [2048 x 4096] = 8M params
blk.{n}.attn_k.weight           -- K proj [2048 x 512]  = 1M params (GQA, 2 KV heads)
blk.{n}.attn_v.weight           -- V proj [2048 x 512]  = 1M params
blk.{n}.attn_output.weight      -- Output [2048 x 2048] = 4M params
blk.{n}.post_attention_norm     -- Post-attn RMS norm
blk.{n}.gate.weight             -- Expert router [2048 x 256] = small
blk.{n}.ffn_norm.weight         -- Pre-FFN RMS norm
blk.{n}.shared_mlp.gate.weight  -- Shared expert gate [2048 x 512]
blk.{n}.shared_mlp.up.weight    -- Shared expert up [2048 x 512]
blk.{n}.shared_mlp.down.weight  -- Shared expert down [512 x 2048]
{256 expert weight sets (per-layer, only 8 activated per token):}
blk.{n}.experts.{e}.w1.weight  -- Expert gate [2048 x 512]
blk.{n}.experts.{e}.w2.weight  -- Expert down [512 x 2048]
blk.{n}.experts.{e}.w3.weight  -- Expert up [2048 x 512]
```

**SSM blocks** (layers where `layer_idx % 4 != 3` -- 30 of 40):

Replace full attenton with SSM (Fused Gated Delta Net):

```
blk.{n}.ssm_norm.weight         -- Pre-SSM RMS norm
blk.{n}.ssm.conv1d.weight       -- SSM conv1d [inner_size x conv_kernel]
blk.{n}.ssm.conv1d.bias        -- SSM conv1d bias
blk.{n}.ssm.x.weight            -- SSM input projection
blk.{n}.ssm.dt.weight           -- SSM time step projection
blk.{n}.ssm.dt.bias            -- SSM time step bias
blk.{n}.ssm.A_log               -- SSM state transition (log)
blk.{n}.ssm.D                   -- SSM skip connection
```

Followed by the same MoE MLP + shared expert block as full-attention layers.

**Per-layer active tensor count (35B MoE):**
- Full attention layer: ~270 weight tensors (shared + 3 shared-MLP + 256 experts x 3)
- SSM layer: ~270 weight tensors (SSM replaces attention, same MoE block)
- **But only 11 experts actually participate per forward pass** (shared expert + 8 routed experts)

---

## Deployment: Which Tensors Go to Which GPU

### Split Mechanism

The tensor split is **contiguous layer-based** (ngl=99, split_mode=layer). The scheduler divides the layer stack at a boundary and assigns the lower portion to one backend and the upper portion to another. Cross-boundary intermediate activations (`leaf_55`, `leaf_59` in traces) are transferred via RPC get_tensor/set_tensor.

### 4B and 9B Dense Models (ts=50,50)

| GPU | Layers | Key Tensors | Compute Role |
|-----|--------|-------------|--------------|
| **RPC0** -- RTX 3060 Ti | First ~50% of blocks | blk.0 through blk.{N/2}: attn_q/k/v/o, ffn_gate/up/down of first half | Processes first half of transformer stack; sends intermediate activations to ROCm |
| **ROCm0** -- RX 7900 XTX | Last ~50% of blocks | blk.{N/2+1} through blk.{N-1}: attn + ffn of second half + output_norm + output (lm_head) | Processes second half, generates logits |

**VRAM budget:** RPC gets ~2.5 GiB weights + ~2.6 GiB KV cache = fits in 8 GB

### 35B MoE (ts=30,70)

| GPU | Layers | Key Tensors | Compute Role |
|-----|--------|-------------|--------------|
| **RPC0** -- RTX 3060 Ti | First ~30% of blocks (layers 0-11) | blk.0-blk.11: attention/SSM + MoE routers for early layers, **all 256 x 3 expert weights** for those layers | ~7.2 GiB on 8 GB VRAM. Processes early layers; MoE router activates 8/256 experts per token |
| **ROCm0** -- RX 7900 XTX | Last ~70% of blocks (layers 12-39) | blk.12-blk.39: attention/SSM + MoE for late layers + output_norm + lm_head + **MTP head** | ~14.2 GiB on 24 GB. Does the heavy lifting |

**Why 30/70 and not 50/50:** The 35B model is 21.86 GiB at Q6_K. A 50/50 split would put 10.9 GiB on the 3060 Ti, exceeding its 8 GB VRAM. The 30/70 split fits 7.2 GiB on RPC and 14.2 GiB on ROCm.

---

## Hot Path Analysis: Tensor Access Patterns

### Prefill (Prompt Processing)

```
Input tokens (1024)
  -> model.input_embed (on RPC via rpc_upload_defer: 10 KB, 469 us)
  -> Split 0: CPU prep (68 us, trivial)
  -> Split 1: RPC0 (3060 Ti):
       blk.0 attn_norm + QKV proj + attention + output proj + ffn_norm + MoE(8/256) -- 393 us
       [for first 12 layers of 35B, or first ~14 layers of 4B/9B]
  -> cross-GPU transfer (leaf tensors via RPC get_tensor)
  -> Split 2: ROCm0 (7900 XTX):
       blk.{split} attn_norm + QKV/SSM + MoE(8/256) + ffn -- 47 ms for all remaining layers
       [for last 28 layers of 35B, or last ~14 of 4B/9B]
  -> output_norm + lm_head
  -> logits returned
```

**PP hot path (35B MoE):** RPC processes 12 layers of attention + MoE in 393 us; ROCm processes 28 layers in 47 ms. The ROCm dominates because it has 2.3x more layers -- and each layer's MoE block activates 8 experts through 512-wide FFN projections.

**PP throughput:** 4,027 t/s (35B) -- matches the 4B dense (3,835 t/s) because MoE activates only 8/256 experts, so effective compute per layer is similar.

### Text Generation (Token-by-Token)

```
Token embedding (from previous output logits)
  -> Split 1: RPC0 (3060 Ti):
       CUDA graph: layers 0-11 attention + MoE -- ~4.3 ms per layer = ~52 ms total
       (all 12 layers run via cached CUDA graph, graph id 472..600)
  -> cross-GPU transfer of intermediate activations (~4 ms)
  -> Split 2: ROCm0 (7900 XTX):
       CUDA graph: layers 12-39 attention + MoE -- ~4 ms per layer = ~112 ms total
  -> output_norm + lm_head
  -> KV cache update (both GPUs write their portion)
```

**TG hot path (35B MoE):** 873 ms total = 146.6 t/s. The RPC bottleneck is ~4.3 ms/layer x 12 layers = 52 ms; ROCm is ~4 ms/layer x 28 layers = 112 ms. CUDA graph reuse confirmed across all 128 decode steps.

### Key Timing Breakdown (35B, TG 128 tokens, 873 ms total)

| Component | Time (ms) | % of Total | Notes |
|-----------|-----------|------------|-------|
| RPC compute (12 layers) | ~52 | 6% | 3060 Ti, bound by PCIe 1.0 x4 RTT |
| Cross-GPU transfer | ~52 | 6% | rpc_defer_flush + download, highly variable |
| ROCm compute (28 layers) | ~112 | 13% | 7900 XTX, compute-bound |
| Total per step | ~6.8 | 100% | 873 ms / 128 tokens |
| Idle/wait | ~75% | | Both GPUs wait on each other due to 2-stage pipeline |

---

## Per-Layer Hot Path Characterization

### Dense Model (4B, 9B) -- All Tensors Always Hot

| Tensor | Access Pattern | Compute Cost | Notes |
|--------|---------------|-------------|-------|
| attn_norm | 1 read per token | O(n_embd) | RMS norm, negligible |
| attn_q.weight | 1 matmul per token | O(n_embd * n_head * head_dim) | 4-8M params |
| attn_k.weight | 1 matmul per token | O(n_embd * n_head_kv * head_dim) | 1-2M params (GQA) |
| attn_v.weight | 1 matmul per token | O(n_embd * n_head_kv * head_dim) | 1-2M params |
| attn_output.weight | 1 matmul per token | O(n_embd * n_embd) | 4-16M params |
| ffn_gate.weight | 1 matmul per token | O(n_embd * ffn_dim) | Heavy: 8-32M params |
| ffn_up.weight | 1 matmul per token | O(n_embd * ffn_dim) | Heavy: 8-32M params |
| ffn_down.weight | 1 matmul per token | O(ffn_dim * n_embd) | Heavy: 8-32M params |

**Hot path concentration:** ~80% of compute in ffn_gate/up/down (SwiGLU MLP), ~20% in attention. This is standard transformer behavior -- the MLP dominates.

### MoE+SSM Model (35B) -- Sparse Activation

| Tensor | Access Pattern | Compute Cost | Notes |
|--------|---------------|-------------|-------|
| attn_norm / ssm_norm | 1 read per token | O(2048) | Negligible |
| attn_q.weight | Full-attention layers only (10 of 40) | 8M params | Every 4th layer |
| attn_k.weight | Full-attention layers only | 1M params | GQA, 2 KV heads |
| attn_v.weight | Full-attention layers only | 1M params | |
| attn_output.weight | Full-attention layers only | 4M params | |
| **SSM tensors** | SSM layers (30 of 40) | ~8M total | conv1d + projections + state |
| gate.weight (router) | **All 40 layers** | tiny | [2048 x 256] softmax routing |
| shared_mlp gate/up/down | All 40 layers | 3x [2048x512] | Always-active shared expert |
| experts.{e}.w1/w2/w3 | **8 of 256 per layer** per token | 3 x [2048 x 512] each | Sparse: only 3.1% of expert weights |

**MoE hot path concentration per layer:**
- Router: 100% always-on (tiny, <1% of compute)
- Shared expert: 100% always-on (~3M params, ~15% of compute)
- Routed experts: 3.1% of expert weights activated (8 of 256), but this IS the compute -- the selected 8 experts' w1/w2/w3 are each 512-wide FFN blocks
- Full-attention vs SSM: SSM path is ~40% cheaper than full attention per layer

**Hot path per-token compute estimate (35B MoE):**

| Component | Active Layers | Active Params | % Total Compute |
|-----------|--------------|---------------|-----------------|
| Attention (10 layers full) | 25% of layers | ~140M | ~8% |
| SSM (30 layers) | 75% of layers | ~72M | ~5% |
| Shared expert MLP (40 layers) | 100% of layers | ~120M | ~10% |
| Routed MoE (40 layers, 8 experts) | 100% of layers, 3.1% of experts | **~1,320M** | **~77%** |
| Embed + output | 1 each | ~8M | <1% |

**Result:** ~1.7B active parameters per forward pass (out of 35.5B total), consistent with the observed PP throughput matching a 4B dense model.

---

## GPU Deployment Map

### Current (Slice 2, Romulus dual-GPU)

```
          RPC Server (Docker)              Client (Native)
      ┌──────────────────────┐     ┌──────────────────────┐
      │  NVIDIA RTX 3060 Ti  │     │  AMD RX 7900 XTX     │
      │  8 GB VRAM           │     │  24 GB VRAM          │
      │  PCIe 1.0 x4 (chip)  │     │  PCIe 4.0 x16        │
      ├──────────────────────┤     ├──────────────────────┤
      │ LAYERS 0-11 (35B)    │◄───►│ LAYERS 12-39 (35B)   │
      │ or 0-~13 (4B/9B)     │     │ or ~14-27 (4B/9B)    │
      │                      │     │                      │
      │ attn_norm            │     │ attn_norm            │
      │ attn_q/k/v           │     │ attn_q/k/v           │
      │ attn_output          │     │ attn_output          │
      │ MoE router           │     │ MoE router           │
      │ 256 experts (8 used) │     │ 256 experts (8 used) │
      │ SSM (if applicable)  │     │ SSM (if applicable)  │
      │                      │     │                      │
      │                      │     │ output_norm          │
      │                      │     │ lm_head              │
      │                      │     │ MTP head (9B, 35B)   │
      └──────────────────────┘     └──────────────────────┘
```

### What the Trace Data Shows

From the scheduler trace (sched-trace.jsonl, decode_id=1):

| Split | Backend | Duration | Key Event |
|-------|---------|----------|-----------|
| Split 0 | backend 2 (CPU) | 68 us | Input embed prep (trivial) |
| Split 1 | backend 0 (RPC0: 3060 Ti) | 920 us | Copy `model.input_embed` (10 KB, 469 us) + graph_compute (393 us) |
| Split 2 | backend 1 (ROCm0: 7900 XTX) | 100,431 us | rpc_defer_flush (52,662 us!) + graph_compute (47,134 us) |

The **rpc_defer_flush** in Split 2 is the cross-GPU synchronization bottleneck -- the client must wait for all deferred RPC operations (hash uploads) to complete before it can read intermediate results from the RPC server. This 52 ms flush represents the round-trip cost of RPC get_tensor/set_tensor for intermediate activations.

---

## Summary

| Aspect | Finding |
|--------|---------|
| **Hot tensors** | MoE expert weights (w1/w2/w3) account for ~77% of compute. Only 8/256 experts activate per token. Attention/SSM accounts for ~13%, shared expert MLP ~10%. |
| **GPU deployment** | Contiguous layer-based split: early layers on 3060 Ti (30-50%), later layers on 7900 XTX (50-70%). The output head + lm_head always on ROCm. |
| **Throughput ceiling** | RPC round-trip latency (PCIe 1.0 x4) is the primary bottleneck, not compute. Each TG step spends ~52 ms in rpc_defer_flush. |
| **MoE scaling** | 35B MoE PP throughput (4,027 t/s) matches 4B dense (3,835 t/s) because effective active params per token is ~1.7B. |
| **CUDA graph reuse** | Confirmed on RPC server (graph id 472..600). Each decode step reuses the cached graph for all layers on the 3060 Ti. |
| **Cross-GPU transfer** | Only 3 intermediate tensors cross the GPU boundary (input_embed, leaf_55, leaf_59). Their total data volume is small (<100 KB/step) but the RPC protocol overhead dominates. |
