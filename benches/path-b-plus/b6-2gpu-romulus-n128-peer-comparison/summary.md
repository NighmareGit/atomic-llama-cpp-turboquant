# Peer Comparison: Gemma 4 26B A4B vs Qwen 3.6 35B A3B

Romulus-local 2-GPU (7900 XTX ROCm + 3060 Ti CUDA RPC), TG-only profiler runs.
GGML_PIPELINE_PLUS=1, q8_0 KV cache, ctx=4096, 128 tokens, 10 reps each.

## TG Baseline (no MTP)

| Model | Arch | Params | MoE | Layers | GGUF | t/s (steady) | wall_ms |
|-------|------|--------|-----|--------|------|-------------|---------|
| Gemma 4 26B A4B | gemma4 | 25.2 B | 128/8 | 30 (pure MoE) | Q4_K 14 GB | **175.1** ± 1.9 | 731.2 |
| Qwen 3.6 35B A3B | qwen35moe | 35.5 B | 256/8 | 41 (MoE + SSM) | Q6_K 22 GB | **148.3** ± 2.3 | 862.9 |

Gemma 26B is 18% faster despite being a 8-expert MoE. Key factors:
- No SSM/hybrid layers (Qwen has Gated DeltaNet on non-attention layers)
- 11 fewer layers (30 vs 41)
- Smaller embedding dimension (2816 vs 2048, but less total compute)

## MTP Overhead (Qwen fused NextN)

| Model | MTP | t/s (steady) | wall_ms | Overhead |
|-------|-----|-------------|---------|----------|
| Qwen 3.6 35B A3B | off | **148.3** | 862.9 | — |
| Qwen 3.6 35B A3B | on (fused) | **147.9** | 865.1 | <0.3% |

MTP nextn extraction cost is negligible — fully amortized by the existing
compute pipeline. The profiler's MTP TG mode measures per-token decode
throughput including the `embeddings_nextn` extraction but excluding the
speculative draft/accept loop (synthetic tokens can't produce meaningful
acceptance sequences).

## Gemma MTP

The Gemma 4 26B uses a separate draft model (gemma4-assistant architecture),
not fused MTP like Qwen. The profiler now supports `--model-draft` for this
mode. However, the available assistant model (12B, n_embd_inp=3840) targets
a different backbone dimension than the 26B target (n_embd=2816), so the
profiler run could not complete. A matching assistant would need
n_embd_inp=2816.

The Gemma 4 31B assistant uses architecture `gemma4_mtp` which is not yet
registered in this build.

## Environment

| Field | Value |
|-------|-------|
| GIT_SHA | f2d627a6a |
| GGML_PIPELINE_PLUS | 1 |
| RPC | 127.0.0.1:50051 |
| TS (Gemma 26B) | 38,62 |
| TS (Qwen 35B) | 24,76 |
| Client | ROCm-native (7900 XTX) |
| RPC backend | CUDA (3060 Ti, docker) |
| Date | 2026-07-14 |

## Profiler

The profiler now supports `--model-draft` for separate-draft MTP models
(Gemma assistant). See `tools/llama-gpipe-profiler/README.md` for usage.

The `sched_need_reserve` bug in `set_embeddings_nextn()` was fixed — without
this, draft acceptance dropped to zero because the graph lacked the h_nextn
output buffer. The fix is a single-line addition matching the existing pattern
in `set_embeddings_layer_inp()`.
