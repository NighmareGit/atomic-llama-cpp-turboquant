# Ticket B Result: Pipeline Flags — H2 Investigation

**Date:** 2026-07-24
**Test:** Strip ALL pipeline flags (PPLUS, RPC_GET_TENSOR_DEFER, WAVEFRONT_CROSS, W2) on Qwen3.5-122B-A10B 5-GPU layer-split
**Script:** `run-122b-noflags.sh` (created from `run-122b-5gpu.sh`)
**Config:** `-sm layer`, `-ngl 43`, `-tensor-split 13,3,9,15,3`, GGML_RPC_UDP=0 (TCP), all pipeline flags disabled

## Flags Disabled

| Flag | Before | After |
|------|--------|-------|
| `GGML_PIPELINE_PLUS` | 1 | **0 (disabled)** |
| `GGML_RPC_GET_TENSOR_DEFER` | 1 | **0 (disabled)** |
| `GGML_SCHED_WAVEFRONT_CROSS` | 1 | **0 (disabled)** |
| `GGML_RPC_UDP` | 0 (TCP) | 0 (TCP, unchanged) |
| `GGML_CUDA_PDL` | 0 | 0 (unchanged) |

## Output Samples

| # | Prompt | max_tokens | Output |
|---|--------|-----------|--------|
| 1 | "Once upon a time," | 50 | `//////////////////////////////////////////////////` |
| 2 | "The capital of France is" | 50 | `//////////////////////////////////////////////////` |
| 3 | "Write a haiku about programming:" | 50 | `//////////////////////////////////////////////////` |

**Verdict: All 3 outputs are GARBLED** — identical slash characters, zero coherent English. **Output is identical to Ticket A (with pipeline flags).**

## Throughput

| # | Prompt | prompt t/s | eval tg t/s | total time |
|---|--------|-----------|-------------|------------|
| 1 | "Once upon a time," | 7.13 | 13.00 | 4.55s |
| 2 | "The capital of France is" | 7.88 | 12.89 | 4.51s |
| 3 | "Write a haiku about programming:" | 9.06 | 12.78 | 4.68s |

Average generation throughput: **~12.89 tg t/s**

Compared to Ticket A (with flags, TCP): **~12.8 tg t/s** — no significant difference.

## Log Warnings/Errors

```
grep -i "error\|warn\|fail\|unsupported" → only 1 line:

1.33.617.628 W sched_reserve: fused Gated Delta Net (chunked) not supported, set to disabled
```

- Zero `send_udp` errors (0 matches)
- Zero RPC errors
- Zero graph errors
- Model loads cleanly, all 4 slots initialized, server listening

## Verdict

**H2 FALSIFIED.** Stripping ALL pipeline flags (PPLUS, RPC_GET_TENSOR_DEFER, WAVEFRONT_CROSS, W2) produces **identical** garbled output to the with-flags configuration. The pipeline flags are NOT the cause of the `///////...` output pattern.

Key observations:
- Output is **deterministically** `0x2F` (`/`) — never varies across prompts, runs, or configurations
- This is NOT random corruption — it's a fixed output regardless of input
- The pattern suggests the logits always produce token ID 47 (`/`) regardless of what the model should be outputting
- This is consistent with a complete failure of the attention mechanism or KV-cache state, not a transport or scheduling issue

## Next Steps

Pipeline flags and transport are both ruled out. The two remaining viable hypotheses are:

1. **H3 (Gated Delta Net not supported)** — Was originally rated 0.10 likelihood (wrong failure mode: fallback was expected to degrade quality, not produce fixed output). However, the deterministic `///////...` output is more consistent with a **complete attention mechanism failure** than degraded quality. The "fused Gated Delta Net (chunked) not supported" warning is the ONLY log anomaly. If Qwen3.5's core attention cannot function without this feature, it would explain fixed-output behavior (the model cannot attend to input tokens). **This should be promoted in priority.**

2. **H4 (MoE router corruption / logits buffer offset)** — Systematic corruption producing fixed token ID 47. Requires code-level investigation of logits buffer allocation in the 5-GPU RPC path.

H5 (CPU offload) and H6 (weak GPU) remain unlikely — their failure modes (degraded quality, throughput drops) don't match deterministic fixed-output corruption.

### Recommended Next Ticket: H3 (Gated Delta Net)

Test whether the model works **single-GPU** (no RPC, no layer split). If it produces coherent output single-GPU but garbled across 5 GPUs, the problem is in the RPC/layer-split interaction with the Gated Delta Net attention mechanism. If it's garbled even single-GPU, the model or GGUF is corrupted.
