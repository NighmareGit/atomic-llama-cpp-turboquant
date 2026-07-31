# Tensor-mode model allow-list (target models)

**Date:** 2026-07-17  
**Branch:** Path-D-Gpipeline-Assembly-Line  
**Wayfinder ticket:** [Tensor-mode model allow-list for target models](../../.scratch/allreduce-timing/issues/02-tensor-mode-model-allow-list.md)  
**Scope:** code/docs gate only; no runtime load of GGUFs.

---

## Gate (source of truth)

`llama_model_create` throws if the first device is a **meta** device (tensor split) and the arch fails:

```cpp
// src/llama-model.cpp ~314-316
if (!devices.empty() && devices[0].is_meta && !llm_arch_supports_sm_tensor(arch)) {
    throw ... "LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '...'";
}
```

`llm_arch_supports_sm_tensor` (`src/llama-arch.cpp` ~921-950) is a **denylist**: listed arches return `false`; **everything else returns `true`**.

Hard context constraints for any tensor-mode run (`src/llama-context.cpp` ~3810-3818):

| Constraint | Behavior |
|------------|----------|
| Flash attention | Required. `auto` is forced **on**; explicit off is a hard error. |
| Backend sampling | Falls back to CPU with a warning if sampler was GPU-bound. |
| `llama_params_fit` | Not implemented for tensor mode (`common/fit.cpp`) — manual VRAM sizing. |

Docs (`docs/multi-gpu.md`) also recommend `-ctk f16 -ctv f16` (or bf16/f32). The exact error string about quantized KV + tensor was **not** found in current C++ sources (doc may be stale); still treat non-quant KV as the operational baseline for TP experiments.

Weight quant (Q6_K / Q4_K_M / Q4_K_S) does **not** change the arch allow-list.

---

## Target models

| User label | Typical GGUF arch string | `llm_arch` | `llm_arch_supports_sm_tensor` | Tensor-mode (arch gate) |
|------------|--------------------------|------------|-------------------------------|-------------------------|
| Qwen3.6-35B-A3B (any quant) | `qwen35moe` | `LLM_ARCH_QWEN35MOE` | **true** (not denylisted) | **YES** |
| Qwen3.6 dense 27B (if used) | `qwen35` | `LLM_ARCH_QWEN35` | **true** | **YES** |
| Qwen3-Next | `qwen3next` | `LLM_ARCH_QWEN3NEXT` | **true** | **YES** |
| Gemma-4 dense | `gemma4` | `LLM_ARCH_GEMMA4` | **true** | **YES** |
| Gemma-4 assistant (if any) | `gemma4-assistant` | `LLM_ARCH_GEMMA4_ASSISTANT` | **true** | **YES** |

**Contrast (not targets, but easy to confuse):**

| Arch | Gate |
|------|------|
| `gemma3n` | **NO** (denylisted) |
| Pure `mamba` / `mamba2` | **NO** |
| `qwen3moe` (older Qwen3 MoE name) | **YES** (not denylisted) |

Qwen3.6-35B-A3B is hybrid MoE + SSM in graph construction, but it is **not** on the code denylist (unlike pure Mamba / several other MoE hybrids listed in docs). Arch gate allows TP; correctness/perf under TP is an empirical baseline question, not a compile-time reject.

---

## Doc vs code denylist

`docs/multi-gpu.md` lists a human-oriented denylist (Grok, MPT, OLMoE, DeepSeek2, Mamba, Gemma-3n, ...). Code denylist is similar but not identical (e.g. includes `deepseek32`, `lfm2`, omits some doc-only names). **Trust `llm_arch_supports_sm_tensor` for this branch.**

Notable for our fleet: **Qwen35/Qwen35MoE/Qwen3Next/Gemma4 are allowed** even though multi-gpu.md groups many MoE/hybrids as unsupported — those named families are a different set.

---

## Recommended TP command shape (all YES models)

```bash
# Same-process multi-GPU (e.g. 3090+3070)
llama-bench -m MODEL.gguf \
  --split-mode tensor \
  -fa on \
  -ctk f16 -ctv f16 \
  -ngl 99

# Provider A/B (from inventory ticket)
GGML_CUDA_ALLREDUCE=internal   # or nccl | none
```

Quant variant is free to vary for TG/PP comparison; it does not re-gate arch support.

---

## Implications for baselines (ticket 07)

| Row type | Models that can appear |
|----------|------------------------|
| **Tensor-mode AR matrix** | All map targets: Qwen3.6-35B (Q6/Q4), Gemma-4, Qwen3-Next — subject to VRAM fit and FA-on-device support |
| **Layer/pipeline only** | Same models; used when topology cannot form a CUDA meta group (see topology ticket) |

**Caveat (empirical, not gate):** Qwen35MoE under TP may stress MoE delay-AllReduce paths in meta (`ggml-backend-meta.cpp`); still allowed. Record quality (acceptance / garbled output) during baselines if MTP is on.

---

## Key sources

| File | Role |
|------|------|
| `src/llama-arch.cpp` `llm_arch_supports_sm_tensor` | Denylist |
| `src/llama-model.cpp` `llama_model_create` | Throw on meta + unsupported arch |
| `src/llama-context.cpp` | FA required for tensor mode |
| `src/llama.cpp` `llama_prepare_model_devices` | Builds meta device from all non-CPU devices |
| `docs/multi-gpu.md` | User docs (partially stale vs code) |
| `NEXTN.md` | Maps Qwen3.6-35B-A3B -> `qwen35moe` |
