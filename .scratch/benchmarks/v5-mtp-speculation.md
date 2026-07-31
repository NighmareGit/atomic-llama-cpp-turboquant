# V5 — MTP Self-Speculation Attempt (2026-07-24)

**Date:** 2026-07-24  
**Build:** 83605ae27 (10202)  
**Status:** BLOCKED — VRAM insufficient for dual model load

## Hypothesis

MTP self-speculation (`--spec-type draft-mtp --spec-draft-n-max 2`) generates 2-3 tokens per decode step, amortizing multi-GPU RPC overhead across multiple tokens. If row-split adds 5% overhead per decode but self-speculation generates 2 tokens per decode, net throughput can exceed single-GPU.

## Attempts

### Attempt 1: Same model as draft (35B self-speculation)

```
-m 35B-model.gguf -md 35B-model.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

**Result:** SIGSEGV during warmup. Root cause: loading the 35B model twice requires ~44 GiB VRAM, far exceeding the 7900 XTX's 24 GiB.

### Attempt 2: Small draft model (9B MTP draft + 35B target)

```
-m 35B-model.gguf -md 9B-MTP-model.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

**Result:** OOM during model load — "cudaMalloc failed: out of memory" (4.1 GiB allocation failed). Combined VRAM: 22 GiB target + 5.5 GiB draft = 27.5 GiB > 24 GiB available on 7900 XTX.

## Analysis

MTP self-speculation requires loading two model instances simultaneously (target + draft). The draft model is either:
- The same model (self-speculation via MTP heads) — requires 2× VRAM
- A smaller model from the same family — still requires significant extra VRAM

For the 35B Q6_K model (21.86 GiB), any dual-model configuration exceeds the 7900 XTX's 24 GiB VRAM. Even with layer-split offloading some target layers to the 5060 Ti, the draft model alone (5.5 GiB for the 9B) consumes the remaining headroom.

## Prerequisites for V5

For MTP self-speculation to work on this hardware:
1. A GPU with ≥30 GiB VRAM (to fit target + draft simultaneously), OR
2. MTP implementation that doesn't load a separate draft model (true self-speculation using the model's internal MTP heads), OR
3. Draft model offloaded to a different GPU (adds cross-GPU overhead, defeating the purpose)

## Verdict

**V5 MTP self-speculation is not viable on current hardware.** The 24 GiB 7900 XTX cannot fit both target and draft models. Even the smallest draft model (9B Q4_K_M, 5.46 GiB) pushes total VRAM beyond 24 GiB when combined with the 35B target.

This vector was already identified as killed for cross-model MTP in Phase 2 (Slice 3C). Self-speculation via same-model MTP heads suffers the same VRAM constraint.

## Next Steps

- V5 can be revisited if the Docker 3060 Ti is fixed and can host the draft model
- Row-split parallelism (V0) would reduce per-GPU VRAM requirements, potentially freeing space for a draft model

## References

- `slice-3c.md`: Phase 2 cross-model MTP research (KILLED)
- AGENTS.md V5 attack vector description
