# Debug-Fix: Qwen3.5-122B Gated Delta Net — Broken Output

**Date:** 2026-07-24  
**Model:** Qwen3.5-122B-A10B-Q4_K_S (70 GiB, 122B/10B MoE)  
**Root cause (confirmed):** Gated Delta Net computation produces wrong output on ALL backends (ROCm fused kernel, CPU non-fused fallback). Not an RPC/multi-GPU issue.

---

## 1. Goal

Fix the Gated Delta Net computation so the 122B model produces coherent output on at least single-GPU (7900 XTX). Multi-GPU 5-GPU is a stretch goal — single-GPU correctness is the minimum bar.

## 2. Evidence (from killed 122b-garbled investigation)

| Config | Output | Implication |
|--------|--------|-------------|
| Single-GPU, fused GDN enabled (`-c 64`) | Variable garbled unicode | Fused ROCm kernel is wrong |
| CPU-only, non-fused fallback | Variable garbled unicode | Non-fused path also wrong |
| 5-GPU, chunked disabled | Deterministic `///////...` | Zeroed recurrent state compounds broken GDN |
| Tokenizer | Works correctly | Token IDs are normal |

**Conclusion:** The GDN computation graph produces incorrect output regardless of backend. Likely a hyperparameter mismatch or tensor dimension error in `qwen35moe.cpp` / `delta-net-base.cpp` / `gated_delta_net.cu`.

## 3. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Orchestrator (YOU)                         │
│  Dispatches each loop phase, feeds results forward           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Loop N (1..5)  │◄──────────────────────┐
              └────────┬────────┘                       │
                       │                                │
    ┌──────────────────┼──────────────────┐             │
    ▼                  ▼                  ▼             │
/research        /systematic-          /plan           │
(investigate      debugging            (write          │
 code paths,      (instrument,         fix plan)       │
 dump tensors,    trace GDN                              │
 compare)         computation)                           │
                       │                                │
                       ▼                                │
                  /prototype                             │
                  (implement fix,                        │
                   build, deploy)                        │
                       │                                │
                       ▼                                │
                  /perf-verification                     │
                  (single-GPU test:                      │
                   coherent output?)                     │
                       │                                │
              ┌────────┴────────┐                       │
              ▼                 ▼                       │
            PASS              FAIL ─────────────────────┘
         (done!)          (loop back with
                           new evidence)
```

## 4. Per-Loop Phases

### Phase 1 — `/research`
Investigate the GDN code paths and identify the specific error:
- Read `qwen35moe.cpp` — GDN tensor dimension computation from GGUF metadata
- Read `delta-net-base.cpp` / `delta-net-base.cuh` — non-fused GDN computation
- Read `gated_delta_net.cu` — fused ROCm/CUDA kernel
- Dump GGUF metadata: `ssm_d_inner`, `ssm_dt_rank`, `ssm_n_group`, `ssm_d_state`
- Compare with known-working smaller Qwen3.5 model (if available)
- Hypothesis: identify which tensor dimension or computation step is wrong

### Phase 2 — `/systematic-debugging`
Build a feedback loop and instrument the GDN computation:
- Instrument: dump intermediate GDN tensors (conv output, Q/K/V extraction, delta, A/B/C matrices)
- Build deterministic single-GPU test harness (fixed prompt, fixed seed)
- Compare tensor values between ROCm fused and CPU non-fused paths
- Identify where the values diverge from expected
- Single hypothesis per loop — fix ONE thing

### Phase 3 — `/plan`
Write a fix plan based on Phase 2 findings:
- Exact code changes with line numbers
- Which file(s) to modify
- Build instructions
- Test procedure

### Phase 4 — `/prototype`
Implement the plan:
- Modify source code
- Build `llama-cli` for single-GPU (fast feedback: `build-rocm-rpc-split`)
- If fix involves CUDA code: build on remus, copy binary

### Phase 5 — `/perf-verification`
Test the fix:
- Single-GPU test with small context (`-c 64`, fused GDN enabled): output coherent?
- Single-GPU test with larger context (`-c 512`, chunked/non-fused): output coherent?
- If PASS → test 5-GPU
- If FAIL → capture garbled output pattern, compare to previous loop, feed back to Phase 1

## 5. Loop Rules

- **Max 5 loops.** If 5 loops fail to produce coherent single-GPU output → KILL with documentation.
- **One change per loop.** Fix one tensor dimension, one computation step, one hypothesis.
- **Regression check.** Each fix must not break the working 35B MoE model (test with `-m Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf` on single-GPU).
- **Fast feedback.** Single-GPU test first (seconds). Only test 5-GPU if single-GPU passes (minutes).
- **Incremental evidence.** Each failed loop must produce new evidence (dumped tensor values, identified mismatch) that guides the next loop.

## 6. Success Criteria

- [ ] Single-GPU (7900 XTX): coherent English output for ≥3 prompts at `-c 64` (fused GDN)
- [ ] Single-GPU (7900 XTX): coherent output at `-c 512` (non-fused/chunked path)
- [ ] 35B MoE regression: working model still produces correct output
- [ ] 5-GPU (stretch): coherent output across all 5 GPUs
- [ ] Code review pass if code changed

## 7. Kill Criteria

- [ ] 5 loops completed without coherent single-GPU output
- [ ] Root cause confirmed to require upstream llama.cpp changes beyond this fork's scope
- [ ] Fix would break the working 35B MoE pipeline

## 8. Loop Results

[To be filled by orchestrator — one section per loop]

---

### Files of Interest (for research agents)

| File | What it does |
|------|-------------|
| `ggml/src/ggml-cpu/qwen35moe.cpp` | Qwen3.5 MoE model implementation, GDN tensor dimension computation |
| `ggml/src/ggml-cpu/delta-net-base.cpp` | Non-fused GDN computation (CPU fallback) |
| `ggml/src/ggml-cpu/delta-net-base.cuh` | GDN CUDA kernel declarations |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | Fused GDN ROCm/CUDA kernel |
| `ggml/include/ggml-qwen35moe.h` | Qwen3.5 MoE header, architecture constants |

### Test Commands

```bash
# Single-GPU fast test (fused GDN, -c 64):
cd /home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant
HIP_VISIBLE_DEVICES=0 timeout 60 ./build-rocm-rpc-split/bin/llama-cli \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 20 -c 64 \
  -p "The capital of France is" -n 32 -t 4 2>&1

# Single-GPU larger test (non-fused, -c 512):
HIP_VISIBLE_DEVICES=0 timeout 120 ./build-rocm-rpc-split/bin/llama-cli \
  -m /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  -ngl 20 -c 512 \
  -p "Once upon a time," -n 32 -t 4 2>&1

# 35B regression test:
HIP_VISIBLE_DEVICES=0 timeout 30 ./build-rocm-rpc-split/bin/llama-cli \
  -m /home/hunter/scratch/prototype-auto/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  -ngl 99 -c 512 \
  -p "Once upon a time," -n 32 -t 4 2>&1
```
