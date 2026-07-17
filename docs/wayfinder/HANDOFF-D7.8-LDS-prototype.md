# Handoff: D7.8 LDS Activation Caching Prototype

**Date:** 2026-07-17 (updated 2026-07-17)
**Branch:** `Path-D-Gpipeline-Assembly-Line` (HEAD: `19db22abb`, +10 ahead of origin)
**Node:** ROMULUS (7900XTX, ROCm 7.2.3, kernel 6.17.0-35-generic)
**See also:** `OPTIMIZATION-LANDSCAPE-L1-L3.md` -- full Layer 1-3 optimization framework with all leads and their status

---

## Pipeline Status

`prototype -> improve-codebase-architecture -> code-review -> testing -> diagnosing-bugs -> implement`

| # | Step | Status |
|---|------|--------|
| 1 | Prototype (D7.8) | Done |
| 2 | Smoke test (baseline) | Done |
| 3 | Benchmark (tg128) | Done |
| 4 | improve-codebase-architecture | Pending |
| 5 | code-review | Pending |
| 6 | diagnosing-bugs | Pending |
| 7 | implement | Pending |

**Steps 1-3 completed.** GPU inference works with both baseline and LDS prototype.
The "garbage tokens" bug was caused by incorrect CMake configuration (see Bug below).
Benchmark shows functional correctness but no significant throughput gain from LDS prototype yet.

---

## What Was Completed

### D7.8 LDS Prototype (`ggml/src/ggml-cuda/mmvq.cu`)
- Inlined `vec_dot_q4_K_q8_1` computation directly into the LDS kernel (lines 745-870)
- Eliminates function-call overhead through `__shared__` pointer -- inlined code reads `y_lds` as `__shared__` directly
- Cooperatively loads 24 `block_q8_1` structures from global into `__shared__` once per MMVQ iteration (16x reduction in redundant reads per half-warp)
- Guarded behind `-DGGML_HIP_MMVQ_LDS_PROTOTYPE=ON` CMake flag

### CMake Integration
- `ggml/CMakeLists.txt` line 222: `GGML_HIP_MMVQ_LDS_PROTOTYPE` option (OFF by default)
- `ggml/src/ggml-hip/CMakeLists.txt` line 148: compile definition guard

### Standalone Test (created, not run)
- `tests/test-lds-mmvq.hip.cu`: HIP standalone test for LDS kernel correctness against CPU reference
- **Status: STILL NOT RUN** as of D7.8 handoff. Requires manual compilation with `-DGGML_HIP_MMVQ_LDS_PROTOTYPE=ON`. Carried forward to Slice 7 as a diagnostic tool to isolate the kernel-level delta. See `docs/wayfinder/D7-REEXAMINATION.md`.

### Documentation
- Wayfinder `TRACKING.md`: D7.1-D7.8 phase status, D7.7 (WMMA) and D7.8 (LDS) tickets
- `MASTER-ORCHESTRATION-PLAN.md`: Updated next actions
- NTFS/FUSE mmap hard-link corruption documented in `src/llama-mmap.cpp:441-460` (separate issue)

---

## The Bug: Garbage Tokens During Decode

**RESOLVED** -- the bug was a CMake misconfiguration, not a kernel bug.

### Root Cause
The original cmake command used `-DGGML_HIPBLAS=ON` which is **not a recognized variable** in this project's CMake files. The correct flag is `-DGGML_HIP=ON`. Additionally:

| Issue | Incorrect | Correct |
|-------|-----------|---------|
| HIP enablement | `-DGGML_HIPBLAS=ON` (silently ignored) | `-DGGML_HIP=ON` |
| HIP compiler path | `/opt/rocm-7.2.3/bin/clang++` | `/opt/rocm-7.2.3/lib/llvm/bin/clang++` |
| GPU target variable | `-DAMDGPU_TARGETS=gfx1100` | `-DCMAKE_HIP_ARCHITECTURES=gfx1100` |

Without `-DGGML_HIP=ON`, the `ggml_add_backend(HIP)` call in `ggml/src/CMakeLists.txt:470` is skipped entirely, and the HIP backend is never compiled. All previous "GPU" tests were silently falling back to CPU inference.

### Correct CMake Command
```bash
cmake -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_RPC=ON \
  [-DGGML_HIP_MMVQ_LDS_PROTOTYPE=ON]
```

### Verified Working
- **CPU-only inference:** Produces correct tokens (model files are intact)
- **GPU baseline** (`19db22abb`): Correct tokens, 174.7 t/s prompt, 65.3 t/s gen
- **GPU LDS prototype** (`19db22abb` + uncommitted LDS diff): Correct tokens, 175.7 t/s prompt, 63.4 t/s gen
- **Both models tested:** Gemma-4-12B Q4_K_M and Qwen3.5-9B Q4_K_M
- **No garbage tokens** on any configuration

### Benchmark Results (tg128, Gemma-4-12B Q4_K_M)

| Metric | Baseline | LDS Prototype | Delta |
|--------|----------|---------------|-------|
| Prompt processing | 174.7 t/s | 175.7 t/s | +1.0 t/s |
| Generation | 65.3 t/s | 63.4 t/s | -1.9 t/s |

The LDS prototype is functionally correct but shows no significant improvement in end-to-end throughput at this stage. The MMVQ kernel is a small fraction of total generation time; the LDS optimization reduces shared-memory load latency for Q4_K quantized weights during single-token decode, but this is not yet the dominant bottleneck. Further profiling and micro-benchmarks (e.g. the standalone `test-lds-mmvq.hip.cu`) are needed to isolate the kernel-level gain.

> **Re-examination note:** The -1.9 t/s regression may be explained by LDS bank conflicts, extra instructions, or compiler artifacts. D7.6 profiling shows quantize_q8_1 = 7.9% of GPU time — the LDS prototype attacks the MMVQ kernel which is a smaller fraction still. Running the standalone test with `rocprofv3 --kernel-trace` (now working per D7.6) is the recommended next step to isolate the kernel-level delta before deciding whether to invest further. See `docs/wayfinder/D7-REEXAMINATION.md`.

---

## Build Configuration

### Active build dir: `build-hip/`
```bash
cmake -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_RPC=ON \
  -DGGML_HIP_MMVQ_LDS_PROTOTYPE=ON
```

### Build output
```
-- HIP and hipBLAS found
-- Including HIP backend
```

### Other build dirs:
- `build-hip-test/` -- 1.6MB binary, built 2026-07-16, older commit `7985f6b90`
- `build-rocm-docker/` -- 18K binary, built 2026-07-15
- `build-cuda/`, `build-cuda-gpu/` -- older, pre-hip

---

## Key Files Modified

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/mmvq.cu` | LDS prototype kernel with inlined vec_dot (lines 1-70 guard, 745-870 kernel, 1060 dispatch) |
| `ggml/CMakeLists.txt` | `GGML_HIP_MMVQ_LDS_PROTOTYPE` option (line 222) |
| `ggml/src/ggml-hip/CMakeLists.txt` | Compile definition guard (line 148) |
| `tests/test-lds-mmvq.hip.cu` | Standalone HIP test for LDS kernel (unused) |
| `docs/wayfinder/TRACKING.md` | D7.x phase status updates |

---

## Next Steps

> **Re-examination update:** The original next steps below (target 80+ t/s) are superseded. The baseline already exceeds 80 t/s on newer models. The real question is whether LDS can improve the MMVQ kernel's contribution to decode time. See `docs/wayfinder/D7-REEXAMINATION.md` for the full retrospective.

### 1. Run Standalone Test (recommended)
Compile and run `tests/test-lds-mmvq.hip.cu` with `-DGGML_HIP_MMVQ_LDS_PROTOTYPE=ON` to isolate the kernel-level delta. Use `rocprofv3 --kernel-trace` (working per D7.6) to compare LDS vs stock MMVQ kernel timing.

### 2. Root-cause the -1.9 t/s regression
Determine whether the degradation is from LDS bank conflicts, extra instructions, or compiler artifact. If fixable, LDS could complement Slice 7's kernel-anvil vectors (multiplicative effect).

### 3. Pipeline review (deferred)
- **improve-codebase-architecture**: Review and refactor LDS kernel structure
- **code-review**: Review all changes
- **diagnosing-bugs**: Any remaining issues
- **implement**: Commit and integrate

---

## Model Files

| Model | Path | Size | Status |
|-------|------|------|--------|
| Gemma-4-12B Q4_K_M | `/home/hunter/models/gemma-4-12b-it-Q4_K_M.gguf` | 6.7G | Verified working |
| Qwen3.5-9B Q4_K_M | `/home/hunter/models/Qwen3.5-9B-MTP-Q4_K_M.gguf` | 5.5G | Verified working |

---

## Session Artifacts

Compaction segments (if needed for deep context recovery):
```
/home/hunter/.grok/sessions/%2Fhome%2Fhunter%2Fprojects%2Fpath-d-gpipeline-assembly-line/019f698e-8a8c-7851-8343-6d3340dde2ba/compaction/
```
See `INDEX.md` in that directory for the table of contents.
