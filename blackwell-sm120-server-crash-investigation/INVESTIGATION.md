# Blackwell sm120 Server Crash Investigation

**Status: CLOSED (2026-07-09)** - resolved by CUDA `-INFINITY` fix. Archived reference only.

## Overview

This folder documents the comprehensive investigation into a llama-server crash bug specific to NVIDIA Blackwell architecture (sm120 / compute capability 12.0) on the RTX 5070 Ti.

**Hardware**: RTX 5070 Ti (Blackwell, sm_120, 16GB GDDR7), i7-13700K
**Build**: llama.cpp with CUDA 12.9.2, MSVC 19.44.35214.0, CMake 4.3 + Ninja Multi-Config
**Build Output**: `build-cuda-b-bin/` (see [../blackwell-windows-build-guide/README.md](../blackwell-windows-build-guide/README.md) for reproducible build steps)

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Environment & SDKs](#environment--sdks)
3. [Code Changes Made](#code-changes-made)
4. [Problem Statement](#problem-statement)
5. [Test Setups](#test-setups)
6. [llama-server Crash Investigation](#llama-server-crash-investigation)
7. [llama-cli Validation](#llama-cli-validation)
8. [All llama-server Args Tested](#all-llama-server-args-tested)
9. [Model Files](#model-files)
10. [Test Scripts](#test-scripts)
11. [Research & AI Prompts](#research--ai-prompts)
12. [Insights & Suggestions](#insights--suggestions)
13. [Related Files](#related-files)

---

## Executive Summary

### What Works
- **llama-cli** works perfectly on Blackwell with all tested models
- Performance: 60-120+ tokens/s generation, no output corruption
- Zero MSVC #221-D warnings after -INFINITY fix
- Core ggml-cuda inference engine is sound

### What Failed (Pre-Fix, Historical)
- **llama-server** crashed silently on Blackwell during/after first request
- Crash pattern: token generation (n_decoded ~100) or post-response cleanup
- No error message, no Windows Event log entry, no crash dump
- 13+ configuration variations failed identically before the CUDA fix

### Resolution (2026-07-09) - CLOSED
The silent crashes were caused by MSVC/nvcc `#221-D` from `-INFINITY` expanding to `-((float)(1e+300))` in CUDA kernels (`softmax.cu`, `topk-moe.cu`, `cross-entropy-loss.cu`). On Blackwell sm_120 with CUDA 12.9, this produced invalid sentinel values in block-reduce and softmax paths used during server batch decoding (multi-slot, `n_seq_max=4`), while single-sequence `llama-cli` was less affected.

**Fix**: IEEE-754 bit-cast helpers in `ggml/src/ggml-cuda/common.cuh` (`neg_inf_f32()`, `neg_inf_f32_host()`, `sentinel<T>()`). Clean rebuild required.

Post-fix verification: all investigation configs pass (see [README.md](README.md), [TEST-RESULTS.md](TEST-RESULTS.md)).

- Official smoke: `scripts/cuda-windows-5070ti/smoke-llama-server.ps1` -> `SMOKE_OK`
- MTP 20k: `-c 20480 -np 1 -ctk turbo3 -ctv q8_0 --spec-type draft-mtp` -> ~91% draft acceptance, ~128 t/s (`mtp-20k-test/`)

**See [../blackwell-windows-build-guide/README.md](../blackwell-windows-build-guide/README.md) for the complete Windows build guide.**

---

## Environment & SDKs

### Hardware
| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 5070 Ti (Blackwell, sm_120, 16GB GDDR7) |
| CPU | Intel i7-13700K |
| OS | Windows 11 |

### Toolchain
| Tool | Version | Path |
|------|---------|------|
| CUDA Toolkit | 12.9.2 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |
| nvcc | 12.9 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe` |
| MSVC | 19.44.35214.0 | `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat` |
| CMake | 4.3 | via `cmake` command |
| Build System | Ninja Multi-Config | |

### Build Workaround
VS 2022's built-in CUDA 13.3 integration has compatibility issues. Workaround:
```powershell
cmd /c 'call "VC\Auxiliary\Build\vcvarsall.bat" x64 && cmake -B build-cuda-b-bin -G "Ninja Multi-Config" -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -DGGML_CUDA=ON ...'
```

---

## Code Changes Made

### 1. MSVC #221-D Warning Fix (COMPLETED)
**Problem**: `-((float)(1e+300))` generates `warning #221-D: ((float)(1e+300)) doesn't fit in (float)` on Blackwell nvcc

**Fix**: Replace with IEEE-754 bit-cast using `__int_as_float(0xFF800000)` for device code, `memcpy`-based for host code

**Files Modified**:
| File | Locations | Change |
|------|-----------|--------|
| `ggml/src/ggml-cuda/common.cuh` | ~lines 570-630 | Added `neg_inf_f32_host()`, `neg_inf_f32()`, `sentinel<T>()` helpers |
| `ggml/src/ggml-cuda/topk-moe.cu` | lines 105, 112, 219, 357 | 4 replacements |
| `ggml/src/ggml-cuda/softmax.cu` | ~line 87 + 7 locations | Removed duplicate local, use common.cuh version |
| `ggml/src/ggml-cuda/cross-entropy-loss.cu` | lines 17, 62 | 2 replacements |
| `ggml/src/ggml-cuda/CMakeLists.txt` | ~line 218 | Removed invalid `-use_fast_math=off` flag |
| `ggml/src/CMakeLists.txt` | ~lines 217-230 | Fixed RPC linker errors - moved RPC source injection |

**Result**: Zero #221-D warnings, 672/672 build steps clean

### 2. RPC Linker Fix (COMPLETED)
Moved RPC source injection outside `GGML_BACKEND_DL` conditional in `ggml/src/CMakeLists.txt`. Resolved 10 unresolved external symbols.

---

## Problem Statement

### Primary Bug: llama-server Silent Crash on Blackwell

**Symptoms**:
- Server starts, loads model, reports "listening on port 8080"
- On first request (via llama-cli or curl), server crashes silently
- No error message, no stdout output, no Windows Event log entry
- No crash dump generated
- Process exits immediately (no stack trace)

**Confirmed NOT to be**:
- BEX64 / STACK_BUFFER_OVERRUN (resolved earlier with clean rebuild)
- Stale build artifacts (verified with full rebuild)
- VRAM exhaustion (16GB sufficient for all tested models)
- FlashAttention specific (crashes with FA on/off)
- MTP specific (crashes with --spec-type none)

### Secondary Bug: Missing Server Flags

Several flags accepted in other llama.cpp builds are invalid in this version:
- `--slot-flash-prefill` -> exit 1
- `--log-err-std` -> exit 1
- `--verbose` -> exit 1
- `--concurrency` -> invalid flag
- `--cache-ram` (without `--cache-idle-slots`) -> invalid flag

---

## Test Setups

### Default Test Configuration
- **Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf (5.47 GB)
- **Context**: 16384
- **Threads**: 4
- **GPU Layers**: 99 (all)
- **MTP**: Enabled (draft-mtp)
- **Flash Attention**: Default / tested both on and off
- **Slot**: 1 (np 1)

### Prompt Categories
| Category | File | Content |
|----------|------|---------|
| Math | `models/test-math.json` | Train catch-up problem (relative speed/distance) |
| Logic | `models/test-logic.json` | 5-house color logic puzzle |
| Coding | `models/test-coding.json` | Python merge sort with type hints |
| Complex | `models/test-complex.json` | Multi-part: Newton's method, set theory, Rust coding |

---

## llama-server Crash Investigation

### Test Matrix

All tests use the same model, GPU, and build. Server always crashes.

| Test ID | Args | Result | Crash Point | Output File |
|---------|------|--------|-------------|-------------|
| default | ctx-size=16384 | CRASH | During startup | `server-16384.log` |
| longctx | ctx-size=16384 | CRASH | During startup | `server-longctx.log` |
| ctx8k | ctx-size=8192 | CRASH | During startup | `server-8192.log` |
| ctx4k-nomtp | ctx-size=4096, --spec-type none | CRASH | n_decoded=100 | `server-4096-nomtp.log` |
| crash-debug | ctx-size=4096 | CRASH | n_decoded=100 | `server-crash-debug.log` |
| nocheck | ctx-size=4096 | CRASH | n_decoded=100 | `server-nocheck.log` |
| faoff | --flash-attn off | CRASH | n_decoded=100 | `server-faoff.log` |
| specnone | --spec-type none | CRASH | n_decoded=100 | `server-specnone.log` |
| 1thread | -t 1 | CRASH | n_decoded=100 | `server-1t.log` |
| 4thread | -t 4 | CRASH | n_decoded=100 | `server-4t.log` |
| 8thread | -t 8 | CRASH | n_decoded=100 | |
| conc1 | --concurrency 1 | INVALID FLAG | exit 1 | |
| sfp-off | --slot-flash-prefill off | INVALID FLAG | exit 1 | |
| log-err | --log-err-std | INVALID FLAG | exit 1 | |
| verbose | --verbose | INVALID FLAG | exit 1 | |

### Successful Responses (Rare)

Only 2 responses captured before crash:

| File | Size | Config | Notes |
|------|------|--------|-------|
| `models/resp-noc.json` | 1441 bytes | Default | Math problem, 121.6 t/s |
| `models/resp-faoff.json` | 1442 bytes | --flash-attn off | Same content, FA off |

### Key Observations

1. **Crash always happens during or after token generation** - never during model loading
2. **Same model + same GPU + same build works perfectly in llama-cli** - proves core inference is sound
3. **Server-specific code path** - slot management, connection handling, or streaming output
4. **Silent crash** - no diagnostic output at all, making debugging extremely difficult
5. **Blackwell-specific** - likely involves new architecture features (GPC/DPC changes, cache hierarchy, or memory controller)

---

## llama-cli Validation

llama-cli works flawlessly on Blackwell. This confirms:
- Core ggml-cuda inference engine is correct
- -INFINITY fix works properly
- CUDA 12.9.2 + sm_120 compilation is sound
- Model loading and KV cache management work

### Test Results

| Model | Prompt Type | Prompt Tokens | Gen Tokens | Speed (t/s) | Output Quality |
|-------|-------------|---------------|------------|-------------|----------------|
| Qwen3.5-9B-MTP | Math/Logic/Coding | 660.6 | 118.6 | 95.6 | Correct answers |
| Qwen3.5-9B-MTP | Long context (16k) | 62.3 | 95.6 | 62.3 | No corruption |
| Ornith-9B | Long context (64k) | - | - | - | No NaN corruption |

### Verified
- Math answers are numerically correct
- Logic puzzle solutions are consistent
- Code output is syntactically valid
- No NaN or garbage values in output
- KV cache integrity maintained across long sequences

---

## All llama-server Args Tested

### Valid Flags (Server Started But Crashed)
```
--ctx-size 16384
--ctx-size 8192
--ctx-size 4096
--n-gpu-layers 99
--threads 4
--threads 1
--flash-attn on
--flash-attn off
--spec-type draft-mtp
--spec-type none
--no-warmup
--no-mmap
--mlock
--ctx-checkpoints 0
```

### Invalid Flags (Server Exited With Code 1)
```
--concurrency 1
--slot-flash-prefill off
--log-err-std
--verbose
--cache-ram 0
--cache-idle-slots (without --cache-ram)
```

### Typical Server Start Command
```powershell
cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\bin\Release
.\llama-server.exe --host 0.0.0.0 --port 8080 -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -ngl 99 -c 4096 -t 4 --flash-attn off
```

---

## Model Files

| Model | Path | Size | Format |
|-------|------|------|--------|
| Qwen3.5-9B-MTP | `D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf` | 5.47 GB | Q4_K_M |
| Ornith-1.0-9B | `D:\models\ornith-1.0-9b-Q5_K_M.gguf` | 6.02 GB | Q5_K_M |
| Gemma 3 12B | `D:\models\gemma-3-12b-it-Q5_K_M.gguf` | 7.87 GB | Q5_K_M |
| Gemma 4 E4B | `D:\models\gemma-4-E4B-Gemini-3.1-Pro-Reasoning-Distill-Q6_K.gguf` | 5.79 GB | Q6_K |

---

## Test Scripts

### Test Script List

| Script | Path | Purpose |
|--------|------|---------|
| `build.ps1` | `scripts/build.ps1` | Main build script (Ninja + CUDA 12.9.2) |
| `test-server.ps1` | `scripts/test-server.ps1` | Server start + request loop |
| `test-cli.ps1` | `scripts/test-cli.ps1` | llama-cli batch testing |
| `romulus-local-up.sh` | `scripts/romulus-local-up.sh` | Cluster local 2-GPU start |
| `cluster-legacy-inventory.sh` | `scripts/cluster-legacy-inventory.sh` | Legacy cluster cleanup |

### Manual Test Commands

```powershell
# Build
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 && cmake -B build-cuda-b-bin -G "Ninja Multi-Config" -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=ON -DGGML_CUDA_MMQ=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-cuda-b-bin --config Release --parallel'

# llama-cli test
cd build-cuda-b-bin\bin\Release
.\llama-cli.exe -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -n 512 -ngl 99 -c 16384 -t 4 --flash-attn off --no-warmup -p "Test prompt" -e

# Server start (crashes)
.\llama-server.exe --host 0.0.0.0 --port 8080 -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -ngl 99 -c 4096 -t 4 --flash-attn off

# Curl test (after server start)
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Research & AI Prompts

### Key Research Questions
1. Does Blackwell (sm_120) have different IEEE-754 behavior for edge cases?
2. Are there CUDA graph compatibility issues with sm_120?
3. Does FlashAttention have sm_120-specific kernels?
4. Is the crash in server's slot management or CUDA context?

### AI Prompts Used
- "Analyze MSVC #221-D warning on Blackwell nvcc"
- "Compare sm_80 vs sm_90 vs sm_120 IEEE-754 behavior"
- "llama.cpp server slot management architecture"
- "CUDA graph capture issues on Blackwell"
- "FlashAttention kernel differences across architectures"

---

## Closure Summary

### Root Cause (Confirmed)

CUDA kernels used `-INFINITY`, which MSVC/nvcc 12.9 expands to `-((float)(1e+300))` (warning #221-D). On Blackwell sm_120 this produced bad softmax/MoE sentinel values. Server multi-slot decoding hit these paths more often than single-sequence `llama-cli`.

### Misleading Hypotheses (Ruled Out Post-Fix)

| Hypothesis | Verdict |
|------------|---------|
| Server-only logic bug (slots, HTTP) | Not root cause |
| FlashAttention | Crashes stopped with FA on/off after CUDA fix |
| MTP / speculative decoding | Works with `--spec-type draft-mtp` (~91% acceptance) |
| CUDA graphs | 497 graph reuses per 500-token run, no crash |
| VRAM exhaustion | 16 GB sufficient through `-c 20480` |

### Operational Notes

1. **Clean rebuild required** after applying the `-INFINITY` fix (stale `ggml-cuda.dll` caused BEX64 earlier).
2. **Smoke script** resolves `build-cuda-b-bin/bin/Release/` (Ninja Multi-Config output path).
3. **Qwen3.5 thinking models** put output in `reasoning_content`; raise `max_tokens` or disable thinking for short answers.
4. **Recommended 5070 Ti flags**: `-np 1 --no-kv-unified -ctk turbo3 -ctv q8_0 --spec-type draft-mtp` (see `advanced_settings_probe.md`).

---

## Related Files

- **Build Guide**: [../blackwell-windows-build-guide/README.md](../blackwell-windows-build-guide/README.md)
- **Build Output**: `build-cuda-b-bin/bin/Release/`
- **Test Logs**: `models/server-*.log`
- **Response Captures**: `models/resp-*.json`
- **Model Files**: `D:\models\*.gguf`
- **Code Fixes**: `ggml/src/ggml-cuda/common.cuh`, `topk-moe.cu`, `softmax.cu`, `cross-entropy-loss.cu`

---

*Last updated: 2026-07-09*
*Status: **CLOSED** - resolved by `-INFINITY` CUDA fix + clean rebuild. No further investigation planned.*
