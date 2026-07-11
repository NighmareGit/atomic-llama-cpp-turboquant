# Code Changes Summary

**Status: CLOSED (2026-07-09)** - all changes below are merged and verified on RTX 5070 Ti.

## All Modifications Made During Blackwell Investigation

### 1. -INFINITY Warning Fix (MSVC #221-D)

**File**: `ggml/src/ggml-cuda/common.cuh` (lines ~570-630)
**Purpose**: Replace `-((float)(1e+300))` with IEEE-754 bit-cast to eliminate #221-D warnings on Blackwell nvcc

**Added Functions**:
```cpp
// Host-side: safe -INF creation via memcpy
static inline float neg_inf_f32_host() {
    uint32_t bits = 0xFF800000U;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// Device-side: direct bit-cast
static __device__ __forceinline__ float neg_inf_f32() {
    return __int_as_float(0xFF800000);
}

// Template sentinel for any type
static __device__ T sentinel() {
    if constexpr (std::is_same_v<T, float>) {
        return __int_as_float(0xFF800000);
    } else if constexpr (std::is_same_v<T, half2>) {
        return make_half2(__float2half(-1e30f), __float2half(-1e30f));
    }
}
```

**Files Modified**:

| File | Lines Changed | Replacements |
|------|---------------|--------------|
| `ggml/src/ggml-cuda/common.cuh` | ~570-630 | Added 3 helper functions |
| `ggml/src/ggml-cuda/topk-moe.cu` | 105, 112, 219, 357 | 4 replacements |
| `ggml/src/ggml-cuda/softmax.cu` | ~87 + 7 others | Removed duplicate, use common version |
| `ggml/src/ggml-cuda/cross-entropy-loss.cu` | 17, 62 | 2 replacements |

### 2. CMakeLists.txt Cleanup

**File**: `ggml/src/ggml-cuda/CMakeLists.txt` (~line 218)
**Change**: Removed invalid `-use_fast_math=off` flag
**Before**: `set(CUDA_FLAGS -fmad=false -use_fast_math=off -extended-lambda)`
**After**: `set(CUDA_FLAGS -fmad=false -extended-lambda)`

### 3. RPC Linker Fix

**File**: `ggml/src/CMakeLists.txt` (~lines 217-230)
**Change**: Moved RPC source injection outside `GGML_BACKEND_DL` conditional
**Result**: 10 unresolved external symbols resolved

---

## Files NOT Modified (but investigated)

- `ggml/src/ggml-cuda/mmq.cu` - Blackwell MMQ kernels, no changes needed
- `ggml/src/ggml-cuda/flash-attn.cu` - FlashAttention, no changes needed
- `ggml/src/ggml-cuda/cuda-graph.cu` - CUDA graph management, no changes needed
- `tools/server/server.cpp` - Not modified; crash was in CUDA kernels, not server layer

### 4. Smoke Script Path Fix

**File**: `scripts/cuda-windows-5070ti/smoke-llama-server.ps1`
**Change**: Resolve `build-cuda-b-bin/bin/Release/` (Ninja Multi-Config) in addition to `portable/` and `bin/`

---

## Build Artifacts (Generated, Not Committed)

All in `build-cuda-b-bin/bin/Release/`:
- `ggml-cuda.dll` (368 MB)
- `llama-server.exe` (10.7 KB stub)
- `llama-server-impl.dll` (12.4 MB)
- `llama-cli.exe`
- All other binaries built 7/9/2026 8:17-8:18 AM
