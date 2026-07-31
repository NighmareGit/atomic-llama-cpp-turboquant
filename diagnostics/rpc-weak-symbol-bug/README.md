# RPC Weak Symbol Bug — Full Diagnostic

**Location**: `diagnostics/rpc-weak-symbol-bug/README.md`
**Index**: [diagnostics/](../README.md)

---

Last updated: 2026-07-19 (update 8 — FULL REGRESSION SUITE VERIFIED)

---

## Root Cause

### The Weak Symbol Bug

`ggml/src/ggml-rpc-stubs.cpp` defines a **WEAK** stub of `ggml_backend_buffer_is_rpc`
that always returns `false`:

```cpp
WEAK bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    (void)buffer;
    return false;
}
```

This stub is compiled into `libggml-base.so`. The **strong** (real) implementation is
in `libggml-rpc.so`:

```cpp
bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}
```

At dynamic link time, the dynamic linker processes `libggml-base.so` BEFORE
`libggml-rpc.so`. Because the weak symbol is resolved first, the strong symbol
in `libggml-rpc.so` NEVER overrides it. All calls to `ggml_backend_buffer_is_rpc()`
resolve to the weak stub, returning `false` for ALL buffers — even legitimate RPC
buffers.

### Impact

In `serialize_tensor` (ggml-rpc.cpp), the check:

```cpp
if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
```

returns `false` for every tensor, even when:
- The buffer name is "RPC0[localhost:50056]"
- The tensor data is a valid CUDA device pointer on the RPC server

This causes `serialize_tensor` to set `result.data = 0` and `result.buffer = 0`
for ALL tensors. On the server, `deserialize_tensor` creates tensors with null
data pointers. `filter_null_src_nodes` then removes nodes that depend on these
tensors. The RPC server computes nothing, and output contains garbage.

## The Fix

**File**: `ggml/src/ggml-rpc/ggml-rpc.cpp`, `serialize_tensor` function

**Change**: Use direct function pointer comparison instead of calling
`ggml_backend_buffer_is_rpc()`:

```cpp
// Direct iface comparison avoids the weak symbol resolution bug
if (tensor->buffer && tensor->buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer) {
```

This works because:
- Both `serialize_tensor` and `ggml_backend_rpc_buffer_free_buffer` are in the
  same translation unit (`ggml-rpc.cpp`) and the same shared library
- The function pointer comparison is resolved at compile time, not through the
  PLT/GOT, bypassing the dynamic linker's weak symbol resolution

## Verification

| Config | Before Fix | After Fix |
|--------|-----------|-----------|
| RPC-only (1 RPC + CPU) | GARBLED (.c.c.c...) | CLEAN (338 chars, 37 words) |
| RPC+ROCm (5 RPC + 15 ROCm) | GARBLED (repeated parens) | CLEAN (275 chars, 31 words) |

### Multi-Model Regression Suite

All tests below use the fixed code (direct iface comparison).

| Model | Size | Config | Layers/RPC | Result |
|-------|------|--------|-----------|--------|
| Qwen2.5-1.5B Q5_K_M | 1.2G | RPC-only (1 RPC + CPU) | 28/1 | CLEAN (215 chars, coherent Chinese/English text) |
| Qwen2.5-1.5B Q5_K_M | 1.2G | RPC+ROCm (5 RPC + 15 ROCm) | 28/1 | CLEAN (coherent English text) |
| Qwen3.5-4B Q5_K_M | 3.0G | RPC-only (1 RPC + CPU) | 32/1 | CLEAN (587 chars, coherent TypeScript code output) |
| Qwen3.5-9B-MTP Q4_K_M | 5.5G | RPC-only (1 RPC + CPU) | 33/1 | CLEAN (coherent Chinese text about real estate) |
| Gemma-4 12B Q4_K_M | 6.7G | RPC-only (1 RPC + CPU) | 48/1 | CLEAN (247 chars, no OOM on 8GB RPC card) |
| TinyLlama 1.1B Q4_0 | 608M | ROCm-only | 22/0 (local) | CLEAN (no regression — fix does not affect ROCm-only path) |

Key observations:
- **No regression on Q4_0 ROCm-only**: The fix has zero impact on pure-ROCm paths because `serialize_tensor` is only called during RPC serialization. ROCm-only inference uses `ggml_backend_hip_buffer_free_buffer` which does not match the RPC function pointer.
- **No OOM on 8GB RPC card**: Gemma-4 12B at Q4_K_M (6.7GB model file) fits comfortably on the RTX 3060 Ti 8GB when using RPC-only mode.
- **Qwen3.5-9B-MTP false positive**: The garble-check heuristic flagged this as "no-words" because the output was in Chinese. Manual inspection confirmed coherent text.

### Debug Verification (with fprintf)

- 412 tensors detected as RPC buffers (RPC_OK)
- 0 tensors missed (NOT_RPC = 0)

---

## Environment

- Client: ROCm 7900 XTX (build-hip/), Linux host
- Server: CUDA RTX 3060 Ti via Docker (build-cuda/bin/rpc-server in nvidia/cuda:12.8.0)
- Model: /mnt/models/qwen2.5-1.5b-instruct-q5_k_m.gguf (28 layers)
- Port: 50056
- RPC protocol: v4.4

## Code Structure Map

### `ggml/src/ggml-rpc/ggml-rpc.cpp` (RPC Backend)
- Line ~1335: **serialize_tensor** — FIXED: uses direct iface.free_buffer comparison
- Line ~1323: `ggml_backend_buffer_is_rpc` — still has correct implementation but
  is unreachable due to weak symbol bug
- Line ~1325: `ggml_backend_rpc_buffer_free_buffer` — static function, internal linkage

### `ggml/src/ggml-rpc-stubs.cpp` (Weak Stubs)
- Line 30: WEAK stub of `ggml_backend_buffer_is_rpc` — returns false unconditionally
- Compiles into `libggml-base.so`, overrides strong symbol in `libggml-rpc.so`

### Other files
- `ggml/include/ggml-rpc.h` line 45: Declaration of `ggml_backend_buffer_is_rpc`
- `ggml/src/ggml-backend.cpp`: Many call sites use `ggml_backend_buffer_is_rpc`
