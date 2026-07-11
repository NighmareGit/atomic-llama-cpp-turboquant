# Research Findings & AI Prompts

**Status: CLOSED (2026-07-09)** - investigation complete. See [INVESTIGATION.md](INVESTIGATION.md) closure summary.

## Research Questions Investigated

### 1. MSVC #221-D on Blackwell nvcc
**Question**: Why does `-((float)(1e+300))` generate warning #221-D on sm_120 but not on older architectures?
**Finding**: Blackwell nvcc (12.9) has stricter range checking during compile-time float evaluation. The cast `(float)(1e+300)` overflows float range (~3.4e38).

**Resolution**: IEEE-754 bit-cast via `__int_as_float(0xFF800000)` bypasses range checking entirely. **This was the root cause of the server crash.**

### 2. IEEE-754 Behavior Across Architectures
**Question**: Does sm_120 have different IEEE-754 edge case handling?
**Finding**: No architectural difference in IEEE-754 behavior at runtime. The issue was nvcc compile-time evaluation of `-INFINITY` literals, not hardware IEEE-754 semantics.

### 3. CUDA Graph Compatibility
**Question**: Are there CUDA graph capture issues with sm_120?
**Finding**: CUDA graphs work on Blackwell post-fix. Post-fix runs show 99-497 graph reuses per request with no crashes. Initial hypothesis (graph re-capture failure) was not the root cause.

### 4. FlashAttention on sm_120
**Question**: Does FlashAttention have sm_120-specific kernels?
**Finding**: Generic sm_80+ kernels. Crash occurred with FA on and off pre-fix; both work post-fix.

### 5. Server Slot Management Architecture
**Question**: What code path does the server take that llama-cli doesn't?
**Finding**: Multi-slot decoding exercises softmax/MoE CUDA paths more heavily. The crash was in those kernels, not HTTP/slot management code.

## AI Prompts Used During Investigation

### Architecture Analysis
- "Explain llama.cpp server slot management architecture"
- "How does llama-server handle multiple concurrent requests?"
- "What's the difference between llama-cli and llama-server execution paths?"

### Bug Hypothesis Generation
- "What could cause silent crashes in CUDA server applications on new GPU architectures?"
- "Common causes of silent process termination without crash dumps on Windows"
- "CUDA graph capture failures on new GPU architectures"

### Build & Toolchain
- "CUDA 12.9 nvcc changes from 12.8 affecting sm_120"
- "MSVC 19.44 changes affecting CUDA interoperability"

## Closure Notes

| Research gap (pre-fix) | Status |
|------------------------|--------|
| CUDA 12.8 comparison | Not needed; 12.9 fix confirmed |
| nsight-systems profiling | Not needed; root cause found via code + repro |
| WinDbg live debugging | Not needed |
| Minimal reproduction | Achieved: `-c 4096` server request post-fix |
| Other sm_120 cards | 5070 Ti verified; fix is compiler-level, likely universal |

---

*Archived 2026-07-09. No further updates planned.*