# D7.13 — Next Steps Analysis: V_DOT8 Feasibility and Scale-Unpack Branch

**Date:** 2026-07-17
**Status:** research-only
**Refs:** `docs/research/d713-dp4a-research-scope.md`, `docs/research/gfx1100-hardware-deep-dive.md`

---

## 1. V_DOT8_I32_IU4 Data Layout Feasibility

### Question
Does Q4_K's packed nibble layout match `V_DOT8_I32_IU4`'s input format?

### Current Q4_K Weight Layout (inner loop)

From `vecdotq.cuh:505-527`:
- Weights loaded as two `int` (32-bit) values: `v[0]`, `v[1]`
- Each int contains **4 nibbles** (4 bits each)
- Nibble packing in a 32-bit int: bits [3:0], [7:4], [11:8], [15:12]
- Loop runs QR4_K=2 iterations:
  - i=0: extract low nibbles via `v[x] & 0x0F0F0F0F`
  - i=1: extract high nibbles via `(v[x] >> 4) & 0x0F0F0F0F`
- Total: 8 nibbles processed via 2x V_DOT4 (4 nibbles each)

### V_DOT8_I32_IU4 Expected Format

From ISA (VOP3P encoding):
- Computes dot product of **8 packed 4-bit elements**
- Two 32-bit input sources, each containing 8 nibbles
- Packing: likely bits [3:0], [7:4], [11:8], [15:12], [19:16], [23:20], [27:24], [31:28]

### Mismatch

| Aspect | Q4_K Layout | V_DOT8 Expected |
|--------|-------------|-----------------|
| Nibble storage | 4 nibbles per int, 2 ints | 8 nibbles in 1 int |
| Total for 8 nibbles | v[0] (4) + v[1] (4) | Single 32-bit register |
| Extraction | Shift + mask per iteration | Direct 8-nibble input |

**Verdict: LOW-MEDIUM feasibility.** The data layout does NOT directly match. Using V_DOT8 would require:
1. Packing 8 nibbles from `v[0]` and `v[1]` into a single 32-bit register
2. This requires shift + OR operations: `packed = v[0] | (v[1] << 16)`
3. The packing cost (2 ops) may negate the benefit of 1 V_DOT8 vs 2 V_DOT4

**Alternative:** If Q4_K weights could be stored in a pre-packed 8-nibble format, V_DOT8 would be a direct drop-in. But this requires changes to the weight layout (breaking compatibility with existing models).

**Recommendation:** Quick prototype to measure packing overhead vs V_DOT8 benefit. If packing is cheap (can be hidden by ALU), V_DOT8 could still win.

---

## 2. Scale-Unpack Branch Uniformity

### Question
Is the `if (j < 2)` branch in scale-unpack divergent within a wave?

### Analysis

From `vecdotq.cuh:888-894`:
```cpp
const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
const int j = bq8_offset/2;
if (j < 2) {
    aux[0] = scales[j+0] & 0x3f3f;
    aux[1] = scales[j+2] & 0x3f3f;
} else {
    aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
}
```

From `mmvq.cu:640-650`:
```cpp
const int kqs = vdr * (tid % (qi/vdr));
// For Q4_K: qi=32, vdr=2, qi/vdr=16
// kqs = 2 * (tid % 16) = 0, 2, 4, ..., 30 (repeating for tid 16-31)
```

**Computing j for each thread in a wave (32 threads):**

| tid | iqs (kqs) | iqs/2 | (iqs/2)/4 | bq8_offset | j |
|-----|-----------|-------|-----------|------------|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 2 | 1 | 0 | 0 | 0 |
| 2 | 4 | 2 | 0 | 0 | 0 |
| 3 | 6 | 3 | 0 | 0 | 0 |
| 4 | 8 | 4 | 1 | 2 | 1 |
| 5 | 10 | 5 | 1 | 2 | 1 |
| 6 | 12 | 6 | 1 | 2 | 1 |
| 7 | 14 | 7 | 1 | 2 | 1 |
| 8 | 16 | 8 | 2 | 4 | 2 |
| 9 | 18 | 9 | 2 | 4 | 2 |
| 10 | 20 | 10 | 2 | 4 | 2 |
| 11 | 22 | 11 | 2 | 4 | 2 |
| 12 | 24 | 12 | 3 | 6 | 3 |
| 13 | 26 | 13 | 3 | 6 | 3 |
| 14 | 28 | 14 | 3 | 6 | 3 |
| 15 | 30 | 15 | 3 | 6 | 3 |
| 16-31 | (same as 0-15) | | | | |

**Result: j takes values 0, 1, 2, 3 within a single wave.**

The branch `if (j < 2)` splits the wave:
- 16 threads (j=0,1): take the if branch
- 16 threads (j=2,3): take the else branch

**The branch IS divergent within a wave.**

### Divergence Cost

On RDNA3 (gfx1100):
- Wave32 divergence penalty: ~4-8 cycles for a simple if-else
- Both paths are short (2 loads + bit manipulation)
- The compiler may convert this to predicated execution (V_CNDMASK_B32) to avoid branch cost

### Optimization Opportunity

The branch selects between two scale-unpacking patterns based on `j`:
- j=0,1: scales are at offsets [j+0, j+2] with mask 0x3f3f
- j=2,3: scales are at offsets [j+2, j-2/j-0] with shift + mask

**Possible simplification:** Pre-compute both paths and select with a conditional move, or restructure the scale table to avoid the branch entirely.

**Verdict: MEDIUM-HIGH feasibility.** The branch is confirmed divergent. Simplifying it could save ~4-8 cycles per vec_dot call. But the compiler may already optimize this. Need to check compiled ISA.

---

## 3. Summary and Recommendations

| Investigation | Finding | Action |
|---------------|---------|--------|
| V_DOT8 data layout | Mismatch: Q4_K uses 4 nibbles/int, V_DOT8 wants 8/int | Quick prototype to measure packing overhead |
| Scale-unpack branch | Confirmed divergent: j=0,1 vs j=2,3 within wave | Check compiled ISA; if not predicated, simplify |

### Recommended Next Actions

1. **Check compiled ISA** for the scale-unpack branch — see if compiler already predicates it
2. **Prototype V_DOT8** with packing overhead measurement
3. **Proceed to benchmark baseline** (needed regardless)
4. **IU4 WMMA prototype** (highest ROI, independent of above)
