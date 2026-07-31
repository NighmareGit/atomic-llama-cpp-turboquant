# D7.13 — V_DOT8_I32_IU4 Feasibility Analysis

**Date:** 2026-07-17
**Status:** **SUPERSEEDED — see correction below**
**Refs:** `docs/research/d713-dp4a-research-scope.md`, `docs/research/gfx1100-hardware-deep-dive.md`

> **CORRECTION (2026-07-17):** This analysis is **incorrect**. It treats `u[0]`, `u[4]`, etc. as single activation values, but each `u[i]` is a 32-bit int containing **4 x 8-bit q8_1 values**. The proposed activation packing (`u[0], u[4], u[1], u[5], ...`) is based on this misunderstanding. The corrected analysis is in `docs/research/d713-v-dot8-layout-analysis.md`, which concludes:
>
> - **Weight layout: COMPATIBLE** — Q4_K nibble packing matches V_DOT8 format exactly
> - **Activation layout: INCOMPATIBLE** — V_DOT8_I32_IU4 requires 4-bit inputs on both operands, but Q8_1 activations are 8-bit. This is a hard blocker.
> - **Overall: CLOSE Idea 7** — V_DOT8 cannot be used for Q4_K + Q8_1 dot products.
>
> The feasibility analysis below is preserved for historical reference but its conclusions are invalid.

---

## Question

Can `V_DOT8_I32_IU4` replace 2x `V_DOT4` in the Q4_K inner loop, despite the data layout mismatch?

## V_DOT8 Instruction Details

From ISA XML:
```
V_DOT8_I32_IU4: Compute the dot product of two packed 8-D signed or unsigned 4-bit
integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value
from the third input and store the result into a vector register.
```

**Operands:**
- SRC0: PK8_IU4 (packed 8-element 4-bit integer, 32 bits)
- SRC1: PK8_IU4 (packed 8-element 4-bit integer, 32 bits)
- SRC2: I32 (32-bit accumulator)
- VDST: I32 (32-bit result)

**Encoding:** VOP3P, Opcode 24

## Data Layout Analysis

### Q4_K Weight Layout
- 256 nibbles per block (128 bytes)
- Packed as bytes: byte[n] = nibble[2n] | (nibble[2n+1] << 4)
- 4 bytes loaded as int32: bits [3:0]=nibble0, [7:4]=nibble1, [11:8]=nibble2, [15:12]=nibble3, ...

### V_DOT8 Expected Layout (PK8_IU4)
- 8 nibbles in 32 bits
- Packing: bits [3:0], [7:4], [11:8], [15:12], [19:16], [23:20], [27:24], [31:28]

### Layout Comparison

**Critical finding:** If we load 4 bytes from Q4_K into a 32-bit register, the nibble positions are:
- bits [3:0] = nibble0
- bits [7:4] = nibble1
- bits [11:8] = nibble2
- bits [15:12] = nibble3
- bits [19:16] = nibble4
- bits [23:20] = nibble5
- bits [27:24] = nibble6
- bits [31:28] = nibble7

**THIS MATCHES V_DOT8's FORMAT EXACTLY!**

A single 32-bit load from Q4_K weight data can be used directly as V_DOT8 input.

### The Catch: Current Code Uses Two Registers

The current inner loop (`vecdotq.cuh:505-527`):
```cpp
v[0] = q4[0];  // 4 bytes = 8 nibbles
v[1] = q4[4];  // 4 bytes = 8 nibbles (next 4 bytes)

for (int i = 0; i < QR4_K; ++i) {  // QR4_K = 2
    const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;  // 4 nibbles
    const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;  // 4 nibbles
    const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],
                       ggml_cuda_dp4a(v0i, u[2*i+0], 0));
    ...
}
```

The current code processes:
- Iteration 0: 4 low nibbles from v[0] + 4 low nibbles from v[1] → chained dp4a
- Iteration 1: 4 high nibbles from v[0] + 4 high nibbles from v[1] → chained dp4a

For V_DOT8, we need all 8 nibbles in a SINGLE 32-bit register. But:
- v[0] contains 8 nibbles (4 low + 4 high) — but they're at different bit offsets
- v[1] contains another 8 nibbles

### Repacking Required

To use V_DOT8 for the low-nibble iteration:
```cpp
// Pack 8 nibbles from v[0] and v[1] into a single 32-bit register
uint32_t weights = (v[0] & 0x0F0F0F0F) | ((v[1] & 0x0F0F0F0F) << 4);
```

This gives:
- [3:0] = v[0] nibble 0 (low)
- [7:4] = v[1] nibble 0 (low)
- [11:8] = v[0] nibble 1 (low)
- [15:12] = v[1] nibble 1 (low)
- [19:16] = v[0] nibble 2 (low)
- [23:20] = v[1] nibble 2 (low)
- [27:24] = v[0] nibble 3 (low)
- [31:28] = v[1] nibble 3 (low)

But the activations must also be in matching order:
- element[0] = v[0] nibble 0 → activation[0] = u[0]
- element[1] = v[1] nibble 0 → activation[1] = u[4]
- element[2] = v[0] nibble 1 → activation[2] = u[1]
- element[3] = v[1] nibble 1 → activation[3] = u[5]
- element[4] = v[0] nibble 2 → activation[4] = u[2]
- element[5] = v[1] nibble 2 → activation[5] = u[6]
- element[6] = v[0] nibble 3 → activation[6] = u[3]
- element[7] = v[1] nibble 3 → activation[7] = u[7]

So the activation packing would be: u[0], u[4], u[1], u[5], u[2], u[6], u[3], u[7]

This requires permuting the 4 activation ints (u[0..3]) into 2 ints with interleaved nibbles.

## Cost-Benefit Analysis

### Current: 2x V_DOT4 per iteration
- V_DOT4 latency: ~4 cycles each
- Chained: inner dp4a result feeds outer dp4a → dependency chain
- Total: ~8 cycles for 2 iterations (chained)

### Proposed: V_DOT8 + repacking
- Weight packing: 2 ops (AND + shift + OR) per iteration
- Activation packing: ~6-8 ops (permuting 4 ints)
- V_DOT8: ~4 cycles
- Total: ~8-10 ops + V_DOT8 latency

### Comparison

| Metric | 2x V_DOT4 | V_DOT8 + repack |
|--------|-----------|-----------------|
| VALU ops | 0 (dp4a is not VALU) | ~8-10 |
| Dot product ops | 2x dp4a (chained) | 1x dot8 |
| Estimated cycles | ~8 (chained) | ~6-8 (repack + dot8) |
| Register pressure | Low | Higher (packed values) |

**Verdict: LOW-MEDIUM feasibility.** The data layout requires repacking both weights and activations. The packing cost (~8-10 VALU ops) may negate the benefit of 1 V_DOT8 vs 2 V_DOT4. The chained dp4a dependency is removed, but the repacking adds its own overhead.

### Potential Optimization

If the activation packing can be done once per k-block (outside the inner loop), the cost could be amortized. But the current code structure has the inner loop processing QR4_K=2 iterations, so the activation packing would need to happen inside the loop.

### Alternative: Use V_DOT8 for Activation Sum

The current code also computes `dot2` (sum of activations):
```cpp
const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1],
                   ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));
```

This sums 8 activation values. V_DOT8 with 0x01010101 as one operand could compute this in 1 instruction instead of 2 chained dp4a. But the same activation repacking issue applies.

## Conclusion

**V_DOT8 is not a drop-in replacement for the current dp4a path.** It requires:
1. Weight repacking: combine v[0] and v[1] into interleaved format
2. Activation repacking: permute u[0..7] into interleaved format
3. Two V_DOT8 calls (one for low nibbles, one for high nibbles) instead of two V_DOT4 iterations

The repacking cost makes this a marginal improvement at best. **Recommendation: Skip V_DOT8, focus on IU4 WMMA (Idea 6) which has higher potential.**
