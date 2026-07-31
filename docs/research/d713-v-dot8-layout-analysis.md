# D7.13 -- V_DOT8_I32_IU4 Layout Analysis: Q4_K Weight Compatibility

**Date:** 2026-07-17
**Status:** research-only (read-only analysis, no code changes)
**Refs:** `ggml/src/ggml-common.h:404-419`, `ggml/src/ggml-cuda/vecdotq.cuh:505-527,862-908`, `docs/research/gfx1100-hardware-deep-dive.md`

---

## 1. Q4_K Weight Layout

### Block structure (`ggml-common.h:404-419`)

```c
typedef struct {
    union {                  // 4 bytes
        struct { ggml_half d; ggml_half dmin; };
        ggml_half2 dm;       // d = super-block scale for quantized scales
    };                       // dmin = super-block scale for quantized mins
    uint8_t scales[12];      // 12 bytes: 6-bit packed scales+mins for 8 sub-blocks
    uint8_t qs[128];         // 128 bytes: 256 nibbles (4-bit quants)
} block_q4_K;                // total = 144 bytes
```

Constants:
- `QK_K = 256` (super-block size)
- `K_SCALE_SIZE = 12`
- `QI4_K = QK_K / (4 * QR4_K) = 256 / (4 * 2) = 32` (elements per quant group)
- `QR4_K = 2` (vec-dot reduction factor for MMVQ)

### Nibble packing in `qs[]`

Each byte stores 2 nibbles: low nibble first, high nibble second.

```
Memory layout (byte address increasing left to right):

qs[0]  qs[1]  qs[2]  qs[3]  |  qs[4]  qs[5]  qs[6]  qs[7]  | ...
+------+------+------+------+  +------+------+------+------+
|n0 n1 |n2 n3 |n4 n5 |n6 n7 |  |n8 n9 |n10n11|n12n13|n14n15|
+------+------+------+------+  +------+------+------+------+
  low high  low high           low high  low high
```

When loaded as a 32-bit int (little-endian) from `qs[0..3]`:

```
Bit:  0-3  4-7  8-11 12-15 16-19 20-23 24-27 28-31
      +----+----+----+----+----+----+----+----+
      | n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7 |
      +----+----+----+----+----+----+----+----+
```

### Dequantization formula

From `ggml-quants.c:1471-1498`:
```
weight[l] = d * sc * nibble[l] - dmin * m
          = dm.x * sc * nibble[l] - dm.y * m
```

Where `sc` and `m` are the per-sub-block scale/min (6-bit values from `scales[12]`).

### How the CUDA code loads weights (`vecdotq.cuh:862-908`)

```cpp
const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
v[0] = q4[0];  // 4 bytes = 8 nibbles (qs[0..3] of this row)
v[1] = q4[4];  // 4 bytes = 8 nibbles (qs[4..7] of this row)
```

Each `v[i]` is a 32-bit int containing 8 weight nibbles.

### Inner loop nibble extraction (`vecdotq.cuh:505-527`)

```cpp
for (int i = 0; i < QR4_K; ++i) {       // QR4_K = 2
    const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;  // 4 nibbles from v[0]
    const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;  // 4 nibbles from v[1]

    const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],
                       ggml_cuda_dp4a(v0i, u[2*i+0], 0));  // chained dp4a
    ...
}
```

- `i=0`: extracts low nibbles (bits 0-3 of each byte) from v[0] and v[1]
- `i=1`: extracts high nibbles (bits 4-7 of each byte) from v[0] and v[1]
- Each dp4a processes 4 weight nibbles x 4 q8_1 values (8-bit each)

---

## 2. V_DOT8_I32_IU4 Input Format Requirements

From `gfx1100-hardware-deep-dive.md`:

```
V_DOT8_I32_IU4: dot product of packed 8-D signed 4-bit integers -> i32
```

Operands:
- SRC0 (Input A): 8 nibbles packed into 32 bits
- SRC1 (Input B): 8 nibbles packed into 32 bits
- SRC2: i32 accumulator
- VDST: i32 result = sum(A[i] * B[i]) + accumulator

Nibble packing in a 32-bit int:
```
Bit:  0-3  4-7  8-11 12-15 16-19 20-23 24-27 28-31
      +----+----+----+----+----+----+----+----+
      | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
      +----+----+----+----+----+----+----+----+
```

NEG[1:0] bits control signed/unsigned per source (0=unsigned, 1=signed).

---

## 3. Compatibility Analysis

### A. Nibble Ordering

**VERDICT: COMPATIBLE -- identical ordering.**

Q4_K stores nibbles in memory as: byte[n] = nibble[2n] | (nibble[2n+1] << 4).

When loaded as a 32-bit int (little-endian), the nibble positions are:
- bits [3:0] = nibble 0
- bits [7:4] = nibble 1
- bits [11:8] = nibble 2
- bits [15:12] = nibble 3
- bits [19:16] = nibble 4
- bits [23:20] = nibble 5
- bits [27:24] = nibble 6
- bits [31:28] = nibble 7

V_DOT8 expects exactly this layout. A single 32-bit load from `qs[]` can be used directly as V_DOT8 input. No nibble reordering needed.

### B. Packing Density

**VERDICT: COMPATIBLE -- clean mapping.**

- Q4_K: QI4_K = 32 elements per quant group, VDR = 2
- Each `int` load covers 8 nibbles = 8 weight elements
- V_DOT8 processes 8 nibbles in one instruction

The 8-nibble `int` load maps 1:1 to V_DOT8's 8-nibble input. No density mismatch.

### C. Scale Interaction

**VERDICT: COMPATIBLE -- post-multiplication.**

Q4_K dequantization: `weight = dm.x * sc * nibble - dm.y * m`

The dot product expands to:
```
sum(weight * q8) = dm.x * sc * sum(nibble * q8) - dm.y * m * sum(q8)
```

V_DOT8 produces `sum(nibble * q8)` as an i32. The scales (`dm.x`, `dm.y`, `sc`, `m`) are applied AFTER the dot product as float32 multiplications. This is independent of the dot-product instruction used.

The `sum(q8)` term (dot2 in the current code) is also computed via dp4a with `0x01010101` as the weight operand. V_DOT8 could compute this too, but the activation format mismatch (see below) prevents this.

### D. The `j<2` Branch

**VERDICT: COMPATIBLE -- can be factored out.**

The branch at `vecdotq.cuh:888-894`:
```cpp
const int j = bq8_offset/2;
if (j < 2) {
    aux[0] = scales[j+0] & 0x3f3f;
    aux[1] = scales[j+2] & 0x3f3f;
} else {
    aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
}
```

This unpacks the 6-bit scale/min values from the packed `scales[12]` array. It is independent of the dot-product computation and does not prevent V_DOT8 usage. The unpacked `sc[]` and `m[]` values are passed to the inner loop as arrays, not baked into the dot product.

---

## 4. Comparison with Current V_DOT4 Path

### Current path (2x dp4a per iteration)

```cpp
// v[0], v[1]: each 8 nibbles loaded as 32-bit int
// u[0..3]: each 4 q8_1 values (8-bit each) loaded as 32-bit int

for (int i = 0; i < 2; ++i) {
    const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;  // 4 nibbles
    const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;  // 4 nibbles
    // Each dp4a: 4 x (4-bit weight * 8-bit activation) -> i32
    const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],
                       ggml_cuda_dp4a(v0i, u[2*i+0], 0));  // chained
}
```

- 2 iterations x 2 dp4a = 4 dp4a calls total
- Each dp4a processes 4 weight nibbles
- Total: 16 weight nibbles processed per call
- dp4a is VOP3P-encoded, not VOPD-encodable (cannot dual-issue)

### V_DOT8 path (hypothetical)

```cpp
// Weight side: v[0] already has 8 nibbles in V_DOT8 format
// But current code splits v[0] into low/high halves across iterations

// To use V_DOT8 for 8 nibbles from v[0]:
// Input A = v[0]  (8 nibbles, already correctly packed)
// Input B = ???  (8 activation values as 4-bit nibbles)
```

### The Activation Problem

**This is the critical blocker.**

V_DOT8_I32_IU4 expects **4-bit inputs on both sides**. But q8_1 activations are **8-bit**:

```c
typedef struct {
    union { struct { ggml_half d; ggml_half s; }; ggml_half2 ds; };
    int8_t qs[32];   // 32 x 8-bit signed values per block
} block_q8_1;
```

Each `u[i]` in the CUDA code is a 32-bit int containing 4 x 8-bit q8_1 values. V_DOT8 cannot process 8-bit values -- it expects 4-bit nibbles.

To use V_DOT8, activations would need to be:
1. Requantized to 4-bit (lossy, defeats the purpose of q8_1)
2. Split into nibbles (doubles the number of dot products)

Neither option is viable for the q8_1 activation format.

---

## 5. Verdict

### Weight layout: COMPATIBLE

Q4_K's nibble packing order (low nibble first, high nibble second, byte-addressable) produces a 32-bit layout that is **identical** to V_DOT8's expected input format. A single `int` load from `qs[]` can be fed directly to V_DOT8 as the weight operand. No rearrangement needed.

### Activation layout: INCOMPATIBLE

V_DOT8_I32_IU4 requires 4-bit inputs on both source operands. Q8_1 activations are 8-bit. This is a fundamental format mismatch that cannot be resolved with simple repacking.

### Overall: INCOMPATIBLE

V_DOT8 cannot be used for Q4_K + Q8_1 dot products. The weight format is perfect, but the activation format is wrong.

---

## 6. Recommendation

**Close Idea 7 (V_DOT8 for Q4_K).**

The existing analysis (`docs/research/d713-vdot8-feasibility.md`) incorrectly treats `u[0]` and `u[4]` as single activation values. In reality, each `u[i]` is a 32-bit int containing 4 x 8-bit q8_1 values. The analysis's proposed activation packing (`u[0], u[4], u[1], u[5], ...`) is based on a misunderstanding of the data layout.

Even if the activation layout were compatible, the current code structure (splitting each 8-nibble register into two 4-nibble halves for dp4a) would require repacking. But this is moot given the activation format mismatch.

### Where V_DOT8 WOULD work

V_DOT8 is suitable for **4-bit weight x 4-bit activation** dot products. This would require:
- A new activation quantization format (e.g., Q4_1 instead of Q8_1)
- Or a GEMM kernel where both operands are 4-bit (e.g., Q4_K x Q4_K)

For the current Q4_K + Q8_1 path, V_DOT4 (dp4a) remains the correct instruction.

---

## 7. Summary Table

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Weight nibble ordering | COMPATIBLE | Low-nibble-first matches V_DOT8 |
| Weight packing density | COMPATIBLE | 8 nibbles per int = V_DOT8 width |
| Scale interaction | COMPATIBLE | Post-multiplication, independent of dot instruction |
| `j<2` branch | COMPATIBLE | Scale unpacking, can be factored out |
| Activation format | INCOMPATIBLE | Q8_1 is 8-bit, V_DOT8 expects 4-bit |
| Overall | INCOMPATIBLE | Activation mismatch is a hard blocker |
