# D7.13 — IU4 WMMA Prototype Design

**Date:** 2026-07-17
**Status:** design doc (implementation pending)
**Refs:** `docs/research/gfx1100-hardware-deep-dive.md`, `docs/research/d713-dp4a-research-scope.md`

---

## Goal

Prototype the `V_WMMA_I32_16X16X16_IU4` path for Q4_K x Q8_1 matrix multiplication on gfx1100 (7900 XTX).

## Approach

Create a standalone test kernel that:
1. Uses rocWMMA's IU4 fragment API for 16x16x16 matrix multiply
2. Feeds Q4_K weights (4-bit) and Q8_1 activations (8-bit) into WMMA fragments
3. Verifies correctness against the existing dp4a path
4. Measures raw throughput vs dp4a baseline

## WMMA IU4 Data Layout Requirements

From ISA Section 7.9:
- **A matrix (weights):** 16x16 IU4, lanes 0-15 replicated into lanes 16-31
- **B matrix (activations):** 16x16 IU4, lanes 0-15 replicated into lanes 16-31
- **C/D matrix (accumulator):** 16x16 I32

### Q4_K Weight Layout for WMMA

Q4_K stores 256 4-bit weights per block (128 bytes). For WMMA:
- Need 16 rows x 16 columns = 256 weights per tile
- Each weight is 4 bits, packed 2 per byte
- The 16x16 tile maps naturally to Q4_K's 256-weight block

**Packing requirement:** WMMA expects weights in a specific lane-to-element mapping. The `AMD Matrix Instruction Calculator` can generate the exact mapping:
```
python3 matrix_calculator.py --architecture rdna3 --instruction v_wmma_i32_16x16x16_iu4 --register-layout --A-matrix
```

### Q8_1 Activation Layout for WMMA

Q8_1 stores 256 8-bit activations + scale/min per block. For WMMA:
- Need 16 rows x 16 columns = 256 activations per tile
- Each activation is 8-bit (signed), but WMMA IU4 expects 4-bit
- **Mismatch:** Q8_1 is 8-bit, WMMA IU4 expects 4-bit

**Resolution:** Use `V_WMMA_I32_16X16X16_IU8` instead of IU4 for the activation matrix:
- A matrix (weights): 16x16 IU4 (Q4_K weights)
- B matrix (activations): 16x16 IU8 (Q8_1 activations, sign handled by NEG bits)
- C/D matrix: 16x16 I32

This is a mixed-precision multiply: 4-bit weights x 8-bit activations -> 32-bit accumulator.

## Prototype Structure

### Phase 1: Synthetic Data Test
- Generate random Q4_K weights and Q8_1 activations
- Run WMMA IU4 x IU8 multiply
- Compare against CPU reference
- Verify correctness

### Phase 2: Q4_K Integration
- Load actual Q4_K block data into WMMA fragments
- Handle scale/min extraction
- Apply post-multiply corrections

### Phase 3: Performance Measurement
- Benchmark WMMA path vs dp4a baseline
- Measure VGPR usage and occupancy
- Determine if WMMA wins for ncols_dst >= 16

## rocWMMA API Mapping

From `fattn-wmma-f16.cu`:
```cpp
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;

// Fragment types for IU4 x IU8 -> I32
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, uint4_t, wmma::row_major> frag_a;  // Q4_K weights
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b;   // Q8_1 activations
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c;                // Accumulator

// Load fragments
wmma::load_matrix_sync(frag_a, weight_ptr, stride);
wmma::load_matrix_sync(frag_b, activation_ptr, stride);

// Multiply-accumulate
wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

// Store results
wmma::store_matrix_sync(result_ptr, frag_c, stride, wmma::mem_row_major);
```

## WMMA Register Layout (from Matrix Instruction Calculator)

**A matrix (weights) — Wave32:**
- 16 rows (i=0..15), 16 columns (k=0..15)
- GPR floor(k/8): k=0..7 in v0, k=8..15 in v1
- Bits [4*(k%8)+3 : 4*(k%8)] per nibble
- Lane: i and i+16 (replicated)
- **Total: 2 VGPRs** for A matrix

**B matrix (activations) — same layout as A**
- **Total: 2 VGPRs** for B matrix

**C/D matrix (accumulator):**
- 16x16 I32 values
- GPR floor(i/2), lane ((16*i)%32)+j
- **Total: 8 VGPRs** for C/D matrix

**Total VGPRs: 12** (wave32)

## Key Finding: Data Layout Mismatch

Q4_K stores 256 nibbles per block in **linear order** (nibble 0..255 packed as bytes).
WMMA expects a **2D layout** (16 rows x 16 columns, 8 nibbles per GPR).

**Resolution:** Weights must be repacked from Q4_K's linear format to WMMA's 2D format.
This repacking can be done once per block (amortized over many multiplies).

## Key Questions to Answer

1. ~~**Data layout compatibility:**~~ **Answered:** Mismatch confirmed. Repacking required.
2. **Sign handling:** Do the NEG bits correctly handle signed Q8_1 activations in IU8 mode?
3. **Scale correction:** How to efficiently apply Q4_K's per-block scale/min after WMMA?
4. **Performance:** Does WMMA's higher throughput (1024 INT4 ops/clock) outweigh repacking cost?
5. **Repacking cost:** Can the repacking be done efficiently with VALU ops or VOPD dual-issue?

## Build Configuration

```cmake
# Enable rocWMMA for MMVQ (currently only enabled for fattn)
GGML_HIP_ROCWMMA_MMVQ=ON
```

## Next Steps

1. Run Matrix Instruction Calculator for exact register layout
2. Create synthetic data test kernel
3. Integrate with Q4_K data format
4. Benchmark vs dp4a baseline
