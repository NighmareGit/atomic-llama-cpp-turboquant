# gfx1100 (RDNA 3) Hardware Deep-Dive: ISA-Level Findings

**Date:** 2026-07-17
**Status:** research reference (no code changes)
**Source:** AMD RDNA 3 ISA Reference Guide (Document 70650, Feb 2023), GPUOpen WMMA guide, rocWMMA docs, AMD machine-readable ISA XML
**Local artifacts:** ISA text extract (`docs/research/gfx1100-isa-reference.txt`), ISA XML (`/tmp/rdna3_isa/amdgpu_isa_rdna3.xml`)
**Relevance:** D7.13 (dp4a micro-optimizations), D7.10 (kernel-anvil), D7.11 (quantize_q8_1 fusion), D7.14 (LDS root-cause)

---

## 1. Execution Model

### Wave32 and Wave64
- RDNA3 supports both wave32 and wave64 execution modes
- Most compute kernels use **wave32** (32 threads per wave)
- VOPD dual-issue is **wave32 only** — illegal in wave64 (skipped)

### WGP (Work-group Processor)
- Each WGP contains 2 SIMDs
- Each SIMD has **256 VGPRs** (vector general-purpose registers)
- 2 SIMDs/CU -> effective 128 VGPRs/thread at full occupancy (512 threads)
- **RDNA3: 2 GPRs per wave** (vs RDNA4: 4 GPRs per wave) — this is why Q4_K cannot use nwarps=8 on RDNA3

### Register Files
- **SGPRs:** Scalar registers, shared across wave (up to 128)
- **VGPRs:** Vector registers, per-lane (256 per SIMD on gfx1100)
- VGPR bank conflict rules: 4 banks (indexed by SRC[1:0]), each with 3 read ports (SRC0, SRC1, SRC2)

---

## 2. VOPD Dual-Issue Mechanism

### Overview
The VOPD (Vector Operation Dual-issue) encoding allows a single shader instruction to encode **two separate VALU operations executed in parallel**.

From ISA Section 7.6:
> "The VOPD instruction encoding allows a single shader instruction to encode two separate VALU operations that are executed in parallel. The two operations must be independent of each other."

### Restrictions
- **Legal only for wave32** — must not be used by wave64 (skipped)
- Each instruction may use up to 2 VGPRs
- At most 2 SGPRs, or 1 SGPR + 1 literal, or share a literal
- SRC0 can be VGPR or SGPR (or constant); VSRC1 must be VGPR
- **VGPR bank rules (hard constraints):**
  - SRCX0 and SRCY0 must use different VGPR banks
  - VSRCX1 and VSRCY1 must use different banks
  - If both use SRC2, one must be even and the other odd
- Dest VGPRs: one must be even, the other odd
- Must not use DPP
- The two instructions must be independent

### VOPD X-Opcodes (Table 91)
| Opcode | Instruction |
|--------|-------------|
| 0 | V_DUAL_FMAC_F32 |
| 1 | V_DUAL_FMAAK_F32 |
| 2 | V_DUAL_FMAMK_F32 |
| 3 | V_DUAL_MUL_F32 |
| 4 | V_DUAL_ADD_F32 |
| 5 | V_DUAL_SUB_F32 |
| 6 | V_DUAL_SUBREV_F32 |
| 7 | V_DUAL_MUL_DX9_ZERO_F32 |
| 8 | V_DUAL_MOV_B32 |
| 9 | V_DUAL_CNDMASK_B32 |
| 10 | V_DUAL_MAX_F32 |
| 11 | V_DUAL_MIN_F32 |
| 12 | V_DUAL_DOT2ACC_F32_F16 |
| 13 | V_DUAL_DOT2ACC_F32_BF16 |

### VOPD Y-Opcodes (Table 92)
Same as X, plus:
| Opcode | Instruction |
|--------|-------------|
| 16 | V_DUAL_ADD_NC_U32 |
| 17 | V_DUAL_LSHLREV_B32 |
| 18 | V_DUAL_AND_B32 |

### Critical Finding: dp4a is NOT VOPD-encodable
**V_DOT4_I32_IU8 and V_DOT4_U32_U8 use VOP3P encoding, NOT VOPDXY.** They cannot be dual-issued via VOPD with any other VALU op. This is confirmed by:
1. V_DOT4 appears in Section 7.5 "Packed Math" (VOP3P encoding), not Section 7.6 "Dual Issue VALU"
2. V_DOT4 is absent from both VOPD X and Y opcode tables
3. The VOP3P encoding covers WMMA, V_DOT4, V_DOT8, V_DOT2, and V_FMA_MIX — none are VOPD-encodable

**Implication for D7.13:** Manual dual-issue scheduling of dp4a is **impossible** at the ISA level. The only dual-issue path is the general VALU+SALU mechanism, which the compiler already exploits.

---

## 3. Packed Math (VOP3P Encoding)

From ISA Section 7.5:
> "V_FMA_MIX_* and WMMA instructions are not packed math, but perform a single MAD operation on a mixture of 16- and 32-bit inputs. They are listed here because they use the VOP3P encoding."

### VOP3P Instruction Space
The VOP3P encoding covers:
- **WMMA** (all variants: F32/F16/BF16/IU8/IU4)
- **V_DOT4_I32_IU8** — signed/unsigned 8-bit dot product (4 elements)
- **V_DOT4_U32_U8** — unsigned 8-bit dot product (4 elements)
- **V_DOT8_I32_IU4** — signed 4-bit dot product (8 elements)
- **V_DOT8_U32_U4** — unsigned 4-bit dot product (8 elements)
- **V_DOT2_F32_F16** — FP16 dot product (2 elements)
- **V_DOT2_F32_BF16** — BF16 dot product (2 elements)
- **V_FMA_MIX_** — mixed-precision FMA

### V_DOT4 Instructions (dp4a)
```
V_DOT4_I32_IU8: dot product of packed 4-D signed/unsigned 8-bit integers -> i32
V_DOT4_U32_U8: dot product of packed 4-D unsigned 8-bit integers -> u32
```
- Uses NEG[1:0] bits for signed/unsigned indication (0=unsigned, 1=signed per source)
- NEG[2] must be zero (undefined behavior otherwise)

### V_DOT8 Instructions (NEW — unexploited)
```
V_DOT8_I32_IU4: dot product of packed 8-D signed 4-bit integers -> i32
V_DOT8_U32_U4: dot product of packed 8-D unsigned 4-bit integers -> u32
```
- These process **8 nibbles in one instruction** — twice the width of V_DOT4
- Same NEG[1:0] signed/unsigned control
- **Implication for D7.13:** Could potentially replace 2x V_DOT4 in the Q4_K inner loop with 1x V_DOT8, but data layout compatibility must be verified

---

## 4. WMMA (Wave Matrix Multiply Accumulate)

From ISA Section 7.9:
> "Wave Matrix Multiply-Accumulate (WMMA) instructions provide acceleration for common matrix arithmetic operations. The instructions are encoded using the VOP3P encoding."

### WMMA Instruction Table (Table 33)
| Instruction | Matrix A | Matrix B | Matrix C | Result |
|-------------|----------|----------|----------|--------|
| V_WMMA_F32_16X16X16_F16 | 16x16 F16 | 16x16 F16 | 16x16 F32 | 16x16 F32 |
| V_WMMA_F32_16X16X16_BF16 | 16x16 BF16 | 16x16 BF16 | 16x16 F32 | 16x16 F32 |
| V_WMMA_F16_16X16X16_F16 | 16x16 F16 | 16x16 F16 | 16x16 F16 | 16x16 F16 |
| V_WMMA_BF16_16X16X16_BF16 | 16x16 BF16 | 16x16 BF16 | 16x16 BF16 | 16x16 BF16 |
| V_WMMA_I32_16X16X16_IU8 | 16x16 IU8 | 16x16 IU8 | 16x16 I32 | 16x16 I32 |
| V_WMMA_I32_16X16X16_IU4 | 16x16 IU4 | 16x16 IU4 | 16x16 I32 | 16x16 I32 |

### Key WMMA Properties
- **IU4/IU8:** NEG[1:0] indicates signed/unsigned per source (0=unsigned, 1=signed)
- **Data replication:** A and B matrices must have lanes 0-15 data replicated into lanes 16-31 (for wave64: also into lanes 32-47 and 48-63)
- **Back-to-back constraint:** Dependent WMMA instructions require one V_NOP (or independent VALU op) between them if the first instruction's matrix D overlaps with the second instruction's matrices A or B
- **No ALU exceptions:** WMMA does not generate exceptions
- **Rounding:** Round-to-nearest-even for float types
- **Inline constants:** Only for C-matrix; for F16/BF16 the inline value is replicated into both halves of the DWORD

### VGPR Usage (from Matrix Instruction Calculator)
- A matrix (iu4): 2 GPRs per thread
- B matrix (iu4): 2 GPRs per thread
- C/D matrix (i32): 8 GPRs in wave32 mode
- **Total: 12 VGPRs** for iu4 WMMA (vs ~20+ for the current dp4a path)
- **Execution:** 16 cycles, 8192 ops, cannot co-execute with VALU

### Register Layout (Wave32)
```
A[i][k]: GPR floor(k/8), bits [4*(k%8)+3:4*(k%8)], lane i and i+16
B[k][j]: GPR floor(k/8), bits [4*(k%8)+3:4*(k%8)], lane j and j+16
C[i][j]: GPR floor(i/2), lane ((16*i)%32)+j
```
- A matrix: v0 = k[0..7], v1 = k[8..15] (8 nibbles per GPR, 16 lanes)
- B matrix: same layout
- C/D matrix: 8 GPRs, 2 rows per GPR

### Throughput (from GPUOpen WMMA guide)
| Data type | RX 7900 XTX FLOPS/clock/CU |
|-----------|---------------------------|
| FP16 | 512 |
| BF16 | 512 |
| IU8 | 512 |
| IU4 | 1024 |

### Why D7.7 Missed This
D7.7 tested FP16 WMMA (dequantizing to fp16 first). The ISA reveals **IU4 WMMA** operates directly on 4-bit data — no dequantization overhead, no FP16 conversion. This is a fundamentally different approach.

---

## 5. Sign-Offset Trick for Signed Activations

Q8_1 activations are signed (-128..127). WMMA IU4 does unsigned x unsigned. The standard trick:
1. Add 128 to activations before WMMA (making them 0..255, fits in iu8)
2. After WMMA, subtract `128 * sum(weights)` per output element
3. This is a per-16x16-tile scalar correction, amortized over 256 output elements

Alternatively, the **NEG bits** on WMMA_IU8/IU4 can indicate signed inputs directly — the hardware handles the sign interpretation. From ISA:
> "For WMMA_*UI8/UI4, NEG[1:0] indicates whether SRC0 and 1 are signed or unsigned"

This means the sign-offset trick may not even be necessary if the NEG bits correctly handle signed iu8/iu4 inputs.

---

## 6. rocWMMA Library

From ROCm docs:
> "rocWMMA is a C++ header library for accelerating mixed-precision matrix multiply-accumulate operations that leverage specialized GPU matrix cores."

### Key Properties
- Header-only C++17 library
- Supports both CDNA (MI-series) and RDNA (Radeon) architectures
- CUDA WMMA-compatible API (easier migration)
- Wavefront-centric programming model
- Supports cooperative loading/storing across waves
- hipRTC compatible

### GEMV Support
rocWMMA includes GEMV (matrix-vector multiply) samples:
- `simple_sgemv`, `simple_dgemv` — simple GEMV kernels
- **Implication for D7.13:** GEMV is the MMVQ analog — rocWMMA may have an optimized path for the vector-matrix case

---

## 7. Implications for D7.x Tickets

### D7.13 (dp4a micro-optimizations)
- **Idea 1 (dual-issue): CLOSED** — ISA proves dp4a is not VOPD-encodable
- **Idea 6 (IU4 WMMA for MMQ): HIGH potential** — V_WMMA_I32_16X16X16_IU4 confirmed
- **Idea 7 (V_DOT8_I32_U4): MEDIUM potential** — 8-element 4-bit dot product, data layout TBD

### D7.10 (kernel-anvil / smithy)
- VOPD dual-issue opportunities exist for the float arithmetic in smithy-tuned kernels
- V_DUAL_FMAC_F32 + V_DUAL_MUL_F32 pairing could improve FMA-heavy kernels
- VGPR bank constraints must be respected for VOPD to function correctly

### D7.11 (quantize_q8_1 fusion)
- WMMA's data replication requirement (lanes 0-15 -> 16-31) aligns with q8_1 quantization patterns
- If activations are quantized to q8_1 with broadcast, WMMA's replication is naturally satisfied

### D7.14 (LDS root-cause)
- Back-to-back WMMA constraint (V_NOP required between dependent WMMAs) may explain LDS prototype issues
- If the LDS prototype used WMMA internally, missing V_NOP could cause stalls

---

## 8. Tools and Resources

### AMD Matrix Instruction Calculator
```
python3 matrix_calculator.py --architecture rdna3 --instruction v_wmma_i32_16x16x16_iu4 --detail-instruction
python3 matrix_calculator.py --architecture rdna3 --instruction v_wmma_i32_16x16x16_iu4 --register-layout --A-matrix
```
Generates: register mappings, VGPR usage, throughput, lane layouts.

### Radeon GPU Analyzer (RGA)
Offline shader compiler that shows ISA of compiled shaders. Useful for verifying VOPD pairing and WMMA code generation.

### Machine-Readable ISA XML
Downloaded to `/tmp/rdna3_isa/amdgpu_isa_rdna3.xml`. Contains all instruction encodings, descriptions, and constraints in parseable form.

---

## 9. References

- **RDNA 3 ISA Reference Guide (Document 70650):** `~/Downloads/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf` (local copy)
- **GPUOpen WMMA guide:** https://gpuopen.com/learn/wmma_on_rdna3/
- **rocWMMA docs:** https://rocm.docs.amd.com/projects/rocWMMA/en/latest/
- **AMD Matrix Instruction Calculator:** https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator
- **Machine-readable ISA XML:** https://gpuopen.com/machine-readable-isa/ (downloaded to `/tmp/rdna3_isa/`)
- **ROCm matrix cores blog:** https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html
