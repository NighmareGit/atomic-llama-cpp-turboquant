# D7.13 — dp4a Micro-Optimizations: Research Scope

**Date:** 2026-07-17
**Status:** research-only (no code changes)
**Refs:** D7.7 findings (`docs/research/d77-wmma-prototype-findings.md`), ticket `docs/tickets/path-d-tickets.md:971`, gfx1100 hardware deep-dive (`docs/research/gfx1100-hardware-deep-dive.md`), nwarps=8 prototype results (`docs/research/d713-nwarps8-prototype-results.md`), ISA reference text (`docs/research/gfx1100-isa-reference.txt`)

> **ISA-LEVEL UPDATE (2026-07-17):** The AMD RDNA 3 ISA PDF and machine-readable XML were
> analyzed. Key findings: (1) dp4a (V_DOT4) is NOT VOPD-encodable — dual-issue is impossible.
> (2) IU4 WMMA (V_WMMA_I32_16X16X16_IU4) exists and operates directly on 4-bit data.
> (3) V_DOT8_I32_IU4 processes 8 nibbles in one instruction. See `docs/research/gfx1100-hardware-deep-dive.md`
> for full ISA analysis.

## 1. Kernel Location

| Item | Path:Lines |
|------|-----------|
| Inner-loop impl | `ggml/src/ggml-cuda/vecdotq.cuh:505-527` |
| Wrapper (loads + calls impl) | `ggml/src/ggml-cuda/vecdotq.cuh:862-907` |
| WMMA alternative (dead, behind ifdef) | `ggml/src/ggml-cuda/vecdotq.cuh:910+` |
| dp4a intrinsic (`sudot4` on RDNA3) | `ggml/src/ggml-cuda/common.cuh:708-742` |
| Dispatch (`get_vec_dot_q_cuda`) | `ggml/src/ggml-cuda/mmvq.cu:86-89` |
| MMVQ kernel + nwarps calc | `ggml/src/ggml-cuda/mmvq.cu:417-520, 544+` |
| `should_use_small_k` | `ggml/src/ggml-cuda/mmvq.cu:1048-1088` |
| Type traits / `QR4_K` | `ggml/src/ggml-cuda/common.cuh:1047-1049`, `ggml-common.h:130-131` |

### Key constants (Q4_K)

| Const | Value | Source |
|-------|-------|--------|
| `QR4_K` | **2** (not 8) | `ggml-common.h:131` |
| `QI4_K` | 32 (= 256/(4*2)) | `ggml-common.h:130` |
| `QI8_1` | 8 (= 32/(4*1)) | `ggml-common.h:121` |
| `VDR_Q4_K_Q8_1_MMVQ` | 2 | `vecdotq.cuh:503` |
| `QK_K` | 256 | `ggml-common.h:89` |
| `K_SCALE_SIZE` | 12 | `ggml-common.h:90` |

> **Correction to ticket D7.13:** the ticket text says "QR4_K=8 inner loop." The actual
> constant is `QR4_K=2`. The impl loop `for (int i=0; i<QR4_K; ++i)` runs **2 iterations**,
> extracting nibbles at bit-offsets 0 and 4 from each of the two loaded `int` weights.

## 2. Current Implementation

### Inner loop (`vec_dot_q4_K_q8_1_impl_vmmq`, vecdotq.cuh:505-527)

```cpp
float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
for (int i = 0; i < QR4_K; ++i) {          // QR4_K = 2 iterations
    const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;  // dequant weight nibbles
    const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

    const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],   // chained sudot4
                       ggml_cuda_dp4a(v0i, u[2*i+0], 0));
    const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1],  // sum of u
                       ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));

    sumf_d += d8[i] * (dot1 * sc[i]);       // scalar float muls
    sumf_m += d8[i] * (dot2 * m[i]);
}
const float2 dm4f = __half22float2(dm4);
return dm4f.x*sumf_d - dm4f.y*sumf_m;
```

### Intrinsic (common.cuh:708-742)

On RDNA3/RDNA4: `__builtin_amdgcn_sudot4(true, a, true, b, c, false)` — unsigned
weights × signed activations, 4-byte dot product accumulated into `c`. CDNA/RDNA2 use
`sdot4`; RDNA1 falls back to `v_mul_i32_i24` inline asm.

### Memory access pattern

- **Weights (`v[2]`):** loaded in wrapper from `bq4_K->qs` (128 bytes/block). Two `int`
  loads per call (`q4[0]`, `q4[4]`), covering 8 nibbles via the 2-iteration shift.
- **Scales (`sc`, `m`):** per-block, derived from `bq4_K->scales[12]` (K_SCALE_SIZE=12)
  with a branch on `j<2` (vecdotq.cuh:888-894). `sc` = low 6 bits packed, `m` = high 2 bits.
  Tiny; likely resident in L1/register once unpacked.
- **Activations (`u[2*QR4_K]`, `d8[QR4_K]`):** the real memory pressure. Loaded in the
  wrapper's `for i<QR4_K` loop from global `block_q8_1` (32 bytes each). Each thread loads
  its own `u[0..3], d8[0..1]` — **not** broadcast. This is the LDS-caching target (D7.8).
- No prefetch, no `.sgpr`/`.vgpr` hints, no manual dual-issue scheduling present.

### Register pressure clues

- Wrapper arrays: `v[2]`, `u[4]`, `d8[2]` (int/float locals) + `aux[2]` uint16.
- Impl locals: `sumf_d/m`, `v0i/v1i`, `dot1/dot2`, `dm4f`.
- gfx1100 has **256 VGPRs per SIMD** (2 SIMDs/CU). At current nwarps=1 (64 threads),
  register pressure is low. The kernel is **not** register-bound at present — but see
  Idea 4 below.

## 3. Dispatch Path (mmvq.cu)

- `get_vec_dot_q_cuda(GGML_TYPE_Q4_K)` returns `vec_dot_q4_K_q8_1` (mmvq.cu:86-89).
- Called from `mul_mat_vec_q<type, ncols_dst>` (mmvq.cu:544) via function pointer inside
  the k-block loop (mmvq.cu:640-660).
- **`calc_nwarps` for RDNA3 + Q4_K + ncols_dst=1 = 1** (mmvq.cu:473, default case — Q4_K
  is **not** on the RDNA3 nwarps=8 whitelist, unlike Q4_0/Q8_0/IQ4_NL).
- `should_use_small_k` returns **false** for all RDNA (mmvq.cu:1083:
  `GGML_CUDA_CC_IS_RDNA(cc)) use = false`).
- Result: `nwarps=1`, `rows_per_block=1`, `small_k=false`.
- `blocks_per_iter = vdr * nwarps * warp_size / qi = 2*1*32/32 = 2`.

## 4. Feasibility Assessment

### Idea 1 — Instruction scheduling for gfx1100 dual-issue

> **CLOSED per ISA analysis — see `docs/research/gfx1100-hardware-deep-dive.md` section 2.**

**Verdict: CLOSED — impossible at ISA level.**

- The RDNA 3 ISA PDF (Section 7.6) and machine-readable XML confirm: **V_DOT4_I32_IU8 and
  V_DOT4_U32_U8 use VOP3P encoding, NOT VOPDXY.** They are absent from both VOPD X and Y
  opcode tables.
- VOPD dual-issue is restricted to 17 specific V_DUAL_* instructions (FMAC, MUL, ADD, SUB,
  MOV, CNDMASK, MAX, MIN, DOT2ACC). dp4a is not among them.
- The only remaining dual-issue path is the general VALU+SALU mechanism, which the HIP
  compiler already exploits. Manual instruction reordering cannot improve on this.
- **Previous assessment (MEDIUM feasibility) was wrong** — it assumed dp4a could participate
  in VOPD pairing. The ISA proves this is impossible.

### Idea 2 — Loop structure (QR4_K=8)

**Verdict: LOW feasibility — ticket premise is wrong.**

- `QR4_K=2`, not 8. The loop is already fully unrolled (2 iterations).
- "Unroll more" is impossible; "unroll less" would hurt (the compiler already sees it
  fully unrolled).
- The real structure question is the **wrapper's dequant setup** (vecdotq.cuh:888-894):
  the `if (j<2)` branch selecting scale-unpack logic is a per-call divergence point.
  Simplifying/removing that branch is a higher-ROI target than loop tinkering.
- Also: the shift `>>(4*i)` with runtime `i` is a variable shift — fully unrolling exposes
  constants (0 and 4), which the compiler already exploits.

### Idea 3 — Prefetch hints for scale tables

**Verdict: LOW feasibility for sc/m; MEDIUM for activations.**

- `sc`/`m` come from a 12-byte `scales[]` array unpacked once per call. Too small to
  prefetch meaningfully; likely L1-hot after first access.
- The **activation loads** (`bq8_1` in the wrapper's `for i<QR4_K` loop) are the real
  memory traffic — 4× `int` + 1× `half2` per iteration from global memory, **per thread,
  not broadcast**. This is exactly what D7.8 LDS prototype caches.
- `__builtin_prefetch` on the *next* `kby` activation block (one k-step ahead) in the
  outer dispatch loop (mmvq.cu:640) is more promising than prefetching inside the impl.
- Note: D7.8 LDS prototype already exists behind `GGML_HIP_MMVQ_LDS_PROTOTYPE`; its
  results should be known before adding prefetch.

### Idea 4 — Register analysis

**Verdict: HIGH informational value, connects to why Q4_K isn't nwarps=8-whitelisted.**

- At nwarps=1, the kernel uses a modest VGPR count — not register-bound.
- **Key question:** Q4_K is the simplest K-quant (no `hmask` like Q3_K/Q5_K, no 3-bit
  base like Q2_K) yet it is **excluded from the RDNA3 nwarps=8 whitelist** (mmvq.cu:473).
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 get nwarps=8; Q6_K gets 2; Q4_K gets only 1.
- Hypothesis: Q4_K's dequant setup (the `sc`/`m` unpack with the `j<2` branch + 12-byte
  scale table) raises VGPR pressure enough that 8 warps (512 threads) would spill.
- Action: compile with `-RPASS,-RPASS2` or check `.vgpr_count` to measure actual register
  usage at nwarps=1 vs forcing nwarps=8. If Q4_K can be made register-frugal, moving it
  to the nwarps=8 whitelist is a **larger win** than any single-instruction tweak.
- gfx1100: 256 VGPRs/SIMD, 2 SIMDs/CU → effective 128 VGPRs/thread at full occupancy
  (512 threads). Spilling threshold is the constraint.

### Idea 5 — Benchmark infrastructure

**Verdict: HIGH feasibility — use existing profiler.**

- `llama-gpipe-profiler` exists at `tools/llama-gpipe-profiler/` with CLI:
  `--model`, `--endpoints`, `--tasks pp,tg`, `--output`, `--repeat`, `--warmup`,
  `--n-prompt`, `--n-gen` (D4.10, path-d-tickets.md:1233, 1304).
- Target model: `gemma-4-12B-Q4_K_M` (Q4_K superblock = the kernel under test).
- Recommended A/B template (romulus, dual-GPU 7900XTX + 3060Ti RPC):

```
llama-gpipe-profiler \
  --model /models/gemma-4-12B-it-Q4_K_M.gguf \
  --endpoints romulus \
  --tasks tg --n-prompt 1024 --n-gen 16 \
  --repeat 3 --warmup 0 \
  --output /tmp/d713-baseline.json
```

- Then rebuild with the micro-opt flag and re-run to `/tmp/d713-opt.json`.
- Control: pin to single-GPU (7900XTX only, `--endpoints local`) to remove RPC variance
  for the tightest dp4a signal; dual-GPU for production-relevant numbers.
- Metric: `tg` tok/s, steady-state (discard run 1). Target: >=3% improvement.

### Idea 6 — IU4 WMMA for MMQ (ncols_dst >= 16)

> **NEW per ISA analysis — see `docs/research/gfx1100-hardware-deep-dive.md` section 4.**

**Verdict: HIGH potential — direct 4-bit hardware path, confirmed by ISA.**

- The RDNA 3 ISA defines `V_WMMA_I32_16X16X16_IU4`: 16x16 IU4 x 16x16 IU4 -> 16x16 I32.
- Operates directly on unsigned 4-bit data — no dequantization, no FP16 conversion.
- **VGPR cost: only 12** (2 for A_frag + 2 for B_frag + 8 for C_frag in wave32).
- **Throughput: 1024 INT4 ops/clock/CU** — ~4x the dp4a path.
- Data replication: lanes 0-15 replicated into lanes 16-31 (natural for read-only weights).
- **Applicability:**
  - MMVQ (ncols_dst=1): Poor — WMMA computes 16x16 tile, need 1 scalar.
  - MMQ (ncols_dst=2-8): Moderate — can tile across columns.
  - MMQ (ncols_dst>=16): Excellent — natural 16x16 tile mapping.
- **Sign handling:** NEG[1:0] bits indicate signed/unsigned per source. Q8_1 activations
  are signed — the NEG bits may handle this directly without sign-offset trick.
- **rocWMMA path:** Header-only C++17 library provides CUDA-compatible API. Preferred over
  raw intrinsics for production code.
- **Back-to-back constraint:** Need V_NOP between dependent WMMA instructions if D overlaps
  with A/B of next. Must account for this in tiled implementations.

### Idea 7 — V_DOT8_I32_IU4 for inner loop

> **NEW per ISA analysis — see `docs/research/gfx1100-hardware-deep-dive.md` section 3.**

**Verdict: MEDIUM potential — 8-element 4-bit dot product, data layout TBD.**

- The RDNA 3 ISA defines `V_DOT8_I32_IU4` and `V_DOT8_U32_U4`: dot product of **8 packed
  4-bit elements** in one instruction — twice the width of V_DOT4.
- Current Q4_K inner loop uses 2 iterations of V_DOT4 (4 nibbles each) to process 8 nibbles.
  V_DOT8 could potentially do this in 1 instruction.
- **Key question:** Does Q4_K's packed nibble layout match V_DOT8's input format? The ISA
  does not specify the exact packing — needs investigation.
- If compatible, this could halve the number of dp4a ops in the inner loop.
- Same NEG[1:0] signed/unsigned control as V_DOT4.
- **Risk:** Data layout mismatch may require shuffling, negating the benefit.

## 5. Recommended Prototype Plan (Updated 2026-07-17)

Priority order (highest ROI / lowest risk first):

| Order | Idea | Effort | Expected signal | Notes |
|-------|------|--------|-----------------|-------|
| 1 | **Benchmark infra + baseline** (Idea 5) | Low | Establishes the measurement | Do first; needed for all A/B tests |
| 2 | **IU4 WMMA for MMQ** (Idea 6) | Medium | **High** — direct 4-bit path, 12 VGPRs | Target ncols_dst>=16 first; use rocWMMA |
| 3 | **V_DOT8_I32_IU4 feasibility** (Idea 7) | Low | Medium if data layout matches | Quick check of Q4_K layout vs V_DOT8 requirements |
| 4 | **Simplify scale-unpack branch** (Idea 2 real target) | Low | Remove `j<2` divergence | Check `bq8_offset` uniformity per warp first |
| 5 | **Register analysis** (Idea 4) | Low | Explains nwarps=1 | Use Matrix Instruction Calculator |

**Closed:** Idea 1 (dual-issue) — ISA proves dp4a is not VOPD-encodable.

**Skip:** literal "QR4_K=8 unroll" (Idea 2 as written) — premise is factually wrong.

**Key dependency:** check D7.8 LDS prototype results (mmvq.cu:10-68) before investing in
Idea 3 (prefetch) — if LDS activation caching already wins, prefetch is moot.

**Reference:** `docs/research/gfx1100-hardware-deep-dive.md` for ISA-level details on all ideas.

## 6. Open Questions

1. ~~Why is Q4_K excluded from the RDNA3 nwarps=8 whitelist?~~ **Answered:** Prototype
   confirmed -4.2% regression. ISA explains: RDNA3 has 2 GPRs/wave vs RDNA4's 4. Q4_K's
   vec_dot is too register-hungry for 8-wave occupancy. Whitelist stands.
2. ~~Does the HIP compiler already dual-issue?~~ **Answered:** ISA proves dp4a cannot
   dual-issue via VOPD. General VALU+SALU dual-issue is already exploited by compiler.
3. What are the D7.8 LDS prototype benchmark results? They gate Ideas 3 (prefetch).
4. Is the `j<2` branch in scale-unpack actually divergent in practice (iqs patterns), or
   does it converge per warp?
5. ~~Does Q4_K's packed nibble layout match V_DOT8's format?~~ **Answered: Requires repacking.**
   Q4_K nibble positions match V_DOT8 format, but current code uses 2 ints (v[0], v[1])
   for 8 nibbles. V_DOT8 needs 1 int with 8 nibbles. Requires weight + activation repacking.
   See `docs/research/d713-vdot8-feasibility.md`.
6. Can the NEG bits on `V_WMMA_I32_16X16X16_IU4` handle signed Q8_1 activations directly
   without sign-offset arithmetic? (Idea 6)
7. What is the exact VGPR count of the current dp4a path? Use Matrix Instruction Calculator.
8. ~~Is the `j<2` branch divergent?~~ **Answered: NO (compiler predicates it).** ISA analysis
   shows the branch is converted to `v_cndmask_b32` — both paths execute, result is selected
   per-lane. No divergence penalty. See `docs/research/d713-scale-unpack-isa-findings.md`.
