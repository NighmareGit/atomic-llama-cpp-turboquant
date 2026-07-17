# D7.13 — dp4a Micro-Optimizations: Research Scope

**Date:** 2026-07-17
**Status:** research-only (no code changes)
**Refs:** D7.7 findings (`docs/research/d77-wmma-prototype-findings.md`), ticket `docs/tickets/path-d-tickets.md:971`

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

**Verdict: MEDIUM feasibility, worth a prototype.**

- gfx1100 can dual-issue scalar (SALU) + vector (VALU) ops in the same cycle.
- Current loop has a **chained dp4a dependency**: `dot1` feeds through two `sudot4` calls
  (result of inner used as `c` of outer). This chains the VALU dot-product pipeline.
- Independent work exists: `dot2` (sum-of-u) is fully independent of `dot1`; the float
  `sumf_d/sumf_m` accumulations are scalar and could be interleaved with the dp4a chain.
- The 2-iteration `#pragma unroll` means all 4 dp4a + 4 scalar muls are visible to the
  scheduler at once — good dual-issue opportunity **if** the compiler isn't already
  doing it. Worth inspecting compiler output (`-RPASS,-RPASS2` / `--llvm-before`).
- Risk: ROCm HIP compiler may already interleave; manual reorder may be no-op.

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

## 5. Recommended Prototype Plan

Priority order (highest ROI / lowest risk first):

| Order | Idea | Effort | Expected signal |
|-------|------|--------|-----------------|
| 1 | **Benchmark infra + baseline** (Idea 5) | Low | Establishes the measurement; do first |
| 2 | **Register analysis** (Idea 4) | Low | Explains nwarps=1; may unlock nwarps=8 (big win) |
| 3 | **Simplify scale-unpack branch** (Idea 2 real target) | Low | Remove `j<2` divergence; may reduce VGPR |
| 4 | **Dual-issue scheduling** (Idea 1) | Medium | Manual sudot4/float interleave; needs compiler-output verify |
| 5 | **Activation prefetch** (Idea 3) | Medium | Only if D7.8 LDS results are negative |

**Skip:** literal "QR4_K=8 unroll" (Idea 2 as written) — premise is factually wrong.

**Key dependency:** check D7.8 LDS prototype results (mmvq.cu:10-68) before investing in
Idea 3/5 — if LDS activation caching already wins, the prefetch/scheduling ideas are moot.

## 6. Open Questions

1. Why is Q4_K excluded from the RDNA3 nwarps=8 whitelist? Register pressure or measured
   regression? No comment in `calc_nwarps` explains it.
2. Does the HIP compiler already dual-issue across the unrolled 2-iteration loop? Need
   compiler report before manual scheduling.
3. What are the D7.8 LDS prototype benchmark results? They gate Ideas 3 and 5.
4. Is the `j<2` branch in scale-unpack actually divergent in practice (iqs patterns), or
   does it converge per warp?
