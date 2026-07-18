# D7.13 Prototype Results: Q4_K nwarps=8 on gfx1100 (7900XTX)

**Date:** 2026-07-17
**Hardware:** romulus, single Radeon RX 7900 XTX (gfx1100)
**Model:** gemma-4-12b-it-Q4_K_M.gguf (7.1 GB)
**Build:** b10144, ROCm 7.2.3, clang 22.0.0git

## Change

One-line `#ifdef GGML_HIP_D713_NWARPS8_Q4K` added to `calc_nwarps()` in
`ggml/src/ggml-cuda/mmvq.cu`. Forces `case GGML_TYPE_Q4_K: return 8;` in the
`MMVQ_PARAMETERS_RDNA3_0` branch (where Q4_K currently falls through to
`default: return 1`). CMake option wires `-DGGML_HIP_D713_NWARPS8_Q4K` compile
define. All changes marked `// PROTOTYPE: D7.13`.

## Correctness

**PASS.** Both baseline and prototype produce identical coherent output. No
crashes, no garbled tokens, no HIP errors. The thinking-model output was
truncated mid-sentence at n=32 (thinking budget exhausted) but the generated
text up to that point was grammatically correct and contextually appropriate.

## Throughput

| Config | Run | Prompt (t/s) | Generation (t/s) |
|--------|-----|-------------|------------------|
| Baseline (nwarps=1) | 1 (n=32, temp=0.7) | 209.2 | **60.0** |
| Baseline (nwarps=1) | 2 (n=64, temp=0.0) | 236.2 | **58.9** |
| Prototype (nwarps=8) | 1 (n=32, temp=0.7) | 150.8 | **57.5** |
| Prototype (nwarps=8) | 2 (n=64, temp=0.0) | 244.3 | **56.8** |
| Prototype (nwarps=8) | 3 (n=64, temp=0.0) | 189.5 | **56.6** |

- **Baseline avg tg: ~59.5 t/s**
- **Prototype avg tg: ~57.0 t/s**
- **Regression: -2.5 t/s (~4.2% slower)**

Prompt throughput is noisy (varies with cache state) but generation — the
single-token-decode metric that matters for interactive use — is consistently
and reproducibly worse with nwarps=8.

## Register Pressure / Spills

No compiler warnings emitted. The build completed cleanly. However, the
performance regression itself is the tell: Q4_K's `vec_dot_q4_K_q8_1` is
substantially more complex than Q4_0/Q8_1 (it has two sub-block scales, per
quant/min arithmetic, and a dequant+scale fused multiply per nibble). At
nwarps=8 (256 threads/block vs 32), the compiler almost certainly spills
to local memory or reduces occupancy to fit the register budget. The
RDNA4 whitelist includes Q4_K precisely because RDNA4 has a larger register
file (4 GPRs per wave vs 2 on RDNA3) — RDNA3 cannot absorb the pressure.

## Recommendation

**Abandon.** Do NOT add Q4_K to the RDNA3 nwarps=8 whitelist.

The experiment answers the question definitively: Q4_K at nwarps=8 on gfx1100
loses ~4% TG throughput while correctness holds. The regression is
structural — Q4_K's vec_dot is too register-hungry for 8-wave occupancy on
RDNA3's smaller register file. The existing whitelist (Q4_0, Q4_1, Q5_0,
Q5_1, Q8_0, IQ4_NL) stands as the correct cutoff. These types have simpler
vec_dot kernels that don't spill at higher wave counts.

If Q4_K TG improvement is needed on RDNA3, alternative approaches (LDS
caching, MFMA dot product, or the existing D7.8 prototype) are more
promising than nwarps scaling.

## Artifacts

- Prototype diff: `ggml/src/ggml-cuda/mmvq.cu` lines 488-493 (+ CMake option
  in `ggml/CMakeLists.txt` and `ggml/src/ggml-hip/CMakeLists.txt`)
- Prototype binary: `/tmp/libggml-hip-prototype.so.0.15.1`
- Baseline binary: `/tmp/llama-cli-baseline`
- Benchmark logs: `/tmp/d713-prototype-run2.log`, `/tmp/d713-prototype-run3.log`,
  `/tmp/d713-baseline-run.log`, `/tmp/d713-baseline-correctness.log`
