# D7.13 — Baseline Benchmark Results

**Date:** 2026-07-17
**Hardware:** romulus, single Radeon RX 7900 XTX (gfx1100), 24GB VRAM
**Model:** gemma-4-12b-it-Q4_K_M.gguf (6.7 GB)
**Build:** a64d78177 (debug build, ROCm 7.2.3, clang 22.0.0git)

## Results

| Test | t/s | Notes |
|------|-----|-------|
| pp512 | 315.41 ± 10.24 | Prompt processing |
| tg1024 | 53.31 ± 0.04 | Generation, 1024 tokens |
| tg16 | 53.82 ± 0.52 | Generation, 16 tokens |

## Comparison to Previous Baseline

| Source | Build type | tg t/s |
|--------|-----------|--------|
| Previous prototype (d713-prototype-results.md) | Release | ~59.5 |
| Current benchmark | Debug | ~53.8 |
| Delta | | ~10% slower (expected for debug) |

## Notes

- Debug build is ~10% slower than release (expected)
- The tg metric (token generation) is the key metric for D7.13 — this is the single-token-decode path where Q4_K vec_dot dominates
- For A/B testing of optimizations, use the same build configuration and compare relative improvements
- Target: >=3% improvement from dp4a/WMMA optimizations

## Benchmark Command

```bash
./build/bin/llama-bench \
  -m /home/hunter/models/gemma-4-12b-it-Q4_K_M.gguf \
  -r 3 --no-warmup \
  -n 1024,16
```

For release builds, add `-DCMAKE_BUILD_TYPE=Release` to cmake configuration.
