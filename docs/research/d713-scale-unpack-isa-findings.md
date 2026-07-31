# D7.13 — Scale-Unpack Branch: Compiled ISA Analysis

**Date:** 2026-07-17
**Method:** Compiled `mmvq-hip-amdgcn-amd-amdhsa-gfx1100.s` with `-save-temps`, analyzed ISA
**Refs:** `docs/research/d713-next-steps-analysis.md`, `docs/research/d713-dp4a-research-scope.md`

---

## Question

Does the `if (j < 2)` branch in `vec_dot_q4_K_q8_1` (`vecdotq.cuh:888-894`) cause wave divergence, or does the compiler predicate it?

## Method

Built `ggml-hip` with `-save-temps` to dump ISA:
```
cmake -DCMAKE_HIP_FLAGS="-save-temps" .
make -j$(nproc) ggml-hip
```

ISA file: `build/ggml/src/ggml-hip/mmvq-hip-amdgcn-amd-amdhsa-gfx1100.s`

## Findings

### Compiler Heavily Uses Predication

| Pattern | Count |
|---------|-------|
| `v_cndmask_b32` / `v_dual_cndmask` | 7,023 |
| Actual branches (`s_cbranch_*`) | 3,486 |

The compiler converts most if-else branches to **predicated execution** using `v_cndmask_b32` (select one of two values based on condition code). This avoids wave divergence entirely — both paths are computed, and the result is selected per-lane.

### dp4a Instructions Confirmed

The ISA contains `v_dot4_i32_iu8` with `neg_lo:[1,1,0]` — confirming the NEG bits are used for signed/unsigned control (src0 signed, src1 signed, src2 unsigned). This matches the `sudot4` intrinsic used in the source.

### Scale-Unpack Branch

The `if (j < 2)` branch is converted to a `v_cndmask_b32` pattern:
- Both scale-unpacking paths are computed (the `if` and `else` cases)
- The result is selected per-lane based on the condition `j < 2`
- **No branch divergence penalty** — both sides execute, result is masked

## Conclusion

**The compiler already handles the branch efficiently.** The `if (j<2)` divergence is converted to predicated execution, so there is no wave divergence cost.

### Implication for Idea 2

Simplifying the scale-unpack branch has **LOWER benefit than expected**:
- No divergence penalty to avoid (already predicated)
- Both paths still execute (predication computes both sides)
- Potential benefit: reducing instruction count for ILP, not avoiding divergence

**Revised verdict for Idea 2: LOW feasibility.** The compiler already optimizes this. Manual simplification may reduce instruction count slightly, but the main cost (divergence) is already eliminated by predication.

---

## ISA Statistics

- Total `v_dot4_i32_iu8` instructions: ~20 (per kernel instance)
- Total `v_cndmask_b32` instructions: ~7,000 (across all template instantiations)
- Total `v_dual_cndmask_b32` instructions: included in above (VOPD-paired with another VALU op)
- VOPD usage: Present (`v_dual_cndmask_b32 :: v_dual_and_b32` pairs observed)

The compiler is also using VOPD dual-issue for other operations (like `v_dual_cndmask_b32` paired with `v_dual_and_b32`), confirming the VOPD mechanism is active in the generated code — just not for dp4a.
