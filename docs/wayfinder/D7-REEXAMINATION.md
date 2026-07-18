# D7.0-D7.8 Re-examination — Retrospective and Open Leads

**Date:** 2026-07-17
**Scope:** Re-examination of D7.0 through D7.8 research docs to identify resolved items, missed leads, and complementary opportunities for Slice 7 (Layer 1-3 Kernel Optimization).
**Triggered by:** Slice 7 research phase — before executing vectors D/E/F/G, verify nothing was missed in prior work.

---

## 1. What Was Already Resolved

Several items flagged as "BLOCKED" or "never fixed" in the D7.x trail were resolved later in the same trail. These are now corrected in the source docs:

| Item | Source Doc | Resolution | Reference |
|------|-----------|------------|-----------|
| rocprofv3 SIGABRT crash | D7.2 | **Fixed in D7.6** — `--kernel-trace` without `--hip-trace` avoids HIP interception | `docs/research/d76-rocprofv3-kernel-profile.md` |
| GPU kernel-level profiling | D7.2 Vector C | **Delivered in D7.6** — per-kernel breakdown for 4 models (MatMul 55.4%, quantize_q8_1 7.9%) | `docs/research/d76-rocprofv3-kernel-profile.md` |
| Skip-SSM +75% upper bound | D7.4 | **Revised to ~5% by D7.6** — SSM is only 2.4% of GPU time (10 us/layer vs 38.8 us attention) | `docs/research/d76-rocprofv3-kernel-profile.md` |
| D6.10 GPU event pipelining | D7.5 pivot | **Shipped as `f29a92eb1`** — `input_copy_slow` 165,000 -> 2,359 us (-98.6%), TPS +4.3% | `docs/wayfinder/D6.10-implementation-analysis.md` |

---

## 2. D6.10 — Not Missing (Re-examination Correction)

The initial re-examination concluded that "D6.10 was never written up." **This was incorrect.** D6.10 documentation lives in `docs/wayfinder/`, not `docs/research/`:

| Doc | Content |
|-----|---------|
| `docs/wayfinder/D6.10-implementation-analysis.md` | 12 sections, ~420 lines: problem, garbling risk, code changes, test plan, D6.10.1 fix, results |
| `docs/wayfinder/D6.10-module-design.md` | 8 sections, ~220 lines: module identity, interface contract, depth analysis, alternatives |
| `docs/tickets/path-d-tickets.md` (lines 662-720) | Original ticket definition |

**D6.10 result:** `input_copy_slow` 165,000 -> 2,359 us (-98.6%), TPS 124.4 -> 129.8 (+4.3%), 12/12 GPipe tests pass.

---

## 3. Open Leads Carried Forward to Slice 7

These leads were identified in D7.0-D7.8 but never fully pursued. They are complementary to Slice 7's official vectors (D/E/F/G).

### 3.1 HIGH Priority — Directly Complementary to Vectors D/E

| Lead | Source | Mechanism | Connection to Slice 7 |
|------|--------|-----------|----------------------|
| **dp4a micro-optimizations (D7.13)** | D7.7 | ~~Dual-issue CLOSED~~ (ISA: dp4a not VOPD-encodable). ~~nwarps=8 CLOSED~~ (regression -4.2%). **Active: IU4 WMMA (Idea 6), V_DOT8_I32_IU4 (Idea 7), scale-unpack branch (Idea 2).** See `docs/research/gfx1100-hardware-deep-dive.md`. | Same kernels as Vector D (kernel-anvil tuning). ISA analysis revealed new hardware paths. |
| **LDS negative result debug** | D7.8 | Run standalone `test-lds-mmvq.hip.cu` with rocprofv3 to isolate -1.9 t/s regression root cause | LDS + kernel-anvil is multiplicative (D7.8 attacks memory traffic, Vector D attacks compute utilization). |

### 3.2 MEDIUM Priority — Quick Wins or Strategic Options

| Lead | Source | Mechanism | Connection to Slice 7 |
|------|--------|-----------|----------------------|
| **FA + Q4_K_M combined** | D7.3 | FA alone gave +7.5%. Q4_K_M leg skipped. Combined estimated 15-25%. 20GB model exists. | Independent of kernel-anvil. One benchmark run could capture this. |
| **MTP acceptance benchmark** | D7.4 | Prerequisite for R1 skip-SSM approach. Never built. | If R1 viable (even with acceptance drop), it competes with L1 optimization ROI. |
| **5:4 FAST/SLOW pattern analysis** | D7.2 | Which layers/tokens cause which step type? Descriptive, never quantified. | Understanding this helps target L1 optimizations to the right step type. |

### 3.3 LOW Priority — Archived Unless Conditions Change

| Lead | Source | Mechanism | Condition to Revisit |
|------|--------|-----------|---------------------|
| R1: skip-SSM verify-only | D7.4 | Dual-context (ctx_tgt + ctx_verify), skip-SSM only for MTP acceptance | MTP acceptance benchmark exists AND SSM cost model revised |
| R2-R5: skip-SSM refinements | D7.4 | Partial skip, simplified SSM, KV-only skip, state checkpoint | R1 viability confirmed |
| WMMA vec_dot revisit | D7.7 | Correct tile layout + shared memory staging | dp4a path exhausted |

---

## 4. The Biggest Insight: D7.7 and D7.8 Abandoned Prematurely

**D7.7 (WMMA)** correctly identified WMMA is wrong for M=1 decode, but the pivot target — dp4a micro-optimizations — was never pursued. The document lists 6 specific optimization ideas (instruction scheduling, loop unrolling, prefetch, register analysis, dual-issue exploitation). **None were attempted.**

**D7.8 (LDS)** produced a net-negative result (-1.9 t/s) and stopped. The standalone test exists but was never executed to isolate whether the problem was kernel structure, bank conflicts, or compiler artifact. D7.6's profiling shows quantize_q8_1 = 7.9% of GPU time — the LDS prototype attacks a kernel in this path, so the negative result matters.

**Both are Layer 1 kernel optimizations that directly feed Slice 7's vectors.** They are not separate work — they are the groundwork.

### Complementarity Matrix

| Approach | Layer 1 Angle | Status | Complementarity |
|----------|--------------|--------|-----------------|
| D7.8 LDS | Activation memory traffic (shared mem) | Complete, negative | Multiplicative with D (compute) |
| D7.7 dp4a | Instruction-level Q4_K vec_dot | Never pursued | Same kernels as D |
| Vector E small_k | Multi-row processing threshold | Ready | Prerequisite for D (fixes threshold) |
| Vector D smithy | nwarps/rows_per_block tuning | Ready | Main L1 lever |
| Vector F fusion | quantize_q8_1 kernel elimination | Ready | Overlaps D7.8 target |
| Vector G autoforge | Custom kernels per shape | Ready | Maximum per-shape perf |

Combined effect is multiplicative, not additive.

---

## 5. Corrected Findings per Source Doc

### D7.2 — GPU Timeline Profiling
- rocprofv3: ~~BLOCKED~~ **RESOLVED (D7.6)**
- Vector C: ~~highest-priority gap~~ **DELIVERED (D7.6)**
- 5:4 FAST/SLOW pattern: still open (descriptive, low priority)
- MTP verification cost vs acceptance: still open (superseded by D7.4/D7.6)

### D7.4 — Skip-SSM Verify
- +75% upper bound: **REVISED to ~5% (D7.6)**
- SSM = 2.4% of GPU time, not a meaningful bottleneck
- R1-R5 refinement catalog: **ARCHIVED** (pending acceptance benchmark)
- Prototype code behind `LLAMA_SKIP_SSM_VERIFY=1`: preserved but inactive

### D7.5 — RPC Overlap
- D6.10 pivot: **SHIPPED** (commit `f29a92eb1`)
- `input_copy_slow` -98.6%: **ACHIEVED** via event pipelining
- H2D async copy premise: **INVALIDATED** (no H2D bottleneck exists)

### D7.7 — WMMA Prototype
- WMMA for M=1: **CLOSED** (correctly dismissed)
- dp4a micro-optimizations: **PARTIALLY PURSUED** — ISA analysis closed Ideas 1 (dual-issue) and 4 (nwarps=8). New Ideas 6 (IU4 WMMA) and 7 (V_DOT8) discovered from ISA. See `docs/research/gfx1100-hardware-deep-dive.md`.
- Dead code behind `#ifdef GGML_HIP_WMMA_VECDOT_EXPERIMENTAL`: preserved

### D7.8 — LDS Prototype
- Functionally correct: **CONFIRMED**
- Throughput delta: **-1.9 t/s** (root cause unknown)
- Standalone test: **NEVER RUN** (carried to Slice 7)
- CMake misconfiguration bug: **RESOLVED**

---

## 6. Action Summary for Slice 7

When executing Slice 7 vectors D/E/F/G, also:

1. **Run the D7.8 standalone test** with `rocprofv3 --kernel-trace` to root-cause the LDS negative result. If fixable, LDS + kernel-anvil is multiplicative.
2. **Pursue D7.7's dp4a checklist** as part of Vector D. The 6 specific ideas attack the same Q4_K/Q6_K kernels that kernel-anvil targets.
3. **Run one FA + Q4_K_M benchmark** on the 2-GPU Romulus config. Model exists (20GB). Independent quick win.
4. **Archive D7.4 skip-SSM** (R1-R5) unless an MTP acceptance benchmark is built.

---

## Cross-References

- **Slice 7 official research:** `docs/research/slice-7-kernel-anvil-integration.md`
- **Optimization landscape:** `OPTIMIZATION-LANDSCAPE-L1-L3.md`
- **D7.6 per-kernel profile:** `docs/research/d76-rocprofv3-kernel-profile.md`
- **D6.10 implementation:** `docs/wayfinder/D6.10-implementation-analysis.md`
- **D7.7 dp4a checklist:** `docs/research/d77-wmma-prototype-findings.md` section 7
- **D7.8 LDS handoff:** `docs/wayfinder/HANDOFF-D7.8-LDS-prototype.md`
