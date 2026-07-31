# ADR-0006: Finalize Adaptive Depth Model

## Status

**Pending** — To be decided during R3.2 (Design step of Advanced Optimization phase).

## Context

ADR-0003 established the initial adaptive pipeline depth model during D5. After D5 implementation and D6 multi-seq testing, real timing data is available to refine the model. This ADR finalizes the adaptive depth behavior based on empirical findings.

### Problem

ADR-0003's decision (whichever option was chosen) needs refinement based on:
- Actual per-backend timing data from D5 traces
- Straggler behavior observed in production-like conditions
- Edge cases: single-backend configs, dense models, MoE models
- Interaction with multi-seq (D6) — does adaptive depth help or hurt with multiple sequences?

### Inputs (to be gathered during R3.1)

| Analysis | Source | What we need |
|----------|--------|--------------|
| D5 adaptive depth traces | D5.7 test results | How adaptive depth performed |
| Straggler patterns | D5 traces | Whether adaptive depth mitigated straggler |
| Edge case behavior | D5/D6 test results | Single-backend, dense, MoE behavior |
| Multi-seq interaction | D6.7 test results | Adaptive depth with multiple sequences |

### Options

#### Option A: Keep ADR-0003 as-is

- Initial decision was correct; no refinement needed
- Only add edge case handling if needed

#### Option B: Update ADR-0003 with empirical tuning

- Adjust adaptive algorithm based on real timing data
- Add edge case fallbacks discovered during testing
- Document lessons learned

#### Option C: Replace with simpler model

- Adaptive depth too complex for the gains achieved
- Revert to static n_stages with better default
- Document why adaptive was abandoned

### Decision

**To be filled during R3.2 based on R3.1 analysis.**

### Consequences

**To be filled after decision.**

---

*ADR-0006 placeholder — decision pending R3.1 analysis*