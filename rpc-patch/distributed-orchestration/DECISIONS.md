# Path C: Locked Architecture Decisions

This file records all major decisions made during the architecture phase.

---

## 2026-06-25

| Decision | Details | Rationale |
|----------|---------|---------|
| Security Priority | Lowest priority during skunkworks | Focus on validating the core idea first. Security can be addressed later by the community. |
| Overall Approach | Evolutionary phased model | Start with centralized but asynchronous orchestration (Phase 1), then gradually move toward decentralized worker autonomy in later phases. |
| KV Cache Direction | Lean toward static layer-range ownership | Avoids expensive KV movement between workers. Aligns with successful patterns in vLLM and other production systems. |

---

*Only record decisions that have been explicitly discussed and agreed upon.*