# Path C: Topics & Decisions Checklist (Living Document)

**Purpose:** Track the status of every major architecture topic.  
**Process:** We grill → discuss → lock decision → update status here.

| # | Topic                                      | Status          | Recommended Approach                          | Decision Locked | Notes / Open Questions |
|---|--------------------------------------------|-----------------|-----------------------------------------------|-----------------|------------------------|
| 1 | KV Cache Ownership & Sharding             | Not Started     | Static layer-range ownership + local KV slice | —               | Most critical decision |
| 2 | Work Package Content                      | Not Started     | Minimal viable activations + metadata         | —               | —                      |
| 3 | Workflow Metadata Schema                  | Not Started     | Lightweight + extensible (`workflow_id`, `next_hop`, etc.) | —     | Foundation for all phases |
| 4 | Routing Table & Discovery                 | Not Started     | Server pushes routing table at start          | —               | —                      |
| 5 | Orchestration Model (Phase 1)             | Not Started     | Server does async, metadata-driven forwarding | —               | —                      |
| 6 | Worker Local Queuing & Execution          | Not Started     | Simple local queue per worker                 | —               | —                      |
| 7 | Failure Modes & Recovery                  | Not Started     | Fail generation in Phase 1                    | —               | —                      |
| 8 | Token Ordering & Consistency              | Not Started     | `workflow_id` + `step_id` + server sequencing | —               | —                      |
| 9 | Data Movement & Activation Strategy       | Not Started     | Minimize transfers, prefer co-located GPUs    | —               | —                      |
| 10| Phasing & Evolution Path                  | Not Started     | Phase 1 centralized-async → later decentralized | —             | Already broadly agreed |
| 11| Observability & Debugging                 | Not Started     | Include workflow/step IDs in logs             | —               | —                      |
| 12| Security & Trust Model                    | **Locked**      | Lowest priority (use existing model)          | Yes             | Revisit post-validation |

---

## Decision Log (Summary)

- **Security** → Lowest priority during skunkworks (2026-06-25)
- **Overall Phasing Model** → Evolutionary (Centralized Async → Hybrid → Decentralized) (2026-06-25)

---

*Update this file after every major decision.*