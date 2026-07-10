# D6 Agent Plan — Mode B Microbatch / Multi-Seq

**Phase:** D6 (Mode B Microbatch)  
**Goal:** Support multiple sequences at different pipeline positions  
**Agent pattern:** research → design → spec → prototype → implement → test → review (loop up to 3x)

---

## Mission Statement

Extend GPipe from single-sequence (Mode A) to multi-sequence pipeline sharing (Mode B). Different sequences occupy different pipeline stages concurrently, improving utilization when serving multiple users.

## Pre-conditions

- Run `bash scripts/safety-check.sh` before every resource-intensive operation
- Models available in `/mnt/models` or `~/models`
- No concurrent llama-server instances
- Commit + push after every completed ticket

## Input Materials

| File | Purpose |
|------|---------|
| `docs/tickets/path-d-tickets.md` | D6.1-D6.7 tickets |
| `docs/path-d-spec.md` section 10.3 | D6 spec |
| `docs/adr/0005-multi-seq-gpipe.md` | ADR placeholder (to be filled in D6.2) |
| `docs/adr/0002-gpipe-kv-ordering.md` | KV ordering model (ADR-0002) |
| D5 implementation | Stage state machine to extend |

## Execution Steps

### D6.1 — Multi-Seq Requirements (Research)

1. Analyze server multi-slot requirements from `llama-server` source
2. Document KV cache interaction with pipeline stages
3. Identify conflicts: KV isolation, stage sharing, sequence lifecycle
4. Write findings to `docs/wayfinder/D6.1-multi-seq-requirements.md`
5. Commit + push

### D6.2 — ADR-0005 (Design)

1. Grill: analyze D6.1 findings, decide multi-seq scheduling model
2. Fill in ADR-0005 decision, consequences, alternatives rejected
3. Commit + push

### D6.3 — Spec Section

1. Add multi-seq spec section to `docs/path-d-spec.md`
2. Define API contracts for multi-seq functions
3. Define acceptance criteria
4. Commit + push

### D6.4 — Prototype

1. Write throwaway code to test multi-seq token tracking
2. Answer: can KV cache handle sequences at different pipeline positions?
3. Document findings (keep findings, delete code)
4. Commit + push

### D6.5 — Extend State Machine

1. Extend stage state machine for multi-seq token tracking
2. Add per-sequence stage state isolation
3. Extend event signaling for multi-seq
4. Use `/tdd` for testable units
5. Commit + push

### D6.6 — Server Multi-Slot Dispatch

1. Implement server-side multi-slot pipeline dispatch
2. Enable different sequences at different stages concurrently
3. Commit + push

### D6.7 — Test

1. Safety check: `bash scripts/safety-check.sh`
2. Run concurrent multi-seq decode
3. Verify: correctness (no KV corruption), throughput scaling
4. Document results in TRACKING.md
5. Commit + push

## Review Loop

After D6.7:
- If acceptance criteria met → milestone commit, handoff to R3
- If test fails with obvious code fix → loop back to D6.5/D6.6
- If design flawed → loop back to D6.2
- If structural problem → loop back to D6.4
- Max 3 loops → mark FAILED SPIKE, document, continue to R3

## Milestone Commit

```bash
git add -A
git commit -m "Path D6: Mode B microbatch — <results>

<Multi-seq pipeline. Correctness. Throughput scaling.>

Assisted-by: Grok"
git push origin Path-D-Gpipeline-Assembly-Line
```

## Safety Reminders

- One llama-server instance at a time
- `bash scripts/safety-check.sh` before every server start
- Do not kill romulus (hosts this session)
- Clean up docker after test builds: `docker system prune -f`
- Watch `/` disk space: `df -h /`

---

*Agent plan for D6 — 2026-07-10*