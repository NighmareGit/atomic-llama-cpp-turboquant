# D5 Agent Plan — Deeper Pipelining

**Phase:** D5 (Deeper Pipelining) — ✅ COMPLETE (2026-07-11)  
**Goal:** Extend GPipe from 2-stage to n_stages > 2 with finer sub-stages  
**Agent pattern:** research → design → spec → prototype → implement → test → review (loop up to 3x)

---

## Mission Statement

Split Stage 0 (RPC compute) into per-backend sub-stages so faster backends can start T+1 while the straggler finishes T. This is the main throughput lever for `global_3bk_pct >= 25%`.

## Pre-conditions

- Run `bash scripts/safety-check.sh` before every resource-intensive operation
- Models available in `/mnt/models` or `~/models`
- No concurrent llama-server instances
- Commit + push after every completed ticket

## Input Materials

| File | Purpose |
|------|---------|
| `docs/tickets/path-d-tickets.md` | D5.1-D5.7 tickets |
| `docs/path-d-spec.md` section 10.2 | D5 spec |
| `docs/adr/0003-adaptive-pipeline-depth.md` | ADR placeholder (to be filled in D5.2) |
| `docs/wayfinder/D0.5-implementation-seam.md` | Implementation seam (section 7: straggler) |
| `docs/wayfinder/D4.1-romulus-baseline-analysis.md` | D4 baseline findings |

## Execution Steps

### D5.1 — Split Timing Analysis (Research)

1. Analyze per-backend split timing from D2.2 and D4.1 traces
2. Identify sub-stage boundaries: embed, RPC0, RPC1, RPC2, RPC3
3. Document straggler pattern and variance
4. Write findings to `docs/wayfinder/D5.1-split-timing-analysis.md`
5. Commit + push

### D5.2 — ADR-0003 (Design)

1. Grill: analyze D5.1 findings, decide adaptive vs static depth
2. Fill in ADR-0003 decision, consequences, alternatives rejected
3. Commit + push

### D5.3 — Spec Section

1. Add deeper pipelining spec section to `docs/path-d-spec.md`
2. Define API contracts for per-backend sub-stage functions
3. Define acceptance criteria
4. Commit + push

### D5.4 — Prototype

1. Write throwaway code to test per-backend sub-stage dispatch
2. Answer: does sub-stage dispatch actually overlap compute?
3. Document findings (keep findings, delete code)
4. Commit + push

### D5.5 — Implement Per-Backend Sub-Stages

1. Split Stage 0 into per-backend sub-stages
2. Add per-sub-stage event signaling
3. Implement straggler isolation
4. Use `/tdd` for testable units
5. Commit + push

### D5.6 — Dynamic Stage Assignment

1. Implement adaptive depth (if ADR-0003 chose adaptive/hybrid)
2. Runtime stage assignment based on backend timing
3. Fallback to static if adaptive fails to converge
4. Commit + push

### D5.7 — Test

1. Safety check: `bash scripts/safety-check.sh`
2. Run benchmark with deeper pipeline
3. Verify: `global_3bk_pct` improves measurably vs 2-stage (D2.2 baseline)
4. Document results in TRACKING.md
5. Commit + push

## Review Loop

After D5.7:
- If acceptance criteria met → milestone commit, handoff to D6
- If test fails with obvious code fix → loop back to D5.5/D5.6
- If design flawed → loop back to D5.2
- If structural problem → loop back to D5.4
- Max 3 loops → mark FAILED SPIKE, document, continue to D6

## Milestone Commit

```bash
git add -A
git commit -m "Path D5: Deeper pipelining — <results>

<n_stages achieved. global_3bk_pct vs 2-stage. Straggler impact.>

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

*Agent plan for D5 — 2026-07-10*