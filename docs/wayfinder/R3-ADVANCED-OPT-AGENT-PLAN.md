# R3 Agent Plan — Advanced Optimization

**Phase:** R3 (Advanced Optimization)  
**Goal:** Adaptive depth refinement + deprecation cleanup  
**Agent pattern:** research → design → implement → test → review (loop up to 3x)

---

## Mission Statement

Refine the adaptive depth model based on D5/D6 empirical findings, and clean up deprecated flags (B+11, B+14, B+7f) with proper deprecation warnings. RDMA is explicitly deferred.

## Pre-conditions

- Run `bash scripts/safety-check.sh` before every resource-intensive operation
- Models available in `/mnt/models` or `~/models`
- No concurrent llama-server instances
- Commit + push after every completed ticket

## Input Materials

| File | Purpose |
|------|---------|
| `docs/tickets/path-d-tickets.md` | R3.1-R3.5 tickets |
| `docs/path-d-spec.md` section 10.4 | R3 spec |
| `docs/adr/0003-adaptive-pipeline-depth.md` | ADR-0003 (to be refined) |
| `docs/adr/0006-finalize-adaptive-depth.md` | ADR placeholder (to be filled in R3.2) |
| `docs/rpc-multi-backend-pipeline-plus/CONFIGURATION.md` | Deprecated flags list |

## Execution Steps

### R3.1 — Adaptive Depth Analysis (Research)

1. Analyze D5 adaptive depth behavior from traces
2. Document straggler patterns observed in D5/D6
3. Identify edge cases: single-backend, dense models, MoE
4. Assess multi-seq interaction with adaptive depth
5. Write findings to `docs/wayfinder/R3.1-adaptive-depth-analysis.md`
6. Commit + push

### R3.2 — ADR-0006 (Design)

1. Grill: analyze R3.1 findings, finalize adaptive depth model
2. Decide: keep ADR-0003 as-is, update it, or replace with simpler model
3. Fill in ADR-0006 decision
4. Commit + push

### R3.3 — Deprecation Warnings

1. Add deprecation warning for `GGML_RPC_DUAL_SOCKET` (B+11)
2. Add deprecation warning for `GGML_SCHED_WAVEFRONT_DISPATCH` (B+14)
3. Add deprecation warning for `GGML_RPC_HASH_DEFER` (B+7f)
4. Ensure warnings are one-time (not spammy)
5. Commit + push

### R3.4 — Adaptive Depth Refinement

1. Refine adaptive depth based on R3.1 findings
2. Handle edge cases discovered during D5/D6
3. Tune for stability across model types
4. Commit + push

### R3.5 — Test

1. Safety check: `bash scripts/safety-check.sh`
2. Run full regression: GPipe ON and OFF
3. Verify: all Path-B+ benchmarks pass
4. Final TRACKING.md update with completion status
5. Commit + push

## Review Loop

After R3.5:
- If acceptance criteria met → milestone commit, plan complete
- If test fails with obvious code fix → loop back to R3.3/R3.4
- If design flawed → loop back to R3.2
- Max 3 loops → mark FAILED SPIKE, document, plan ends

## Milestone Commit

```bash
git add -A
git commit -m "Path R3: Advanced optimization — adaptive depth + deprecation cleanup

<Final adaptive depth model. Deprecation warnings shipped. Regression status.>

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

*Agent plan for R3 — 2026-07-10*