# D4 Agent Plan — Path C Stepping Stone

**Phase:** D4 (Path C Stepping Stone)  
**Goal:** Prove server-side scheduling on romulus dual-GPU (7900 XTX + 3060 Ti)  
**Agent pattern:** research → design → spec → prototype → implement → test → review (loop up to 3x)

---

## Mission Statement

Establish whether server-side scheduling on romulus's dual-GPU setup (AMD 7900 XTX client + NVIDIA 3060 Ti RPC server, models at `/mnt/models`) is feasible and beneficial, before attempting deeper client-side pipelining (D5). This is a de-risking phase: smaller blast radius, localized to one machine. Cluster deployment (triton 5-GPU) is deferred to a later session.

## Pre-conditions

- Run `bash scripts/safety-check.sh` before every resource-intensive operation
- Models available in `/mnt/models` or `~/models`
- No concurrent llama-server instances
- Commit + push after every completed ticket

## Input Materials

| File | Purpose |
|------|---------|
| `docs/tickets/path-d-tickets.md` | D4.1-D4.6 tickets |
| `docs/path-d-spec.md` section 10.1 | D4 spec |
| `docs/adr/0004-server-side-scheduling.md` | ADR placeholder (to be filled in D4.2) |
| `docs/rpc-multi-backend-pipeline-plus/DESIGN-path-d-layer-pipeline.md` | Parent design |
| `rpc-patch/docs/rpc-path-c-plan.md` | Path C implementation reference |

## Execution Steps

### D4.1 — C1 Baseline (Research)

1. Safety check: `bash scripts/safety-check.sh`
2. Document current per-device RPC splits on romulus (7900 XTX + 3060 Ti)
3. Capture RTT count and GPU utilization (`rocm-smi` for AMD, `nvidia-smi` for NVIDIA)
4. Write findings to `docs/wayfinder/D4.1-romulus-baseline-analysis.md`
5. Commit + push

### D4.2 — ADR-0004 (Design)

1. Grill: analyze D4.1 findings, decide server-side scheduling model
2. Fill in ADR-0004 decision, consequences, alternatives rejected
3. Commit + push

### D4.3 — Spec Section

1. Add Path C spec section to `docs/path-d-spec.md`
2. Define API contracts for Path C functions
3. Define acceptance criteria for C2
4. Commit + push

### D4.4 — Prototype

1. Write throwaway code to test `GRAPH_COMPUTE_ALL` concept
2. Answer: is server-side scheduling feasible on romulus dual-GPU?
3. Document findings (keep findings, delete code)
4. Commit + push

### D4.5 — Implement C2

1. Implement server-side scheduler with `GRAPH_COMPUTE_ALL`
2. Target: 2x server GPU duty cycle
3. Use `/tdd` for any testable units
4. Commit + push

### D4.6 — Test

1. Safety check: `bash scripts/safety-check.sh`
2. Run C2 on romulus, compare vs C1 baseline
3. Verify: server GPU duty improved, G non-regression
4. Document results in TRACKING.md
5. Commit + push

## Review Loop

After D4.6:
- If acceptance criteria met → milestone commit, handoff to D5
- If test fails with obvious code fix → loop back to D4.5
- If design flawed → loop back to D4.2
- If structural problem → loop back to D4.4
- Max 3 loops → mark FAILED SPIKE, document, continue to D5

## Milestone Commit

```bash
git add -A
git commit -m "Path D4: Path C stepping stone — <results>

<Server-side scheduling feasibility. What worked. What didn't.>

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

*Agent plan for D4 — 2026-07-10*

---

## Completion (2026-07-11)

All D4.1-D4.10 tickets complete. Key outcomes:

- **D4.1**: Romulus dual-GPU baseline — 3 models (9B MTP, Llama 8B, 35B APEX), trace capture, 7 findings documented
- **D4.2**: ADR-0004 — Option B+ (GRAPH_COMPUTE_ALL + weighted weight placement)
- **D4.3**: Path C spec section 12 integrated into `docs/path-d-spec.md`
- **D4.4**: GRAPH_COMPUTE_ALL prototype — 7 findings documented for D4.5
- **D4.5**: Production implementation — serialization consolidation, scheduler cache, EVENT_RECORD, `--rpc-multidevice` CLI arg
- **D4.6**: Romulus dual-GPU validation — pp32=991 t/s (+21%), tg32=81.7 t/s (+32%), draft-mtp n_max=2: 113.2 t/s gen (+82%)
- **D4.7**: Profiler research — heatmap schema, binary design, KV cache scope decision
- **D4.8**: RPC telemetry — 6-field struct, `collect_telemetry()` on both GRAPH_COMPUTE and GRAPH_COMPUTE_ALL paths, HELLO cap negotiation
- **D4.9**: ADR-0004b — binary arch (`llama-bench` pattern), CLI surface, heatmap JSON schema v1
- **D4.10**: `llama-gpipe-profiler` binary (17,928 bytes), task-stratified profiling, 3-model benchmark suite, hot paths analysis at `docs/hot-paths-analysis.md`
- **D4.11-D4.14**: Pareto Optimizer — planned + ticketed, NOT built

**Commit range:** `eb1e5a261` .. `eb1e5a261` (pure docs + testing milestone; D4.1-D4.10 work committed incrementally throughout Slice 2)