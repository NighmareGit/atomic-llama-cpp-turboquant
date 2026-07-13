# D6 Agent Plan — Mode B Microbatch / Multi-Seq

**Phase:** D6 (Mode B Microbatch) — ✅ COMPLETE (2026-07-13)  
**Goal:** Support multiple sequences at different pipeline positions  
**Agent pattern:** research → design → spec → prototype → implement → test → review (loop up to 3x)  
**Status:** D6.1-D6.9 complete. D6.10 (GPU event pipelining fix) ticketed as follow-up.

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

### D6.8 — Per-Sequence Events ✅ complete

1. Migrate from double-buffered (2 banks) to per-sequence event arrays
2. `gpipe_events` flattened to `[n_gpipe_seqs * GGML_SCHED_MAX_STAGES]` row-major
3. `ggml_sched_gpipe_wait_seq`/`record_seq` accept `seq_id` parameter
4. Remove old bank API (`record_bank`, `wait_bank`, `toggle_bank`)
5. Commit `5c408b052`

### D6.9 — GRAPH_COMPUTE_STAGE RPC ✅ complete

1. Add `RPC_CMD_GRAPH_COMPUTE_STAGE` (value 23) to RPC protocol
2. Server-side per-stage split filtering with telemetry
3. Thread-local `tls_gpipe_active_stage` signals stage to RPC backend
4. Profiler verified: `device_timings_us` per-stage
5. Commits `c19c9f917`, `c91743d32`

### D6.10 — GPU Event Pipelining Fix 📋 planned

1. Move gpipe_events from CPU gather backend to GPU backend
2. Fall back to full sync when no GPU backends present
3. Revert temporary `ggml_backend_sched_synchronize` workaround
4. See `docs/tickets/path-d-tickets.md` D6.10
5. See `docs/path-d-spec.md` section 10.3.6 (KL-D6.1)

## Review Loop

After D6.9:
- Acceptance criteria met → D6.1-D6.9 complete, milestone commits done
- D6.10 follow-up ticketed (GPU event pipelining)
- Handoff to R3

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

*Agent plan for D6 — 2026-07-10. Updated 2026-07-13 with D6.1-D6.9 completion and D6.10 planning.*