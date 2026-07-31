# Path C: RPC Server Multi-GPU Aggregation — Tracking

## Status: NOT STARTED

### Build: N/A  |  Test: N/A  |  Benchmark: N/A

### Current Phase: Planning / Scope Definition
### Phase Status: NOT STARTED

---

## Implementation Log

| Date | Phase | Action |
|------|-------|--------|
| 2026-06-25 | Planning | Created initial `rpc-path-c-plan.md` and this tracking file |

---

## Issues Log

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|

---

## Benchmark Results

| Date | Model | Config | Prompt t/s | Gen t/s | Notes | vs Path B Baseline |
|------|-------|--------|------------|---------|-------|--------------------|

---

## Phase Checklist

### Phase C1: Multi-Device RPC Server Foundation

- [ ] `rpc-server` accepts multiple devices at startup (e.g. `--devices CUDA0,CUDA1`)
- [ ] Internal `ggml_backend` array + unified scheduler created on server
- [ ] `RPC_CMD_DEVICE_COUNT` and device properties report multi-GPU capability
- [ ] Basic aggregated memory reporting works
- [ ] Server starts cleanly with 2+ GPUs

### Phase C2: New Multi-GPU Protocol Commands

- [ ] `RPC_CMD_GRAPH_COMPUTE_MULTI` defined and implemented (server + client)
- [ ] `RPC_CMD_GRAPH_RECOMPUTE_MULTI` defined and implemented
- [ ] `RPC_CMD_SET_TENSOR_MULTI` (optional for first iteration)
- [ ] Wire format finalized and version negotiated via HELLO
- [ ] Backward compatibility with single-device clients preserved

### Phase C3: Internal Multi-GPU Scheduler on Server

- [ ] Server uses internal `ggml_backend_sched` across local GPUs
- [ ] Layer-wise splitting logic implemented inside `rpc-server`
- [ ] Local GPU synchronization (CUDA/ROCm events) working
- [ ] Memory allocation strategy across multiple devices
- [ ] Basic error handling when one GPU fails

### Phase C4: Client-Side Aggregation & Discovery

- [ ] HELLO response includes multi-GPU aggregation capability
- [ ] Client can collapse multiple endpoints into one logical device (explicit config first)
- [ ] Tensor split / layer offload logic updated for aggregated devices
- [ ] Configuration option added (`--rpc-aggregate` or similar)

### Phase C5: Hardening, Testing & Polish

- [ ] Multi-device graph caching implemented
- [ ] Full regression testing on single-GPU + Path B fallback paths
- [ ] Heterogeneous GPU support (stretch goal)
- [ ] Production stability + error handling
- [ ] Documentation and example configurations updated

---

## Open Questions (to be resolved before heavy implementation)

| # | Question | Decision | Date |
|---|----------|----------|------|
| 1 | Scope for v1: Homogeneous GPUs only, or attempt heterogeneous? | | |
| 2 | Discovery: Explicit config (`--rpc-aggregate`) or automatic? | | |
| 3 | Who decides the split? Server suggests, or client stays in control? | | |
| 4 | Priority: Focus on 70B+ or also improve 30-40B workloads? | | |

---

## Conclusion

Path C is in the **planning phase**. Core Path B is complete and validated. This tracking file will be updated as we move into implementation.

**Next Milestone:** Resolve open questions → Finalize MVP scope → Start Phase C1.