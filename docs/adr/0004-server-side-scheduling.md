# ADR-0004: Server-Side Scheduling Model for Co-Localized GPUs

## Status

**Pending** — To be decided during D4.2 (Design step of Path C Stepping Stone phase).

## Context

Path C explores server-side scheduling on triton's co-localized dual-GPU (`:50054` 3090 + `:50055` 3070). Currently, the client `ggml_backend_sched` drives every split; each `rpc-server` is a passive compute endpoint. For co-localized GPUs on the same node, server-side scheduling could reduce client RTT overhead and enable parallel dispatch to local GPUs.

This ADR establishes the model for server-side scheduling on co-localized GPUs.

### Problem

Current model:
- Client scheduler dispatches splits serially to each RPC endpoint
- Each `rpc-server` is passive — waits for client commands
- For co-localized GPUs (triton 3090+3070), client RTT adds overhead per split
- No server-side parallelism between local GPUs

Path C question:
- Can the server schedule its own GPUs in parallel?
- What changes to the RPC protocol are needed?
- What is the blast radius on existing Path-B+ behavior?

### Inputs (to be gathered during D4.1)

| Analysis | Source | What we need |
|----------|--------|--------------|
| Per-device RPC splits | C1 baseline on triton | How splits are currently assigned |
| RTT count | C1 baseline | Client RTT overhead per split |
| Server GPU util | C1 baseline | How busy each GPU is today |
| Protocol capabilities | `ggml-rpc.cpp` | What RPC commands already exist |

### Options

#### Option A: Client-driven (no change)

- No server-side scheduling
- Client continues to drive all splits
- Simplest; no RPC protocol changes
- Does not reduce client RTT overhead

#### Option B: Server-side `GRAPH_COMPUTE_ALL`

- Server receives full graph, schedules its own GPUs
- Requires new RPC command or semantic change
- Reduces client RTT (one command vs many)
- More complex; needs graph partitioning logic on server

#### Option C: Server-side parallel dispatch

- Server dispatches to its GPUs in parallel
- Client still specifies split assignment
- Moderate complexity; no graph partitioning needed
- Partial RTT reduction (parallel vs serial)

### Decision

**To be filled during D4.2 based on D4.1 baseline findings.**

### Consequences

**To be filled after decision.**

---

*ADR-0004 placeholder — decision pending D4.1 research*