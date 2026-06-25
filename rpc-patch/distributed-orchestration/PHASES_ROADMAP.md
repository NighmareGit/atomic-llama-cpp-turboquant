# Path C: Phased Evolution Roadmap

**Goal:** Move from centralized orchestration to a more distributed, efficient model over multiple phases while keeping risk manageable.

---

## Phase Overview

| Phase | Name                              | Orchestrator     | Worker Role                     | Key Capability                     | Status      |
|-------|-----------------------------------|------------------|----------------------------------|------------------------------------|-------------|
| **1** | Centralized Async                 | Server           | Passive / Execute + Return      | Async metadata-driven forwarding   | Planning    |
| **2** | Hybrid Controlled Forwarding      | Server + Worker  | Limited autonomous forwarding   | Workers can forward to known peers | Future      |
| **3** | Decentralized Workflow            | Distributed      | Active participants             | Workers maintain local routing     | Future      |

---

## Phase 1: Centralized Async Orchestration (Current Focus)

**Description:**  
The server remains the central brain but stops blocking on every hop. It uses `WorkflowMetadata` to know where to send work next and forwards packages asynchronously.

**Main Wins:**
- Better overlapping of work across workers
- Foundation metadata layer is built
- Relatively low risk

**Key Deliverables:**
- `WorkflowMetadata` + `RoutingTable`
- Async forwarding logic on server
- Basic observability

---

## Phase 2: Hybrid Controlled Forwarding

**Description:**  
Workers gain the ability to forward work packages directly to the next hop (when the next hop is known and trusted), while the server still maintains overall coordination and can intervene when needed.

**Main Wins:**
- Reduced load on server
- Lower latency for some hops
- Still relatively safe (controlled forwarding)

**Key Changes:**
- Workers can resolve `next_hop` using the routing table
- Lightweight validation that a worker is allowed to forward to another
- Server can still override or collect results

---

## Phase 3: Decentralized Workflow Participation

**Description:**  
Workers become more autonomous. They can participate in workflow decisions, potentially handle local queuing, rerouting in case of failure, and maintain their own view of the routing topology.

**Main Wins:**
- Highest utilization and scalability
- Better fault tolerance potential
- Server becomes more of a coordinator / ingress point

**Key Challenges:**
- Much higher complexity
- Requires good failure handling and consistency model
- Security and trust model becomes more important

---

## Migration Strategy Between Phases

- All phases should be **backward compatible** with previous phases.
- The `WorkflowMetadata` structure should be designed to support Phase 3 features even if they are not used in Phase 1.
- Routing table distribution starts simple (server → workers) and can evolve to peer exchange in Phase 3.

---

## Decision Points Between Phases

| Decision Point             | Phase 1 → 2 Trigger                     | Phase 2 → 3 Trigger                     |
|---------------------------|-----------------------------------------|-----------------------------------------|
| Worker forwarding         | When server becomes bottleneck          | When we need higher scalability         |
| Routing ownership         | Server pushes table                     | Workers can exchange / update routes    |
| Failure handling          | Simple fail-generation                  | Partial recovery + replay               |
| Security model            | Existing (low priority)                 | Must be strengthened                    |

---

*This roadmap is intentionally high-level. Details will be filled in as we complete Phase 1 architecture.*

---

## Future Concepts / Phase 3+ Ideas

### Hot Layer Replication

Once we have profiling data about which layers/experts are frequently used ("hot"), we can explore duplicating those layers across multiple workers.

**Potential Benefits:**

- **Failure resilience:** If a worker fails, work can be redeployed to another worker that holds a copy of the required hot layers.
- **Performance boost:** Hot layers can be treated similarly to extra execution units. The orchestrator could route tokens to any available replica, increasing parallelism on the critical path.

This concept would require coordinated KV cache handling and more advanced scheduling. It is considered a Phase 3+ idea.