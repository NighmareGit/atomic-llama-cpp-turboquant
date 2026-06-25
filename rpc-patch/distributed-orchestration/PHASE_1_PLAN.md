# Path C Phase 1: Implementation Plan

**Phase Name:** Centralized Async Orchestration  
**Goal:** Improve utilization by making the server forward work packages asynchronously using metadata, without blocking on every hop.  
**Risk Level:** Medium  
**Estimated Effort:** Medium

---

## 1. Objectives for Phase 1

- Enable asynchronous work forwarding from the server.
- Introduce `WorkflowMetadata` that travels with work packages.
- Push a `RoutingTable` from the server to workers at the start of generation.
- Keep workers relatively simple (low intelligence) in this phase.
- Achieve measurable improvement in hardware utilization compared to current serialized dispatch.
- Maintain full backward compatibility with existing single-worker usage.

---

## 2. Scope (What is included)

- Server-side changes to support async forwarding logic.
- Definition and serialization of `WorkflowMetadata`.
- Basic `RoutingTable` distribution mechanism.
- Minimal changes to `rpc-server` and `llama-server` / `llama-cli`.
- Logging and basic observability (`workflow_id`, `step_id`).

**Out of Scope for Phase 1:**
- Worker-to-worker direct forwarding
- Complex failure recovery / replay
- Dynamic routing table updates during generation
- Heavy optimization of activation transfer size

---

## 3. Key Components to Modify

| Component          | Changes Needed                                      | Difficulty |
|--------------------|-----------------------------------------------------|------------|
| `ggml-rpc.cpp`     | Add support for new metadata structures             | Medium     |
| `rpc-server`       | Handle routing table distribution + async logic     | Medium     |
| `llama-server`     | Build and send routing table, manage async flow     | Medium-High |
| Protocol           | Minor additions (new commands or extended metadata) | Low-Medium |

---

## 4. Implementation Steps (Recommended Order)

### Step 4.1 – Define Core Data Structures
- Finalize `WorkflowMetadata` struct
- Finalize `WorkPackage` structure
- Define serialization format (keep it simple and binary)

### Step 4.2 – Routing Table Distribution
- Add mechanism for server to send `RoutingTable` to workers
- Decide whether to extend `RPC_CMD_HELLO` or create a new command (`RPC_CMD_WORKFLOW_ANNOUNCE`)

### Step 4.3 – Async Forwarding Logic (Server Side)
- Modify server dispatch loop to be non-blocking where possible
- Use existing deferred send patterns from Path B where applicable
- Implement metadata update when forwarding packages

### Step 4.4 – Worker-Side Changes
- Workers should be able to read and update `WorkflowMetadata`
- Minimal change: Workers update `step_id` and return result to server

### Step 4.5 – Integration & Testing
- Test with 2 workers (one RPC + local GPU)
- Measure utilization improvement vs current behavior
- Validate correctness on small and medium models first

---

## 5. Success Criteria

- Server can send work to multiple workers without blocking on every hop.
- Measurable reduction in idle time on secondary workers.
- No regression in generation quality or correctness.
- Clean logs showing `workflow_id` and `step_id` flow.

---

## 6. Risks & Mitigations

| Risk                              | Mitigation                                      |
|-----------------------------------|-------------------------------------------------|
| Increased complexity in dispatch  | Keep Phase 1 logic as simple as possible        |
| Metadata design is insufficient   | Design metadata to be extensible from the start |
| Performance gain is small         | Measure early with realistic workloads          |
| Breaking existing single-worker path | Maintain full backward compatibility         |

---

*This plan will be refined after key architecture decisions (especially KV Cache and Metadata) are locked.*