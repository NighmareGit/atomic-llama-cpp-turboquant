# Path C: Distributed Orchestration — Global Overview

**Date:** 2026-06-25  
**Status:** Architecture Definition Phase  
**Parent:** Path B (Event-Based Pipeline Parallelism)

---

## 1. Executive Summary

Path C aims to improve hardware utilization when using multiple RPC workers by introducing **metadata-driven asynchronous work distribution**.

Current limitation (even after Path B):
- The server (llama-server) still acts as a strict serial dispatcher.
- Workers largely sit idle while waiting for the previous hop to complete and return to the server.

Path C introduces a workflow-oriented model where work packages carry metadata that describes the next steps, enabling better overlap across workers.

---

## 2. Core Philosophy & Constraints

- **Evolutionary, not revolutionary**: Start with centralized but async orchestration (Phase 1), then gradually decentralize.
- **Metadata as foundation**: All future flexibility comes from rich but lightweight metadata.
- **Security is lowest priority** during skunkworks phase (use existing RPC trust model).
- **Prefer static KV ownership**: Avoid moving full KV cache between workers when possible.
- **Minimize data movement**: Only send what is strictly necessary between hops.

---

## 3. Major Architecture Topics (Pre-Flight Checklist)

Below is the complete list of topics that need to be addressed, along with the **recommended best approach** for each.

### 3.1 KV Cache Ownership & Sharding Strategy
**Why it matters:** This decision affects almost everything else (data movement, correctness, complexity).

**Recommended Approach:**
- Use **static layer-range ownership**.
- Each worker owns a fixed set of layers **and** maintains the KV cache for the sequences it processes.
- Do **not** move full KV history between workers.
- Only send new token projections when moving to the next worker.

### 3.2 Work Package Content
**Why it matters:** Defines what travels over the network.

**Recommended Approach:**
- Keep packages as small as possible.
- Include: Token info, metadata, and only the **minimal intermediate activations** needed by the next stage.
- Avoid sending full KV states unless strictly necessary.

### 3.3 Workflow Metadata Schema
**Why it matters:** The control plane of the entire system.

**Recommended Approach:**
- Design a lightweight but extensible metadata structure from the beginning.
- Core fields: `workflow_id`, `step_id`, `next_hop`, `layer_range`, `sequence_id`.
- This metadata should support both Phase 1 (server forwarding) and future decentralized phases.

### 3.4 Routing Table & Discovery
**Why it matters:** How workers know where to send work.

**Recommended Approach:**
- Server pushes a small routing table once at the start of generation.
- Use extended `RPC_CMD_HELLO` or a new `RPC_CMD_WORKFLOW_ANNOUNCE`.
- Keep discovery centralized in Phase 1.

### 3.5 Orchestration Model (Phase 1)
**Why it matters:** Defines how work actually flows in the first version.

**Recommended Approach:**
- Server remains the logical orchestrator.
- Server performs **asynchronous, non-blocking forwarding** based on metadata.
- Workers execute their portion and either return results or (later) forward directly.

### 3.6 Worker Local Queuing & Execution
**Recommended Approach:**
- Workers maintain a simple local queue for incoming work packages.
- Start with straightforward single-threaded execution per worker.

### 3.7 Failure Modes & Recovery
**Recommended Approach:**
- Phase 1: Simple fail-the-generation model on worker failure.
- Add partial recovery and replay capabilities in later phases.

### 3.8 Token Ordering & Consistency
**Recommended Approach:**
- Use `workflow_id` + `step_id` in metadata for ordering.
- Enforce strict sequencing at the server level during Phase 1.

### 3.9 Data Movement & Activation Strategy
**Recommended Approach:**
- Prefer co-located GPUs (one rpc-server managing multiple devices) when possible.
- When crossing machines, minimize transferred tensor size.
- Defer heavy optimization (compression, zero-copy, etc.) until needed.

### 3.10 Phasing & Evolution Path
**Recommended Phasing:**

| Phase | Model                        | Worker Role                     | Key Feature                     |
|-------|------------------------------|----------------------------------|---------------------------------|
| 1     | Centralized + Async          | Relatively passive               | Server does async forwarding    |
| 2     | Hybrid                       | Limited autonomous forwarding    | Controlled worker-to-worker     |
| 3     | Decentralized                | Active participants              | Workers use local routing       |

### 3.11 Observability & Debugging
**Recommended Approach:**
- Include `workflow_id`, `step_id`, and worker identity in all relevant logs from the start.
- Add optional tracing information in metadata.

### 3.12 Security & Trust Model
**Decision:** Lowest priority during skunkworks.
- Use existing RPC trust model.
- Revisit only after core functionality is validated.

---

## 4. Next Steps

1. Complete architecture decisions using structured grilling process.
2. Document locked decisions in `DECISIONS.md`.
3. Create detailed `PHASE_1_PLAN.md`.
4. Begin implementation only after key topics (especially KV Cache and Metadata) are sufficiently clear.

---

*This document is the living global view. It will be updated as decisions are made.*