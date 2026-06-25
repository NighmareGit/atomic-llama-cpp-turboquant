# Path C: Core Architecture & Data Models

**Status:** Initial Draft  
**Date:** 2026-06-25

This document defines the core data structures and component responsibilities for Path C.

---

## 1. Guiding Principles

- Metadata-driven workflow
- Evolutionary design (Phase 1 = centralized async)
- Minimize data movement between workers
- Static KV cache ownership where possible
- Reusable foundation for later decentralized phases

---

## 2. Core Data Models

### 2.1 Workflow Metadata

This is the most important structure. It travels with every work package.

```cpp
struct WorkflowMetadata {
    // === Identification ===
    uint64_t workflow_id;           // Unique identifier for this generation/session
    uint64_t step_id;               // Monotonic step counter in the workflow

    // === Sequence Information ===
    uint32_t sequence_id;           // Which sequence this work belongs to
    uint32_t token_position;        // Current token position in the sequence

    // === Routing ===
    uint32_t source_worker_id;      // Worker that processed this package before
    uint32_t destination_worker_id; // Worker that should process this next

    // === Layer Range ===
    uint32_t layer_start;           // Start of the layer range this package targets
    uint32_t layer_end;             // End of the layer range this package targets

    // === Status & Control ===
    uint32_t status;                // 0 = Success, >0 = Error/Warning
    uint32_t error_code;            // Specific error identifier
    uint32_t flags;                 // Bitfield for optional/future behavior
};
```

**Design Notes:**
- Must be small and efficient to serialize.
- `destination_worker_id` is resolved via the routing table.
- Designed to support both server-orchestrated and future worker-driven forwarding.

### 2.2 Work Package

The unit of work that moves between nodes.

```cpp
struct WorkPackage {
    WorkflowMetadata metadata;
    // Payload (flexible)
    std::vector<uint8_t> payload;   // Can contain:
                                    // - Token IDs / batch info
                                    // - Intermediate activations (minimal)
                                    // - KV updates (if needed)
};
```

**Phase 1 Recommendation:**
- Keep `payload` as small as possible.
- Prefer sending only what the next stage strictly needs.

### 2.3 Routing Table

Pushed by the server at the beginning of a generation.

```cpp
struct RoutingEntry {
    uint32_t worker_id;
    std::string endpoint;           // e.g. "127.0.0.1:50052" or logical ID
    uint32_t layer_start;
    uint32_t layer_end;
    bool is_local;                  // true if this is the server's own GPU
};

struct RoutingTable {
    uint64_t workflow_id;
    std::vector<RoutingEntry> entries;
};
```

**Usage in Phase 1:**
- Server uses this table to resolve `destination_worker_id` → actual destination.
- In later phases, workers can also use this table for direct forwarding.

---

## 3. Component Responsibilities (Phase 1)

| Component       | Responsibilities                                                                 | Intelligence Level |
|-----------------|----------------------------------------------------------------------------------|--------------------|
| **Server**      | Build routing table, create initial metadata, async forwarding, collect results | High (orchestrator) |
| **Worker**      | Execute assigned layers, update metadata `step_id`, return or forward result    | Low–Medium        |
| **Metadata**    | Carries routing + execution state                                                | Core              |

---

## 4. High-Level Flow (Phase 1)

1. Server loads model and builds `RoutingTable`.
2. Server sends `RoutingTable` to all workers (via extended HELLO or new command).
3. For each token:
   - Server creates `WorkPackage` with initial metadata.
   - Server sends package to first worker asynchronously.
   - Worker processes its layers → updates `step_id` + `destination_worker_id`.
   - Worker returns result to server (or forwards in future phases).
4. Server collects completed packages and continues.

---

## 5. Open Questions (to be resolved during grilling)

- Exact fields needed in `WorkflowMetadata`
- What belongs in `payload` vs what can be referenced
- How to handle KV cache updates in the package
- Size and serialization format of `WorkPackage`

---

*This is a living document. It will be refined as we lock decisions on KV Cache and Metadata.*