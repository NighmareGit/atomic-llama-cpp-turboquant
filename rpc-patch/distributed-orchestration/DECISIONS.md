# Path C: Locked Architecture Decisions

This file records all major decisions made during the architecture phase.

---

## 2026-06-25

### Topic 1: KV Cache Ownership & Sharding Strategy
- **Decision**: Use **static layer-range ownership** in Phase 1.
- Each worker owns a fixed set of layers + maintains its own KV cache.
- Smart/dynamic hot-cold expert placement and runtime profiling = **Phase 2/3** topic.
- Optional lightweight profiling hooks (behind an argument) can be added for future use.

### Topic 2: Work Package Content
- **Decision**: Keep the Work Package **lean** in Phase 1.
- Must include `WorkflowMetadata` + basic token/sequence information.
- `payload` field exists as a **stub** for future richer content.
- Do **not** implement heavy intermediate activation transfer or KV movement in Phase 1.
- Design for future extensibility, but keep implementation minimal.

### Topic 3: Workflow Metadata Schema
- **Decision**: Use the following structure for Phase 1:

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

This provides good tracing, basic error propagation, and cheap future extensibility while staying lightweight for Phase 1.