# SPEC: Per-Split Graph UID Tracking for Interleaved RPC Placement

## Problem

The `GRAPH_RECOMPUTE` fire-and-forget optimization in `ggml-rpc.cpp` is dead code
for any RPC device with 2+ splits (interleaved layer placement). Only the last
split per device gets `reuse=true`; all others pay the full blocking TCP cost
every token.

## Root Cause

`ggml_backend_rpc_device_context` tracks a single `uint64_t last_graph_uid` per
device (line 919). The scheduler assigns each split a unique UID
(`ggml_backend_sched_split_graph`, line 2036 of `ggml-backend.cpp`). When a device
owns multiple splits, the single slot is overwritten each split, so only the
final split's UID is remembered.

Failure timeline for an interleaved device with splits A(uid=100), B(uid=200):

| Token | Split | cgraph->uid | last_graph_uid before | reuse | last_graph_uid after |
|-------|-------|-------------|-----------------------|-------|----------------------|
| 1     | A     | 100         | 0                     | false | 100                  |
| 1     | B     | 200         | 100                   | false | 200                  |
| 2     | A     | 100         | 200                   | false | 100                  |
| 2     | B     | 200         | 100                   | false | 200                  |

Every gen token pays full `GRAPH_COMPUTE` cost for both splits. The
`GRAPH_RECOMPUTE` path (fire-and-forget + EVENT_RECORD) is never taken.

## Target Behavior

Every split on every device must independently track whether its graph has been
uploaded to the server. After the first full `GRAPH_COMPUTE` for a given UID,
all subsequent calls with the same UID take the `GRAPH_RECOMPUTE` fast-path.

## Design Options

### Option (a): Hash set of seen UIDs — RECOMMENDED

```cpp
struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    std::unordered_set<uint64_t> seen_graph_uids;
};
```

- Reuse check: `cgraph->uid != 0 && rpc_dev_ctx->seen_graph_uids.count(cgraph->uid)`
- On full compute: `rpc_dev_ctx->seen_graph_uids.insert(cgraph->uid)`
- Bounded: splits are fixed after first scheduler alloc, so the set holds at
  most `n_splits` entries per device (typically 2-8).
- O(1) lookup, no collision risk, minimal memory.

### Option (b): Fixed array indexed by split position

```cpp
uint64_t last_graph_uids[GGML_SCHED_MAX_SPLITS];
```

- Requires knowing split position at compute time (not directly available in
  `ggml_backend_rpc_graph_compute` — only `cgraph` is passed).
- Wastes memory on `GGML_SCHED_MAX_SPLITS` slots even when a device has 1 split.
- Rejected: needs extra plumbing to recover split index.

### Option (c): Per-backend split cache (store graph alongside UID)

- Stores serialized graph bytes alongside UID — redundant, since the server
  already caches the graph by hash.
- Rejected: over-engineered, no benefit over a simple seen-set.

## Recommendation

Option (a). `std::unordered_set<uint64_t>` is the minimal correct structure:
it directly models "which graph UIDs have I already uploaded to this device."
It is already in the translation unit's include set (line 16), requires no new
dependencies, and the struct remains trivially aggregate-initialized except for
the set (which default-constructs empty).

## Affected Code Paths

Both reuse sites must be updated:

1. **Multi-device path** (line 2550): `GRAPH_COMPUTE_ALL` / `GRAPH_RECOMPUTE_ALL`
2. **Single-device path** (line 2609): `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE`

Both use the identical pattern (`cgraph->uid != 0 && last == uid`), so the fix
is symmetric.

## Backward Compatibility

- Wire protocol unchanged — same commands, same serialization.
- Server unchanged — it already caches graphs by hash.
- The `cgraph->uid != 0` guard still correctly disables reuse for UIDs that
  haven't been assigned (e.g. non-scheduler direct calls).
