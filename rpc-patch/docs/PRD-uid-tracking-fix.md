# PRD: Per-Split Graph UID Tracking for Interleaved RPC Placement

## Goal

Enable `GRAPH_RECOMPUTE` fire-and-forget for every split on every RPC device,
including devices with 2+ interleaved splits. Eliminate the redundant blocking
`GRAPH_COMPUTE` TCP round-trip on gen tokens.

## Requirements

1. **Per-split independence**: Each split's reuse decision is based on whether
   that specific split's `cgraph->uid` has been seen before, independent of any
   other split on the same device.

2. **All reuse paths updated**: Both the single-device path (`GRAPH_COMPUTE` /
   `GRAPH_RECOMPUTE`, line ~2609) and the multi-device path (`GRAPH_COMPUTE_ALL`
   / `GRAPH_RECOMPUTE_ALL`, line ~2550) must use the new tracking.

3. **First-token guard preserved**: The `cgraph->uid != 0` check must remain —
   UIDs that are 0 (graph not assigned by the scheduler) must never trigger
   reuse.

4. **Bounded memory**: The tracking structure must not grow unbounded. After
   the scheduler's first allocation, the number of splits per device is fixed,
   so the structure holds at most `n_splits` entries per device.

5. **Thread safety**: The tracking structure is accessed from the scheduler's
   graph_compute thread only (same threading model as the existing
   `last_graph_uid`), so no new synchronization is required.

6. **Debug observability**: A `LOG_DBG` line on reuse hit/miss so the fix can
   be verified in logs without a wire trace.

## Non-Requirements

- **No scheduler changes**: `ggml-backend.cpp` split UID assignment is already
  correct (each split gets a unique UID). Do not touch it.
- **No wire protocol change**: Same RPC commands, same serialization, same
  server-side graph cache.
- **No server changes**: The server already caches graphs by hash.
- **No eviction policy**: UIDs are never invalidated. Graph shape changes
  (e.g. context shift) produce a new scheduler graph with new split UIDs, so
  stale entries are never matched.

## Success Criteria

- `grep "RPC-REUSE" log | grep "reuse=1"` shows ~100% for gen tokens (after
  warmup). The exact token count depends on split count: a device with N splits
  shows N `reuse=0` entries on the first token, then N `reuse=1` entries on every
  subsequent token.
- No regression on single-split devices (they already worked; the set behaves
  identically to the scalar for N=1).

## Edge Cases

### Warmup vs request
The first `GRAPH_COMPUTE` for any request is always a full upload (reuse=false)
because each new request produces a new scheduler graph with fresh split UIDs.
This is correct — the server must receive the graph at least once.

### Prompt vs gen
Prompt evaluation uses a separate scheduler graph (different UIDs) from gen.
Both get full uploads on first encounter. During gen, UIDs are stable across
tokens, so reuse fires every token after the first.

### Graph shape changes (context shift / KV resize)
When the scheduler re-splits (e.g. after a context shift), new split UIDs are
assigned. These won't match any entry in the set, so the first token after a
reshaping correctly does a full upload. Old UIDs remain in the set but are
never matched again — harmless because the set is bounded by the number of
live splits.

### Multi-device (2+ GPUs behind one RPC endpoint)
The `GRAPH_COMPUTE_ALL` / `GRAPH_RECOMPUTE_ALL` path (line ~2550) has the same
single-slot bug. The fix applies identically — `seen_graph_uids` lives in the
device context, which is per-device, so both paths benefit.

### Concurrent graph_compute calls
Not a concern — the RPC backend's `graph_compute` is called serially by the
scheduler for a given backend. The existing `last_graph_uid` had no
synchronization; the set inherits the same safe access pattern.
