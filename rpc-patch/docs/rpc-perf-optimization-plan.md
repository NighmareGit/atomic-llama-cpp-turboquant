# RPC Performance Optimization Plan

## Context

### Symptom
70B model (40 GB): **35+ tok/s locally** (7900XTX + 3060 Ti) drops to **<15 tok/s** in RPC mode over 2.5 GbE. Network traffic stays low (~700 KB/s at 15 tok/s) — this is **not** a bandwidth problem.

### Root Cause Analysis

#### Primary: Pipeline Parallelism Disabled
The RPC backend reports `caps.async=false, caps.events=false` (`ggml-rpc.cpp:2501-2506`). The scheduler checks (`llama-context.cpp:358-374`) and forces `pipeline_parallel=false`, which means `n_copies=1` — **all backends run strictly sequentially**, zero overlap between RPC server and local GPU.

**Local dual-GPU** → pipeline_parallel enabled, GPUs overlap → 35+ tok/s
**RPC mode** → pipeline_parallel disabled, sequential execution → <15 tok/s

#### Secondary: 7-22+ Blocking RTTs per Token
| RPC Call | Count | Blocks? |
|----------|-------|---------|
| `SET_TENSOR` | 5-20 (embeddings, positions, mask, KV indices) | Yes |
| `GRAPH_RECOMPUTE` | 1 | No (fire-and-forget) |
| `GET_TENSOR` | 1 | Yes |

Each SET_TENSOR serializes metadata + sends + waits for ACK before the next can begin.

#### Tertiary: KV-Cache Offload Multiplies the Problem
`--no-kv-offload` (`offload_kqv=false`) puts KV cache on CPU while attention runs on RPC GPU. For each of N layers, K and V cache tensors cross the CPU→RPC boundary — **2N extra SET_TENSOR calls**. For an 80-layer 70B: 160 extra RTTs per token. This is why you see massive differences across `no-kv-offload` / `kv-offload` / `kv-unified` toggles: they directly amplify or reduce the RTT count.

### "All GPUs via RPC Workers" — Assessment
You suggested running llama-server on a CPU/iGPU host and attaching all GPUs via RPC workers. Here's what that actually does:

- **If GPUs are on separate machines** → they're RPC devices, same protocol limits apply. No benefit.
- **If all GPUs are on one RPC server but separate devices** → client creates splits between them → **more RTTs, worse**.
- **If all GPUs on one RPC server as ONE backend** → requires Path C multi-GPU aggregation → could help, but significant engineering.
- **If you eliminate the local-GPU/RPC split** (CPU-only client, single RPC server with best GPU) → **helpful** — 0 code changes, reduces split count. Worth testing.

The core issue isn't where GPUs are located — it's the **synchronous RPC protocol + disabled pipeline parallelism**. Fix those and any topology benefits.

---

## Implementation Paths

### Path A: Protocol Optimization — Reduce RTTs & Enable Async
**Effort:** 3-5 days | **Risk:** Low | **Gain:** +30-50% estimated

| Phase | What | Where | Detail |
|-------|------|-------|--------|
| **A1** | Batch `SET_TENSOR` calls | `ggml-rpc.cpp` | Thread-local buffer accumulates tensor sets. Flush before compute/get/copy. New `RPC_CMD_SET_TENSOR_BATCH` (cmd 18). Server loops over entries calling existing `set_tensor`. Cuts 5-20 send() calls → 1. |
| **A2** | Async `get_tensor` | `ggml-rpc.cpp` | Add deferred-response tracker to `socket_t`. `get_tensor_async` sends request, returns immediately. `synchronize` drains pending response. Set `caps.async=true`. |
| **A3** | Scheduler overlap | `ggml-backend.cpp` | Avoid redundant `synchronize` before RPC→GPU copies. Let async GET_TENSOR response arrive during other backend work. |

### Path B: Full Pipeline Parallelism for RPC
**Effort:** 5-10 days | **Risk:** Medium | **Gain:** +80-150% estimated (requires A2)

| Phase | What | Where | Detail |
|-------|------|-------|--------|
| **B1** | Event support | `ggml-rpc.cpp` | `RPC_CMD_EVENT_RECORD` leverages TCP ordering: sent after `GRAPH_COMPUTE`, server responds immediately (meaning prior compute finished). Implement `event_new/free/record/wait/synchronize` on device + backend interfaces. |
| **B2** | Enable pipeline_parallel | `ggml-rpc.cpp` | Set `caps.events=true`. Detection code in `llama-context.cpp` now passes. `n_copies=4`, pipeline scheduling activates. |
| **B3** | Tune & verify | `ggml-backend.cpp` | Fix `synchronize` to not interfere with events. Verify GPU utilization overlap. |

### Path C: RPC Server Multi-GPU Aggregation
**Effort:** 10-15 days | **Risk:** High | **Gain:** +100-200% estimated (independent, can parallel A/B)

| Phase | What | Where | Detail |
|-------|------|-------|--------|
| **C1** | Baseline | Setup | CPU-only client, 1 RPC server with both GPUs. Measure split overhead when copies are server-local PCIe. |
| **C2** | Multi-device server scheduling | `ggml-rpc.cpp` server | `RPC_CMD_GRAPH_COMPUTE_MULTI`. Server creates internal scheduler across its GPUs for one graph. Client sends ONE compute command for all layers. |
| **C3** | Client endpoint grouping | `ggml-rpc.cpp` client | Detect same-endpoint devices. Send `GRAPH_COMPUTE_MULTI` instead of per-device `GRAPH_COMPUTE`. Handle scheduler split merging. |
| **C4** | Multi-graph caching | `ggml-rpc.cpp` server | `RPC_CMD_GRAPH_RECOMPUTE_MULTI`. Cache combined multi-device graph on server. |

---

## Tracking File Format

File: `.opencode/plans/rpc-path-{A|B|C}-tracking.md`

Each tracking file follows:

```
# Path {A|B|C} Status

**Overall:** {NOT STARTED | IN PROGRESS | COMPLETE | BLOCKED}
**Build:** {PASS | FAIL}
**Current Phase:** {name}
**Phase Status:** {NOT STARTED | IN PROGRESS | COMPLETE | BLOCKED}

## Implementation Log

### YYYY-MM-DD HH:MM — Phase N: {name}
- **Done:** what was implemented
- **Files:** paths
- **Blockers:** none or description
- **Debug:** issues found and fixed
- **Next:** what to do next

## Issues
| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|

## Benchmarks
| Run | Config | Baseline tok/s | Modified tok/s | Delta |
|-----|--------|----------------|----------------|-------|
```

---

## Recommended Execution

1. **Quick test (0 code changes):** CPU-only client, `--rpc` to 7900XTX-only RPC server, `-ngl 80`. If this alone narrows the gap, the bottleneck was the local-GPU/RPC split.
2. **Path A1** — batching. Immediate benefit for ALL RPC modes.
3. **Path A2 + B1 + B2** — async + events. Enables pipeline parallelism. Highest ROI code change.
4. **Path C** — only if you need combined GPU memory from multiple GPUs and Paths A+B aren't enough.

---

## System Prompt for Implementation AI

```
You are implementing a phase of the atomic-llama-cpp-turboquant RPC optimization plan.

**Rules:**
1. MEASURE before and after — bit-exact output must match baseline
2. MINIMAL changes — only the files and functions specified
3. SAFE fallback — every new feature gracefully degrades if unsupported
4. PROGRESSIVE — one phase at a time, update tracking file after each
5. CODING — follow existing style, use GGML_LOG_DEBUG for diagnostics,
   no undefined behavior, no compiler extensions

**Verification before marking phase complete:**
- `cmake --build build` succeeds with no warnings
- `git diff` shows only intended changes
- Run server + client, same prompt: output matches baseline character-for-character
- Performance not regressed (>2% slower = debug and fix)

**Debug mode:** Set GGML_RPC_DEBUG=1, compare traces, fix, retest, remove traces.

**Safety:** Never change wire format without version negotiation. Never remove
bounds checks. Always validate tensor dimensions/types server-side.
New RPC commands increment RPC_PROTO_MINOR_VERSION.
```
