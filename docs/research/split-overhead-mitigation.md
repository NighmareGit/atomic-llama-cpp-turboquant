# Split Overhead Mitigation in Multi-Backend GPipe Dispatch

**Date**: 2026-07-16 (updated after D7.1 prototype)
**Model**: Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf (Q6_K, 21.86 GB, 41 layers, 256 experts/8 active)
**Config**: 3 backends (CPU, CUDA/3060Ti RPC, HIP/7900XTX), GPipe stages=3, n_copies=1
**Prototype**: `GGML_SCHED_FORCE_N_COPIES` env var, A/B tested n_copies=1/4, 1-GPU/2-GPU, n_max=1/2
**Profiler**: `llama-gpipe-profiler` with `GGML_SCHED_TRACE=1`

---

## Key Finding: Split Count Is 2, Not 26-31

The task premise assumed 26-31 splits. **Actual measurement shows only 2 splits** for this GPipe configuration. This is because GPipe's D6.9 stage filtering (`gpipe_active_stage`) assigns each pipeline stage to a single backend, and the model's 41 layers are partitioned into 2 active backends (RPC=stage with MTP draft head, ROCm=main model). The CPU backend exists but hosts no compute splits.

This changes the optimization target: the problem is NOT split count but **per-split synchronization overhead**, specifically the `event_wait_slot` barrier that consumes ~90% of wall time.

---

## Section 1: Per-Split Timing Breakdown

### Steady-State Decode (traces 10-17, warmup excluded)

| Split | Backend | Count | Mean total_us | Mean compute_us | Mean wait_slot_us | Wait/Compute Ratio |
|-------|---------|-------|---------------|-----------------|-------------------|---------------------|
| 0     | 1 (RPC)| 8     | 11.4          | 5.4             | 0.0               | 0x                  |
| 1     | 0 (ROCm)| 8    | 9,373         | 64.0            | 9,194             | **143x**            |

### Phase Breakdown (split 1, ROCm, all 68 traces)

| Phase | Mean us | Total us | % of split_total |
|-------|---------|----------|-------------------|
| event_wait_slot | 8,594 | 585,593 | 89.9% |
| graph_compute_async | 1,758 | 119,539 | 2.6% |
| input_wait_copy | 8,370 | 569,143 | (includes wait_slot) |
| input_copy_slow | 250 | 2,496 | 0.4% |
| host_h2d_issue | 2.7 | 544 | 0.01% |
| event_record | 0.0 | 0 | 0% |

### Critical Insight

Split 1 (ROCm, the main model) spends **89.9% of its time in `event_wait_slot`** — waiting for the previous decode's ROCm event to signal. The actual GPU compute submission (`graph_compute_async`) averages 64 us. The wait/compute ratio is **143:1**.

This is the pipeline serialization bottleneck: with `n_copies=1`, each decode must fully complete before the next begins.

---

## Section 2: Split Classification

With only 2 splits, classification is straightforward:

| Split | Type | Trigger | Backend | Nodes | Inputs |
|-------|------|---------|---------|-------|--------|
| 0 | Backend-transition | `node_backend_id != cur_backend_id` | RPC (1) | ~small | 0 |
| 1 | Backend-transition | `node_backend_id != cur_backend_id` | ROCm (0) | ~3727 | varies |

Both splits are **backend-transition splits** created at pass 5 of `ggml_backend_sched_split_graph()` (line 1628). No weight-incompatibility or input-ceiling splits occur because:
- All model weights are on ROCm (backend 0) or RPC (backend 1)
- The MTP draft head is small enough to not hit the GGML_SCHED_MAX_SPLIT_INPUTS=30 ceiling

### Split Creation Logic (lines 1570-1700)

Three triggers create splits in pass 5:

1. **Backend-transition** (line 1628): `node_backend_id != cur_backend_id`
   - This is the dominant trigger. Each time the assigned backend changes between adjacent non-view nodes, a new split begins.

2. **Weight-incompatibility** (lines 1600-1611): `src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(src, cur_backend_id)`
   - A weight tensor lives on a backend whose buffer type is incompatible with the current split's backend.
   - Creates a split to allow memory reuse of previously offloaded weights.

3. **Input-ceiling** (lines 1613-1621): `split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS` (30)
   - The split already has 30 inputs and a new incompatible input is encountered.
   - Forces a split to bound memory and copy overhead.

---

## Section 3: Coalescing Opportunity Map

### Current State

With only 2 splits on different backends (RPC vs ROCm), **no same-backend coalescing is possible**. The splits are inherently on different physical devices.

### Hypothetical 26-31 Split Scenario

If the model were distributed such that 26-31 splits occurred (e.g., without GPipe stage filtering, or with per-layer backend assignment), coalescing opportunities would arise when:

- **Adjacent splits share backend_id**: A weight-incompatibility or input-ceiling split that targets the SAME backend as the previous split could be merged. The sync barrier between them is unnecessary since they use the same device stream.

- **Estimation**: In a 31-split scenario, if ~40% of splits are same-backend adjacent pairs (typical for MoE with expert weights on different backends), ~12 splits could be eliminated, removing 12 sync barriers.

### Coalescing Barrier Analysis

The sync at each split boundary (lines 2340-2368 in `ggml_backend_sched_compute_splits`):

```cpp
// ggml_backend_sched_wait_copy_slot (line 1950)
if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
    if (split_backend->iface.event_wait != NULL) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } else {
        ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
    }
}
```

For same-backend adjacent splits, this wait is redundant — the previous split's compute is already ordered on the same stream.

---

## Section 4: Ranked Mitigation Strategies

### Strategy 1: Increase n_copies (Pipeline Depth) — HIGH IMPACT

**Mechanism**: Change `sched->n_copies` from 1 to 2-4. This allows multiple decode iterations to be in-flight simultaneously, overlapping compute of iteration N with data transfer of iteration N+1.

**Code location**: `ggml_backend_sched_new()` (line 2820), `n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1`

**Current state**: `n_copies=1` (from log: "sched copies = 1"). The profiler runs single-copy.

**Impact estimate**:
- Current: 95 t/s TG with ~9.4ms per decode (dominated by wait)
- With n_copies=2: overlap ROCm compute with RPC+next-decode setup. Theoretical 1.5-2x throughput.
- Per-ms of sync saved = ~0.8 t/s gain (from task formula: 28 t/s / 35.7ms = 0.78 t/s per ms)

**Risk**: Increases memory (each copy has its own input tensors). With 35B model + 256 experts, memory pressure is real.

**Effort**: Low. Set `parallel=true` in sched config or use `--parallel` flag.

### Strategy 2: Same-Backend Split Coalescing — MEDIUM IMPACT (for high split counts)

**Mechanism**: In pass 5 of `split_graph()`, when a new split is triggered by weight-incompatibility or input-ceiling AND the target backend equals the previous split's backend, merge them instead of creating a new split.

**Code location**: Lines 1628-1648 in `ggml_backend_sched_split_graph()`

**Implementation sketch**:
```cpp
if (node_backend_id != cur_backend_id || need_new_split) {
    // COALESCE: if same backend and trigger is weight-incompatibility or input-ceiling,
    // extend current split instead of creating new one
    if (split->backend_id == node_backend_id && split->n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS - 5) {
        // Keep same split, just continue
        cur_backend_id = node_backend_id;
    } else {
        // Create new split (existing code)
        split->i_end = i;
        i_split++;
        ...
    }
}
```

**Impact estimate**: Only relevant if split count is high (>10). For current 2-split config, zero impact. For hypothetical 31-split scenario, could eliminate ~12 sync barriers = ~12 * 9ms = ~108ms saved per decode (but this is speculative).

**Risk**: Increases per-split input count, may hit GGML_SCHED_MAX_SPLIT_INPUTS (30) more often. Could increase peak memory for split input buffers.

**Effort**: Medium. Requires careful handling of input counting and the memory-reuse logic.

### Strategy 3: Lazy Sync (Defer event_wait_slot) — MEDIUM IMPACT

**Mechanism**: The `event_wait_slot` at lines 2340-2368 waits for the previous copy's event before starting the current split's input copies. For non-RPC backends with async copy support, this could be deferred until the actual backend transition.

**Code location**: `ggml_backend_sched_wait_copy_slot()` (line 1950), called from line 2360.

**Current behavior**: Always waits on `sched->events[split_backend_id][sched->cur_copy]` before starting input copies.

**Proposed change**: Only wait if the previous split used a DIFFERENT backend (true dependency). For same-backend sequential splits, the stream ordering guarantees correctness without explicit wait.

**Impact estimate**: Saves ~5-50us per deferred wait (not the full 9ms — that's inter-decode, not intra-decode). Modest for 2-split config.

**Risk**: Requires tracking which backend last used each copy slot. The event_wait also serves to ensure input buffer reuse safety — deferring it could cause corruption if the buffer is still in use.

**Effort**: Medium-high. Requires careful dependency tracking.

### Strategy 4: Event Gating Extension (D6.10 for intra-GPU splits) — LOW IMPACT

**Mechanism**: Extend the D6.10 GPipe event gating (lines 2930-2960) to gate intra-GPU splits within a single sequence. Currently D6.10 only handles inter-stage (inter-backend) gating.

**Code location**: `ggml_sched_gpipe_wait_seq()` (line 2930)

**Why low impact**: With only 2 splits on different backends, there are no intra-GPU splits to gate. This would only help in the hypothetical 26-31 split scenario.

**Effort**: High. Requires new event allocation and wait logic.

### Strategy 5: Split Reordering — LOW IMPACT

**Mechanism**: Group same-backend splits together to minimize backend transitions. Requires dependency analysis.

**Why low impact**: Split order is largely fixed by the graph topology (layers execute sequentially). Reordering across layers would break KV-cache dependencies.

**Effort**: High. Requires topological sort with backend-grouping heuristic.

---

## Section 5: Recommended Next Experiment

### Experiment: Enable n_copies=2 (Double-Buffering)

**Hypothesis**: Increasing pipeline depth from 1 to 2 copies will overlap ROCm compute with RPC+setup work, hiding the 9ms event_wait_slot latency.

**Expected impact**: 1.4-1.8x TG throughput (95 -> 130-170 t/s).

**Implementation**:

1. The profiler uses `n_copies=1`. To test n_copies=2:
   - Option A: Add `--parallel` flag to profiler (if supported)
   - Option B: Modify `llama-gpipe-profiler.cpp` to pass `parallel=true` to `ggml_backend_sched_new()`
   - Option C: Use `llama-cli` with `--parallel` and compare

2. Measurement:
```bash
GGML_CUDA_GRAPHS=0 GGML_SCHED_TRACE=1 build/bin/llama-gpipe-profiler \
  -m /mnt/980pro/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  -ngl 99 \
  --gpipe-stages 3 \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 --spec-draft-n-min 1 \
  -ctk q8_0 -ctv q8_0 \
  --ctx-size 4096 -n 32 -b 1 -ub 256 \
  --trace \
  -o /tmp/research-split-overhead/heatmap-copies2.json 2>&1 | grep -E 'split_total|TG|t/s'
```

3. Success criteria:
   - event_wait_slot for split 1 drops from ~9ms to <1ms (overlapped with compute)
   - TG increases from ~95 to >130 t/s
   - No OOM (monitor with `rocm-smi` or `nvidia-smi`)

4. If successful, extend to n_copies=3 (triple-buffering) for speculative decoding with draft model.

### Backup Experiment: Same-Backend Coalescing Prototype

If n_copies increase is insufficient, prototype the coalescing logic from Strategy 2:

1. Add a `split_reason` enum to `ggml_backend_sched_split`
2. In pass 5, tag each split with its trigger (backend-transition / weight-incompatibility / input-ceiling)
3. Post-process: merge adjacent same-backend splits where the trigger is NOT backend-transition
4. Measure split count reduction and TG improvement

---

## Appendix: Trace Output Format

The `sched_trace_emit()` function (line 251) outputs JSON lines:

```json
{"ts_us":<steady_clock_us>,"trace_id":<decode_id>,"split":<split_id>,"backend":<backend_id>,"copy":<copy_id>,"phase":"<phase_name>","elapsed_us":<duration_us>}
```

Phases emitted:
- `split_total`: Total time for the split (from split_t0 to end)
- `graph_compute_async`: Time for `ggml_backend_graph_compute_async`
- `event_record`: Time to record the completion event
- `event_wait_slot`: Time waiting on the copy-slot event (sync barrier)
- `input_wait_copy`: Time spent in input copy loop (includes wait_slot)
- `input_copy_slow`: Individual slow input copies (>50us)
- `rpc_prefetch_start`: RPC prefetch issued at split start
- `rpc_gather_flush`: Flush of pending RPC downloads
- `rpc_defer_flush`: Deferred RPC download flush
- `copy_async_ok`: Successful async copy
- `host_h2d_issue`: Host-to-device async copy issue

### Raw Trace Stats (68 decodes, 2 splits each)

| Metric | Split 0 (RPC) | Split 1 (ROCm) |
|--------|---------------|----------------|
| split_total mean | 14.6 us | 10,134.6 us |
| split_total median | 12.0 us | 9,177.0 us |
| split_total min | 9 us | 2,404 us |
| split_total max | 70 us | 58,153 us (warmup) |
| compute mean | 6.6 us | 1,757.9 us |
| wait_slot mean | 0.0 us | 8,593.7 us |

---

## Section 6: D7.1 Prototype Results (2026-07-16)

### Hypothesis

Increasing n_copies from 1 to 4 would overlap the `event_wait_slot` sync barrier,
hiding the 9 ms latency behind concurrent decode iterations (predicted 1.4-1.8x TG).

### Method

Added `GGML_SCHED_FORCE_N_COPIES=N` env var override in `ggml_backend_sched_new()`
(line 2823). A/B tested 5 configs on Qwen35B MTP, n_gen=32, GPipe stages=3.

### Results

| Config | GPUs | n_copies | n_max | PP (t/s) | TG (t/s) | vs baseline |
|--------|------|----------|-------|----------|----------|-------------|
| baseline | 1 | 1 | 1 | 106.3 | 102.5 | — |
| n_copies only | 1 | 4 | 1 | 106.5 | 103.3 | +0.8% |
| n_copies + nmax2 | 1 | 4 | 2 | 106.2 | 102.8 | +0.3% |
| **RPC + nmax2** | 2 | 1 | 2 | 143.8 | **131.1** | **+27.9%** |
| **RPC + nmax2 + ncp4** | 2 | 4 | 2 | 143.4 | **133.0** | **+29.8%** |

### Key Findings

1. **n_copies has negligible impact regardless of GPU count or n_max.**
   +0.8% to +1.4% is within measurement noise. The hypothesis is disproven.

2. **Root cause: GPipe bypasses the pipeline_barrier path entirely.**
   The n_copies>1 optimization targets `ggml_backend_sched_pipeline_barrier()` (line
   3148). But GPipe decode calls `llama_decode_gpipe_multi_impl()` (line 1640),
   which uses `ggml_sched_gpipe_wait_seq()` — a completely different synchronization
   mechanism. These are orthogonal code paths.

3. **The real bottleneck is GPipe's per-stage `event_wait_slot`, not copy-slot contention.**
   With GPipe stages=3 and single-seq, stages execute sequentially (stage0→stage1→stage2)
   within each decode. The `event_wait_slot` (9,194 µs avg) is the inter-decode
   wait for the previous decode's stage N to complete. Since stages are sequential
   in single-seq, there is no intra-decode overlap to exploit.

4. **Dual-GPU + n_max=2 is the actual performance lever.**
   Going from 1-GPU/n_max=1 to 2-GPU/n_max=2 gives +29.8% TG (102.5→133.0 t/s).
   The RPC backend splits model work across both GPUs; stronger speculative decoding
   increases draft token acceptance.

5. **The mitigation for single-seq GPipe is already built: multi-seq (D6.5-D6.9).**
   Multi-seq fills the pipeline with concurrent sequences so that sequence B's
   stage 0 can execute while sequence A's stage 1 is still in GPU compute. This is
   the GPipe-equivalent of copy-slot overlap.

### Updated Strategy Ranking

| # | Strategy | Prototype Result | Verdict |
|---|----------|-----------------|---------|
| 1 | n_copies > 1 | +0.8% to +1.4% (within noise) | ❌ CLOSED — GPipe bypasses pipeline_barrier |
| 2 | Same-backend coalescing | 0 impact (2-split GPipe config) | ⏸️ Deferred to non-GPipe high-split-count scenarios |
| 3 | Lazy sync | Modest, requires dependency tracking | ⏸️ Deferred |
| 4 | Event gating (intra-GPU) | 0 impact (2-split GPipe config) | ⏸️ Deferred |
| 5 | Split reordering | Minimal, breaks KV-cache deps | ⏸️ Deferred |

### Recommendation

No further investment in n_copies. The existing 2-GPU + n_max=2 config at 133.0 t/s
is the single-seq GPipe production baseline. For higher throughput, multi-seq
(D6.5-D6.9, already implemented) is the correct lever — it enables the GPipe
equivalent of copy-slot overlap by running multiple sequences concurrently through
the pipeline stages.
