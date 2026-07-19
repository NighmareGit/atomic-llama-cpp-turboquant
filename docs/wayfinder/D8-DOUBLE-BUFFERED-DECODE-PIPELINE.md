# D8 - Double-Buffered Decode Pipeline (Rotation-Aware Graph Pointer Update)

**Date:** 2026-07-19
**Status:** FUTURE ENHANCEMENT (design sketch, not yet scheduled)
**Prerequisite:** `f68e17b9b` (PPLUS multi-GPU garble fix) is merged and stable
**Parent diagnostic:** [diagnostics/pipeline-plus-dual-gpu-garble/README.md](../../diagnostics/pipeline-plus-dual-gpu-garble/README.md)
**Scope class:** Lateral enhancement (does not alter placement control plane, RPC TP-unit, or kernel-anvil work)

---

## 1. Context - what `f68e17b9b` left on the table

The PPLUS multi-GPU garble fix (`f68e17b9b`) removed copy-slot rotation from
`ggml_backend_sched_pipeline_barrier` because rotating during graph reuse
desynchronised the copy target from the graph's frozen tensor pointers. The fix
restores correctness but disables **double-buffered decode pipelining**, which
was the original point of the B+16 rotation machinery.

Concretely, post-fix on the canonical 27B workload (triton, RTX 3090+3070,
`-sm layer -ts 75,25`, 64-token decode):

| Config | PP (t/s) | TG (t/s) |
|--------|----------|----------|
| Plus=0 (no pipeline parallelism) | 82.69 | 29.94 |
| Plus=1 post-`f68e17b9b` | 76.28 | 29.89 |

TG is at parity with Plus=0 (within 0.2%, noise-dominated). The pipelining
benefit Plus was supposed to provide - overlapping the next step's cross-GPU
copy with the current step's compute - is gone, because the same copy slot is
reused every decode step. Async compute overlap via events is preserved.

This doc sketches how to restore the pipelining benefit safely.

---

## 2. Goal

Re-enable copy-slot rotation during decode such that:

- While backend A computes on slot N, backend B can begin copying inputs for
  the next decode step into slot N+1.
- The graph's `node->src` tensor pointers track the rotation, so compute reads
  from the slot the scheduler copied into.
- All correctness invariants preserved: same output as Plus=0 at `temperature=0`.

---

## 3. Why it was deferred

The fix in `f68e17b9b` chose correctness over pipelining because:

1. The bug it fixed was a silent output-garble - the worst failure mode (users
   get nonsense with no error). A targeted fix to the rotation machinery
   requires touching both `alloc_graph` and `split_graph`, with nontrivial
   validation that the new pointer-update path is sound.
2. The TG delta vs Plus=0 was zero, so there was no immediate throughput cliff
   to recover.
3. The original B+16 design assumed graph pointers were re-derived every step;
   the hot path during generation skips that. Closing the gap properly means
   either making rotation cheap-and-frequent or making pointer update cheap.

A rushed re-enablement would risk reintroducing the garble in a more subtle
form. Worth doing right, not in the same commit as the fix.

---

## 4. Design sketch

### 4.1 Where the frozen pointers live

In `ggml_backend_sched_split_graph` (`ggml/src/ggml-backend.cpp:1840-1900`):

```cpp
if (src_backend_id != cur_backend_id && ...) {
    ...
    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
}
```

`node->src[j]` is set to the `cur_copy` slot's tensor at split-graph time.
This runs inside `alloc_graph`, which is skipped during reuse. Result: the
graph's tensor pointer is stuck on the slot that was current at the last
`alloc_graph` call.

### 4.2 Candidate approaches (ranked)

**A. Re-derive pointers each decode (cheap rotation).**
On every `compute_splits` call (not just alloc), walk split inputs and rewrite
`node->src[j]` to `tensor_id_copy(id, backend_id, cur_copy)`. Cost: one hash
lookup + pointer write per split input per decode (~tens of ns per node).
Benefit: rotation becomes safe because the graph always matches `cur_copy`.

Pros: small, localised patch; no graph rebuild.
Cons: every decode step touches the graph struct (may invalidate CUDA graph
capture if used); needs care around the `split_graph` invariant.

**B. Two pre-built graphs, alternate between them.**
Keep two split graphs in the scheduler, one pinned to copy slot 0 and one to
slot 1, and switch the active graph each decode step. Rotation limited to
depth=2 (matching `n_backends` on the typical 2-GPU case).

Pros: zero per-decode pointer churn; CUDA graph capture friendly.
Cons: doubles scheduler graph storage; complicates alloc/reseve paths.

**C. Slot-stable views over a rotating backing buffer.**
Make the graph tensor pointers fixed; rotate only the underlying buffer
bindings. Each `tensor_copy(id, backend, c)` becomes a view whose `buffer`
field is rebound on rotation.

Pros: graph never changes; cleanest semantically.
Cons: invasive change to `ggml_tensor` lifecycle; touches alloc, views, and
buffer management. Largest blast radius of the three.

**Recommendation:** Approach A first. It's a minimal patch, directly addresses
the root cause, and the per-decode cost is negligible (~tens of ns vs ~33 ms
per-token compute on 27B). If CUDA graph capture becomes a problem later,
fall back to B.

### 4.3 Validation plan

The PPLUS garble ticket already shipped the right feedback loop:

```bash
bash scripts/triton-pplus-garble-loop.sh --plus 1 \
  --model /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf \
  --port 8096 --tokens 64 --expect "Paris" --label "d8-rotation-test"
```

Plus the full 5-case falsification matrix in
`scripts/triton-pplus-falsification-matrix.sh`:
- All Plus=1 variants must stay CLEAN.
- Run 27B and 1.5B for at least 256 decode tokens (longer than the test that
  caught the original garble, which sometimes only manifested after ~30 steps).
- TG throughput should improve over post-`f68e17b9b` Plus=1 baseline.

---

## 5. Expected performance impact

### 5.1 Ceiling analysis (where the time goes)

For 27B Q5_K_M decode on RTX 3090 + 3070 (post-fix measurements):

- Per-token TG compute: ~33.4 ms
- Cross-GPU split-input copy: ~1-2 ms (single hidden state, ~32 MB at PCIe
  Gen4 x4 peer-copy bandwidth, ~24 GB/s)
- Theoretical ceiling from overlap: ~3-6% TG improvement

For 1.5B decode (smaller hidden state, faster compute):

- Per-token TG compute: ~3.4 ms
- Cross-GPU copy: ~0.17 ms (~4 MB hidden state)
- Theoretical ceiling: ~5% TG improvement

### 5.2 Cluster RPC impact (Path-B+)

On RPC deployments (romulus HIP+CUDA, remus/triton workers), cross-node copy
goes over Ethernet and is much slower relative to compute. The overlap win is
larger in absolute ms but smaller in percentage, because copy dominates the
decode loop anyway. Order of magnitude: 5-10% TG improvement at the high end,
0-2% in bandwidth-constrained layouts.

### 5.3 When the enhancement is NOT worth pursuing

- Single-GPU topologies (no cross-GPU copy to overlap).
- Workloads where TG compute is already tiny relative to scheduling overhead.
- If kernel-anvil Slice 7 work (small_k fix, shape tuning) delivers 15-30% TG,
  the D8 ceiling of ~5% becomes a smaller incremental win and should be
  prioritised behind it.

---

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| Reintroduce the exact garble we just fixed | Land behind the falsification matrix; require 256+ decode tokens on 27B before merge |
| CUDA graph capture invalidated by per-decode pointer churn | Approach B as fallback; gate on `GGML_CUDA_DISABLE_GRAPHS` or equivalent capture toggle |
| Wavefront (B+14) interactions: rotation + wavefront depth guards share state | Re-read `ggml-backend.cpp:3399-3418` W2 depth guard before enabling rotation |
| RPC defer paths (B+13/B+9) assumed specific copy slot semantics | Verify `g_rpc_producer_ready_mask` and `barrier_slot_pending` still make sense under rotation |

---

## 7. Acceptance criteria

- [ ] 27B Plus=1 on triton: TG improves by >=2% over post-`f68e17b9b` baseline
      (measured over >=256 decode tokens, median of 3 runs)
- [ ] 1.5B Plus=1: full 5-case falsification matrix stays CLEAN
- [ ] Plus=0 TG unchanged (no regression)
- [ ] Single-GPU Plus=1 unchanged (no regression)
- [ ] RPC cluster smoke (Path-B+): at least parity, ideally improvement
- [ ] No `[DEBUG-...]` tags left in the rotation path

---

## 8. Sequencing recommendation

1. Finish kernel-anvil Slice 7 vectors D/E (expected 15-30% TG, much larger
   win). D8 would be optimising on top of an unoptimised baseline.
2. Land and validate `f68e17b9b` on the 4-GPU cluster (per the original
   ISSUE.md acceptance criteria) to confirm no topology-specific regression.
3. If Slice 7 lands and TG is still the bottleneck on multi-GPU deployments,
   pick up D8 with Approach A.

---

## 9. Code map (where to look first when picking this up)

| Area | File | Notes |
|------|------|-------|
| Frozen pointer write | `ggml/src/ggml-backend.cpp:1900` | `node->src[j] = tensor_id_copy(...)` |
| Rotation skipped by fix | `ggml/src/ggml-backend.cpp:3421-3492` | `pipeline_barrier` no longer rotates |
| `alloc_graph` rotation site | `ggml/src/ggml-backend.cpp:3330-3346` | Where rotation still happens |
| Reuse skip in `graph_compute_async` | `ggml/src/ggml-backend.cpp:3357-3369` | Why rotation was unsound |
| Detector + loop | `rpc-patch/patch/loop-check-garble.sh`, `scripts/triton-pplus-garble-loop.sh` | Validation tooling |
| Falsification matrix | `scripts/triton-pplus-falsification-matrix.sh` | 5-case regression suite |
| Diagnostic pack | `diagnostics/pipeline-plus-dual-gpu-garble/README.md` | Root cause + verification evidence |
