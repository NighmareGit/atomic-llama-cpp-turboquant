# V0 Phase 2 vs Current: Row-Split Output Correctness Regression

**Date:** 2026-07-24
**Investigator:** v0-phase2-comparison research subagent
**Scope:** `ggml/src/ggml-rpc/ggml-rpc.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-backend.cpp`, `src/llama-model.cpp`
**Builds compared:** Phase 2 (pre-`8cb644877`, ~build 9964) vs Current (`bf38bb1c6` / `83605ae27`)

---

## Summary Table (visible if truncated)

| Question | Answer |
|----------|--------|
| Phase 2 build commit? | No single commit. Phase 2 = benchmark result (108.81 t/s) from ~build 9964 era (late June, pre-`8cb644877`). Used the **native CUDA split buffer**, NOT the RPC split buffer. |
| What broke correctness? | The RPC split buffer type (commits `8cb644877` + `bf38bb1c6`) is entirely new and has **never produced correct output**. Two compounding bugs: (1) `serialize_tensor` split-dimension adjustment is clobbered by an unconditional overwrite loop; (2) even if fixed, activation tensors keep full `ne[1]` while weight tensors get slice `ne[1]`, creating graph-wide dimension mismatches. |
| Minimal fix? | Two-part: (A) Fix the `serialize_tensor` clobber bug (move the split adjustment AFTER the loop, ~5 lines). (B) Make the server use the **native CUDA split buffer** path instead of a plain CUDA buffer, so `mul_mat`'s internal `row_low`/`row_high` slice resolution handles weights AND activations consistently. |
| Feasible in ~100 lines? | Part A (clobber fix) alone: ~5 lines, but insufficient (only fixes weights, not activations). Full fix (Part A + Part B server-side CUDA split buffer reuse): ~80-120 lines, feasible but touches `alloc_buffer_split`, `set_tensor`, `deserialize_tensor`. |

---

## 1. Phase 2 Build Commit

**There is no single "Phase 2 build commit."** The 108.81 t/s figure is a benchmark result, not a code state.

### Evidence

- The Phase 2 result (108.81 t/s, 70/30 row-split, 7900 XTX + 5060 Ti) is referenced in `.scratch/CONTEXT.md:15` and `.scratch/benchmarks/T2-5gpu-multi-backend.md:128` as a baseline, with no associated git hash.
- `.scratch/research/v3-kernel-profiling.md:394` labels it "Phase 2 (older build)".
- The RPC split buffer type (`ggml_backend_rpc_split_buffer_type`) was introduced in commit `8cb644877` (2026-07-23 16:19) and has **never produced a successful compute run** (per `.scratch/benchmarks/T5-row-split-gc-diag.md:104-106`).

### Key finding: Phase 2 used the native CUDA split buffer, not RPC

`.scratch/benchmarks/T5-row-split-gc-diag.md:104-106` resolves the ambiguity definitively:

> "The RPC split buffer type (commit `8cb644877`, 2026-07-23) is *newer* than the Phase 2 row-split result and has **never produced a successful compute run**... Phase 2's 108.81 t/s row-split therefore used the **native CUDA split buffer** (local multi-GPU) or a pre-split-build path - **not** the RPC split buffer."

The T2 report (`.scratch/benchmarks/T2-5gpu-multi-backend.md:53-59`) confirms: "Row split aborts at model load... Row split is deprecated and incompatible with RPC `split_buffer_type`."

### Timeline of the relevant commits

| Commit | Date | Description | Role |
|--------|------|-------------|------|
| `5460affb4` | 2026-06-27 | Phase 2 trace fixes + 2gpu bench (G=48.9) | Phase 2 era (build ~9964) |
| `8cb644877` | 2026-07-23 16:19 | Add RPC split buffer type (first version) | Crashed at model load (set_tensor) |
| `4b30e9aa1` | 2026-07-23 16:37 | Fix socket lifetime across alloc/set_tensor | Crash fixed, but graph_compute OOB |
| `bf38bb1c6` | 2026-07-24 11:04 | Row-split buffer support (split_buffer_context) | No crash, but **garbled output** (this commit) |
| `83605ae27` | 2026-07-24 11:05 | Enable GGML_RPC_UDP by default | Unrelated to correctness (throughput only) |

**Conclusion:** The "regression" is not a regression of working code. The RPC split buffer path is brand new (2026-07-23/24) and was never correct. Phase 2's correct output came from the native CUDA split buffer, which resolves row splits **inside** `mul_mat` via `tensor->extra` and `row_low`/`row_high` - a mechanism the RPC path does not replicate.

---

## 2. What Changed in the Split-Buffer Code Path

### 2.1 The RPC split buffer is entirely new code

Commit `8cb644877` introduced `ggml_backend_rpc_split_buffer_type` (+456 lines). Before this, the RPC backend returned `NULL` when `llama-model.cpp:935` queried for `"ggml_backend_split_buffer_type"`, so row-split over RPC was impossible (model load failed).

Commit `bf38bb1c6` rewrote/extended it (+241/-126 lines) with `split_buffer_context`, slice tracking, and the `serialize_tensor` dimension adjustment.

### 2.2 Bug #1: serialize_tensor dimension adjustment is clobbered (PRIMARY BUG)

**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:1593-1648`

The `serialize_tensor` function has a split-buffer branch (added in `bf38bb1c6`) that sets `result.ne[1] = nrows_split`:

```cpp
// Lines 1608-1624 (split buffer branch)
} else if (tensor->buffer && tensor->buffer->iface.free_buffer == ggml_backend_rpc_split_buffer_free_buffer) {
    ggml_backend_rpc_split_buffer_context * buf_ctx = ...;
    auto it = buf_ctx->slices.find(const_cast<ggml_tensor*>(tensor));
    if (it != buf_ctx->slices.end()) {
        result.buffer = it->second.remote_ptr;
        int64_t nrows_split = it->second.row_high - it->second.row_low;
        result.ne[1] = nrows_split;           // <-- sets slice dimension
        result.nb[2] = result.ne[1] * result.nb[1];
        result.nb[3] = result.ne[2] * result.nb[2];
    }
    ...
}
```

**But immediately after all branches**, an unconditional loop overwrites ALL dimensions:

```cpp
// Lines 1628-1630 (CLOBBERS the split adjustment above)
for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
    result.ne[i] = tensor->ne[i];   // <-- full ne[1] overwrites nrows_split
    result.nb[i] = tensor->nb[i];  // <-- full nb[] overwrites adjusted nb[2]/nb[3]
}
```

**Result:** The server ALWAYS receives full `ne[1]` regardless of the split-buffer logic. The split adjustment is dead code.

**Consequence on the server:** `alloc_buffer_split` (line 3340) sizes the buffer with `ggml_nbytes(tensor)` which uses the full `ne[1]`, so it allocates a **full-size buffer** (not slice-sized). It then zeroes it (line 3350). The stored `split_buffer_meta` has full `ne[]`. The server computes with full-dimension tensors on a full-size buffer - no OOB crash, but the zeroed rows outside `[row_low, row_high)` corrupt every downstream operation (norms, attention, MoE routing).

This matches `.scratch/benchmarks/T5-row-split-fix.md` "Approach 2: Full-size buffer" result: "No crashes... 35B produces garbage. Root cause: Zeroed rows propagate through the computation graph."

### 2.3 Bug #2: Activation tensors keep full ne[1] (STRUCTURAL BUG)

Even if Bug #1 were fixed (so weight tensors get `ne[1] = nrows_split`), the graph would still be broken:

- **Weight tensors** (e.g., `blk.0.attn_q.weight`) are assigned to the split buffer type by the scheduler (`llama-model.cpp:947` via `make_gpu_buft_list` with `LLAMA_SPLIT_MODE_ROW`).
- **Activation tensors** (inputs/outputs of ops like `mul_mat`, `rms_norm`, `residual`) are NOT pre-allocated. The scheduler assigns them to the **default buffer type** of whichever backend runs the op (`ggml-backend.cpp:1407-1412`, `ggml_backend_sched_backend_from_buffer`).

So a `mul_mat` would receive:
- `src0` (weight): `ne[1] = nrows_split` (e.g., 300)
- `src1` (activation): `ne[1] = full_rows` (e.g., 1000)

This dimension mismatch produces wrong results or crashes. This is exactly what `.scratch/benchmarks/T5-row-split-fix.md` "Approach 3" found: "Only weight tensors on the split buffer get adjusted ne[1], but activation tensors on the regular buffer keep full ne[1]."

The commit message of `bf38bb1c6` itself acknowledges this:
> "Known issue: model loads without crash but output is garbled. Root cause: graph dimension inconsistency - weight tensors on split buffer get adjusted ne[1] but activation tensors keep full dimensions."

### 2.4 How the native CUDA split buffer avoids both bugs

In `ggml/src/ggml-cuda/ggml-cuda.cu`, the native split buffer resolves slices **inside** the backend, not at serialization:

- `ggml_backend_cuda_split_buffer_init_tensor` (line 1027): allocates per-device pointers in `tensor->extra` for exactly `nrows_split` rows.
- `mul_mat` (line 2111): `split = ggml_backend_buft_is_cuda_split(src0->buffer->buft)` -> **true**.
- It computes `dev[id].row_low/row_high` from `get_row_split` (line 2156) and launches the kernel with those bounds.
- `dst` for off-device results is allocated per-slice at `(row_high - row_low)*ne1` (line 2233) and written at `dev[id].row_low` offset.

**Key insight:** the slice bounds are carried by the buffer **type** and `tensor->extra`, NOT by `ne[1]`. The tensor keeps full `ne[1]` everywhere; the split is a buffer-level concept resolved by the CUDA backend internally. The RPC split buffer discards this by using a plain CUDA buffer + attempting (and failing) to mangle `ne[1]`.

---

## 3. Scheduler Behavior

### 3.1 Tensor-to-buffer assignment

`src/llama-model.cpp:929-950` (`make_gpu_buft_list`):
- For `LLAMA_SPLIT_MODE_ROW`, queries the backend reg for `"ggml_backend_split_buffer_type"`.
- If found, creates a split buffer type with the `tensor_split` ratios.
- Only **weight tensors** (pre-allocated during model load) get assigned to this buffer type.

`ggml/src/ggml-backend.cpp:1407-1412` (`ggml_backend_sched_backend_from_buffer`):
- For each tensor, finds the backend that `supports_buft` for the tensor's buffer type.
- Activation tensors have `buffer == NULL` (not pre-allocated), so they fall through to the op-supports path and get assigned to the **default buffer type** of the running backend.

### 3.2 The scheduler does NOT decompose ops

`ggml/src/ggml-backend.cpp:1572+` (`ggml_backend_sched_split_graph`): splits graphs by **backend assignment**, not by **op decomposition**. There is no code that splits a single `MUL_MAT` into partial `MUL_MAT`s for row-split. This decomposition is expected to be handled **inside the backend** (as CUDA does).

### 3.3 supports_buft (current state)

`ggml/src/ggml-rpc/ggml-rpc.cpp:5361-5378`: correctly accepts both regular RPC buffer types and RPC split buffer types by comparing endpoint + device. This part works (the scheduler can assign tensors). The problem is downstream compute, not assignment.

---

## 4. Minimal Fix

### Part A: Fix the serialize_tensor clobber (REQUIRED, ~5 lines, but INSUFFICIENT alone)

Move the split-buffer dimension adjustment to **after** the unconditional overwrite loop, or guard the loop to skip `ne[1]`/`nb[2]`/`nb[3]` for split tensors:

```cpp
// After the for loop at line 1628-1630, re-apply split adjustment:
if (tensor->buffer && tensor->buffer->iface.free_buffer == ggml_backend_rpc_split_buffer_free_buffer) {
    auto * buf_ctx = (ggml_backend_rpc_split_buffer_context *)tensor->buffer->context;
    auto it = buf_ctx->slices.find(const_cast<ggml_tensor*>(tensor));
    if (it != buf_ctx->slices.end()) {
        int64_t nrows_split = it->second.row_high - it->second.row_low;
        result.ne[1] = nrows_split;
        result.nb[2] = result.ne[1] * result.nb[1];
        result.nb[3] = result.ne[2] * result.nb[2];
    }
}
```

This makes the server allocate **slice-sized** buffers and compute with slice dimensions for weights. But it does NOT fix the activation mismatch (Bug #2), so output will still be wrong.

### Part B: Use the native CUDA split buffer on the server (REQUIRED for correctness, ~80-100 lines)

The server must replicate the native CUDA split buffer's internal slice resolution so that `mul_mat` takes the `split == true` path. Two options:

**Option B1 (recommended): Allocate a CUDA split buffer type on the server.**
In `alloc_buffer_split` (line 3309), instead of `ggml_backend_get_default_buffer_type`, use `ggml_backend_cuda_split_buffer_type` configured so the server's single GPU owns `[row_low, row_high)`. Then:
- `ggml_backend_buft_is_cuda_split(...)` -> true -> `mul_mat` uses the correct `row_low`/`row_high` path.
- `dst` is allocated per-slice and offset correctly.
- No manual `ne[1]` rewriting needed for weights OR activations.

Required adaptations (per `T5-row-split-gc-diag.md` Section 5):
- `alloc_buffer_split`: create CUDA split buffer type (not default buft) with derived `tensor_split`/`split_id`.
- `set_tensor` (line 3510): special-case split buffers - write into `tensor->extra->data_device[id]` instead of `tensor->data` (CUDA split `get_base` returns dummy `0x1000`).
- `deserialize_tensor` (line 3446): detect split buffers, don't clobber `result->data` with dummy base.

**Option B2 (alternative): Server-side graph rewriting.**
Rewrite `ne[1]` for ALL tensors (weights AND activations) in the split subgraph at `graph_compute` time (line 3981, before `ggml_backend_graph_compute`). This requires propagating the row dimension through the entire block graph (residuals, norms, attention outputs all share `ne[1] = ne01`). More fragile; the T5 doc recommends Option B1.

### Feasibility

| Part | Lines | Risk | Sufficient? |
|------|-------|------|-------------|
| A: serialize_tensor clobber fix | ~5 | Low | No (only weights) |
| B1: Server CUDA split buffer reuse | ~80-100 | Medium | Yes (full correctness) |
| B2: Server graph rewriting | ~100-150 | High | Yes but fragile |
| **Total (A + B1)** | **~85-105** | **Medium** | **Yes** |

**Verdict:** Feasible within ~100 lines if Option B1 is chosen. The fix centralizes slice logic in the already-correct `mul_mat` path rather than hand-mangling dimensions per node.

---

## 5. Key Code References

| Location | Role | Issue |
|----------|------|-------|
| `ggml-rpc.cpp:1608-1624` | `serialize_tensor` split branch | Sets `ne[1]=nrows_split` |
| `ggml-rpc.cpp:1628-1630` | `serialize_tensor` overwrite loop | **CLOBBERS** the split adjustment (Bug #1) |
| `ggml-rpc.cpp:3309-3370` | `alloc_buffer_split` (server) | Uses `ggml_nbytes(tensor)` with full `ne[1]` -> full-size buffer |
| `ggml-rpc.cpp:3340` | buffer sizing | `ggml_nbytes(tensor)` = full (due to Bug #1) |
| `ggml-rpc.cpp:3350` | `buffer_clear` | Zeroes full buffer -> zeroed rows corrupt compute |
| `ggml-rpc.cpp:3446-3510` | `deserialize_tensor` (server) | Rebuilds with full `ne[1]`, no slice resolution |
| `ggml-rpc.cpp:3981` | `graph_compute` -> `ggml_backend_graph_compute` | Plain CUDA path, `split == false` |
| `ggml-cuda.cu:2111` | `mul_mat` split detection | `ggml_backend_buft_is_cuda_split` -> false on RPC server |
| `ggml-cuda.cu:1027,2156,2233` | Native split path | Per-device `row_low/row_high`, slice dst (the reference design) |
| `ggml-backend.cpp:1407-1412` | `sched_backend_from_buffer` | Activations get default buft, not split buft (Bug #2) |
| `llama-model.cpp:929-950` | `make_gpu_buft_list` | Only weights get split buffer type |

---

## 6. Conclusion

The Phase 2 "regression" is a misnomer. The RPC split buffer path (`8cb644877` + `bf38bb1c6`) is brand new and has never worked. Phase 2's correct 108.81 t/s came from the **native CUDA split buffer**, which resolves row slices inside `mul_mat` via `tensor->extra` and `row_low`/`row_high` - a mechanism the RPC path does not replicate.

The current garbled output has two compounding causes:
1. **Bug #1 (code bug):** `serialize_tensor` split adjustment is dead code - clobbered by the unconditional overwrite loop. Server allocates full-size buffers with zeroed rows.
2. **Bug #2 (design gap):** Even with Bug #1 fixed, activation tensors keep full `ne[1]` while weight tensors get slice `ne[1]`. The scheduler does not create views or decompose ops for row-split; this is expected to happen inside the backend.

The minimal correct fix (~100 lines) is to make the RPC server allocate a **native CUDA split buffer type** (Option B1) so `mul_mat`'s existing `row_low`/`row_high` path handles both weights and activations consistently, plus the 5-line `serialize_tensor` clobber fix (Part A) so the wire protocol carries correct dimensions during model load.
