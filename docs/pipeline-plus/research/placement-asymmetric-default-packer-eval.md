# Placement: Asymmetric Default Packer + GGUF Offline Structure

**Date:** 2026-07-19
**Purpose:** Evaluate two report halves against primary sources in this repo (code + locked design docs), not the report's claims. Part A = asymmetric capacity-weight default algorithm (Phase 1, no heat). Part B = GGUF metadata dump tools for offline layer/tensor discovery.

## Summary table

| Claim / question | Verdict | Primary evidence |
|---|---|---|
| Four packers exist: capacity / min-hop / attn_local / heat | **Supported** | `common/placement-plan.h:150-220` declares all four; `arg.cpp:2336-2380` wires CLI |
| pack_capacity = proportional, min-1, contiguous, large-first | **Supported** | `placement-plan.cpp:588-740` |
| pack_capacity_attn_local = attn-on-local, SSM-on-remote, interleaved ranges | **Supported** | `placement-plan.cpp:752-1013` |
| pack_capacity_min_hop = uncommitted / WIP | **Contradicted** | Shipped in working tree: `placement-plan.cpp:1020-1298`, `arg.cpp:2351`, `common.cpp:1306` |
| pack_heat "not yet active" / issue 12/13 open | **Contradicted** | Issues 12 AND 13 both resolved; `placement-plan.cpp:1300-1663`, `arg.cpp:2364-2380` |
| Usable weight subtracts static pads + KV reserves | **Supported** | `placement-capacity.cpp:78-97`; design §1 |
| "1.5 GB pad for 8GB card" matches table-v1 | **Contradicted** | table-v1 = 256 + KVpad(64-512) + 256 = ~0.55-1.0 GiB, not 1.5 |
| overrides[] can pin embeddings / LM-head / non-layer tensors | **Supported** | `placement-plan.h:27-33`; design §5; `llama-model.cpp:1320-1325` |
| layer_start/end includes embeddings | **Contradicted** | Covers `[0, n_layer)` = transformer blocks only; embeddings via overrides/cpu default |
| Mid layers 0.3N-0.8N = "semantic engine room", early 0.05-0.25 resilient | **Untested / folklore** | Zero primary evidence; no heatmap/research doc supports depth zones |
| Asymmetric mid-core-on-fast composable with min-hop + attn_local | **Partial** | Mutually exclusive assignment policies; can't all be default |
| Report targets `common/placement-pack.cpp` | **Contradicted** | File does NOT exist; packers live in `placement-plan.cpp` |
| gguf-py has gguf_dump.py for offline structure | **Supported** | `gguf-py/gguf/scripts/gguf_dump.py`, entry `gguf-dump` |
| llama_model_layer_is_recurrent answers hybrid without custom GGUF | **Supported** | `llama.h:586`, `llama-model.cpp:2357`; generate path already calls it |
| generate path already has n_layer + is_layer_recurrent pre-load | **Supported** | `common.cpp:1228-1247` (no_alloc metadata load) |
| Offline GGUF inspect needed for generate-only before load | **Contradicted** | generate-only still loads model metadata via llama APIs |

## Part A — Asymmetric algorithm vs shipped packers

### A1. Shipped packers (functions, CLI, issues)

| Packer | Function | CLI flag | Issue | Status |
|---|---|---|---|---|
| Capacity | `placement_plan_pack_capacity` | `--placement-generate` | 11 | resolved/shipped |
| Min-hop contiguous | `placement_plan_pack_capacity_min_hop` | `--placement-generate-min-hop` | (after 11) | shipped in working tree |
| Attention-local | `placement_plan_pack_capacity_attn_local` | (default for --placement-generate when recurrent) | (after 11) | shipped; default |
| Heat-aware | `placement_plan_pack_heat` | `--placement-generate-heat --placement-heatmap PATH` | 13 | resolved/shipped |

All four declared at `common/placement-plan.h:150-220`. Note: report claims min-hop is "uncommitted" and heat "not yet active" — both contradicted by working-tree code + resolved issues 12/13.

### A2. pack_capacity assignment logic (`placement-plan.cpp:588-740`)

- Candidates sorted by `usable_weight_mib` desc, then `backend_id` asc.
- Proportional counts = `usable * n_layer / sum_u`, **min 1** per candidate when `n_layer >= n_backends`.
- Over-allocation trimmed from largest first; remainder to index 0.
- **Contiguous ranges** emitted in sort order: big GPU gets layers `[0, c0)`, next gets `[c0, c0+c1)`, etc. Small cards get late contiguous blocks.
- Backends with `usable_weight_mib == 0 && free_mib == 0` skipped.

### A3. pack_capacity_attn_local (`placement-plan.cpp:752-1013`)

- Fallback to pack_capacity when `is_layer_recurrent` empty or all-false, or no local GPU.
- Sort: local GPU (tier 2) first, then RPC (tier 1), CPU (tier 0); within tier by usable desc.
- Per-layer loop: attention (non-recurrent) -> local if space; SSM (recurrent) -> remote preferred, spill to local if needed.
- Emits **multiple contiguous sub-ranges per backend** (interleaved) when type affinity splits a backend's assignment.
- This is the **current default** for `--placement-generate` (see `common.cpp:1339-1342`).

### A4. pack_capacity_min_hop (`placement-plan.cpp:1020-1298`)

- Same candidate/tier sort as attn_local.
- **Non-hybrid:** simple contiguous split, local first.
- **Hybrid:** computes smoothed attention density (window=3); if a remote block has density >= 0.25 higher than local's block, swaps local block with that remote block, then re-sorts by layer_start to restore contiguity.
- Exactly `(n_backends - 1)` hops by construction.
- Trim over-allocation from **slowest** first (reverse order) — differs from capacity packer.

### A5. pack_heat (`placement-plan.cpp:1300-1663`)

- Reads heatmap via `placement_plan_parse_heatmap_file` (tasks.tg.layer_rollup or tasks.tg.layers).
- Sorts backends speed-tier desc then usable desc; sorts layers hot-to-cold.
- Hottest `counts[0]` layers -> fastest backend, next `counts[1]` -> next, etc.
- Falls back to pack_capacity when rollup empty.
- Sets `heat.status = "full"` only when rollup covers all n_layer with unique indices; stub detection included.
- **Issue 12 (rollup) and 13 (heat-aware generate) both resolved** — see `.scratch/placement-control-plane/issues/12-heatmap-per-layer-rollup.md` and `13-heat-aware-plan-generate.md`.

### A6. Usable weight + table-v1 pad math

`placement-capacity.cpp:78-97`:
```
usable = max(0, free - sum(static_pads) - client_reserves)
client_reserves (table-v1) = graph_pad(256) + kv_pad + small_floor(256 if total<10240)
kv_pad = ceil(n_ctx * n_parallel * 256 / 1MiB), min 64 (48 with FA)
```
For an 8 GiB card (total < 10240 MiB) at default ctx=4096, n_par=1, F16: ~256 + 64 + 256 = **~576 MiB**, up to ~1 GiB at large ctx. **Report's "1.5 GB pad" is contradicted** by table-v1 arithmetic; actual pad is roughly half that. Design doc confirms at `placement-capacity-discovery-design.md:270`.

### A7. Residual / embedding / LM-head pinning

- `overrides[]` (`placement-plan.h:27-33`) = `{match: regex, backend_id}`. Applied via `tensor_buft_overrides` at `placement-plan.cpp:528-577`.
- Design doc §5 (`placement-plan-ir-apply-design.md:110-128`) shows example: `{ "match": "token_embd", "backend_id": "cpu" }` and MoE expert tensors to CPU.
- **layer_start/end covers `[0, n_layer)` only** = transformer block layers (`blk.N.*`). Embeddings (`token_embd`) and LM-head (`output.weight`) are NOT in the layer range.
- At apply, `llama-model.cpp:1320-1325`: output layer follows plan's last-layer device; input/embedding defaults to CPU (`dev_input = cpu`). So non-layer tensors are handled by engine defaults + overrides[], not by layer ranges.

### A8. Mid-layer "semantic engine room" / depth-dependent heat zones

**No primary evidence.** Searched `docs/`, `benches/`, profiler outputs, `node_timings` rollup docs, research docs. Zero mentions of "0.3N-0.8N semantic engine room", "early 0.05-0.25 resilient", or any depth-dependent heat zones. The `heatmap-rollup.cpp` produces per-layer ms but no research doc interprets them into depth zones. **Verdict: untested hypothesis / external folklore.** Any asymmetric depth-based placement would require empirical heatmap validation (issue 12 rollup gives the signal; no analysis consumes it yet).

### A9. Conflict analysis — asymmetric mid-core vs min-hop vs attn_local

| Dimension | Asymmetric mid-core (report) | min-hop | attn_local |
|---|---|---|---|
| Block shape | contiguous mid-range on fast | contiguous, local-first | interleaved sub-ranges |
| Hybrid awareness | none (capacity-only) | density swap >= 0.25 | type affinity (attn/SSM) |
| Hop count | n_backends-1 | n_backends-1 | variable (type-driven) |
| Default candidate | ? | `--placement-generate-min-hop` | current default |

- They are **mutually exclusive assignment policies** applied to the same plan. Only one packer can own a given generation call.
- Composability: attn_local already subsumes "attention on fast" for hybrid models. min-hop subsumes "contiguous sequential". The report's "mid layers on fast" is a density heuristic that min-hop's swap threshold approximates for hybrid, but has no equivalent for dense models.
- For dense (non-hybrid) models, the report's depth zones have no hook — attn_local and min-hop both fall back to plain pack_capacity.

### A10. Report targets `common/placement-pack.cpp`

**File does NOT exist.** grep across the whole tree finds only `tests/test-placement-pack.cpp` (the unit test). All packers live in `common/placement-plan.cpp`. Report's code references are stale/wrong.

### A11. Recommendation tiers

| Option | Verdict | Reasoning |
|---|---|---|
| Implement as default | **Reject** | No empirical support for depth zones; attn_local already default for hybrid; min-hop available for contiguous. Adding a 4th default without heatmap evidence risks regressing the shipped working-tree default. |
| Implement as optional 4th packer | **Research-only hypothesis** | Could be a `--placement-generate-asymmetric` flag IF validated against real heatmaps first. Low priority — attn_local covers the stated goal (attn-on-fast) more rigorously. |
| Research-only hypothesis | **Preferred** | Run issue-12 rollup heatmaps on 2-3 models, then check whether depth-correlated heat clusters actually exist. Validate before coding. |
| Reject | **Acceptable** | For dense models, depth-based placement has no architectural justification in this repo. |

**Empirical validation required before default:** (1) per-layer heatmap rollup across models (issue 12 signal); (2) cluster analysis for depth-correlated zones; (3) TG/PP benchmark with mid-core-on-fast vs attn_local; (4) confirm 8 GB card behavior with real pad numbers (table-v1 ~0.55 GiB, not 1.5).

## Part B — GGUF offline structure

### B1. gguf-py path + gguf_dump.py usage

- Path: `gguf-py/gguf/scripts/gguf_dump.py` (entry: `gguf-dump` per `gguf-py/pyproject.toml:23`).
- Usage: `python gguf_dump.py model.gguf [--json] [--markdown] [--no-tensors] [--json-array]`.
- Dumps metadata key/value pairs + tensor inventory (name, shape, type, offset). Supports JSON/markdown/plain. Confirmed primary tool for offline GGUF inspection.

### B2. Architecture keys for hybrid/recurrent vs dense

- `gguf-py/gguf/constants.py:221` `class SSM:` (conv_kernel, inner_size, state_size, time_step_rank, group_count, dt_b_c_rms) and `class WKV:` for RWKV.
- `llama.h:586` `llama_model_layer_is_recurrent(model, layer_idx)` — per-layer recurrent flag.
- `llama.h:647-650` `llama_model_is_recurrent`, `llama_model_is_hybrid`.
- `llama-arch.cpp:865-877` `llm_arch_is_recurrent` returns true for MAMBA/MAMBA2/RWKV6/RWKV6QWEN2/RWKV7/ARWKV7.
- **Yes, `llama_model_layer_is_recurrent` already answers hybrid composition without a custom GGUF script** — after load.

### B3. generate path already has n_layer + is_layer_recurrent

`common/common.cpp:1228-1247` (`common_placement_generate`):
- Loads model metadata with `no_alloc = true`, `use_mmap = false`, `n_gpu_layers = 0` (no weight load, no full VRAM commit).
- `n_layer = llama_model_n_layer_all(meta)` (falls back to `llama_model_n_layer`).
- `is_layer_recurrent[i] = llama_model_layer_is_recurrent(meta, i)` for all layers.
- Frees metadata handle. No separate GGUF dump needed.

### B4. Value of offline GGUF inspect for packer

- **Generate-only before load:** NOT needed — generate-only still does the no_alloc metadata load (it needs n_layer + recurrent flags). A pure-GGUF pre-parse would only avoid the llama metadata load, which is already weightless.
- **overrides patterns:** Moderately useful — `gguf_dump --json` reveals tensor names (e.g. `blk.N.*.ffn_*_exps` for MoE, `token_embd`, `output.weight`) so operators can write `overrides[]` regexes without loading. This is the one genuine ops use case.
- **MoE expert tensors:** `gguf_dump` shows expert tensor names; `ffn_gate_inp` / `ffn_*_exps` naming is in `gguf_dump.py:200-203`. Useful for crafting MoE-to-CPU overrides.

### B5. Recommendation

| Option | Verdict | Reasoning |
|---|---|---|
| Integrate GGUF pre-parse into packer | **Reject** | generate path already gets n_layer + is_layer_recurrent via llama metadata load (no_alloc). Adding a parallel GGUF parser duplicates `llama_model_layer_is_recurrent` and risks schema drift. |
| Keep as ops script only | **Preferred** | `gguf-dump` is the right tool for operators writing overrides[] patterns (MoE expert names, embedding/output tensor names). Document it as such. |
| Already covered by llama APIs after load | **Supported** | For the packer itself, yes — `common.cpp:1228-1247` covers it. Offline GGUF only adds value for human-authored override patterns. |

## Integration options (A1/A2/A3) ranked

1. **A3 (research-only hypothesis)** — Run issue-12 rollup heatmaps on multiple models first. Validate depth-correlated heat zones before writing any packer. Cheapest, de-risks the other two.
2. **A2 (optional 4th packer)** — Only after A3 validates. Implement as `--placement-generate-asymmetric` with a clear "experimental, no heatmap required" caveat. Reuse min-hop's density-swap primitive.
3. **A1 (implement as default)** — Do not do this. attn_local is already the default for hybrid models and has architectural justification (type affinity). Depth zones for dense models are unsupported.

## Open questions for grill session

1. Does the report's "1.5 GB pad for 8 GiB" come from a different reserve_mode (auto/fit projection) or is it simply wrong? table-v1 math says ~0.55 GiB.
2. Is there ANY profiler run (even external) showing depth-correlated heat clusters, or is the 0.3N-0.8N claim purely literature-based?
3. For dense (non-hybrid) models, what signal would drive mid-core-on-fast placement without heat? Layer index alone?
4. Should `gguf-dump` be documented in `docs/ops.md` as the overrides[] authoring aid, or is that out of scope?
5. Does min-hop's 0.25 density-gap threshold need tuning, and is it the right knob to subsume the report's "mid-core on fast" goal?
6. Why does the report reference `common/placement-pack.cpp` which does not exist — is there a stale branch?
