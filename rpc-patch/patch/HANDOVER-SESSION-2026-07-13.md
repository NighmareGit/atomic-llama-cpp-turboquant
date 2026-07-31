# Handover: Large-Model 3-GPU Multi-Node Testing (2026-07-13)

**Purpose:** Validate prev_copy KV-cache fix across 1/2/3-GPU configs, then scale to 27B-35B models with MTP speculative decoding on 3-GPU RPC cluster.
**Branch:** `Path-D-Gpipeline-Assembly-Line` @ `87357519e` (post MTP fix: next commit)
**Prior handover:** [HANDOVER-SESSION-2026-07-03.md](HANDOVER-SESSION-2026-07-03.md)

| Location | Path | Notes |
|----------|------|-------|
| **romulus (primary client)** | `user@192.168.8.108` -> `~/projects/path-d-gpipeline-assembly-line` | 7900XTX ROCm + docker 3060Ti CUDA |
| **remus** | `user@192.168.8.22` | RPC1 5060Ti CUDA `:50052` |
| **rpc-server on remus** | `~/rpc-server-bin/` | Built in `nvidia/cuda:12.8.0-devel` for Blackwell sm_120 |

**Remotes:**
- Gitea: `gitea/Path-D-Gpipeline-Assembly-Line`
- GitHub: `origin/Path-D-Gpipeline-Assembly-Line` (`NighmareGit/atomic-llama-cpp-turboquant`)

---

## Session summary

### 1. prev_copy KV-cache fix (8B model) — ALL PASS

**Symptom before fix:** KV-cache corruption on multi-GPU with copy-slot rotation; stateful tensors copied from frozen original instead of previous graph execution slot.

**Fix:** Added `prev_copy` field to scheduler (`ggml/src/ggml-backend.cpp`):
- Tracks which copy slot was active in previous graph execution
- Split input copy uses `prev_copy` as source when `n_copies > 1`
- Set in `reset`, `alloc_graph`, `pipeline_barrier`
- Removed `GET_TENSOR_DEFER` guard: current-split downloads always flush

**Test matrix (8B): 54/54 PASS** across 6 configs (1/2/3 GPU x PPLUS/NOPLUS)

| Config | TG (t/s) |
|--------|----------|
| 1-GPU ROCm | 102 |
| 2-GPU (ROCm + 3060Ti RPC) | 137 |
| 3-GPU (ROCm + 3060Ti + remus 5060Ti) | 103 |

3-GPU degrades vs 2-GPU because 8B model (5.5 GB) is too small to amortize network RPC latency to remus. PPLUS vs NOPLUS identical at this scale.

### 2. Large-model 3-GPU testing

**VRAM budget** (from `pathb-72b-vram-calc.py --config config-c`):

| Device | GPU | Usable |
|--------|-----|--------|
| RPC0 (docker) | 3060 Ti | 7.0 GB |
| RPC1 (remus) | 5060 Ti | 15.5 GB |
| ROCm0 (local) | 7900 XTX | 22.0 GB |
| **Total** | | **44.5 GB** |

Recommended TS: `35,15,50`

**Results:**

| Model | Size | TG (t/s) | PP (t/s) | Correct? |
|-------|------|----------|----------|----------|
| Qwen3.5-27B-Q5_K_M | 19 GB | 24.4 | - | YES |
| Qwen3.5-35B-A3B.i1-Q4_K_M | 20 GB | 133-137 | - | YES |
| Qwen3.6-35B-A3B-APEX-MTP-I-Quality | 22 GB | 55 avg | 145 | NO (garbled) |
| meta-llama-3-70b-instruct.Q4_K_M | 40 GB | - | - | OOM |

**27B dense:** Works perfectly. 24.4 t/s TG. 3-GPU overhead amortized by 19 GB model size.

**35B MoE (no MTP):** Fastest: 133-137 t/s TG. MoE architecture (256 experts, 8 active) means only ~3B active params per token. 5.6x faster than 27B dense.

**35B MTP:** MTP speculative decoding activates (n_max=3 draft tokens) but:
- Output is garbled (repetitive nonsense)
- TG is slower than non-MTP (55 t/s vs 137 t/s)
- Draft acceptance rate: ~6.6% (5/76) — nearly all drafts rejected
- Likely APEX quantization damaged MTP heads, or cross-device RPC MTP path has a bug

**70B:** OOM with pure GPU offload. Auto-split tries to allocate 40 GB on ROCm0 first (only 22 GB available).

### 3. MTP garbled output root cause + fix

**Root cause:** MTP draft context (`ctx_dft`, `LLAMA_CONTEXT_TYPE_MTP`) gets its own independent scheduler with separate `prev_copy`/`cur_copy`/`next_copy` cycle. When `GGML_PIPELINE_PLUS=1` enables copy-slot rotation (`n_copies > 1`), the draft scheduler's rotation is uncoordinated with the target scheduler. Both operate on the same underlying KV-cache buffer (shared via `v_cells_impl`), producing stale reads and garbled draft tokens.

This explains why "classic vanilla multi-GPU works": without PPLUS, `n_copies=1`, so `prev_copy` never activates — the condition `prev_copy != cur_copy` short-circuits.

**Fix** (`src/llama-context.cpp`, line ~694): Disable `pipeline_parallel` for MTP draft contexts unconditionally:
```cpp
const bool pipeline_parallel = cparams.pipeline_parallel &&
    (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP);
```

The MTP draft graphs are tiny (MTP head only, a few layers), so the pipeline parallelism benefit from copy-slot rotation is negligible. The target context retains full PPLUS with copy-slot rotation.

**Effect:** Draft acceptance rate improved from 6.6% to 25% (tested with 9B MTP). Full 3-GPU validation with 35B APEX MTP pending (remus was shut down after session).

**Alternative fix directions (if pivot needed):**

| Approach | Pros | Cons |
|----------|------|------|
| 1. **Disable copy-slot rotation for ctx_dft** (implemented) | Simple, safe, minimal change | If MTP heads ever span multiple backends, loses theoretical PPLUS benefit |
| 2. **Share copy-slot state** between target and draft schedulers | Clean, no lost optimization | Invasive; two `llama_context` objects sharing scheduler state breaks encapsulation |
| 3. **Skip prev_copy for shared-memory contexts** (check `ctx_other`) | Conceptually correct — "shared buffer, single owner" | Requires `ctx_other` propagation through `llama_cparams` (currently only set for Gemma4Assistant/EAGLE3) |
| 4. **Force `n_copies=1` at ggml-backend level** when context type is MTP | Same effect as option 1, different insertion point | Couples ggml scheduler to llama-level context type |

**Note:** The `Qwen3.5-9B-MTP-Q4_K_M.gguf` model (5.5 GB) produces garbled output when loaded **without** `--spec-type draft-mtp` on the Path-D branch — this is a Path-D regression, not damaged weights (the model works correctly on vanilla atomic with MTP enabled). However, this edge case is not a production concern: MTP-combined models are always used with `--spec-type draft-mtp`. The `Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf` model (22 GB) produces correct output without MTP and is the canonical test target.

### 4. VRAM fit scan (what else fits)

| Model | Size | Fit? | Notes |
|-------|------|------|-------|
| Gemma-4-31B Q4_K_M | 18 GB | Easy | Pure GPU offload |
| Qwen3-72B IQ4_XS | 37 GB | Tight | ngl=56, ~13 GB CPU offload |
| Kimi-Dev-72B IQ4_XS | 37 GB | Tight | ngl=56, ~13 GB CPU offload |
| Qwen3-Next-80B-A3B Q5_K_M | 53 GB | Very tight | ngl=33, ~19 GB CPU offload |

### 5. Profiler limitation

`llama-gpipe-profiler` sweeps 1-GPU -> 2-GPU -> 3-GPU configs. For models >22 GB, the 1-GPU step OOMs before reaching the working 3-GPU config. Use `llama-server` benchmarks directly for large models.

### 6. New/modified files

| File | Change |
|------|--------|
| `ggml/src/ggml-backend.cpp` | prev_copy fix + GET_TENSOR_DEFER guard removal |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | RPC pipeline plus adjustments |
| `src/llama-context.cpp` | **MTP fix**: disable pipeline_parallel for MTP draft contexts |
| `rpc-patch/patch/test-kv-fix.sh` | Fixed garble false positive (punctuation filter) |
| `rpc-patch/patch/run-profiler.sh` | Multi-GPU profiler runner (1/2/3 GPU x PPLUS/NOPLUS) |
| `rpc-patch/patch/test-large-models.sh` | Large model test harness (27B, 35B MoE, 35B MTP, 70B) |
| `rpc-patch/patch/loop-check-garble.sh` | Garble detection loop for correctness testing |

### 7. Remus RPC server

- Binary: `~/rpc-server-bin/rpc-server` on remus (192.168.8.22)
- Built in docker `nvidia/cuda:12.8.0-devel-ubuntu22.04` targeting `86-real;120-real`
- Needs NCCL: `~/rpc-server-bin/libnccl.so.2` (copied from docker)
- Start: `cd ~/rpc-server-bin && LD_LIBRARY_PATH=. ./rpc-server -H 0.0.0.0 -p 50052 -d CUDA0 --telemetry`

---

## Next steps

1. **Validate MTP fix on 3-GPU**: Restart remus, test `Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf` with `--spec-type draft-mtp` + `GGML_PIPELINE_PLUS=1`. Verify output is coherent and TG speed >= non-MTP baseline (133-137 t/s).
2. Test Gemma-4-31B as a mid-size dense benchmark
3. Try 72B with VRAM calculator's CPU-offload recommendations (ngl=56, TS=35,15,50)
4. Investigate `Qwen3.5-9B-MTP-Q4_K_M.gguf` — produces garbage even without `--spec-type draft-mtp`. Likely damaged weights; re-download or re-quantize.
5. Fix `llama-gpipe-profiler` to skip non-viable GPU configs for large models
6. If MTP fix approach 1 proves insufficient, consider approach 3 (skip prev_copy when `ctx_other != nullptr`) — requires adding `ctx_other` propagation for MTP to `llama_cparams`

---

*Handover prepared 2026-07-13. Assisted-by: Grok.*
