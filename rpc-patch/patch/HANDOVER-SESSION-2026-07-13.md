# Handover: Large-Model 3-GPU Multi-Node Testing (2026-07-13)

**Purpose:** Validate prev_copy KV-cache fix across 1/2/3-GPU configs, then scale to 27B-35B models with MTP speculative decoding on 3-GPU RPC cluster.
**Branch:** `Path-D-Gpipeline-Assembly-Line` @ `c92be33fe`
**Prior handover:** [HANDOVER-SESSION-2026-07-03.md](HANDOVER-SESSION-2026-07-03.md)

| Location | Path | Notes |
|----------|------|-------|
| **romulus (primary client)** | `hunter@192.168.8.108` -> `~/projects/path-d-gpipeline-assembly-line` | 7900XTX ROCm + docker 3060Ti CUDA |
| **remus** | `hunter@192.168.8.22` | RPC1 5060Ti CUDA `:50052` |
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

### 3. VRAM fit scan (what else fits)

| Model | Size | Fit? | Notes |
|-------|------|------|-------|
| Gemma-4-31B Q4_K_M | 18 GB | Easy | Pure GPU offload |
| Qwen3-72B IQ4_XS | 37 GB | Tight | ngl=56, ~13 GB CPU offload |
| Kimi-Dev-72B IQ4_XS | 37 GB | Tight | ngl=56, ~13 GB CPU offload |
| Qwen3-Next-80B-A3B Q5_K_M | 53 GB | Very tight | ngl=33, ~19 GB CPU offload |

### 4. Profiler limitation

`llama-gpipe-profiler` sweeps 1-GPU -> 2-GPU -> 3-GPU configs. For models >22 GB, the 1-GPU step OOMs before reaching the working 3-GPU config. Use `llama-server` benchmarks directly for large models.

### 5. New/modified files

| File | Change |
|------|--------|
| `ggml/src/ggml-backend.cpp` | prev_copy fix + GET_TENSOR_DEFER guard removal |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | RPC pipeline plus adjustments |
| `rpc-patch/patch/test-kv-fix.sh` | Fixed garble false positive (punctuation filter) |
| `rpc-patch/patch/run-profiler.sh` | Multi-GPU profiler runner (1/2/3 GPU x PPLUS/NOPLUS) |
| `rpc-patch/patch/test-large-models.sh` | Large model test harness (27B, 35B MoE, 35B MTP, 70B) |
| `rpc-patch/patch/loop-check-garble.sh` | Garble detection loop for correctness testing |

### 6. Remus RPC server

- Binary: `~/rpc-server-bin/rpc-server` on remus (192.168.8.22)
- Built in docker `nvidia/cuda:12.8.0-devel-ubuntu22.04` targeting `86-real;120-real`
- Needs NCCL: `~/rpc-server-bin/libnccl.so.2` (copied from docker)
- Start: `cd ~/rpc-server-bin && LD_LIBRARY_PATH=. ./rpc-server -H 0.0.0.0 -p 50052 -d CUDA0 --telemetry`

---

## Next steps

1. Test Gemma-4-31B as a mid-size dense benchmark
2. Try 72B with VRAM calculator's CPU-offload recommendations (ngl=56, TS=35,15,50)
3. Investigate MTP garbled output — test with non-APEX Qwen3.5-9B-MTP-Q4_K_M (5.5 GB) on 3-GPU to isolate whether the bug is APEX-specific or RPC-MTP-general
4. Fix `llama-gpipe-profiler` to skip non-viable GPU configs for large models

---

*Handover prepared 2026-07-13. Assisted-by: Grok.*
