# Pipeline Plus Multi-GPU KV-Cache Garble

**Location**: `diagnostics/pipeline-plus-dual-gpu-garble/README.md`
**Index**: [diagnostics/](../README.md)

Last updated: 2026-07-19

---

## Symptom

On multi-GPU native CUDA (no RPC), enabling `GGML_PIPELINE_PLUS=1` (pipeline
parallelism with copy-slot rotation, `n_copies=4`) causes **KV-cache garbling**
across both dense and MoE models. Output degrades to repetitive nonsense:

```
"Thinking Process Process Process Process::::


1111....     **222+++"
```

The same models produce **clean, coherent output** with `GGML_PIPELINE_PLUS=0`.

This affects **all multi-GPU configs**, not just one node:
- **romulus**: 7900 XTX (ROCm) + 3060 Ti (CUDA docker RPC)
- **triton**: 3090 + 3070 (native CUDA)
- **4-GPU cluster**: 7900 XTX + 3060 Ti + 5060 Ti (remus) + 5070 (jupiter)

This is a broader manifestation of the MTP draft-context garbling documented in
the 2026-07-13 handover — the issue is not limited to MTP draft contexts. Any
target context running pipeline parallelism on multi-GPU is affected.

---

## Root Cause Analysis

### Background: Copy-Slot Rotation

`GGML_PIPELINE_PLUS=1` enables pipeline parallelism in `ggml-backend.cpp`. When
active, the scheduler uses `n_copies=4` slot rotation so the next graph can be
prepared while the current one executes. A `prev_copy` field tracks which copy
slot was active in the previous graph execution, and split input copies use
`prev_copy` as their source.

The `prev_copy` fix (commit `87357519e`, 2026-07-13 handover) addressed the
MTP draft context case: the draft scheduler's copy-slot rotation was
uncoordinated with the target scheduler, causing stale reads from shared
KV-cache buffers. The fix disabled `pipeline_parallel` for MTP draft contexts:

```cpp
// src/llama-context.cpp
const bool pipeline_parallel = cparams.pipeline_parallel &&
    (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP);
```

### The Broader Bug

The 2026-07-13 fix only addressed **draft** contexts. The **target** context
still runs with full pipeline parallelism (`n_copies=4`, `prev_copy`
rotation). On single-GPU or single-copy configs this is harmless, but on
multi-GPU the copy-slot rotation interacts with the KV-cache in ways that
produce garbled output.

Hypothesis: the `prev_copy` source selection for split inputs reads from a slot
that has been partially overwritten by a concurrent graph execution on another
GPU, or the KV-cell mapping (`v_cells_impl`) returns stale indices under
rotation. The exact mechanism requires a focused bisect, but the symptom is
reproducible and the workaround (disable Plus) is clear.

### Why Single-GPU Works

On single-GPU, the compute buffer is shared but the scheduling is simpler —
there is no cross-device synchronization of copy slots. The bug is specific to
multi-GPU where `ggml-backend.cpp` creates per-device scheduling graphs with
independent copy-slot cycles.

---

## Fix-and-Verify Plan

The fix should be developed and validated locally first, then rolled out to
other nodes.

### Phase 1: Reproduce and Fix Locally (romulus)

**Why local first:** romulus is the primary dev node with ROCm build
infrastructure already in place. Fastest iteration cycle.

**Config:** 7900 XTX (ROCm, 24GB) + 3060 Ti (CUDA docker RPC, 8GB)

```bash
# Start RPC worker (3060 Ti docker)
cd ~/docker/Atomic-Llama-Romulus-PathB && docker compose up -d

# Reproduce garbling (GGML_PIPELINE_PLUS=1)
GGML_PIPELINE_PLUS=1 llama-server \
  -m /mnt/models/Qwen3.5-27B-Q5_K_M.gguf \
  --rpc 127.0.0.1:50051 -sm layer -ts 75,25 \
  -ngl 99 --ctx-size 4096 \
  --host 127.0.0.1 --port 8080

# Confirm clean (GGML_PIPELINE_PLUS=0)
GGML_PIPELINE_PLUS=0 llama-server ...
```

**Tasks:**
1. Confirm garbling reproduces on romulus with `GGML_PIPELINE_PLUS=1`
2. Bisect to identify the exact commit that introduced target-context garbling
3. Implement fix (either extend `pipeline_parallel` disable to all contexts on
   multi-GPU, or fix `prev_copy` source selection for split inputs)
4. Verify clean output on romulus with `GGML_PIPELINE_PLUS=1`
5. Run full benchmark matrix (27B, 35B, 35B+MTP) to confirm no regression

### Phase 2: Verify on triton

**Config:** 3090 (24GB) + 3070 (8GB), native CUDA

Already reproduced (see [Results](#results) below). After local fix:

```bash
# Pull fix, rebuild CUDA, verify
cd ~/projects/path-d-gpipeline-assembly-line
git pull gitea
cmake -B build-cuda-fresh -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build-cuda-fresh --target llama-server -j$(nproc)

# Verify clean with GGML_PIPELINE_PLUS=1
GGML_PIPELINE_PLUS=1 llama-server \
  -m /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf \
  -sm layer -ts 75,25 -ngl 99 --ctx-size 4096 \
  --host 127.0.0.1 --port 8080
```

### Phase 3: Verify on 4-GPU Cluster

**Config:** 7900 XTX (romulus ROCm) + 3060 Ti (docker) + 5060 Ti (remus) + 5070 (jupiter)

After triton passes, scale to full cluster using existing bench scripts:

```bash
cd rpc-patch/scripts
./pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary
```

---

## Environments

| Node | GPUs | Backend | Notes |
|------|------|---------|-------|
| **romulus** | 7900 XTX (24GB) + 3060 Ti (8GB) | ROCm + CUDA RPC | Primary dev node, fix here first |
| **triton** | 3090 (24GB) + 3070 (8GB) | Native CUDA | Already reproduced |
| **remus** | 5060 Ti (16GB) | CUDA RPC | 4-GPU cluster worker |
| **jupiter** | 5070 (12GB) | CUDA RPC | 4-GPU cluster worker |
| **4-GPU cluster** | 7900+3060+5060+5070 | ROCm + 3x CUDA RPC | Full production config |

---

## Config

### Server invocation (clean)
```
GGML_PIPELINE_PLUS=0 llama-server \
  -m <model.gguf> \
  -sm layer -ts 75,25 \
  -ngl 99 --ctx-size 4096 \
  --host 127.0.0.1 --port 8080 \
  --no-warmup
```

### Server invocation (garbled)
Same but `GGML_PIPELINE_PLUS=1`.

### Benchmark
```bash
curl -s --max-time 120 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"The quick brown fox jumps over the lazy dog."}],"max_tokens":128,"temperature":0,"stream":false}'
```

### Garbling check
```bash
bash rpc-patch/patch/loop-check-garble.sh 8080 'What is the capital of France?' 64 42
```

Note: `loop-check-garble.sh` only inspects `content`, not `reasoning_content`.
Thinking models (35B) false-positive as GARBLED because `content` is empty.

---

## Results (triton reproduction)

### 27B Q5_K_M (Qwen3.5-27B, 19GB dense, 64 layers)

| Run | PP (t/s) | TG (t/s) | Correct? |
|-----|----------|----------|----------|
| 1 | 56.2 | 29.8 | CLEAN |
| 2 | 58.4 | 29.8 | CLEAN |
| 3 | 58.7 | 29.8 | CLEAN |

Median: **PP=58.4, TG=29.8 t/s** (GGML_PIPELINE_PLUS=0)

GPU memory: 3090=14241 MiB, 3070=5667 MiB

With `GGML_PIPELINE_PLUS=1`: garbled (`"Thinking Process Process Process Process:::: ... 1111.... **222+++"`)

### 35B APEX (Qwen3.6-35B-A3B, 22GB MoE, thinking model)

| Run | PP (t/s) | TG (t/s) | Correct? |
|-----|----------|----------|----------|
| 1 | 303.4 | 125.8 | CLEAN |
| 2 | 164.3 | 128.2 | CLEAN |
| 3 | 166.6 | 128.5 | CLEAN |

Median: **PP=166.6, TG=128.2 t/s** (GGML_PIPELINE_PLUS=0)

GPU memory: 3090=16097 MiB, 3070=6305 MiB

With `GGML_PIPELINE_PLUS=1`: garbled.

Output field: `reasoning_content` (thinking model). Example clean output:
```
Here's a thinking process:

1.  **Identify the User's Question**: The user is asking "What is the capital of France?"
2.  **Retrieve Knowledge**: I know from general knowledge that the capital o...
```

### 0.8B sanity check

Also garbled with `GGML_PIPELINE_PLUS=1`, clean with `=0`. Confirms the issue
is not model-size dependent.

---

## Relation to Prior Work

| Handover / Commit | Issue | Fix |
|-------------------|-------|-----|
| 2026-07-13 handover, commit `87357519e` | MTP draft context garbling under PPLUS | Disable `pipeline_parallel` for `LLAMA_CONTEXT_TYPE_MTP` |
| This diagnostic (2026-07-19) | Target context garbling under PPLUS on multi-GPU | **OPEN** — workaround: `GGML_PIPELINE_PLUS=0` |

The 2026-07-13 fix was necessary but insufficient. The target context still
runs pipeline parallelism, and on multi-GPU this produces garbling for all
model types (dense, MoE, thinking, non-thinking).

---

## Open Items

1. **Reproduce on romulus**: Confirm the bug exists on the primary dev node.
2. **Bisect**: Identify the exact commit where target-context PPLUS garbling
   appeared. The MTP fix (`87357519e`) is a candidate ancestor.
3. **Single vs multi-GPU**: Confirm the bug is absent on single-GPU with
   `GGML_PIPELINE_PLUS=1`. This would isolate the issue to cross-device
   copy-slot synchronization.
4. **Implement fix**: Either extend the `pipeline_parallel` disable to all
   contexts on multi-GPU, or fix the `prev_copy` source selection for split
   inputs so it reads coherently under rotation.
5. **Verify on triton**: After local fix, pull and verify on triton.
6. **Verify on 4-GPU cluster**: Scale to full cluster using existing bench
   scripts.
7. **Profiler bug**: `llama-gpipe-profiler` segfaults (exit 139) on dual-GPU
   after `sched_reserve`. Separate investigation needed.
8. **MTP unavailable on triton**: The `Qwen3.6-35B-A3B-APEX-I-Quality.gguf`
   model has no MTP layers. Cannot verify MTP speculative decoding on triton.
   A `-MTP-`-suffixed 35B model does not exist on `/mnt/980pro/models`.
9. **Garbling check false positive**: `loop-check-garble.sh` needs an update to
   also inspect `reasoning_content` for thinking models.

---

## Investigation Log (triton)

1. Built CUDA binaries (`llama-server`, `llama-gpipe-profiler`) for arch 86
   on triton (3090+3070).
2. Attempted to run `llama-gpipe-profiler` — it **segfaults** (exit 139) on
   dual-GPU after `sched_reserve` completes. Works on single-GPU. This is a
   separate profiler bug.
3. Fell back to `llama-server` + curl benchmark.
4. Ran 27B with `GGML_PIPELINE_PLUS=1` → garbled output.
5. Ran 27B with `GGML_PIPELINE_PLUS=0` → clean output.
6. Ran 35B with `GGML_PIPELINE_PLUS=1` → garbled output.
7. Ran 35B with `GGML_PIPELINE_PLUS=0` → clean output.
8. Verified the 35B model is a thinking model that outputs to
   `reasoning_content` (not `content`) — the `loop-check-garble.sh` script
   false-positives on thinking models.
