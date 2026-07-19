# Fix multi-GPU target-context KV garble under GGML_PIPELINE_PLUS

Type: bugfix
Status: resolved
Triage: resolved in `f68e17b9b`
Blocked by:

## Parent / forensic pack

- Diagnostic report (required reading): [README.md](README.md)
- Index: [diagnostics/README.md](../README.md)
- Prior related fix (draft only): commit `87357519e` (disable `pipeline_parallel` for `LLAMA_CONTEXT_TYPE_MTP`)
- Skill for process: diagnosing-bugs (tight red loop first; no theory without a command that goes red)

## Problem (one sentence)

With `GGML_PIPELINE_PLUS=1`, multi-GPU layer-split decode corrupts KV / split inputs on the **target** context so generation becomes repetitive nonsense; `GGML_PIPELINE_PLUS=0` is clean on the same binary, model, and topology.

## Scope

**In scope**

- Target-context copy-slot / `prev_copy` / multi-GPU scheduling under Plus
- Repro on dual native CUDA (triton 3090+3070) and dual HIP+CUDA if easy
- Fix that restores coherent decode with Plus=1 on multi-GPU
- Regression feedback loop that another agent can re-run unattended
- Update `loop-check-garble.sh` so thinking models do not false-positive (inspect `reasoning_content` when `content` empty)

**Out of scope**

- Placement control plane / plan IR (orthogonal)
- Full Path D GPipe stage redesign
- Profiler dual-GPU segfault (note as separate if hit; do not block this ticket)
- Claiming "MTP model" when the GGUF has no NextN/MTP layers (report used non-MTP APEX 35B)

## Non-goals

- Leaving Plus permanently off as the "fix"
- Heuristic-only "fixes" that hide garble in the detector without curing logits/text quality
- Enabling experimental envs (`MULTI_BACKEND_SEQ`, dual-socket, wavefront) as part of the minimal repro

## Known facts (do not re-litigate)

| Fact | Source |
|------|--------|
| Plus=1 garbles multi-GPU; Plus=0 clean | README results 27B / 35B / 0.8B on triton |
| No RPC required to reproduce | native dual CUDA on one node |
| Draft-context race was already mitigated | `llama-context.cpp`: `pipeline_parallel && ctx_type != MTP` |
| Target still uses `n_copies` / `prev_copy` under Plus | `ggml-backend.cpp` + `llama-context.cpp` sched construction |
| Thinking models put answer in `reasoning_content` | 35B clean example in README |
| `loop-check-garble.sh` only reads `content` | false-positive on thinking models |

## Environment (canonical repro)

Prefer **triton** (or any dual-GPU CUDA node matching the report):

```text
Branch tip near: Path-D-Gpipeline-Assembly-Line @ 57dca6483+ (or current tip)
Build: CUDA, arch matching GPUs (e.g. 86 for 3090/3070)
Binary: llama-server
Models (examples from report):
  /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf          # non-thinking preferred for loop
  /mnt/980pro/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf  # thinking; use reasoning_content
Topology: -sm layer -ts 75,25 -ngl 99 -c 4096
```

Romulus HIP+CUDA RPC is optional secondary confirmation only. Do not mix inverted preflight TS (preflight RPC-first vs server ROCm-first) into the minimal loop.

## Agent workflow (debug and fix loop)

Follow phases in order. Do not skip Phase 1.

### Phase 0 - Claim and orient

1. Set this file `Status: in_progress`, `Triage: claimed`.
2. Read [README.md](README.md) fully.
3. Skim seams:
   - `src/llama-context.cpp` - where `pipeline_parallel` is set / cleared for MTP
   - `ggml/src/ggml-backend.cpp` - `n_copies`, `prev_copy`, split input copies under Plus
4. Confirm models and dual GPU free. Prefer 27B for the primary loop (non-thinking, stable `content`).

### Phase 1 - Tight red-capable feedback loop

**Goal:** one agent-runnable command that is RED when Plus=1 multi-GPU is broken and GREEN when fixed (or when Plus=0).

**Minimum loop (already sketched in README):**

```bash
# Terminal A - RED server (expect garble)
GGML_PIPELINE_PLUS=1 ./build-cuda-fresh/bin/llama-server \
  -m /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf \
  -sm layer -ts 75,25 -ngl 99 --ctx-size 4096 \
  --host 127.0.0.1 --port 8080 --no-warmup --jinja

# Terminal B
bash rpc-patch/patch/loop-check-garble.sh 8080 'What is the capital of France?' 64 42
# expect FAIL / GARBLED on current tip with Plus=1
```

**GREEN control (same binary, same model, same TS):**

```bash
GGML_PIPELINE_PLUS=0 ...same flags...
bash rpc-patch/patch/loop-check-garble.sh 8080 'What is the capital of France?' 64 42
# expect PASS / CLEAN
```

**Harden the loop before deep theory:**

1. Fix detector: if `content` empty/whitespace, also score `message.reasoning_content` (and/or concatenate). Commit detector fix separately if pure tooling.
2. Assert **semantic** green: e.g. response mentions `Paris` (or `4` for `2+2`) under `temperature=0` `seed=42`, not only "no repetition".
3. Prefer 27B for CI-style loop; keep 35B as secondary once detector is thinking-safe.
4. Record one command + sample RED and GREEN output in this ticket Answer section when claiming.

**Phase 1 done when:** you have run the command, pasted RED (Plus=1) and GREEN (Plus=0) evidence, and the assert matches the user symptom (nonsense / repetition under Plus multi-GPU).

### Phase 2 - Reproduce + minimise

Keep cutting until every remaining factor is load-bearing:

| Cut | Keep if still red with Plus=1 |
|-----|-------------------------------|
| Drop MTP / draft flags | yes expected |
| Drop RPC | yes expected (native dual GPU) |
| Drop SEQ / dual-socket / wavefront envs | yes - leave unset |
| Smaller model (0.8B / 1.5B) | if still red, prefer for speed |
| Single GPU (`-ts 100` or one device) | if **green**, multi-GPU is load-bearing |
| Plus=0 | must go **green** (control) |

Document minimal matrix in Answer:

```text
minimal RED: Plus=1, multi-GPU layer split, model=..., ts=...
minimal GREEN: same but Plus=0
single-GPU Plus=1: RED|GREEN
```

### Phase 3 - Ranked falsifiable hypotheses

Propose 3-5 before testing. Examples to start from (replace with evidence-backed ranking):

1. **Target `prev_copy` reads stale split inputs** across devices under `n_copies>1`.  
   If true: forcing `n_copies=1` while Plus env stays on goes green; instrumented copy indices disagree across GPUs.
2. **KV view / cell mapping races under rotation** on multi-backend sched.  
   If true: KV dumps differ Plus=0 vs Plus=1 after same prompt tokens.
3. **Over-broad pipeline_parallel enable** on multi-device even when unsafe.  
   If true: gating `pipeline_parallel` off when `n_devices>1` goes green (may be temporary fix).
4. **Draft fix incomplete** (shared buffers still rotated from target path).  
   If true: only appears when MTP draft also constructed; pure target-only still green.
5. **Regression after a known commit** near `87357519e` / copy-slot work.  
   If true: `git bisect` with Phase 1 loop finds a clear first-bad.

### Phase 4 - Instrument

- One variable at a time.
- Tag logs `[DEBUG-pplus-garble]` (or similar); remove before resolve.
- Prefer logging: `n_copies`, `cur_copy`, `prev_copy`, split id, backend name, decode/token index.
- Avoid flooding every tensor.

### Phase 5 - Fix + lock

**Acceptable fixes (in order of preference):**

1. Correct multi-GPU copy-slot / split-input coherence under Plus (preserves Plus benefit).
2. Safer enablement: disable pipeline_parallel (or n_copies>1) only when multi-device and unsafe, with clear log.
3. Temporary hard disable of Plus multi-GPU with loud warning - only if (1)(2) blocked; document follow-up.

**Before merging fix:**

- [ ] Phase 1 loop GREEN with Plus=1 multi-GPU on 27B (or agreed minimal model)
- [ ] Plus=0 still GREEN (no regression)
- [ ] Secondary: 35B thinking path coherent (check `reasoning_content` if needed)
- [ ] Single-GPU Plus=1 still OK if it was OK before
- [ ] Detector updated for thinking models if you touched it
- [ ] No leftover `[DEBUG-...]` tags

### Phase 6 - Close out

1. Update [README.md](README.md): Status FIXED, commit SHA, final root cause one-liner.
2. Update [diagnostics/README.md](../README.md) index row to FIXED + SHA.
3. This file: all AC checked, `Status: resolved`, short `## Answer`.
4. Commit message states the winning hypothesis and the one command that went green.

## Acceptance criteria

- [ ] Red-capable unattended loop exists (document path + example RED/GREEN output)
- [ ] Garble check handles thinking models (`reasoning_content` fallback)
- [ ] Minimal multi-GPU repro documented (Plus=1 red / Plus=0 green)
- [ ] Root cause stated with evidence (not only "disable Plus")
- [ ] Fix lands: Plus=1 multi-GPU produces coherent text on 27B (primary) and 35B (secondary)
- [ ] Regression path: loop stays green; Plus=0 unchanged
- [ ] Diagnostic README + diagnostics index updated
- [ ] Debug instrumentation removed

## Code map (starting points)

| Area | File | Notes |
|------|------|-------|
| Disable PP for MTP draft | `src/llama-context.cpp` | ~`pipeline_parallel && ctx_type != MTP` |
| Where PP is enabled on context | `src/llama-context.cpp` | search `pipeline_parallel` |
| Copy slots / prev_copy | `ggml/src/ggml-backend.cpp` | Plus path, split copies |
| Plus env gate | search `llama_pipeline_plus_enabled` / `GGML_PIPELINE_PLUS` | |
| Feedback script | `rpc-patch/patch/loop-check-garble.sh` | fix content-only check |

## Anti-patterns (do not)

- Declare victory because detector says CLEAN while text is empty (thinking model)
- Expand scope into RPC TP-unit / placement tickets
- "Fix" by permanently defaulting Plus off without documenting residual risk
- Bisect without automating the Phase 1 loop (`git bisect run`)

## Handoff checklist for the next agent

```text
[ ] Claim ticket (status in_progress / claimed)
[ ] Read README.md in this directory
[ ] Run GREEN control Plus=0 then RED Plus=1 on 27B dual-GPU
[ ] Harden loop (thinking-safe + semantic assert)
[ ] Minimise matrix
[ ] Rank hypotheses; test one change at a time
[ ] Fix; re-run loop; update READMEs; resolve ticket
```

## Agent context (ops + docs + history)

Copy this section when starting a session on another machine. Prefer keys over passwords when available; defaults below match in-tree scripts.

### Cluster access

| Node | IP | SSH user | Default password (scripts) | Role for this bug |
|------|-----|----------|----------------------------|-------------------|
| **triton** (canonical repro) | `192.168.8.23` | `hunter` | `12345` (`B6_TRITON_PASS` / `PATHB_TRITON_SSH_PASS`) | Dual native CUDA 3090+3070; report was run here |
| **romulus** | `192.168.8.108` | `hunter` | see gitignored secrets | ROCm client + optional local 3060 docker RPC |
| **remus** | `192.168.8.176` (CONTEXT) / `.22` in older handover | `hunter` | see secrets | CUDA RPC worker (optional secondary) |

**Secrets file (gitignored, not in git):** `.scratch/cluster-access.env`  
**Layout:** [rpc-patch/patch/CLUSTER-NODE-LAYOUT.md](../../rpc-patch/patch/CLUSTER-NODE-LAYOUT.md)

```bash
# Triton SSH (from scripts/b6-gate-triton-remote.sh / pathb-triton-rpc.sh)
export PATHB_TRITON_SSH=hunter@192.168.8.23
export PATHB_TRITON_SSH_PASS=12345   # or B6_TRITON_PASS=12345
# Prefer: ssh-copy-id once, then drop password env

ssh hunter@192.168.8.23
# password default in scripts: 12345
```

**Canonical repo path on every Linux node:**

```text
~/projects/atomic-llama-cpp-turboquant
```

This workspace may also live as `~/projects/path-d-gpipeline-assembly-line` (same branch family). Sync both if clone names differ; do not invent a third path.

**Triton build / RPC docker:**

| Item | Path / note |
|------|-------------|
| Host CUDA build | `~/projects/atomic-llama-cpp-turboquant/build-cuda-fresh` or `build-cuda-b-bin` |
| Report binary | `build-cuda-fresh/bin/llama-server` (CUDA arch 86) |
| Docker deploy | `~/docker/Atomic-Llama-Triton-PathB` (`:50054` 3090, `:50055` 3070) |
| Lifecycle script | `rpc-patch/scripts/pathb-triton-rpc.sh start|stop|status|rebuild` |
| Remote helper | `scripts/b6-gate-triton-remote.sh sync|status|rpc|stop|rebuild|audit` |
| Models on triton | `/mnt/980pro/models/` (nvme mount) |

**Romulus-local (secondary only - HIP+CUDA RPC, not minimal repro):**

```bash
cd ~/projects/atomic-llama-cpp-turboquant   # or path-d checkout
bash scripts/romulus-local-build.sh
bash scripts/romulus-local-up.sh -c 4096 --mtp off   # or --mtp on for MTP GGUF
# listen 0.0.0.0:8080; RPC docker pathb-rpc-romulus :50051
```

### Env flags (what matters for this bug)

#### Primary switch (load-bearing)

| Flag | RED repro | GREEN control |
|------|-----------|---------------|
| `GGML_PIPELINE_PLUS` | `1` | `0` |

Plus enables pipeline parallelism / copy-slot rotation (`n_copies=4`, `prev_copy`) in `ggml-backend.cpp`.

#### Blessed Path-B+ stack (when Plus=1 for perf work - not required for minimal garble loop)

From [docs/pipeline-plus/CONFIGURATION.md](../../docs/pipeline-plus/CONFIGURATION.md) and [IMPLEMENTATION.md](../../docs/pipeline-plus/IMPLEMENTATION.md):

```bash
export GGML_PIPELINE_PLUS=1
export GGML_PIPELINE_BARRIER_PARTIAL=1
export GGML_PIPELINE_BARRIER_PARTIAL_STRICT=1
export GGML_RPC_EVENT_DEFER_BARRIER=1
export GGML_RPC_GET_TENSOR_DEFER=1
export GGML_RPC_MULTI_SOCKET_FLUSH=1
export GGML_SCHED_MOE_ASYNC_COPY=1
```

These are **performance** mitigations for multi-backend RPC. The dual-GPU garble reproduces with **native multi-GPU only** (no RPC). For Phase 1 loop, keep the matrix minimal: only toggle `GGML_PIPELINE_PLUS`.

#### Do not enable for the minimal loop (unless testing a hypothesis)

| Flag | Why |
|------|-----|
| `GGML_PIPELINE_MULTI_BACKEND_SEQ` | TSC seq-repair (romulus-local default); not in CONFIGURATION blessed set; confounds |
| `GGML_RPC_DUAL_SOCKET` | Deprecated / regression on 4-GPU (ADR path) |
| `GGML_SCHED_WAVEFRONT_DISPATCH` | Deprecated null effect |
| Extra trace spam | Optional later: `GGML_SCHED_TRACE`, `GGML_PIPELINE_TRACE` |

#### CLI shape (report-proven)

```bash
# GREEN
GGML_PIPELINE_PLUS=0 llama-server \
  -m /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf \
  -sm layer -ts 75,25 -ngl 99 --ctx-size 4096 \
  --host 127.0.0.1 --port 8080 --no-warmup --jinja

# RED
GGML_PIPELINE_PLUS=1 ...same flags...
```

**TS note:** `-ts` follows **server device order** (list order in load log). On triton CUDA dual-GPU, report used `75,25` (more on 3090). Do not paste preflight RPC-first splits onto a client that registers local GPU first without remapping.

### Project doc refs (read in this order)

| Doc | Why |
|------|-----|
| [README.md](README.md) (this directory) | Forensic symptom, results, false-positive detector note |
| [diagnostics/README.md](../README.md) | Index of diagnostic packs |
| [PIPELINE.md](../../PIPELINE.md) | How Plus / copy slots / pipeline_parallel work |
| [docs/pipeline-plus/CONFIGURATION.md](../../docs/pipeline-plus/CONFIGURATION.md) | Blessed env set + deprecations |
| [docs/pipeline-plus/IMPLEMENTATION.md](../../docs/pipeline-plus/IMPLEMENTATION.md) | Flag table B+8..B+13 |
| [docs/pipeline-plus/CONTEXT.md](../../docs/pipeline-plus/CONTEXT.md) | Node names (Romulus/Remus/Triton), glossary |
| [rpc-patch/patch/CLUSTER-NODE-LAYOUT.md](../../rpc-patch/patch/CLUSTER-NODE-LAYOUT.md) | Paths, docker dirs, remotes |
| [rpc-patch/patch/HANDOVER-SESSION-2026-07-13.md](../../rpc-patch/patch/HANDOVER-SESSION-2026-07-13.md) | prev_copy + MTP draft garble session |
| [docs/pipeline-plus/TRACKING.md](../../docs/pipeline-plus/TRACKING.md) | Bench gate labels / history |

### Related commits (what already tried to fix garble)

| Commit | Summary | Relation to this ticket |
|--------|---------|-------------------------|
| `87357519e` | `rpc: prev_copy KV-cache fix + large-model 3-GPU test results` | Added `prev_copy` so split inputs copy from previous graph slot under `n_copies>1`. Fixed classic multi-GPU PPLUS for many models at 8B-scale matrix. |
| `470798451` | `llama: disable pipeline_parallel copy-slot rotation for MTP draft contexts` | Stops **draft** (`LLAMA_CONTEXT_TYPE_MTP`) from independent copy-slot race on shared KV. **Does not disable PP on target.** |
| `95e21952d` | MTP draft graph reserve / profiler draft flags | Adjacent MTP reliability; not the multi-GPU target fix. |
| `72c2aa3cd` | RPC weak-symbol garble | **Different bug** (RPC buffer type detection). See [rpc-weak-symbol-bug/](../rpc-weak-symbol-bug/). Do not confuse. |
| `d1df5f436` | placement Shape A TP-unit (issue 14) | Orthogonal; placement opt-in. Pre-issue retest showed multi-GPU Plus garble without it. |

**2026-07-13 handover takeaway:** MTP garble under Plus was explained as draft vs target uncoordinated `prev_copy` cycles on shared KV. Fix: force `pipeline_parallel=false` for MTP draft sched. **2026-07-19 diagnostic:** target context still garbles on multi-GPU with Plus=1 even without MTP (27B dense, 0.8B, thinking 35B).

### Models (triton paths from report)

| GGUF | Role |
|------|------|
| `/mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf` | **Primary loop** (non-thinking `content`) |
| `/mnt/980pro/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf` | Secondary; **thinking** -> check `reasoning_content` |
| Small dense (0.8B / 1.5B if present) | Minimise loop speed |

Note: APEX-I-Quality 35B in the report has **no MTP layers**. For real draft-mtp tests use a `*MTP*` GGUF under `/mnt/models` or `/mnt/980pro/models` after `scripts/verify-qwen36-nextn-gguf.py`.

### Feedback scripts

| Script | Role |
|--------|------|
| `rpc-patch/patch/loop-check-garble.sh` | Phase 1 loop; **update** for `reasoning_content` |
| `scripts/b6-gate-triton-remote.sh` | Remote status / rebuild helpers |
| `rpc-patch/scripts/pathb-triton-rpc.sh` | Triton docker RPC lifecycle (not needed for native dual-GPU minimal loop) |

### One-shot agent start (triton)

```bash
ssh hunter@192.168.8.23   # pass default in scripts: 12345
cd ~/projects/atomic-llama-cpp-turboquant
git fetch && git checkout Path-D-Gpipeline-Assembly-Line && git log -1 --oneline
# build if needed: CUDA 12.8, arch 86, llama-server -> build-cuda-fresh

# GREEN
GGML_PIPELINE_PLUS=0 ./build-cuda-fresh/bin/llama-server \
  -m /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf \
  -sm layer -ts 75,25 -ngl 99 -c 4096 --host 127.0.0.1 --port 8080 --no-warmup --jinja &
sleep 20
bash rpc-patch/patch/loop-check-garble.sh 8080 'What is the capital of France?' 64 42

# RED (same port after restart with PLUS=1)
```

## Answer

**Commit**: `f68e17b9b` (2026-07-19)
**Root cause**: During graph reuse (the token-generation hot path),
`ggml_backend_sched_pipeline_barrier` (`ggml/src/ggml-backend.cpp:3484-3486`)
rotated copy slots (`cur_copy`/`next_copy`) but `alloc_graph` was skipped
(`is_alloc=true`), so the graph's tensor pointers remained frozen at the
alloc-time slot. `compute_splits` then copied split inputs to the rotated slot
while the graph read from the frozen slot, garbling every cross-backend input.

**Fix**: The barrier now waits on `cur_copy` events (not `next_copy`) and does
not rotate. Copy-slot rotation happens only in `alloc_graph`, which also calls
`split_graph` to update `node->src` to match.

**First-bad commit**: `87357519e` (introduced B+16 rotation + partial barrier).
The MTP-only fix `470798451` worked around the symptom for draft contexts.

**Verification command** (triton, RTX 3090+3070):

```bash
bash scripts/triton-pplus-garble-loop.sh --plus 1 \
  --model /mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf \
  --port 8096 --tokens 64 --expect "Paris" --label "post-fix-27B"
# verdict=CLEAN exit_code=0
```

**Evidence** (pre-fix RED / post-fix GREEN):

| Model | Plus | Pre-fix | Post-fix |
|-------|------|---------|----------|
| 1.5B  | 1    | GARBLED ("capital capital capital...") | CLEAN ("The capital of France is Paris.") |
| 27B   | 1    | GARBLED ("Thinking Process Process...") | CLEAN (coherent thinking-process output) |
| 1.5B  | 0    | CLEAN | CLEAN (no regression) |

Full 5-case falsification matrix (Plus=0 control, Plus=1 baseline, Plus=1 +
BARRIER_PARTIAL=0, Plus=1 + PIPELINE_DEPTH=2, Plus=1 + MULTI_BACKEND_SEQ=1):
all CLEAN post-fix. Logs in `benches/pplus-garble/matrix-*.log` on triton.

**Trade-off**: copy-slot pipelining disabled during generation (same slot per
decode). Async compute overlap preserved via events. Matches existing design
intent at `ggml-backend.cpp:3377` ("same copy is used every time during
generation").
