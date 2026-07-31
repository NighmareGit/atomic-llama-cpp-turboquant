# Scratchpad: PPLUS multi-GPU garble fix

**Purpose**: Combat compaction amnesia. Update after each mentionable step.
Re-parse this file first after any context compaction.
**Ticket**: [ISSUE.md](ISSUE.md)  |  **Forensic pack**: [README.md](README.md)
**Started**: 2026-07-19

---

## Current mode / authorization

- I orchestrate + decide; agents do legwork; fixes land locally on Romulus,
  commit, push, pull on triton, rebuild, continue loop on triton.
- Authorized to edit source on Romulus working tree; NO git commit / push /
  PR without explicit per-action approval.
- triton access: `sshpass -p <redacted> ssh user@192.168.8.23` (verified working).
- ASCII only in code + commits. No emdash / unicode arrows.

## Environment snapshot (verified Phase 0)

| Item | Romulus (here) | triton (repro node) |
|------|----------------|---------------------|
| Hostname | Romulus | Triton |
| IP | 192.168.8.108 | 192.168.8.23 |
| GPUs | 1x 3060 Ti + Radeon 7900 XTX (hetero RPC, OUT OF SCOPE) | 3090 + 3070 native dual-CUDA (canonical repro) |
| Repo | `~/projects/path-d-gpipeline-assembly-line` @ `d1df5f436` (1 ahead of gitea) | same path @ `57dca6483` (1 behind Romulus) |
| CUDA build | none working | `build-cuda-fresh/bin/llama-server` 253MB, ready |
| 27B model | present | present at `/mnt/980pro/models/Qwen3.5-27B-Q5_K_M.gguf` |

triton ssh: works via `sshpass -p <redacted> ssh user@192.168.8.23`.

## Key code sites (verified by reading)

- `src/llama-context.cpp:573-600` - `pipeline_parallel` computed
  (n_devices>1 && ngl>n_layer_all && all backends have async+events)
- `src/llama-context.cpp:699-707` - MTP exclusion: `pipeline_parallel && ctx_type != MTP`
- `src/llama-context.cpp:864-867` - retry without PP if alloc fails
- `src/llama-context.cpp:972` - Plus gate: `!pp || !plus || p1_full_sync` -> sync
- `src/llama-context.cpp:1607-1609` - reuse + plus -> pipeline_barrier
- `ggml/src/ggml-backend.cpp:1296-1362` - sched struct: `n_copies`, `prev_copy`,
  `hv_tensor_copies`, `tensor_id_copy`, `ggml_sched_pipeline_depth_cfg`
- `ggml/src/ggml-backend.cpp:1860-1923` - split input copies under rotation
- `ggml/src/ggml-backend.cpp:1975-2010` - n_copies>1 init path
- `rpc-patch/patch/loop-check-garble.sh:46` - reads `content` only (thinking-model bug)

## Bisect candidates (commits of interest)

| SHA | Date | Why interesting |
|-----|------|----------------|
| `e2fcdaf63` | Jul 16 | "copy_event overwrite ... root cause of async H2D crash but defers overlap" - direct async event overwrite suspect |
| `251f1a157` | Jul 19 | "serialize_graph_for_all buffer overflow causing heap corruption" - heap corruption could garble tensors |
| `f29a92eb1` | Jul 16 | "D6.10 GPU event pipelining for GPipe multi-seq dispatch" - new event machinery |
| `b145d6fce` | Jul 16 | "D6.10.1 skip GPU event wait for host-to-GPU INPUT copies" - event-skip on INPUT copies |
| `470798451` | Jul ? | "disable pipeline_parallel copy-slot rotation for MTP draft contexts" - the prior MTP-only fix |
| `87357519e` | Jul 13 | "prev_copy KV-cache fix" - the original ancestor fix |
| `57dca6483` | recent | triton tip; per `d1df5f436` notes, garble reproduces here in blessed smoke |

## Key prior-art note (from local unpushed `d1df5f436`)

> "Multi-backend blessed smoke (FA on, -nkvo, ctk q8_0, ctv turbo3, draft-mtp,
> GGML_PIPELINE_PLUS + MULTI_BACKEND_SEQ + barrier/defer) still garbles on
> 9B/35B MTP. Rebuilt client+RPC at same git SHA (57dca6483); failure remains.
> Pre-issue clean tree retest (WIP stashed) reproduces the same garble: NOT
> introduced by this change."

So: garble is reproducible at `57dca6483` independent of placement Shape A.
Placement work is orthogonal. Issue is on the multi-backend SEQ / RPC path
PLUS the simpler native dual-CUDA layer-split path described in README.

---

## TODO tracker (mirror of todo_write)

- [x] 1. Phase 0: triton ssh + GPU + repo verified
- [x] 2. Agent A1: patch loop-check-garble.sh (reasoning_content + EXPECT_SUBSTR)
      -> subagent 019f7a67-...2c5 DONE. bash -n OK. smoke tests pass.
- [x] 3. Agent A2: rank hypotheses from code read
      -> subagent 019f7a67-...798 DONE. Artifact /tmp/pplus-hypotheses.md.
         Top hypothesis: partial pipeline_barrier (ggml-backend.cpp:3421-3444)
         waits only on barrier_copy_src_mask, but B+16 rotation copy
         (ggml-backend.cpp:2660-2665) reads prev_copy slot whose producer
         backend may not be waited on. Falsification: GGML_PIPELINE_BARRIER_PARTIAL=0.
- [x] 4. Agent A3: write scripts/triton-pplus-garble-loop.sh
      -> subagent 019f7a67-...165 DONE. bash -n OK. GPU-gated.
- [x] 5. Commit tooling + push (1c048ff5b, fd160fe88)
- [x] 6. Push mirrors + pull on triton - DONE
- [x] 7. triton Phase 1 RED/GREEN loop with patched detector (1.5B model, 5s repro)
- [x] 8. triton Phase 3 falsification matrix (env toggles, NO rebuild) - DONE
- [x] 9. Root cause + source fix (f68e17b9b) - DONE
- [x] 11. Phase 6 close-out: README + ISSUE + diagnostics index + final commit (802ccb4c6)
- [x] 12. D8 future-enhancement doc written + TRACKING pointer (906468aac)

## PERF DATA (apples-to-apples, triton 27B Q5_K_M, 64 tokens)

| Config | PP t/s | TG t/s | Notes |
|--------|--------|--------|-------|
| Plus=0 baseline (post-fix) | 82.69 | 29.94 | working baseline |
| Plus=1 post-fix (f68e17b9b) | 76.28 | 29.89 | TG parity with Plus=0; PP -7.8% |
| Plus=1 pre-fix | n/a (garbled) | n/a | unusable |
| README baseline (Plus=0) | 58.4 | 29.8 | different test setup |

TG regression from fix: ~0% (within noise). PP overhead: ~8% (Plus sync, expected).
D8 future enhancement ceiling: ~3-6% TG on dual-GPU, ~5-10% on RPC cluster.

## FIX VERIFICATION (2026-07-19 15:28 UTC, triton, post-fix f68e17b9b)

### Post-fix matrix (1.5B model, 5 cases) - ALL CLEAN

| Case | Toggle | Before fix | After fix |
|------|--------|-----------|-----------|
| 1 | Plus=0 (control) | CLEAN | CLEAN (no regression) |
| 2 | Plus=1 (baseline) | GARBLED | CLEAN |
| 3 | Plus=1 + BARRIER_PARTIAL=0 | GARBLED | CLEAN |
| 4 | Plus=1 + PIPELINE_DEPTH=2 | GARBLED | CLEAN |
| 5 | Plus=1 + MULTI_BACKEND_SEQ=1 | CLEAN (Plus off) | CLEAN |

### Post-fix canonical 27B test (Plus=1)

Output: "Thinking Process: 1. Identify the core question... The capital of France is Paris."
Verdict: CLEAN, tokens=64. Previously garbled.

## FALSIFICATION MATRIX RESULTS (2026-07-19 15:01 UTC, triton, 1.5B model)

| Case | Toggle | Verdict | Hypothesis | Status |
|------|--------|---------|------------|--------|
| 1 | Plus=0 (control) | CLEAN | baseline green | OK |
| 2 | Plus=1 (baseline) | GARBLED ("capital capital capital...") | baseline red | OK |
| 3 | Plus=1 + GGML_PIPELINE_BARRIER_PARTIAL=0 | GARBLED | H1: partial mask too narrow | **FALSIFIED** |
| 4 | Plus=1 + GGML_SCHED_PIPELINE_DEPTH=2 | GARBLED | H5: over-rotation | **FALSIFIED** |
| 5 | Plus=1 + GGML_PIPELINE_MULTI_BACKEND_SEQ=1 | CLEAN | H6: reuse full-sync bypass | **SUSPICIOUS** |

### Interpretation

H1 FALSIFIED: BARRIER_PARTIAL=0 forces wait_mask = ALL backends at pipeline_barrier
(line 3422-3430). Still garbles. So the race is NOT in the barrier wait mask.
Even waiting on all backends' events doesn't fix it -> data is stale BEFORE
the barrier, or wrong data is being copied.

H5 FALSIFIED: DEPTH=2 (matching n_backends) doesn't help. Over-rotation ruled out.

H6 SUSPICIOUS: MULTI_BACKEND_SEQ=1 goes clean, BUT this env var likely forces
Plus=0 internally (see ggml-backend.cpp:60 `else if multi_backend_seq: v=0`
inside ggml_sched_pipeline_plus_enabled). So this "CLEAN" is just Plus disabled
by another name, NOT evidence that the reuse bypass is the fix locus. NEED TO
VERIFY by reading the full function.

### Pivot: the issue is in the rotation copy mechanism itself, not the barrier

Since BARRIER_PARTIAL=0 (full sync) doesn't fix it, the data being read from
prev_copy is wrong at copy time, not at sync time. Focus shifts to:
- ggml-backend.cpp:2660-2665 B+16 rotation copy (prev_copy -> cur_copy on consumer)
- ggml-backend.cpp:2680+ second pass for non-FLAG_INPUT split tensors
- Whether the B+16 copy sources from the right tensor/backend
- Whether split-data (hidden state) tensors get rotation-copied at all

Next step: read the second pass (non-INPUT split tensors, line 2680+) and the
full Plus-gating function to verify MULTI_BACKEND_SEQ=1 indeed forces Plus=0.

## ROOT CAUSE FOUND (2026-07-19, static analysis confirmed)

### The bug: copy-slot/graph-slot mismatch during graph reuse

During token generation, the graph is REUSED (llama-context.cpp:1603 can_reuse).
In the reuse path:
1. `pipeline_barrier` runs (ggml-backend.cpp:3385), which ROTATES copy slots
   at lines 3484-3486: `prev_copy=cur_copy; cur_copy=next_copy; next_copy=(next_copy+1)%n_copies`
2. `graph_compute_async` runs (ggml-backend.cpp:3357), but since `is_alloc=true`
   (from the previous alloc_graph), alloc_graph is SKIPPED (line 3363)
3. `compute_splits` runs, copying split inputs to `sched->cur_copy` (rotated)
   at line 2688: `tensor_copy(input, split_backend_id, sched->cur_copy)`
4. But the GRAPH's tensor pointers are FROZEN at the alloc-time cur_copy
   (set at line 1900: `node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy)`
   during the LAST alloc_graph call)
5. MISMATCH: copy writes to rotated slot, compute reads from frozen slot -> garble

### Why single-GPU is immune

Single-GPU has zero cross-backend split inputs. The mismatch between copy slot
and graph slot is invisible because no split input copies happen.

### Why the env toggles didn't fix it

- BARRIER_PARTIAL=0: doesn't affect rotation (lines 3484-3486 are unconditional)
- PIPELINE_DEPTH=2: only affects wavefront depth guard, not rotation
- MULTI_BACKEND_SEQ=1: disables Plus entirely (ggml-backend.cpp:60)

### The fix

In `ggml_backend_sched_pipeline_barrier` (ggml-backend.cpp:3385-3487):
- When is_alloc=true (graph reuse), the barrier must NOT rotate cur_copy
- Event wait should target cur_copy (the slot the graph uses), not next_copy
- Rotation only happens in alloc_graph (which also updates graph pointers)

This matches the existing design intent at line 3377-3380:
"if the graph is not already allocated, always use copy 0 after a synchronization
this ensures that during generation the same copy is used every time"

The pipeline_barrier violates this intent by rotating during reuse.

### Trade-off

Fix sacrifices copy-slot pipelining during generation (same slot reused each
decode). The async compute overlap benefit of Plus is preserved (events still
prevent blocking), but true double-buffered pipelining is disabled. This is
the same trade-off the synchronize() path already makes. A future enhancement
could update graph tensor pointers on rotation to re-enable pipelining.

## Agent findings (condensed)

### A1 - detector patch (DONE)
- 5th arg EXPECT_SUBSTR (case-insensitive substring assert)
- reasoning_content fallback when content empty/whitespace
- content+reasoning concatenated for scoring when both present
- Smoke-tested with mocked payloads: all 6 cases pass

### A2 - hypothesis ranking (DONE)
Most likely first-bad commit: 87357519e (introduced B+16 rotation + partial barrier).
H1 (top): partial barrier wait_mask = barrier_copy_src_mask only; B+16 reads
prev_copy slot whose producer may not be in mask. Single-GPU clean because
empty mask -> full sync fallback (ggml-backend.cpp:3385-3389).
H4: B+16 sources from CONSUMER prev_copy, not PRODUCER (provenance question).
H5: n_copies=4 over-rotates for 2-GPU; GGML_SCHED_PIPELINE_DEPTH=2 tests this.
H6: Plus-gated reuse full-sync bypass (src/llama-context.cpp:1607-1609).

### A3 - repro loop script (DONE)
scripts/triton-pplus-garble-loop.sh: GPU-gated, port-clean, /health poll,
detector call, exit-code propagation, EXIT-trap server kill, log capture.

## WIP tree situation

Working tree has WIP files NOT mine (timestamps = d1df5f436 commit time):
  common/arg.cpp, common/common.cpp, common/common.h
  common/placement-plan.cpp, common/placement-plan.h
  docs/wayfinder/PLACEMENT-CONTROL-PLANE-PLAN.md
  tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp
These are placement/profiler leftovers from the prior commit. Leave alone.
Plus untracked: .rocprofv3/, Config:, benches/allreduce-baselines/,
  docs/pipeline-plus/research/placement-asymmetric-default-packer-eval.md,
  docs/rpc-multi-backend-pipeline-plus/PIPELINE_PLUS.md,
  scripts/allreduce-t2-baseline.sh, scripts/check-garble-feedback-loop.sh,
  tests/test-iu4-wmma-mmq.hip.cu
Also not mine; leave alone.

My commit scope (selective git add):
- rpc-patch/patch/loop-check-garble.sh
- scripts/triton-pplus-garble-loop.sh
- diagnostics/pipeline-plus-dual-gpu-garble/ (README, ISSUE, SCRATCHPAD)
- diagnostics/README.md (only my ticket's index row + Agent tickets section)

## Decision log

- 2026-07-19 start: chose orchestrator+agents pattern. Fixes land Romulus-first.
- 2026-07-19 prep done: 3 agents landed clean. Hypothesis ranked.
- 2026-07-19 next: run env-toggle falsification on triton BEFORE writing source
  fix. This needs NO rebuild - just env vars. Saves a build cycle.
- 2026-07-19 WIP tree: confirmed unrelated to garble ticket. Selective git add.

## Answer (filled at resolve)

<!-- root cause, fix summary, commit SHA, loop command, evidence -->
