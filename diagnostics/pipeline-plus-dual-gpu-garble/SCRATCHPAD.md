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
- triton access: `sshpass -p 12345 ssh hunter@192.168.8.23` (verified working).
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

triton ssh: works via `sshpass -p 12345 ssh hunter@192.168.8.23`.

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
- [ ] 5. Commit tooling (selective add: detector, loop script, diag dir, index)
- [ ] 6. Push to gitea + github; pull on triton
- [ ] 7. triton Phase 1 RED/GREEN loop with patched detector
- [ ] 8. triton Phase 3 falsification matrix (env toggles, NO rebuild needed):
       - Plus=1 baseline (RED sanity)
       - Plus=1 + GGML_PIPELINE_BARRIER_PARTIAL=0   (tests H1: mask too narrow)
       - Plus=1 + GGML_SCHED_PIPELINE_DEPTH=2       (tests H5: over-rotation)
       - Plus=1 + GGML_PIPELINE_MULTI_BACKEND_SEQ=1 (tests H6: reuse bypass)
       - Plus=0 baseline (GREEN control)
- [ ] 9. Synthesize source fix from winning hypothesis
- [ ] 10. Commit fix; push; pull on triton; rebuild; verify loop GREEN
- [ ] 11. Phase 6: README.md + ISSUE.md Answer; final commit

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
