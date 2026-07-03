# BUGFIX — Plus=1 Total Semantic Collapse (output stutter)

| Field | Value |
|-------|-------|
| **Status** | Active — identification phase (mitigation bisect) |
| **Branch** | `Path-B-Event-Support-Pipeline-Plus` @ `2393538ae` |
| **Related** | [CONTEXT.md](CONTEXT.md) (TSC glossary), [IMPLEMENTATION.md](IMPLEMENTATION.md) (mitigation flags) |
| **Ops** | [HANDOVER-CLUSTER-OPS.md](../../rpc-patch/patch/HANDOVER-CLUSTER-OPS.md) |

## Problem statement

With `GGML_PIPELINE_PLUS=1`, fox-bench generation completes at full throughput but output degrades into low-entropy repetition (`known-known-known`, `pangramramramram`, `HereHereHere`). With `GGML_PIPELINE_PLUS=0` on the same topology and binary, output is coherent.

This is **Gate B (stutter)** per TSC composite gate — not Gate A (hang). RPC protocol regression (`recv failed` / slot-init hang) was fixed separately in `2393538ae`.

**Constraint:** `GGML_PIPELINE_PLUS=0` is a diagnostic baseline only. Production must keep Plus=1; fix must preserve Path-B+ pipeline work.

## Evidence (2026-07-03 TSC matrix @ `2393538ae`)

| Topology | Plus=1 | Plus=0 |
|----------|--------|--------|
| 2-GPU | PASS hang, **stutter** (~47 t/s) | PASS, **coherent** (~46 t/s) |
| 3-GPU | PASS hang, **stutter** (~44 t/s) | PASS, **coherent** (~44 t/s) |
| 4-GPU | PASS hang, **stutter** (~45 t/s) | PASS, **coherent** (~41-44 t/s) |

Artifacts: `rpc-patch/patch/bench-results/rpc-server-bench/trace-g-*-tsc-plus*-merged.*` on romulus; local logs `/tmp/phase1e-*gpu-plus*.log`.

## Ruled out (do not re-investigate as primary cause)

| Signal | Verdict |
|--------|---------|
| RPC EVENT_RECORD / PATCH v3 hang | **Fixed** @ `2393538ae` (three commits) |
| `restored context checkpoint (pos_min=17...)` | **Normal** — appears on Plus=0 clean runs too (fox prompt checkpointing) |
| Frontier tensor names `.attn_q`, `.attn_k` | **Wrong for this fork** — GDN uses `__fgdn_ar__-<il>` / `__fgdn_ch__-<il>` (`src/models/delta-net-base.cpp`) |

## Strategy: D — mitigation bisect before design

Do **not** implement GDN pin-to-ROCm0 or name-regex guards until bisect identifies which Plus mitigation surface correlates with stutter.

**Hypothesis classes (deferred until bisect):**

- **A** Recurrent GDN state diverges across backends / copy-slots under Plus
- **B** Logits stale due to P1 narrow `synchronize_sampling()`
- **C** KV / slot position mismatch

**Working method:** one mitigation flag `=0` per run, `GGML_PIPELINE_PLUS=1` held constant. See [IMPLEMENTATION.md](IMPLEMENTATION.md) rollback table.

## Phase 1f — mitigation bisect matrix

**Gate topology:** 2-GPU fox (fastest loop) — romulus 7900 + remus `:50051`

```bash
set -a && source .scratch/cluster-access.env && set +a
./rpc-patch/scripts/pathb-plus1-tsc-mitigation-bisect.sh
```

| Arm | Env override | Mitigation | Code area |
|-----|--------------|------------|-----------|
| `canonical` | (none) | all ON | baseline stutter repro |
| `no-partial` | `GGML_PIPELINE_BARRIER_PARTIAL=0` | B+8 OFF | `ggml_backend_sched_pipeline_barrier` |
| `no-defer` | `GGML_RPC_EVENT_DEFER_BARRIER=0` | B+9 OFF | `ggml-rpc.cpp` EVENT defer |
| `no-async-copy` | `GGML_SCHED_MOE_ASYNC_COPY=0` | B+10 OFF | MoE copy-slot path |
| `no-get-defer` | `GGML_RPC_GET_TENSOR_DEFER=0` | B+12 OFF | GET recv defer |

**Bench env:** `BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_TRACE=0 GGML_PIPELINE_PLUS=1`

**Pass criterion (per arm):** fox preview coherent (pangram explanation, no morpheme loops); G within ~5% of canonical.

**Verdict rules:**

| Outcome | Next step |
|---------|-----------|
| One arm PASS coherent | Root cause narrowed to that mitigation; design minimal guard or default-OFF for that flag only |
| All arms stutter | Escalate to state-divergence witness on `GGML_OP_GATED_DELTA_NET` I/O (GDB/hash), not blanket sync |
| All arms coherent | Stutter is interaction of multiple flags; run pairwise bisect |

## After bisect (no code until here)

1. Document winning arm in this file (Results section below).
2. Grill fix design against IMPLEMENTATION.md / `ggml-backend.cpp` call sites.
3. Implement smallest change that restores coherence with Plus=1.
4. Re-run full TSC matrix D1-D3 with Plus=1.

## Results (2026-07-03 bisect @ `2393538ae`, 2-GPU fox)

| Arm | RESULT | G t/s | Stutter (B) | Notes |
|-----|--------|-------|-------------|-------|
| canonical | PASS | ~47 | **YES** | `following...HereHere` |
| no-partial | PASS | ~48 | **YES** | B+8 OFF |
| no-defer | PASS | ~47 | **YES** | B+9 OFF |
| no-async-copy | PASS | ~47 | **YES** | B+10 OFF |
| no-get-defer | PASS | ~44-47 | **YES** | B+12 OFF |
| all-off | PASS | ~47 | **YES** | all four rollback flags `=0`, Plus=1 |

Logs: `/tmp/plus1-tsc-bisect/*.log`

### P0/P1 split results (2026-07-03 @ `4ee47dcc7`, 2-GPU fox)

| Arm | P0 | P1 | G t/s | Stutter (B) |
|-----|----|----|-------|-------------|
| canonical | barrier | narrow | ~48 | **YES** |
| p0-legacy | full sync | narrow | ~47-52 | **YES** |
| p1-legacy | barrier | full sync | ~48 | **YES** |
| p0-p1-legacy | full sync | full sync | ~48 | **YES** |

Logs: `/tmp/plus1-tsc-p0p1-bisect/*.log`

**Verdict:** Tier-0 **llama-context** P0/P1 alone do not restore coherence. `p0-p1-legacy` matches Plus=0 behavior in `llama-context.cpp` but leaves `ggml_sched_pipeline_plus_enabled()` true in `ggml-backend.cpp`.

**Narrowed to:** scheduler/backend Plus path (`ggml-backend.cpp` copy-slot / split compute) while `GGML_PIPELINE_PLUS=1`.

### Bisect verdict

**No single B+8..B+12 mitigation is the stutter root cause.** Disabling all rollback flags together while keeping `GGML_PIPELINE_PLUS=1` still stutters.

**Narrowed to Tier-0 Plus path** (always active when Plus=1, not gated by per-flag env):

| ID | Code | Effect |
|----|------|--------|
| **P0** | `ggml_backend_sched_pipeline_barrier` on graph reuse | `cur_copy` rotation vs full `sched_synchronize` |
| **P1** | `llama_context::synchronize_sampling` | narrow logits sync vs full `synchronize()` |

`GGML_PIPELINE_PLUS=0` disables both (legacy full sync) and restores coherent output — consistent with P0/P1 surface.

### P0/P1 split bisect (Tier-0 knobs)

Romulus ROCm client must be rebuilt after pulling knobs in `src/llama-context.cpp`.

| Env | Effect |
|-----|--------|
| `GGML_PIPELINE_P0_FULL_SYNC=1` | Graph reuse: `sched_synchronize` instead of `pipeline_barrier` (Plus=1) |
| `GGML_PIPELINE_P1_FULL_SYNC=1` | Sampling: full `synchronize()` instead of narrow logits sync (Plus=1) |

```bash
set -a && source .scratch/cluster-access.env && set +a
# rebuild romulus ROCm client at current branch first
./rpc-patch/scripts/pathb-plus1-tsc-p0p1-bisect.sh
```

| Arm | P0 | P1 |
|-----|----|----|
| `canonical` | barrier | narrow |
| `p0-legacy` | full sync | narrow |
| `p1-legacy` | barrier | full sync |
| `p0-p1-legacy` | full sync | full sync |

### Next identification step (no production fix until identified)

1. **Sched vs client split knob** — `GGML_PIPELINE_SCHED_LEGACY=1` (proposed): force `ggml_sched_pipeline_plus_enabled()` false while `GGML_PIPELINE_PLUS=1` in llama-context; falsifies whether backend scheduler alone causes stutter.
2. If sched-legacy restores coherence: bisect inside `ggml_backend_sched_compute_splits` / copy-slot path with trace.
3. State witness on `GGML_OP_GATED_DELTA_NET` I/O if sched bisect inconclusive.
4. Do not implement frontier Stratum 2 (pin GDN to ROCm0) until bisect completes.

---

## Appendix — Frontier AI original analysis prompt (2026-07-02)

Preserved verbatim for grill / design reference. Implementation suggestions below are **unvalidated** until Phase 1f bisect completes.

```
ROLE: High-IQ Systems Architect & C++ Compiler Engineer
CONTEXT: Custom llama.cpp multi-node RPC fork (Branch: Path-B-Event-Support-Pipeline-Plus)
HARDWARE TOPOLOGY: Asymmetric heterogeneous cluster (7900 XTX Head Node ROCm, RTX 3060 Ti, RTX 5060 Ti, Windows RTX 5070 CUDA over network)
MODEL UNDER TEST: Qwen3.6-35B-A3B-APEX-I-Quality.gguf (Hybrid Gated DeltaNet Linear Attention + MoE architecture)
ACTIVE RUNTIME FLAGS: ctk=q8_0, ctv=q8_0, ngl=99, ts=36,24,24,16, GGML_PIPELINE_PLUS=1, ncmoe=none

ENVIRONMENT CRITICAL UPDATE: 
The TCP socket desynchronization (cmd_child_to_router:error due to un-drained EVENT_RECORD bytes during GRAPH_RECOMPUTE) has been completely patched via central-drain enforcement in ggml-rpc.cpp. Sockets are rock solid. Raw physical throughput is fully saturated at 43.07 t/s.

THE ISSUE: TOTAL SEMANTIC COLLAPSE (INFINITE REPETITION LOOP)
Despite hitting 43 t/s generation speed, the text token output immediately degrades into infinite, low-entropy repetition cascades during the evaluation phase:
- Example A: "...well-known-known-known-known..."
- Example B: "...ppppangangangangramramramram..."
- Example C: "...is sentence is sentence is is is is..."

DIAGNOSTIC HYPOTHESIS & MECHANICAL ROOT CAUSE:
1. Qwen3.6-35B-A3B is NOT a standard static-weight transformer. Its non-MoE blocks utilize Gated DeltaNet layers. Unlike traditional attention, DeltaNet relies on a recurrent linear attention matrix where hidden states are updated sequentially across step boundaries.
2. GGML_PIPELINE_PLUS=1 implements an aggressive asynchronous assembly loop designed to pre-fetch and interleave execution chunks to maximize cross-node hardware utilization.
3. Because the scheduler assumes all offloaded tensors represent static weights, the asynchronous overlap is executing tensor state updates out of order or dropping critical hidden state scaling coordinates across different GPU/RPC backends. 
4. The moment these recurrent state updates lose mathematical alignment, the model's global positional awareness and RoPE coordinate space collapse, trapping the sampler layer in a permanent feedback loop.
5. This is confirmed by a slot initialization warning: "restored context checkpoint (pos_min = 17, pos_max = 17, n_tokens = 18, n_past = 18)" indicating a corrupted/misaligned KV position boundary boundary step.

TARGET REPAIR PARADIGM (THE PATCH ROUTE):
We must implement a strict "Selective Synchronization Boundary" inside the tensor routing and scheduling layers. We cannot treat DeltaNet linear attention states like static MoE weights.
- Stratum 1: Identify all Gated DeltaNet state/attention tensors via string pattern matching (e.g., matching `.attn_q`, `.attn_k`, `.gate`, or explicit recurrent state cache buffers).
- Stratum 2: Force these specific tensors to execute strictly and synchronously on the primary Head Node (ROCm0, 7900 XTX) local PCIe bus. They must be explicitly barred from asynchronous pre-fetch interleaving.
- Stratum 3: Allow GGML_PIPELINE_PLUS=1 to remain asynchronous ONLY for the highly sparse, static MoE expert layers where out-of-order execution does not corrupt a recurrent hidden history state.

EXECUTION INSTRUCTION: DEBUGS -> FIX -> REVIEW -> DEBUG LOOP
1. Phase 1 (Locate & Grill): Inspect `ggml-backend-sched.c`/`.cpp`, `ggml-rpc.cpp`, and our custom `GGML_PIPELINE_PLUS` loop handler. Identify exactly where tensor evaluation order is shuffled or split across backends.
2. Phase 2 (Design the Guard): Draft a structural intercept that reads tensor names or allocation context during the graph compile phase. If a tensor belongs to the recurrent attention state path, force a synchronous backend fence or lock it to the local head node.
3. Phase 3 (Review & Dry-Run): Audit the patch for race conditions. Ensure the introduction of these synchronous fences doesn't re-introduce the TCP socket desync or induce a deadlocking pipeline stall.
4. Phase 4 (Compile & Verification): Generate the precise C++ patch lines ready for direct ingestion into our build targets.

Let's step directly into Grill Mode. Inspect the scheduling loop boundaries first and point out exactly where the out-of-order state leakage is occurring.
```

**Codebase notes on frontier Stratum 1-3 (for post-bisect design only):**

- GDN recurrent state: `ggml_gated_delta_net(..., s, K=1)` in `src/models/delta-net-base.cpp`; node op `GGML_OP_GATED_DELTA_NET`.
- Plus barrier: `ggml_backend_sched_pipeline_barrier` in `ggml/src/ggml-backend.cpp` — `cur_copy` rotation, B+8 partial `wait_mask` from `barrier_copy_src_mask`.
- P1 sampling: `llama_context::synchronize_sampling()` in `src/llama-context.cpp` — narrow logits sync when Plus=1.