# ROOT CAUSE ANALYSIS — Plus=1 TSC (semantic stutter)

| Field | Value |
|-------|-------|
| **Date** | 2026-07-03 |
| **Branch** | `Path-B-Event-Support-Pipeline-Plus` @ `09e3f0fe3` |
| **Evidence** | [BUGFIX-plus1-tsc-semantic-collapse.md](BUGFIX-plus1-tsc-semantic-collapse.md), [BUGFIX-plus1-tsc-REDTEAM.md](BUGFIX-plus1-tsc-REDTEAM.md) |
| **Status** | Mechanistic model + fix candidates (not ship-ready) |

## 0. Model specificity (MTP / MoE / GDN) — grounded proof

### 0.1 What the TSC bench actually runs

Romulus server log (`trace-g-2gpu-tsc-pair-canonical-server.log`):

| Field | Value | Implication |
|-------|-------|-------------|
| `general.architecture` | `qwen35moe` | MoE trunk, not dense qwen35 |
| `block_count` / `n_layer_all` | **40** | No extra MTP block in this GGUF |
| `n_layer_nextn` | **0** (implicit: `n_layer_all` = trunk) | **MTP weights not present** |
| `full_attention_interval` | 4 | 30 GDN + 10 full-attn layers |
| `expert_count` / `expert_used` | 256 / 8 | MoE routing active every layer |
| `draft: 0.000 MiB` | speculative init has no draft | **MTP decode path not used** |
| `common_speculative_init: no implementations` | — | No speculative/MTP inference |
| RS buffers | ROCm0 29.31 MiB + RPC 33.50 MiB | **GDN recurrent state split across backends** |
| `pipeline parallelism enabled` | yes | `n_copies=4`, graph reuse 272/token |
| Bench `ncmoe` | none (omit `--n-cpu-moe`) | Experts stay on GPU; MoE `MUL_MAT_ID` still runs |

Main decode graph (`qwen35moe.cpp`):

```180:181:src/models/qwen35moe.cpp
    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
```

MTP graph is separate (`LLM_GRAPH_TYPE_DECODER_MTP` / `graph_mtp`) and only runs when draft/speculative context is active.

### 0.2 Static simulation

```bash
python3 rpc-patch/scripts/pathb-plus1-tsc-graph-sim.py --n-layer 40 --n-layer-nextn 0
# -> 30 gdn_layers, 10 full_attn_layers, moe_ffn_all_layers=true, mtp_in_main_pass=false
```

Dense qwen35 27B (`--n-layer 64 --dense`): 48 GDN + 16 full-attn, **no MoE** — use as falsification target on cluster.

### 0.3 Specificity verdict

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| **MTP / NextN execution bug** | **Ruled out** for TSC bench | `n_layer_nextn=0`, no draft, main loop `il < n_layer` only |
| **MoE-only bug** | **Unlikely primary** | `ncmoe=none` still routes 8 experts/layer; original B+10 bisect alone did not fix; MoE is necessary but not sufficient |
| **GDN + cross-backend recurrent** | **Primary** | 30 GDN layers; RS split ROCm/RPC; Plus copy-slot + defer hits recurrent input path |
| **Generic Plus bug (any transformer)** | **Open** | Need dense non-GDN or pure-attention model A/B |

**Conclusion:** TSC on `APEX-I-Quality.gguf` is a **qwen35moe GDN + multi-backend** issue, **not** an MTP-inference issue. MoE adds async `MUL_MAT_ID` surface area but is not the sole root cause. MTP-specific guards would not fix this bench.

### 0.5 Blast-radius matrix (phase1+2, 2026-07-03)

Script: `rpc-patch/scripts/pathb-plus1-tsc-blast-radius.sh`. Logs: `/tmp/plus1-tsc-blast-radius/`.

**Fixed baseline:** `ctk=q8_0`, `ctv=q8_0`, 2-GPU `ts=50,50`, fox, `gen=64`, `runs=1`, `ncmoe=none`.

Human-audited previews (GATE_B unreliable on word-split repetition):

| Model | GDN | MoE | MTP GGUF | Plus=1 | Plus=0 |
|-------|-----|-----|----------|--------|--------|
| qwen35moe-apex | yes | yes | no | **stutter** | coherent |
| qwen35moe-mtp | yes | yes | yes | **stutter** | coherent |
| qwen35-dense | yes | no | no | **stutter** | coherent |
| gemma4-moe | no | yes | no | **stutter** | coherent |
| gemma3-12b | no | no | no | **stutter** | coherent |
| llama3-8b | no | no | no | **stutter** | coherent |
| qwen35moe-9b-mtp | yes | yes | yes | **stutter** | coherent |

**kv_unified (phase2, Plus=1):** `--kv-unified` vs `--no-kv-unified` — **no effect** (apex, dense, llama3 still stutter).

**Blast-radius verdict:**

| Scoped hypothesis | Result |
|-----------------|--------|
| MTP inference / NextN graph | **Ruled out** — no draft; MTP-weight GGUF same as non-MTP |
| qwen35moe only | **Ruled out** — gemma + llama affected |
| MoE only | **Ruled out** — qwen35-dense stutters |
| GDN only | **Ruled out** — llama3-8b, gemma3 stutter |
| **Multi-backend `pipeline_parallel` + Plus=1** | **Confirmed** — all loaded models show Plus=1 stutter, Plus=0 coherent |

GDN recurrent state is an **amplifier** (worst on qwen35 family) not the **exclusive** trigger.

### 0.4 S factorial @ `P0_FULL_SYNC=1` (2026-07-03)

Script: `rpc-patch/scripts/pathb-plus1-tsc-s-factorial.sh`. Logs: `/tmp/plus1-tsc-s-factorial/`.

| Arm | Extra env | GATE_B | Human audit |
|-----|-----------|--------|-------------|
| `p0-only` | (none) | stutter | stutter |
| `no-get` | `GET_DEFER=0` | stutter | stutter |
| `no-event` | `EVENT_DEFER=0` | stutter | stutter |
| `no-partial` | `BARRIER_PARTIAL=0` | stutter | stutter |
| `no-moe-copy` | `MOE_ASYNC_COPY=0` | coherent | **stutter** (`!!` loops) — GATE_B false positive |
| `no-get-no-event` | both OFF | stutter | stutter |
| `no-get-no-partial` | both OFF | stutter | stutter |
| `all-s-off` | all four S flags OFF | stutter | stutter |
| `sched-legacy` | `SCHED_LEGACY=1` | coherent | **coherent** (pangram) |

**Factorial lesson:** No single S flag decomposition matches `SCHED_LEGACY` coherence. Individual `=0` arms keep other Plus async paths on; combined `all-s-off` still stutters without master `SCHED_LEGACY`. The sched master switch is an **atomic bundle**, not a simple sum of parts.

---

## 1. Symptom recap

- `GGML_PIPELINE_PLUS=1`: fox bench completes at ~45-48 t/s but output collapses into low-entropy loops.
- `GGML_PIPELINE_PLUS=0`: coherent pangram explanation at similar throughput.
- Not a hang (Gate A PASS). Not a throughput cliff. **Semantic state corruption.**

## 2. Decode-step timeline (where bugs can live)

Single-token generation (`n_tokens=1` per `llama_decode`) with graph reuse (`graphs reused` ~250+ in logs):

```
llama_decode()
  process_ubatch()
    [P0] graph reuse gate:
         Plus=1 default -> ggml_backend_sched_pipeline_barrier()  # rotates cur_copy
         P0_FULL_SYNC   -> ggml_backend_sched_synchronize()       # no rotation
    ggml_backend_sched_compute_splits()  # async S surface: partial barrier, RPC defer, copy slots
    [P1] synchronize_sampling() on llama_get_logits:
         Plus=1 default -> sync logits-backend only
         P1_FULL_SYNC   -> full synchronize()
  sample / server emits token
```

**Three coupled surfaces** (bisect + pairwise):

| ID | Knob | Code |
|----|------|------|
| **S** | `GGML_PIPELINE_SCHED_LEGACY=1` | `ggml_sched_pipeline_plus_enabled()`, `rpc_pipeline_plus_enabled()` |
| **P0** | `GGML_PIPELINE_P0_FULL_SYNC=1` | `process_ubatch` graph-reuse path |
| **P1** | `GGML_PIPELINE_P1_FULL_SYNC=1` | `synchronize_sampling()` |

**Pairwise winner:** `sched-p0-legacy` = S legacy + P0 full sync + P1 narrow. **Coherent with Plus=1.**

## 3. Mechanistic model (working hypothesis)

### 3.1 Pipeline parallelism primitives

When `pipeline_parallel` is enabled (`llama-context.cpp`):

- `n_copies = GGML_SCHED_MAX_COPIES` (4 in this build).
- Cross-backend split inputs get per-copy tensor duplicates (`tensor_id_copy`, `ggml_set_input`).
- `pipeline_barrier` advances `cur_copy` after waiting on `events[backend][new_copy]` with `wait_mask`.
- **B+8 partial barrier:** `wait_mask = barrier_copy_src_mask` (cross-backend producers only), not all backends.

```2946:3046:ggml/src/ggml-backend.cpp
void ggml_backend_sched_pipeline_barrier(ggml_backend_sched_t sched) {
    // ...
    uint32_t wait_mask = (1u << sched->n_backends) - 1;
    if (ggml_sched_barrier_partial_enabled() && sched->barrier_copy_src_mask != 0) {
        wait_mask = sched->barrier_copy_src_mask;
        // ...
    }
    // event wait on wait_mask, RPC defer drain if enabled
    sched->cur_copy  = new_copy;
    sched->next_copy = (new_copy + 1) % sched->n_copies;
}
```

### 3.2 GDN / recurrent path (Qwen3.6)

GDN layers (`hparams.is_recr(il)`) use **two** recurrent stores per layer:

1. **`ssm_states_all`** — DeltaNet matrix state `s`, read via `build_rs`, updated after `GGML_OP_GATED_DELTA_NET`.
2. **`conv_states_all`** — SSM conv ring, via `build_conv_state`.

```360:414:src/models/qwen35moe.cpp
ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
// ... ggml_gated_delta_net / build_recurrent_attn writes back via ggml_cpy into ssm_states_all
```

These buffers are **persistent memory**, not per-copy split buffers. Corruption is therefore unlikely to be "wrong copy slot of ssm_states" directly. More plausible:

- **Cross-backend activation copies** (hidden states, conv/qkv tensors) use `cur_copy` and async RPC GET defer.
- GDN reads **garbage inputs** or **stale prior-token activations** while recurrent stores get updated from those bad inputs.
- Low-entropy loops are a downstream sampler symptom, not the first divergence site.

### 3.3 Reconciling pairwise arms (critical)

| Arm | P0 at reuse | S during compute | Result |
|-----|-------------|------------------|--------|
| `canonical` | `pipeline_barrier` (partial wait + rotate) | Plus async | STUTTER |
| `p0-legacy` | **full sync** | Plus async | STUTTER |
| `sched-p1-legacy` | `pipeline_barrier` (full wait + rotate) | Plus **off** | STUTTER |
| `sched-p0-legacy` | **full sync** | Plus **off** | COHERENT |

**Implications:**

1. **P0 full sync alone is not sufficient** (`p0-legacy`). Corruption happens during `compute_splits` when S is async.
2. **S legacy alone is not sufficient** (`sched-p1-legacy`). `pipeline_barrier` copy rotation at reuse still breaks coherence even with full `wait_mask`.
3. **Both S legacy and P0 full sync are required** for the observed 2-GPU fix. Interaction, not independent bugs.

**Refined root cause statement:**

> Plus=1 TSC is a **cross-token ordering bug** at the intersection of (a) copy-slot rotation / partial quiescence at graph reuse (P0) and (b) async split execution with RPC tensor defer and partial producer waits (S). GDN-heavy models amplify it because recurrent updates compound small per-step input errors into collapsed logits.

P1 narrow sync is **not** on the critical path (`sched-p0-legacy` coherent with P1 async).

## 4. Ruled-in / ruled-out causes

| Cause | Verdict |
|-------|---------|
| RPC PATCH v3 / EVENT_RECORD hang | **Ruled out** (Phase 1e) |
| Single B+8..B+12 flag | **Ruled out** (all-off still stutters) |
| `pos_min=17` checkpoint | **Ruled out** (coherent arms too) |
| Logits-only stale (P1) | **Weak** — p1-legacy stutters but sched-p0 keeps P1 narrow |
| Partial `pipeline_barrier` wait_mask | **Contributing** — canonical path |
| `cur_copy` rotation at reuse without full drain | **Contributing** — sched-p1-legacy |
| RPC GET/EVENT defer during gather | **Contributing** — p0-legacy vs sched-p0 |
| Direct `ssm_states` copy-slot aliasing | **Unlikely** — persistent mem, not INPUT-flag dup |
| GDN on wrong backend (frontier pin) | **Unproven** — auto_fgdn checks device match at load |

## 5. Investigation tools (audited from red-team)

Priority order. **Do not ship a fix until at least #1 and #2 complete.**

### 5.1 GDN state / activation witness (P0 — must do)

**Goal:** Prove whether divergence is recurrent `s`, conv state, or upstream activations.

**Hook:** `ggml_backend_sched_set_eval_callback` already splits graph compute at tensor granularity and **forces `ggml_backend_synchronize(split_backend)`** after each sub-graph (`ggml-backend.cpp` ~2659).

**Minimal witness patch (debug-only, ~40 lines):**

1. In `llama_context` ctor after `sched_reserve`, register callback when `GGML_PIPELINE_GDN_WITNESS=1`.
2. On `GGML_OP_GATED_DELTA_NET` node `before=false`, hash `src[5]` (`s`) and optionally `src[0..2]` via `ggml_backend_tensor_get` on host.
3. Log `{decode_id, trace_id, il, token_idx, hash_s, hash_q}` to `GGML_PIPELINE_GDN_WITNESS_FILE`.

**Run:**

```bash
# canonical vs sched-p0-legacy, 2-GPU fox, stop at first preview divergence (~run 1 token 20-30)
GGML_PIPELINE_GDN_WITNESS=1 GGML_PIPELINE_TRACE=1 GGML_SCHED_TRACE=1 \
  ./rpc-patch/scripts/pathb-romulus-2gpu-bench.sh witness-canonical
```

**Pass/fail:** If hashes diverge at same `(trace_id, step)` between arms before preview diverges → GDN state path confirmed. If `s` matches but `q` differs → activation copy bug.

### 5.2 Split-S factorial (P0 — must do)

`SCHED_LEGACY` couples sched + RPC. Run **untested** arms on 2-GPU fox:

| Arm | Env |
|-----|-----|
| `p0-only` | `P0_FULL_SYNC=1` (baseline stutter) |
| `p0-no-get` | `P0_FULL_SYNC=1 GGML_RPC_GET_TENSOR_DEFER=0` |
| `p0-no-event` | `P0_FULL_SYNC=1 GGML_RPC_EVENT_DEFER_BARRIER=0` |
| `p0-no-partial` | `P0_FULL_SYNC=1 GGML_PIPELINE_BARRIER_PARTIAL=0` |
| `p0-no-get-no-event` | both OFF + P0_FULL |

**Script:** extend `pathb-plus1-tsc-pairwise-bisect.sh` or add `pathb-plus1-tsc-s-factorial.sh`.

**Purpose:** Find minimal S subset that + P0_FULL restores coherence without full `SCHED_LEGACY`.

### 5.3 Trace join (P1 — high value, existing infra)

```bash
export GGML_SCHED_TRACE=1 GGML_RPC_TRACE=1 GGML_PIPELINE_TRACE=1
# run canonical + sched-p0-legacy single fox run
bash rpc-patch/scripts/pathb-rpc-trace-parse.sh ...
```

Compare per `trace_id`:

- `pipeline_barrier_mask.wait_mask` at reuse (canonical partial vs full).
- `rpc_gather_prefetch_early` / `rpc_defer_flush` before GDN splits.
- `copy_from` / `copy_to` rotation pattern.

### 5.4 Full-output bench gate (P1)

Extend `rpc-server-bench.sh` to save full 128-token generation (not 200-char preview). Add per-run GATE_B + optional trigram entropy.

### 5.5 Topology confirmation (P2 — before ship)

Re-run minimal winning env on **3-GPU** TSC cell. Pairwise was 2-GPU only.

### 5.6 Non-GDN control (P3 — optional)

If a pure-attention model is available on cluster, Plus=1 canonical. Stutter → generic Plus bug; coherent → GDN-specific guard scope.

## 6. Fix tiers (after blast-radius + S factorial)

See [BUGFIX-plus1-tsc-REPAIR-PATH.md](BUGFIX-plus1-tsc-REPAIR-PATH.md) for surgical repair (not gate-off).

### Tier 0 — Config (proven interim)

`P0_FULL_SYNC=1` + `SCHED_LEGACY=1` or `Plus=0`.

### Tier 1 — Gate-off auto (interim ship)

`RECURRENT_GUARD` / `MULTI_BACKEND_PLUS_GUARD` when `pipeline_parallel && n_devices > 1` (widened from GDN-only after blast-radius).

### Tier 2 — Surgical repair (target)

| ID | Fix | Scope |
|----|-----|-------|
| **R1** | `sched_synchronize` on graph reuse (no `cur_copy` rotate) | all multi-backend pipeline |
| **R2** | Full `wait_mask` at barrier (no B+8 partial skip) | all multi-backend pipeline |
| **R3** | GDN/conv split: no RPC GET prefetch + full copy-slot wait | `is_recr` models extra |

**R1+R2** implements frontier invariant ("complete before consume handoff") without disabling all Plus defer globally.

### Tier 3 — Deferred

Pin GDN to ROCm0; MTP-specific code (blast-radius ruled out).

## 7. Recommended path

1. Implement **R1+R2** behind env flag; default ON for `pipeline_parallel && n_devices > 1`.
2. Add **R3** for `is_recr` if b6 overlap still short.
3. Verify: human fox preview + 3-GPU TSC + b6 gate.

## 8. Devil's advocate (revised)

| Attack | Response |
|--------|----------|
| "T1 is just SCHED_LEGACY in disguise" | Yes. Factorial proved no smaller S decomposition at P0_FULL fixed. |
| "no-moe-copy looked like a fix" | GATE_B false positive; human audit shows `!!` stutter. |
| "MTP might matter on UDT-MTP GGUF" | Different artifact; guard predicate is GDN in main pass, not MTP weights. |
| "MoE-free model might not need guard" | Falsify with qwen35 27B dense on cluster. |
| "T1 kills Plus gains" | sched-p0-legacy holds ~45-46 t/s; overlap loss bounded but must measure on b6. |
| "Why not Plus=0" | Violates Path-B+ constraint; T1 keeps Plus=1 flag and non-recurrent future path open. |

## 9. What not to do

- Do not merge permanent `SCHED_LEGACY=1` as the fix.
- Do not implement tensor-name regex from frontier prompt (`.attn_q` wrong for this fork).
- Do not pin GDN to ROCm0 before witness + factorial.
- Do not treat GATE_B heuristic as ship gate without full-output capture.

## 10. Doc cross-links

- Pairwise results: [BUGFIX-plus1-tsc-semantic-collapse.md](BUGFIX-plus1-tsc-semantic-collapse.md)
- Evidence critique: [BUGFIX-plus1-tsc-REDTEAM.md](BUGFIX-plus1-tsc-REDTEAM.md)
- Mitigation map: [IMPLEMENTATION.md](IMPLEMENTATION.md)