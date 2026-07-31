# Repair path — Plus=1 TSC (not gate-off)

Companion to [BUGFIX-plus1-tsc-ROOTCAUSE.md](BUGFIX-plus1-tsc-ROOTCAUSE.md). Goal: **restore sequential consistency for recurrent state** while keeping Plus overlap on stateless paths (MoE weights, static activations).

## Frontier AI proposal — what is sound vs wrong

| Frontier claim | Verdict | Codebase truth |
|----------------|---------|----------------|
| TSC is low-entropy repetition at full t/s | **Sound** | Confirmed |
| GDN is recurrent; state must be sequential across tokens | **Sound** | `state` tensor `[S_v,S_v,H,n_seqs]` in `ggml_gated_delta_net`; updated via `ggml_cpy` into `ssm_states_all` |
| Plus async treats tensors like static weights | **Sound (mechanism)** | Copy-slot rotation + RPC defer on cross-backend splits |
| `pos_min=17` checkpoint = root cause | **Wrong** | Appears on coherent runs |
| Match `.attn_q` / `.attn_k` for GDN | **Wrong** | Use `GGML_OP_GATED_DELTA_NET`, `__fgdn_ar__-<il>`, `__fgdn_ch__-<il>` |
| Pin all GDN to ROCm0 | **Over-broad** | RS already split ROCm/RPC by layer; pin may fight tensor split |
| Keep Plus async for MoE only (Stratum 3) | **Sound goal** | Aligns with blast-radius + factorial evidence |

**Mathematical invariant to enforce:**

For each decode step `t` and GDN layer `l`, let `s_{t,l}` be recurrent state. Before evaluating GDN at step `t+1`:

```
forall backends B that wrote s_{t,l} or inputs q,k,v,g,beta at step t:
  complete(B) before read(s_{t+1,l})
```

Plus=1 violates this when `pipeline_barrier` rotates `cur_copy` with partial `wait_mask` (B+8) or RPC GET prefetch delivers step `t` inputs while step `t` GDN write still in flight on another backend.

## Gate-off vs repair

| Approach | What it does | Problem |
|----------|--------------|---------|
| `RECURRENT_GUARD` / `SCHED_LEGACY` + `P0_FULL_SYNC` | Disables Plus async bundle | Works; throws away overlap on MoE + attention |
| **Repair** | Fence only recurrent + cross-backend handoff | Preserves Plus elsewhere |

## Surgical repair tiers (implementation)

### R1 — P0 recurrent fence (llama-context.cpp, ~15 lines)

On graph reuse when model has `is_recr` layers:

```cpp
ggml_backend_sched_synchronize(sched.get());  // not pipeline_barrier
```

**Necessary** (proven). **Not sufficient** alone (`p0-only` stutters).

### R2 — Recurrent-expanded barrier mask (ggml-backend.cpp, ~25 lines)

At `ggml_backend_sched_update_barrier_src_mask` or start of `pipeline_barrier`:

- Scan allocated graph for `GGML_OP_GATED_DELTA_NET` and `GGML_OP_SSM_CONV` (conv path).
- OR backends from `llama_memory_recurrent` RS buffer assignment.
- Set `wait_mask |= recurrent_backend_mask` before `cur_copy` advance (disable partial skip for those backends).

**Frontier Stratum 1 done correctly** — op/tag based, not `.attn_q` regex.

**Preserves:** MoE defer on backends not in mask. **Cost:** extra event wait on ROCm+RPC each token at barrier only.

### R3 — GDN split input hardening (ggml-backend.cpp, ~40 lines)

In `ggml_backend_sched_compute_splits`, for splits whose first GDN/conv node is within N nodes:

- Skip `rpc_gather_prefetch_early` for that split.
- Force `ggml_backend_sched_wait_copy_slot` (not producer-deferred gather).

**Frontier Stratum 2 partial** — fence at op boundary without pinning device.

### R4 — Post-GDN backend fence (eval callback or split tail, ~20 lines)

After subgraph containing `GGML_OP_GATED_DELTA_NET` completes on backend B:

- `ggml_backend_synchronize(B)` before next split uses tensors derived from GDN output.

Narrowest sequential guarantee; highest sync cost on GDN layers only.

## Recommended repair stack (ship order)

1. **R1 + R2** behind `GGML_PIPELINE_GDN_SEQ=1` (default ON when `is_recr` detected).
2. Measure fox + b6 gate; if still stutter, add **R3**.
3. **R4** only if witness shows divergence inside same token (unlikely).

Do **not** ship blanket `SCHED_LEGACY` once R1+R2 passes blast-radius + b6.

## Witness before narrowing R2 mask

`GGML_PIPELINE_GDN_WITNESS=1`: hash `src[5]` at each `GGML_OP_GATED_DELTA_NET` via eval callback. Confirms repair fixes state not just logits.

## Blast-radius drives scope (2026-07-03 phase1)

Config: 2-GPU `ts=50,50`, `ctk=q8_0` `ctv=q8_0`, fox, `gen=64`, `runs=1`, `-np 1` (kv_unified default off).

| Model | GDN | MoE | MTP in GGUF | Plus=1 | Plus=0 |
|-------|-----|-----|-------------|--------|--------|
| qwen35moe-apex | yes | yes | no | **stutter** | coherent |
| qwen35moe-mtp | yes | yes | yes | **stutter** | coherent |
| qwen35-dense | yes | no | no | **stutter** | coherent |
| gemma4-moe | no | yes | no | **stutter** | coherent |
| gemma3-12b | no | no | no | **stutter** | coherent |
| llama3-8b | no | no | no | **stutter** | coherent |
| qwen35moe-9b-mtp | yes | yes | yes | **stutter** | coherent |

**Not MTP-inference:** MTP-weight GGUF stutters same as non-MTP; no speculative draft.

**Not GDN-only:** llama3-8b and gemma3 dense stutter — **generic multi-backend Plus=1 bug**.

**Not MoE-only:** qwen35-dense (no MoE) stutters; gemma4-moe also stutters.

**Implication for repair:** R1/R2 must apply to **`pipeline_parallel && n_devices > 1`**, not only `is_recr`. GDN-specific R3 is an **extra** fence for recurrent amplification, not the whole fix.

Phase2 (`kv_unified` on/off): pending — see `/tmp/plus1-tsc-blast-radius/run-phase2.log`.