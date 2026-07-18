# AllReduce optimization plan (locked)

**Date:** 2026-07-17  
**Map:** `.scratch/allreduce-timing/map.md`  
**Ticket:** [Rank residual gap and lock ALLREDUCE plan](../../.scratch/allreduce-timing/issues/08-rank-gap-lock-plan.md)  
**Status:** **LOCKED** (planning complete; implementation of speedups is a follow-on effort)

This document is the destination of the AllReduce wayfinder map. It ranks residual performance gaps from measured baselines and recommends follow-on work. It does **not** implement AR kernel changes.

---

## 1. Executive summary

| Question | Locked answer |
|----------|----------------|
| Where is residual gap on **layer / Path-D** (T1, T3)? | **Orchestration / RPC / straggler**, not AllReduce. `split_timing` shows compute-dominated heavy splits; multi-node LAN adds idle. |
| Where is residual gap on **tensor** (T2)? | **AllReduce + TP overhead**: tensor TG loses to single-GPU and dual **layer** on the measured model; among AR providers **internal wins TG**. |
| Production rail by topology? | See section 4. |
| Hybrid TP-unit + pipeline now? | **No-go for immediate execution** as an AR win; optional later via **placement map** if VRAM/layout needs it. |
| Top follow-on levers? | See section 6 (ranked). |

**Bottom line:** Do **not** treat “improve AllReduce” as the primary Path-D fleet win. Keep **layer** for heterogeneous multi-GPU VRAM. On dual-CUDA tensor islands, prefer **`GGML_CUDA_ALLREDUCE=internal`** over butterfly; treat NCCL as ops-dependent and re-validate on large models before preferring it.

---

## 2. Evidence base (sources)

| Asset | Role |
|-------|------|
| `docs/research/allreduce-providers-inventory.md` | NCCL / internal / butterfly / knobs |
| `docs/research/allreduce-tensor-mode-model-allow-list.md` | Arch gate (qwen35moe, gemma4, qwen3next OK) |
| `docs/research/allreduce-topology-feasibility.md` | T0–T5 rails |
| `docs/research/allreduce-instrumentation-design.md` | Trace schema (shipped) |
| `docs/research/allreduce-multi-topology-baselines.md` | T0/T1/T2/T3 numbers |
| `benches/allreduce-baselines/*` | Raw logs + AR jsonl |

### Caveats (do not over-claim absolute TG)

- Romulus T0/T1/T3 client was often **debug + asserts** (absolute t/s depressed vs Path-D release ~143 t/s lore).
- T2 used **Qwen3.5-0.8B** (smoke); map-target 35B not run on dual-CUDA this effort.
- NCCL host AR timers are **async-enqueue**; rank providers by **E2E TG/PP**.

---

## 3. Residual gap ranking

### 3.1 Layer rail (T1 local dual, T3 multi-node)

| Observation | Evidence |
|-------------|----------|
| Single ROCm beats dual layer TG for small dense models (debug) | T0 Llama 105 vs T1 ~65 vs T3a remus ~57 |
| Idle << compute on heavy split (co-located) | T1 split_timing: idle hundreds us, compute ms–tens of ms |
| LAN remus increases idle | T3a Llama idle p50 ~1 ms; Gemma ~5 ms |
| Adding remus to local dual does not help TG vs T1 | T3b Llama ~65 ≈ T1 |
| No specialized AllReduce on these paths | Feasibility: mixed HIP+RPC / multi-node |

**Gap label:** **G1 — Layer orchestration / placement / RPC cost**  
Owner for most of G1: Path-D ops + **placement control plane** sibling map (not AR kernels).

### 3.2 Tensor rail (T2 3090+3070)

| Observation | Evidence |
|-------------|----------|
| Best TG: single 3090 | T0 455–457 t/s (0.8B, release) |
| Dual layer near single 3090 | Layer ~437–455 |
| Tensor internal TG ~278 | Loses ~39% vs single 3090 |
| Tensor butterfly TG ~225–226 | Worse than internal |
| Tensor NCCL TG ~222 (after install) | True `provider=nccl`; worst TG despite low host AR us |
| AR host p50: nccl 5 / internal 38 / butterfly 71 | NCCL async undercount |

**Gap label:** **G2 — Tensor AR + TP overhead**  
Only relevant when deliberately using `-sm tensor` on same-process multi-CUDA.

### 3.3 Cross-gap priority (fleet)

| Priority | Gap | Why |
|----------|-----|-----|
| **P0** | G1 layer / placement | Production Path-D is layer+RPC; VRAM needs all cards including 8 GB |
| **P1** | G2 only if tensor is a product path | Today tensor loses TG on measured dual-CUDA; AR provider choice still matters if TP is used |
| **P2** | Hybrid TP-unit | Not justified as TG win from current data; design lives on placement map |

---

## 4. Production rail policy (locked)

| Topology | Prefer | Avoid |
|----------|--------|--------|
| **T1** Romulus HIP + local CUDA RPC | **Layer** Path-D | Client `-sm tensor` over HIP+RPC |
| **T3** multi-node RPC | **Layer** Path-D | End-to-end tensor / cross-RPC AR |
| **T2** dual CUDA same process | **Layer** or **single fastest GPU** for TG; tensor only for experiments / VRAM-shard TP | Butterfly as default |
| **Tensor on T2** (if used) | `GGML_CUDA_ALLREDUCE=internal` until large-model NCCL re-proof | `none`/butterfly for production |

**Pareto rule (map success bar):** Do not recommend a rail or provider that regresses TG or PP beyond noise relative to the best measured alternative on that topology.

---

## 5. Hybrid topology go/no-go

| Option | Decision |
|--------|----------|
| Ship hybrid “3090+3070 TP-unit pipelined to 7900” **now** as AR optimization | **NO-GO** |
| Keep hybrid / RPC TP-unit as **architecture candidate** | **YES** — on [Placement control plane](../../.scratch/placement-control-plane/map.md) |
| Reopen hybrid if | Larger models show tensor TG ≥ layer on T2 **and** VRAM needs multi-node fill |

Rationale: T2 tensor underperforms layer/single; T3 layer residual is LAN/orchestration; VRAM aggregation is placement, not AR.

---

## 6. Ranked follow-on levers (recommend only)

Execution is **out of this map**. Order is ROI for this fleet given measurements.

| Rank | Lever | Type | Rationale | Owner |
|------|-------|------|-----------|--------|
| **1** | Placement control plane (capacity query, plan IR, hot/cold) | System | G1 dominant; `-ts`/`-fit` brittle; 8 GB cards | Placement map |
| **2** | Layer/Path-D stall reduction (events, RPC drain, straggler `-ts`) | Pipeline | T1/T3 residual; already partially shipped (D6.10 etc.) | Path-D ops |
| **3** | Keep dual-CUDA default AR = **internal**; document NCCL install + re-bench on 35B | Ops / config | T2 TG: internal > NCCL/butterfly | Ops + small doc |
| **4** | Re-bench T2 tensor on **Gemma-4 / Qwen3.6-35B** release | Measurement | 0.8B may not represent large TP | Bench follow-up |
| **5** | AR micro-opts (chunk thresholds, BF16 wire knobs) **only if** #4 shows AR-bound TG | Kernel | Premature until large-model tensor is competitive | Future |
| **6** | HIP internal AR / RCCL enablement | Backend | No multi-AMD same-process production path today | Low / defer |
| **7** | Distributed AR over RPC | New subsystem | Explicitly rejected as first expansion | Out of scope |

**Do not prioritize:** NCCL-only optimization on consumer PCIe dual-GPU without large-model proof; global heterogeneous client tensor.

**Agent slices (published):** `docs/tickets/allreduce-followon-slices.md` — Track A (F1–F4, F6 gated) + Track B placement (P1–P5).

---

## 7. Instrumentation (already shipped — keep)

| Feature | How |
|---------|-----|
| Split idle/compute | `GGML_SCHED_TRACE=1` → `event=split_timing` |
| AR provider + duration | `GGML_ALLREDUCE_TRACE=1` (+ `_FILE`) |
| Force provider | `GGML_CUDA_ALLREDUCE=nccl\|internal\|none` |

Use **TG/PP + traces** for any future lever acceptance; do not accept AR host us alone for NCCL.

---

## 8. Acceptance criteria for a future “AR speedup” PR (if any)

A change claims success only if:

1. Measured on **T2 dual-CUDA release** build, model ≥ map-target class when possible.  
2. **TG and PP** both non-regress vs best baseline (layer or single or prior tensor) beyond multi-run noise.  
3. Provider identity logged (`allreduce_provider`).  
4. Does not strand 8 GB cards in the fleet story (layer/placement still owns VRAM pool).

---

## 9. Open follow-ups (not blocking this lock)

- Release re-baseline of T0/T1/T3 absolute TG.  
- T2 large-model tensor matrix (Gemma-4, Qwen3.6-35B).  
- qwen35moe numerical quality under `-sm tensor`.  
- Layer dual CUDA crash flakes (mmq) seen mid-run — separate bug if reproducible.  
- Placement map continues independently.

---

## 10. Decision log (this lock)

| Decision | Choice |
|----------|--------|
| Primary fleet residual | G1 layer/RPC/placement |
| Tensor AR provider default (T2) | **internal** |
| Butterfly | Debug / last resort only |
| NCCL | Optional; re-prove on large models; do not prefer from 0.8B TG |
| Hybrid TP-unit now | **No-go** (placement map may design later) |
| Map complete | Yes — destination plan file written |

---

## References

- Wayfinder map: `.scratch/allreduce-timing/map.md`  
- Baseline master: `docs/research/allreduce-multi-topology-baselines.md`  
- Sibling: `.scratch/placement-control-plane/map.md`  
