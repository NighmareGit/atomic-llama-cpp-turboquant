# Handover: Plus=1 TSC spike + open-points validation (2026-07-03)

**Purpose:** Save project state after TSC identification, repair spike, and open-point validation. Next session: surgical R3, default-on policy, b6 overlap re-gate.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus` @ `6ac048b97`  
**Mission root:** [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/)  
**Prior handover:** [HANDOVER-SESSION-2026-07-01-phase-d.md](HANDOVER-SESSION-2026-07-01-phase-d.md)

| Location | Path | Notes |
|----------|------|-------|
| **romulus (primary client)** | `hunter@192.168.8.108` -> `~/atomic-llama-cpp-turboquant` | 5-GPU prod client; benches run here |
| **remus** | `hunter@192.168.8.176` | RPC0 5060 `:50051` |
| **triton** | `hunter@192.168.8.23` | RPC2 3090 `:50054`, RPC3 3070 `:50055` |
| **dev / edit** | `/home/hunter/projects/atomic-llama-cpp-turboquant` | this repo |

**Remotes (both synced @ `6ac048b97`):**

- Gitea: `gitea/Path-B-Event-Support-Pipeline-Plus`
- GitHub: `github/Path-B-Event-Support-Pipeline-Plus` (`NighmareGit/atomic-llama-cpp-turboquant`)

---

## Session summary (done)

### Problem: Plus=1 Total Semantic Collapse (TSC)

**Symptom:** `GGML_PIPELINE_PLUS=1` -> full generation t/s but semantic stutter/repetition; `Plus=0` coherent.

**Blast radius (generic, not MTP/MoE/GDN-only):** `pipeline_parallel` + multi-backend (ROCm+RPC) + `Plus=1`. GDN amplifies on qwen35; llama3/gemma3 dense also stutter. `kv_unified` on/off has no effect.

**Interim bisect arm (gate-off):** `sched-p0-legacy` (`GGML_PIPELINE_SCHED_LEGACY=1` + `GGML_PIPELINE_P0_FULL_SYNC=1`) restores coherence; disables Plus async bundle.

### Spike fix shipped (`9eb10c5ca`)

**Knob:** `GGML_PIPELINE_MULTI_BACKEND_SEQ=1`

**Effect (R1+R2 bundle):**

- R1: `llama_pipeline_reuse_full_sync()` -> full `ggml_backend_sched_synchronize` on graph reuse (no `cur_copy` rotation).
- R2: `ggml_pipeline_multi_backend_seq_enabled()` disables Plus async in sched + RPC (same compute path as `SCHED_LEGACY`).

**Code touchpoints:**

| File | Change |
|------|--------|
| `ggml/include/ggml-backend.h` | `ggml_pipeline_multi_backend_seq_enabled()` |
| `ggml/src/ggml-backend.cpp` | env read; disables `ggml_sched_pipeline_plus_enabled` when set |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | disables `rpc_pipeline_plus_enabled` when set |
| `src/llama-context.cpp` | `llama_pipeline_reuse_full_sync()` OR multi_backend_seq |

**Not yet shipped:** selective R3 (GDN split input hardening), `GGML_PIPELINE_GDN_WITNESS=1`, auto-default when `n_devices > 1`.

### Open-points validation (`6ac048b97`)

Script: `rpc-patch/scripts/pathb-plus1-tsc-open-points.sh`  
Prompts: `rpc-patch/bench-prompts/tsc-spike-validation.json` (8 prompts: fox, math, logic, coding, human_eval, long_context, long_gen, niche_json)  
Logs: `/tmp/plus1-tsc-open-points/` (local); romulus artifacts under `rpc-patch/patch/bench-results/rpc-server-bench/`

**Fixed baseline:** `ctk=q8_0`, `ctv=q8_0`, `ctx=8192`, `-np 1`, `--fit off`, `--reasoning off`

| Test | canonical-plus1 | seq-repair (`MULTI_BACKEND_SEQ=1`) |
|------|-----------------|-------------------------------------|
| 8-prompt suite (apex, 2-GPU ts=50,50) | 0/8 | **8/8** |
| Multi-turn KV fill (single slot -> 8192) | 2453 tok, stutter | **8149 tok PASS** (truncated @ 8191) |
| llama3-8b cross-arch | 2/8 | **7/8** (niche_json: truncated Python, not TSC) |
| gemma3-12b cross-arch | 0/8 | **8/8** |
| 3-GPU `primary-5060-3060` apex | 0/8 | **8/8** @ ~42-44 G t/s |

**Discard:** `seq-repair-v2` run (broken prompt parser).

**Spike validation (prior commit):** 2-GPU fox suite 8/8 under `MULTI_BACKEND_SEQ=1`. Logs: `/tmp/plus1-tsc-spike-validate/`.

### Frontier AI report verdict

| Claim | Verdict |
|-------|---------|
| Cross-token ordering invariant for recurrent state | **Sound** |
| MoE can stay async if handoffs fenced | **Sound goal** |
| `ggml-backend-sched.cpp`, `.attn_q`/`.attn_k` matching, `pos_min=17` root cause | **Wrong** |
| Pin all GDN to ROCm0 | **Over-broad** |
| KV rollback as fix | **Wrong** |

**Target repair path:** R1+R2 ship path (done as spike); R3 op-based GDN fences if overlap insufficient.

### Docs shipped

| Doc | Role |
|-----|------|
| [BUGFIX-plus1-tsc-semantic-collapse.md](../../docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-semantic-collapse.md) | TSC glossary + bisect matrix |
| [BUGFIX-plus1-tsc-ROOTCAUSE.md](../../docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-ROOTCAUSE.md) | Mechanistic model + blast-radius |
| [BUGFIX-plus1-tsc-REPAIR-PATH.md](../../docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-REPAIR-PATH.md) | R1-R4 tiers + frontier red-team |
| [BUGFIX-plus1-tsc-REDTEAM.md](../../docs/rpc-multi-backend-pipeline-plus/BUGFIX-plus1-tsc-REDTEAM.md) | Pairwise bisect + identification limits |

---

## Commits this session

| SHA | Summary |
|-----|---------|
| `9eb10c5ca` | Spike: `MULTI_BACKEND_SEQ`, validation harness, blast-radius scripts, TSC docs |
| `6ac048b97` | Open-points: KV fill, llama3/gemma3, 3-GPU; base64 prompt transport; `BENCH_SAVE_FULL`, `BENCH_MULTITURN_KVFILL` |

**Earlier same day (context):** `09e3f0fe3` pairwise bisect docs; `2393538ae` RPC PATCH v3 + trace_id regression fix.

---

## Cluster / ops state (end of session)

Per [HANDOVER-CLUSTER-OPS.md](HANDOVER-CLUSTER-OPS.md):

- **romulus:** `pathb-rpc-romulus` stopped; no `llama-server`; `:50051` closed.
- **Host still up:** `sudo shutdown` failed (SSH password != sudo password).
- **Creds:** `.scratch/cluster-access.env` (gitignored)
- **Uncommitted / intentionally left:** `benches/path-b-plus/_archived-from-nested-2026-07-03/` (untracked archive)

---

## Path-B+ production state (unchanged policy)

- Phase 1c **complete** -- deploy-ready for overlap-unrelated work.
- B+6 overlap hunt **closed** (structural ceiling 2026-07-01).
- Phase D fork **paused** while TSC repair completes.
- **Do not** scope TSC fix to GDN-only or MTP-only per blast-radius evidence.

**Safe deploy today:** `Plus=0` or `GGML_PIPELINE_MULTI_BACKEND_SEQ=1` on multi-backend RPC topologies.

---

## Next session checklist

| ID | Task | Priority |
|----|------|----------|
| N1 | Implement selective **R3** (GDN split fences) -- preserve MoE async where possible | high |
| N2 | `GGML_PIPELINE_GDN_WITNESS=1` state hash at `GGML_OP_GATED_DELTA_NET` | high |
| N3 | Default-on `MULTI_BACKEND_SEQ` when `pipeline_parallel && n_devices > 1` (with escape hatch) | medium |
| N4 | b6 gate re-run under seq-repair -- measure overlap vs `sched-p0-legacy` | medium |
| N5 | 4-GPU primary (`trace-g-4gpu-primary`) TSC suite | medium |
| N6 | Fix `PLAN.md` merge conflict markers (if not fixed in handover commit) | done in handover |
| N7 | Romulus physical shutdown (sudo on console) | ops |
| N8 | Resume Phase D0 fork after TSC ship criteria met | backlog |

**Ship criteria (proposed):** blast-radius PASS + 8/8 suite on 2/3/4-GPU + KV fill PASS + b6 G within 5% of Plus=0 baseline.

---

## Commands for next session

```bash
cd /home/hunter/projects/atomic-llama-cpp-turboquant
set -a && source .scratch/cluster-access.env && set +a

# Bring cluster up
./rpc-patch/scripts/pathb-cluster-up.sh start

# Re-run open points (after rebuild/sync)
./rpc-patch/scripts/pathb-plus1-tsc-open-points.sh

# Quick 2-GPU seq-repair smoke
GGML_PIPELINE_PLUS=1 GGML_PIPELINE_MULTI_BACKEND_SEQ=1 \
  ./rpc-patch/scripts/pathb-romulus-2gpu-bench.sh tsc-smoke-seq \
  BENCH_MODEL=/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf \
  BENCH_PROMPTS_FILE=rpc-patch/bench-prompts/tsc-spike-validation.json \
  BENCH_CTX=8192 BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TS=50,50 BENCH_GEN_TOKENS=512

# Blast radius matrix
./rpc-patch/scripts/pathb-plus1-tsc-blast-radius.sh

# Pairwise bisect (identification, not fix)
./rpc-patch/scripts/pathb-plus1-tsc-pairwise-bisect.sh
```

**Rebuild after code changes (romulus):**

```bash
cmake --build build-rocm-docker --target llama-server -j$(nproc)
cmake --build build-cuda-b-bin --target rpc-server -j$(nproc)
# redeploy docker workers per HANDOVER-CLUSTER-OPS.md
```

---

## Script index (TSC)

| Script | Purpose |
|--------|---------|
| `pathb-plus1-tsc-mitigation-bisect.sh` | Single-knob bisect |
| `pathb-plus1-tsc-p0p1-bisect.sh` | P0/P1 surface bisect |
| `pathb-plus1-tsc-pairwise-bisect.sh` | Pairwise combo matrix |
| `pathb-plus1-tsc-s-factorial.sh` | S x P0 x P1 factorial |
| `pathb-plus1-tsc-blast-radius.sh` | Multi-model blast radius |
| `pathb-plus1-tsc-seq-repair-spike.sh` | Spike validation launcher |
| `pathb-plus1-tsc-spike-validate.sh` | 8-prompt suite runner |
| `pathb-plus1-tsc-spike-validate.py` | Prompt scoring gate |
| `pathb-plus1-tsc-open-points.sh` | KV fill + cross-arch + 3-GPU |
| `pathb-plus1-tsc-multiturn-kvfill.py` | Multi-turn KV saturation |
| `pathb-romulus-2gpu-bench-host.sh` | SSH wrapper 2-GPU |
| `pathb-romulus-3gpu-bench-host.sh` | SSH wrapper 3-GPU |

---

*Handover prepared 2026-07-03. Assisted-by: Grok.*