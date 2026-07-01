# Path-B-Plus performance rules (auto-tuning)

Navigation: [PLAN.md](PLAN.md) | [TRACKING.md](TRACKING.md)

**Branch:** Path-B-Event-Support-Pipeline-Plus  
**Last updated:** 2026-07-01

Client-side env tuning for multi-RPC inference. Overlap gate is **closed**; these rules optimize **G_tps** and stability.

---

## Auto-switch script

```bash
source scripts/b6-gate-performance-env.sh
b6_apply_performance_tuning "$BENCH_RPC_ENDPOINT"
```

Disable auto rules: `B6_PERF_AUTO=0` and set flags explicitly.

Presets:

| Label | Topology | Auto dual | Notes |
|-------|----------|-----------|-------|
| `b6-2gpu-f` | 1 RPC + local | OFF | remus 5060 |
| `b6-3gpu-g` | 2 RPC + local | **ON** | +18% G n=384 (matrix R2) |
| `b6-4gpu-g-triton` | 3 RPC + local | OFF | -9% G when dual ON |
| `b6-5gpu-g` | 4 RPC + local | OFF | bisect baseline |
| `b6-5gpu-g-prod` | 4 RPC + local | **ON** | production ship |

---

## `GGML_RPC_DUAL_SOCKET` (B+11)

| RPC endpoints | dual=ON evidence | Rule |
|---------------|------------------|------|
| 1 (2-GPU) | -9% G @ n=384 | **OFF** |
| 2 (3-GPU) | retest in flight | **OFF** until `b6-3gpu-g` matrix |
| 3 (4-GPU) | -9% G, overlap flat | **OFF** |
| 4 (5-GPU) | +18% n128, +10% n2048 | **ON** for prod |

Log: `RPC host:port: proto 4.4 ... dual=yes|no`

Requires proto 4.4 rpc-servers on all endpoints when ON.

---

## `GGML_RPC_HASH_DEFER` (B+7f)

| Topology | Evidence | Rule |
|----------|----------|------|
| 5-GPU | +1-5% G, overlap flat | **OFF** default; opt-in `B6_5GPU_HASH_DEFER=1` |

---

## Core env (all production paths)

| Var | Value |
|-----|-------|
| `GGML_PIPELINE_PLUS` | `1` |
| `GGML_RPC_EVENT_DEFER_BARRIER` | `1` (default when Plus + multi-RPC) |

---

## Instrumentation (every run)

Written to `{out_dir}/env.txt` by `llama-pipeline-profiler-cluster.sh`:

- `GIT_SHA`, `RPC`, `TS`, `N_GEN`, `client_kind`
- `GGML_PIPELINE_PLUS`, `GGML_RPC_DUAL_SOCKET`, `GGML_RPC_HASH_DEFER`
- Mitigation flags: `GGML_RPC_EVENT_DEFER_BARRIER`, `GGML_RPC_MULTI_SOCKET_FLUSH`, etc.

Post-run checklist:

```bash
bash rpc-patch/scripts/pathb-hotpath-summary.sh "$OUT/telemetry"
cat "$OUT/telemetry/diagnose.json" | python3 -m json.tool | head -40
```

Key diagnose fields for performance (not overlap gate):

| Field | Use |
|-------|-----|
| `G_tps` | Throughput |
| `blocking_ms` | RPC drain cost |
| `drain_flush_ms` | EVENT/GET flush |
| `straggler_backend` / `straggler_ms_per_token` | Worker balance |

Key fields for overlap reopen:

| Field | Use |
|-------|-----|
| `overlap_pct` | Assembly-line metric |
| `stall_ratio` | Idle fraction |
| sched `input_wait_copy_ms` | Scheduler copy-wait |
| sched `assembly_overlap_count` | Raw overlap pairs |

---

## Retest matrix (2026-07-01)

Script: `scripts/b6-gate-retest-matrix.sh`

| Run | Label | n | dual | Purpose |
|-----|-------|---|------|---------|
| R1 | `b6-3gpu-g` | 384 | OFF | 3-GPU baseline |
| R2 | `b6-3gpu-g` | 384 | ON | 3-GPU dual bisect |
| R3 | `b6-5gpu-g-prod` | 384 | ON | prod @ gate depth |
| R4 | `b6-5gpu-g-prod` | 2048 multiturn | ON | long prefill + gen |
| R5 | `b6-5gpu-g` | 2048 multiturn | OFF | single-socket compare |

Results appended to [TRACKING.md](TRACKING.md) retest section after matrix completes.