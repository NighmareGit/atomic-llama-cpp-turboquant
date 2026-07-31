# Handover: B+6 gate session end (2026-06-29)

**Purpose:** Resume Path-B-Plus B+6 overlap gate after triton A/B, ts sweep, and diagnosis.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Mission doc root (2026-06-30):** [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/) — supersedes fragmented notes; see TRACKING + PLAN Phase 2 (B+8–B+13).  
**Session transcript:** `C:\Users\nightmare\.grok\sessions\C%3A%5CUsers%5Cnightmare%5C.grok%5Cworktrees%5Catomic-llama-cpp-5070ti-atomic-llama-cpp-turboquant%5Cpath-b-plus-5070ti-triton\019f0f97-e000-7f70-9c1d-4eb3c4a41e2b\updates.jsonl`

| Location | Git | Notes |
|----------|-----|-------|
| **Primary scratchpad (Windows)** | `D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant` | edit/commit/push here |
| Gitea | `gitea/Path-B-Event-Support-Pipeline-Plus` | source of truth remote |
| Grok worktree | `C:\Users\nightmare\.grok\worktrees\...` | ephemeral; do not use as primary |
| romulus client | sync after push | `git fetch gitea && git reset --hard gitea/Path-B-Event-Support-Pipeline-Plus` |

---

## Session summary (done today)

1. **JUPITER abort() fixed** -- portable must include `120a-real`; `b6-gate-jupiter-rebuild-rpc.cmd` + schtask `:50053`.
2. **Profiler link fix** -- `GGML_BACKEND_DL` + `pipeline-rpc-validate.cpp` uses backend registry API.
3. **Canonical `b6-4gpu-g` n=384** -- JUPITER alive; overlap 0.1%, drain **50.4s**, straggler backend3 5070 @ 11.6 ms/tok, G=77.2.
4. **Triton `b6-4gpu-g-triton` n=384** -- swap RPC2 to triton 3090 `:50054`; overlap 0.1%, drain **5.9s**, straggler backend3 3090 @ 9.0 ms/tok, G=63.1.
5. **ts sweep** -- 5 rows n=128 + confirm G2/G4 n=384 on JUPITER topology. Best grid G2 legacy `36,24,24,16` **0.7%** overlap; confirm G4 `30,14,16,40` **0.2%**, drain 4.8s, G=75.8.
6. **Diagnosis** -- mission verdict **D3 + D1** (drain-bound on canonical split; triton swap fixes drain class, not overlap). **M1 not reached** at n=384.
7. **Scripts shipped** -- `b6-4gpu-g-triton` preset, `b6-gate-ts-sweep-4gpu.sh`, `b6-gate-diagnose-runs.sh`, triton/jupiter ops helpers.

---

## Cluster map (current)

Live inventory: `bash scripts/b6-gate-cluster-gpu-inventory.sh` (`nvidia-smi` + `rocm-smi` per node).

| Node | IP | GPUs on host | Port | Role |
|------|-----|--------------|------|------|
| romulus | 192.168.8.108 | 7900 XTX + 3060 Ti | 7900 local / `:50051` docker | ROCm client + local CUDA RPC |
| remus | 192.168.8.176 | 5060 Ti + RX 6600 (unused) | `:50051` / `:50052` | CUDA RPC worker |
| JUPITER | 192.168.8.21 | 5070 Ti | :50053 | CUDA RPC (deprecated RPC2) |
| triton | 192.168.8.23 | 3090 + 3070 | `:50054` / `:50055` | CUDA RPC (canonical RPC2) |

**Hard rule:** one profiler at a time; never two profilers on triton `:50054`.

---

## Recommended `-ts` (RPC-first order)

| Preset | `-rpc` | `-ts` |
|--------|--------|-------|
| 4-GPU canonical (gate default) | remus 5060, romulus 3060 docker, JUPITER | `25,12,25,38` |
| 4-GPU triton A/B | remus 5060, romulus 3060 docker, triton :50054 | `22,11,34,33` |
| ts sweep best n=128 overlap | same canonical | `36,24,24,16` (G2) |
| ts confirm best n=384 G | same canonical | `30,14,16,40` (G4) |

---

## B+6 gate state

| Milestone | Target | Best measured | Status |
|-----------|--------|---------------|--------|
| M1 | overlap >= 1% | 0.7% (G2 n=128 only) | **FAIL** |
| M3 | overlap >= 5% | 0.3% (2-GPU triton) | **FAIL** |

Living checklist: [b6-gate/TRACKING.md](../docs/b6-gate/TRACKING.md)  
Diagnosis matrix (romulus): `benches/path-b-plus/b6-diagnosis-matrix.tsv`

---

## Next session (priority order)

See [docs/rpc-multi-backend-pipeline-plus/PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) Phase 2 (updated 2026-06-30).

1. **B+8 bisect** -- partial `pipeline_barrier` in `ggml-backend.cpp`; re-bench `b6-2gpu-f` n=384.
2. **B+9** -- EVENT defer-to-barrier in `ggml-rpc.cpp`.
3. **B+10** -- MoE `input_wait_copy` de-sync (`ggml-backend.cpp` ~1682).
4. **B+7a′** -- 4-RPC socket drain on canonical `25,12,25,38` (`b6-4gpu-g` n=384).
5. **Ops eval** (parallel) -- triton `:50054` vs G4 `-ts` (throughput only).
6. **Romulus sync** after push.

---

## Startup checklist

```powershell
# JUPITER
.\scripts\b6-gate-jupiter-start-rpc-task.ps1

# triton (if A/B needed)
.\scripts\b6-gate-triton-remote.ps1 -Action rpc
```

```bash
# romulus
pgrep -af llama-pipeline-profiler   # must be empty
bash scripts/b6-gate-run-remote.sh b6-4gpu-g-triton   # or ts sweep / diagnose
bash scripts/b6-gate-diagnose-runs.sh b6-4gpu-g-triton b6-4gpu-ts-sweep/G2-confirm
```

Push scripts to romulus when local tree ahead of remote:
```powershell
.\scripts\b6-gate-push-and-run.ps1 -Label b6-4gpu-g
```

---

## Key scripts (this session)

| Script | Purpose |
|--------|---------|
| `b6-gate-profiler-romulus.sh` | Presets incl. `b6-4gpu-g`, `b6-4gpu-g-triton` |
| `b6-gate-run-remote.sh` | Run on romulus (respects `PROFILER_OUT_DIR`) |
| `b6-gate-ts-sweep-4gpu.sh` | `--phase grid\|confirm` |
| `b6-gate-diagnose-runs.sh` | Post-run matrix + verdict TSV |
| `b6-gate-push-and-run.ps1` | scp + run label on romulus |
| `b6-gate-jupiter-start-rpc-task.ps1` | JUPITER :50053 schtask |
| `b6-gate-triton-remote.ps1` | triton sync/status/rpc/stop |
| `b6-gate-triton-stop-rpc-task.ps1` | stop triton :50054/:50055 schtasks |
| `b6-gate-cluster-shutdown.ps1` | stop remus + romulus + triton (not JUPITER) |

---

## References

- [CLUSTER-4GPU-PRIMARY.md](../../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md)
- [pathb-sync-site-audit.md](../docs/pathb-sync-site-audit.md)
- [HANDOVER-CLUSTER-4GPU-2026-06-28.md](HANDOVER-CLUSTER-4GPU-2026-06-28.md) (prior cluster handover)

## Cluster shutdown (2026-06-29 session end)

**Status:** **STOPPED** (except JUPITER -- user shuts down workstation locally).

| Node | Service | Shutdown action | Verified |
|------|---------|-----------------|----------|
| remus | `pathb-rpc-remus` docker :50051 | `pathb-cluster-up.sh stop` | :50051 refused |
| romulus | `pathb-rpc-romulus` docker :50051 | `pathb-cluster-up.sh stop` | :50051 refused |
| romulus | profiler | `pkill llama-pipeline-profiler` | no hung profiler |
| triton | `PathB-Triton-RPC-50054/50055` | taskkill + schtasks delete | no rpc-server |
| **JUPITER** | `PathB-Jupiter-RPC-50053` | **not stopped** | user workstation shutdown |

One-command shutdown (next time):

```powershell
.\scripts\b6-gate-cluster-shutdown.ps1
# triton only: .\scripts\b6-gate-triton-remote.ps1 -Action stop
```

Startup (next session): see [Startup checklist](#startup-checklist) above; start JUPITER locally first.

Profiler artifacts remain on romulus only (`benches/path-b-plus/`); not in git.

**Git tip at shutdown:** `git log -1 --oneline` on `Path-B-Event-Support-Pipeline-Plus` (gitea); cluster stopped 2026-06-29.