# Handover: B+6 gate session end (2026-06-29)

**Purpose:** Resume Path-B-Plus B+6 overlap gate after triton A/B, ts sweep, and diagnosis.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Session transcript:** `C:\Users\nightmare\.grok\sessions\C%3A%5CUsers%5Cnightmare%5C.grok%5Cworktrees%5Catomic-llama-cpp-5070ti-atomic-llama-cpp-turboquant%5Cpath-b-plus-5070ti-triton\019f0f97-e000-7f70-9c1d-4eb3c4a41e2b\updates.jsonl`

| Location | Git | Notes |
|----------|-----|-------|
| Gitea (push target) | `gitea/Path-B-Event-Support-Pipeline-Plus` | commit at session end |
| Windows worktree | this tree | scripts + docs committed this session |
| romulus client | `3efe5b3d8` (may lag gitea) | profiler runs via scp'd scripts; env override OK |

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

| Node | IP | GPU | Port | Role |
|------|-----|-----|------|------|
| romulus | 192.168.8.108 | 7900 | - | ROCm client / profiler |
| remus | 192.168.8.176 | 5060 | :50051 | CUDA RPC |
| romulus docker | 127.0.0.1 | 3060 | :50051 | CUDA RPC |
| JUPITER | 192.168.8.21 | 5070 Ti | :50053 | CUDA RPC (canonical RPC2) |
| triton | 192.168.8.23 | 3090 | :50054 | CUDA RPC (A/B swap) |
| triton | 192.168.8.23 | 3070 | :50055 | deferred (8 GB) |

**Hard rule:** one profiler at a time; never two profilers on triton `:50054`.

---

## Recommended `-ts` (RPC-first order)

| Preset | `-rpc` | `-ts` |
|--------|--------|-------|
| 4-GPU canonical (gate default) | remus, 3060, JUPITER | `25,12,25,38` |
| 4-GPU triton A/B | remus, 3060, triton :50054 | `22,11,34,33` |
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

1. **B+7 bisect** -- 4-RPC socket drain on canonical `25,12,25,38` (reproduce 50s vs 5s delta in `ggml-rpc.cpp`).
2. **Ops eval** -- triton `:50054` as production RPC2 vs tuned G4 `-ts` on JUPITER.
3. **Romulus sync** -- `git fetch gitea && git reset --hard gitea/Path-B-Event-Support-Pipeline-Plus` after push.
4. **Optional** -- re-run single `b6-4gpu-g` n=384 after drain fix to refresh canonical baseline artifact.

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
| `b6-gate-triton-remote.ps1` | triton sync/status/rpc |

---

## References

- [CLUSTER-4GPU-PRIMARY.md](../../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md)
- [pathb-sync-site-audit.md](../docs/pathb-sync-site-audit.md)
- [HANDOVER-CLUSTER-4GPU-2026-06-28.md](HANDOVER-CLUSTER-4GPU-2026-06-28.md) (prior cluster handover)

**Session end:** cluster RPC workers may be left running (JUPITER + triton schtasks). Stop before power-off if desired. Profiler artifacts on romulus only (not in git).