# Handover: 4-GPU Config G Primary Cluster (2026-06-28)

**Purpose:** Resume Path B+ multi-GPU cluster work and pipeline documentation.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Session transcript (latest):** `C:\Users\nightmare\.grok\sessions\C%3A%5CUsers%5Cnightmare%5C.grok%5Cworktrees%5Catomic-llama-cpp-5070ti-atomic-llama-cpp-turboquant%5Cpath-b-plus-5070ti-triton\019f0f70-046f-7501-8583-70961641efef\updates.jsonl`  
**Prior session:** `C:\Users\nightmare\.grok\sessions\D%3A%5Cprojects%5Catomic-llama-cpp-5070ti%5Catomic-llama-cpp-turboquant\019f06be-da15-75d2-a3ba-ca266a57fcda\updates.jsonl`

| Location | Git commit | Notes |
|----------|------------|-------|
| Gitea tip (romulus synced) | `1dfdf1b6f` | `rpc: fix handover git tip to d9c3e6754` |
| Windows worktree | `1dfdf1b6f` + **local uncommitted** | See [Uncommitted local work](#14-uncommitted-local-work-2026-06-28) |
| Prior handover commit | `d9c3e6754` / `6363d6f02` | 2026-06-27 cluster gate + snapshots |

---

## 0. What we did this session (2026-06-28 timeline)

1. **Parsed prior handover** and ran startup checklist from 2026-06-27 shutdown state.
2. **Synced romulus** `833ad4429` -> `1dfdf1b6f` via gitea fetch/reset.
3. **Brought cluster online:** romulus `pathb-rpc-romulus` (:50051), remus `pathb-rpc-remus` (:50051, was offline at first), Windows 5070 RPC (:50053).
4. **Windows RPC gotcha reproduced:** `Start-Process -WindowStyle Hidden` without portable cwd exits silently; fixed by foreground/background shell from `build-cuda-b-bin/portable/`.
5. **Connectivity:** romulus initially timed out to `192.168.8.21:50053` until worker was listening on `0.0.0.0`; all 3 hops OK before bench.
6. **4-GPU smoke bench** `trace-g-4gpu-primary-resume` -- **RESULT=PASS**, load 95s, G 23.4 / 29.1 / 22.7 t/s (below 2026-06-27 peak 38-43).
7. **Created PIPELINE.md** -- Layer A (depth-2) + Layer B (Path-B Plus); external event injection in Future work only.
8. **Created BENCHMARKING.md** -- T0-T3 taxonomy, run contract, profiling decision tree; excludes unshipped Grok items.
9. **Created bench stubs** -- `scripts/bench-pipeline-depth2.sh`, `scripts/bench-pipeline-plus-ab.sh`.
10. **Created `benches/path-b-plus/`** -- README, plus-ab/depth-2/composition summaries; seeded Matrix B rows.

**Not done this session:** T0/T1 live A/B runs, 5-GPU bring-up, RX6600 gdb, bench curl parser fix, git commit/push of new docs.

---

## 1. Executive summary

**4-GPU primary topology remains production default** (no RX6600). Cluster resumed cleanly; smoke gate passed but throughput regressed vs cold-start baseline from 2026-06-27.

| Milestone | Status |
|-----------|--------|
| 4-GPU primary (7900+3060+5060+5070) | **Stable** -- smoke PASS 2026-06-28 |
| 4-GPU gate (2026-06-27) | **3/3 PASS**, G 38-43 t/s, load ~85s |
| 4-GPU smoke (2026-06-28) | **PASS**, G 22.7-29.1 t/s, load ~95s |
| 4-GPU + RX6600 | **HANG** at slot init -- parked |
| Pipeline + bench docs | **Local uncommitted** -- PIPELINE.md, BENCHMARKING.md |
| T0 depth-2 / T1 Plus A/B matrices | **Stubs only** -- harness not run yet |

**Production 4-GPU topology (romulus client):** no RX6600.

---

## 2. Cluster map

| Node | IP | GPU | Role | Port | Service |
|------|-----|-----|------|------|---------|
| **romulus** | 192.168.8.108 | 7900 XTX | ROCm **client** + gitea | - | native `llama-server` |
| romulus | 127.0.0.1 | 3060 Ti | CUDA RPC worker | **50051** | docker `pathb-rpc-romulus` |
| **remus** | 192.168.8.176 | 5060 Ti | CUDA RPC worker | **50051** | docker `pathb-rpc-remus` |
| remus | 192.168.8.176 | RX6600 | ROCm RPC (experimental) | **50052** | docker `rx6600-rpc` |
| **Windows** | 192.168.8.21 | 5070 Ti | CUDA RPC worker | **50053** | `rpc-server.exe` portable |

SSH: `hunter` / <redacted> (use `sshpass` or keys).  
Env: `PATHB_ROMULUS_SSH_PASS=<redacted>`, `PATHB_REMUS_SSH_PASS=<redacted>`.

---

## 3. Stable 4-GPU production config

```
Client:  romulus 7900 XTX (ROCm native)
RPC:     192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053
ts:      36,24,24,16   # 7900, 5060, 5070, 3060
Model:   /mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf
KV:      q8_0 / q8_0
Flags:   GGML_PIPELINE_PLUS=1, ncmoe=none, --fit off, --no-warmup -np 1
```

**Launcher:**
```bash
export PATHB_ROMULUS_SSH_PASS=<redacted>
# Windows first (see section 6 -- must run from portable cwd):
#   cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\portable
#   .\rpc-server.exe -H 0.0.0.0 -p 50053 -d CUDA0
cd rpc-patch/scripts
BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_NCMOE= BENCH_TRACE=0 \
  ./pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary
```

**Legacy (do not use for production):**
```bash
BENCH_4GPU_PRESET=legacy-6600 ./pathb-romulus-4gpu-bench.sh trace-g-4gpu-legacy-6600
```

---

## 4. Investigation evidence (RX6600 vs 5070) -- 2026-06-27

### Legacy 6600 (HANG) -- `trace-g-4gpu-romulus-q8-pp1`

- Endpoint: `192.168.8.176:50051,192.168.8.176:50052,127.0.0.1:50051`
- `ts=28,12,28,32`
- Load completes: `sched_reserve: graph splits = 5`, `sched copies = 4`
- **Stall:** `initializing slots, n_slots = 1` (no further progress)
- Meta saved: `romulus-host/trace-g-4gpu-romulus-q8-pp1.meta`

### Primary 5070 (PASS) -- `trace-g-4gpu-primary` (2026-06-27 gate)

- Endpoint: `192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053`
- `ts=36,24,24,16`
- `server ready at 85s`; G=43.0 t/s on run 3
- Buffer split: 7900 3696 / 3060 4381 / 5060 8289 / 5070 4986 MiB

### Model buffer split (load_tensors, primary)

| Device | MiB |
|--------|-----|
| ROCm0 7900 | 3696 |
| RPC 3060 | 4381 |
| RPC 5060 | 8289 |
| RPC 5070 | 4986 |
| CPU mapped | 398 |

---

## 5. Bench results

### Gate runs (2026-06-27, in git)

| Label | RESULT | Load | G (t/s) |
|-------|--------|------|---------|
| trace-g-4gpu-primary | PASS | 85s | 43.0 |
| trace-g-4gpu-primary-r2 | PASS | 85s | 38.2 |
| trace-g-4gpu-primary-r3 | PASS | 85s | 43.1 |
| trace-g-4gpu-primary-trace | FAIL* | 85s | ~37 |

*Run 3 `curl error (http=200)` -- inference OK; bench parser flake.

Trace hotpath (per-token backend ms): total 20.71; backend1 (5060) 9.63; backend2 (3060) 5.51; backend3 (5070) 5.41.

### Smoke resume (2026-06-28, romulus host)

| Label | RESULT | Load | G (t/s) fox x3 |
|-------|--------|------|----------------|
| trace-g-4gpu-primary-resume | PASS | 95s | 23.4 / 29.1 / 22.7 |

Artifacts: `rpc-patch/patch/bench-results/rpc-server-bench/trace-g-4gpu-primary-resume.*` (romulus only, not pulled to git).

Throughput below 2026-06-27 peak -- investigate with `BENCH_TRACE=1` + `pathb-hotpath-summary.sh` before treating as regression.

### Artifact index

| Artifact | Location |
|----------|----------|
| Canonical doc | `docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md` |
| Summary | `rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md` |
| Pipeline doc | `PIPELINE.md` (local) |
| Benchmarking doc | `BENCHMARKING.md` (local) |
| Plus bench tables | `benches/path-b-plus/plus-ab/summary.md` (local) |
| Tracking | `rpc-patch/docs/rpc-path-b-plus-tracking.md` |
| Pull script | `rpc-patch/scripts/pull-romulus-bench-snapshot.sh` |

---

## 6. Node state at handover (2026-06-28)

### Windows (dev workstation)

| Item | State |
|------|-------|
| Git worktree | `1dfdf1b6f` + uncommitted docs (see section 14) |
| Portable build | `D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\portable\` |
| `rpc-server.exe` | Was running in background shell during session; **stop before shutdown** |
| Worktree | No local CUDA build -- use `D:\projects\...` binary |

**Start (reliable):**
```powershell
cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\portable
.\rpc-server.exe -H 0.0.0.0 -p 50053 -d CUDA0
# verify: netstat -ano | findstr :50053
```

**Stop:**
```powershell
Get-Process -Name rpc-server -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Romulus (head + gitea + ROCm client)

| Item | State |
|------|-------|
| Git | `1dfdf1b6f` (synced this session) |
| `llama-server` | none after bench (clean) |
| 3060 RPC | docker `pathb-rpc-romulus` Up |
| Gitea | docker `gitea_beast` Up |

**Resume sync (if behind):**
```bash
ssh user@192.168.8.108
cd ~/atomic-llama-cpp-turboquant
./rpc-patch/scripts/pathb-node-git-sync.sh
```

### Remus (RPC workers)

| Item | State |
|------|-------|
| `pathb-rpc-remus` | Up (5060 Ti, :50051) -- was offline at session start |
| `rx6600-rpc` | Parked (:50052) |

---

## 7. Safe shutdown procedure

Run in this order:

```bash
# 1. Romulus -- ensure no hung client
ssh user@192.168.8.108 'pgrep -a llama-server || echo clean'

# 2. Windows -- stop 5070 worker
# Get-Process -Name rpc-server | Stop-Process -Force

# 3. Optional -- stop RPC docker
ssh user@192.168.8.176 'docker stop pathb-rpc-remus rx6600-rpc'
ssh user@192.168.8.108 'docker stop pathb-rpc-romulus'

# 4. Power off nodes
```

Optional: pull resume bench artifacts before wiping romulus:
```bash
./rpc-patch/scripts/pull-romulus-bench-snapshot.sh
```

---

## 8. Startup checklist (next session)

1. Power on: romulus, remus, Windows.
2. **Windows:** start `rpc-server.exe` from portable cwd (section 6); verify `nc -zv 192.168.8.21 50053` from romulus.
3. **Remus:** `pathb-remus-rpc.sh status`.
4. **Romulus:** `pathb-romulus-rpc.sh status`; git sync if needed.
5. **Verify from romulus:**
   ```bash
   nc -zv 192.168.8.176 50051
   nc -zv 127.0.0.1 50051
   nc -zv 192.168.8.21 50053
   ```
6. **Smoke:** `./pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary`
7. **Docs:** read `PIPELINE.md` + `BENCHMARKING.md` (commit locally first if starting fresh worktree).
8. Optional: `BENCH_TRACE=1` hotpath to explain G regression.

---

## 9. Known issues

| Issue | Impact | Workaround |
|-------|--------|------------|
| RX6600 in 4-GPU | Hang at `initializing slots` | Use primary topology (5070 :50053) |
| `pathb-rpc-server.ps1` / hidden start | Worker exits silently | Run from portable cwd; verify `netstat :50053` |
| Romulus -> Windows :50053 timeout | Bench fails RPC preflight | Ensure `0.0.0.0` bind + Windows firewall allows LAN |
| Bench `curl error (http=200)` | RESULT=FAIL on fox run 3 | Inference OK; fix parser in `bench-5gpu.sh` |
| G regression smoke vs gate | 23-29 vs 38-43 t/s | Rerun traced gate; check 5060 straggler |
| `q4_0` + `n-cpu-moe 8` on ROCm 35B | Segfault (older repro) | Use `q8_0`, omit ncmoe |
| Windows portable stale vs Linux | proto mismatch risk | Rebuild `build.ps1` if RPC code changes |
| Worktree vs D:\projects build | Agent may target wrong binary | Always use `D:\projects\...\portable\` for RPC |

---

## 10. Next work (prioritized)

| Priority | Task | Notes |
|----------|------|-------|
| P0 | Commit + push docs | PIPELINE.md, BENCHMARKING.md, benches/, scripts/ |
| P0 | Pull resume bench snapshot | `trace-g-4gpu-primary-resume` into git |
| P1 | Populate T0 Matrix A | `scripts/bench-pipeline-depth2.sh` on MTP/NextN server |
| P1 | Populate T1 paired A/B | `scripts/bench-pipeline-plus-ab.sh` on cluster |
| P2 | Traced 4-GPU rerun | Chase 38-43 t/s baseline; `BENCH_TRACE=1` |
| P3 | 5-GPU without 6600 | Bring 3090+3070 node online |
| P4 | RX6600 slot-init debug | gdb `BENCH_GDB_PRESET=4gpu-legacy-6600` |
| P5 | Fix bench curl false FAIL | `bench-5gpu.sh` parser |
| P6 | Windows portable rebuild | If RPC code changes |

---

## 11. Key scripts

| Script | Purpose |
|--------|---------|
| `pathb-romulus-3gpu-bench.sh` | 3-GPU presets |
| `pathb-romulus-4gpu-bench.sh` | 4-GPU primary default; `legacy-6600` preset |
| `pathb-romulus-gdb-repro.sh` | gdb repro presets |
| `pathb-node-git-sync.sh` | Remote git fetch + reset |
| `pull-romulus-bench-snapshot.sh` | Pull meta/results from romulus |
| `pathb-rpc-server.ps1` | Windows :50053 worker (prefer manual portable start) |
| `bench-pipeline-depth2.sh` | T0 depth-2 A/B stub (local, uncommitted) |
| `bench-pipeline-plus-ab.sh` | T1 Plus A/B stub (local, uncommitted) |
| `pathb-hotpath-summary.sh` | Parse trace telemetry per-backend ms/tok |

---

## 12. Documentation added (2026-06-28)

| File | Scope |
|------|-------|
| `PIPELINE.md` | Layer A depth-2 + Layer B Path-B Plus; observability env vars |
| `BENCHMARKING.md` | T0-T3 taxonomy, matrices, profiling tree |
| `benches/path-b-plus/README.md` | Published results home |
| `benches/path-b-plus/plus-ab/summary.md` | Matrix B seeded rows |
| `benches/path-b-plus/depth-2/README.md` | T0 placeholder |
| `benches/path-b-plus/composition/README.md` | T3 placeholder |

Cross-links patched: `README.md`, `docs/speculative.md`, `rpc-patch/README.md`.

---

## 13. Git remotes

| Remote | URL |
|--------|-----|
| gitea | `http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git` |
| origin | `https://github.com/NighmareGit/atomic-llama-cpp-turboquant.git` |

Fetch gitea on romulus before cluster work.

---

## 14. Uncommitted local work (2026-06-28)

Windows worktree at `1dfdf1b6f` with **untracked / modified** (not committed this session):

```
?? PIPELINE.md
?? BENCHMARKING.md
?? benches/path-b-plus/
?? scripts/bench-pipeline-depth2.sh
?? scripts/bench-pipeline-plus-ab.sh
 M README.md
 M docs/speculative.md
 M rpc-patch/README.md
```

Commit before pushing to gitea or syncing another machine.

---

## 15. References

- [PIPELINE.md](../../PIPELINE.md) (local)
- [BENCHMARKING.md](../../BENCHMARKING.md) (local)
- [CLUSTER-4GPU-PRIMARY.md](../../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md)
- [RPC-BUG-HUNT.md](../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md) -- RX6600 slot-init section
- [PROFILING.md](../../docs/cuda-windows-5070ti/PROFILING.md)
- [rpc-path-b-plus-tracking.md](../docs/rpc-path-b-plus-tracking.md)
- [cluster-4gpu-primary/summary.md](bench-results/cluster-4gpu-primary/summary.md)

**Session end (2026-06-28):** cluster was left running during doc work. Stop Windows `rpc-server` and optional docker before power-off. Pull `trace-g-4gpu-primary-resume` artifacts if preserving smoke numbers in git.

---

## 16. Continuation (2026-06-29 session end)

**Superseded ops detail:** [HANDOVER-SESSION-2026-06-29.md](HANDOVER-SESSION-2026-06-29.md)

| Item | Status |
|------|--------|
| JUPITER :50053 schtask + 120a-real rebuild | DONE |
| triton :50054 A/B (`b6-4gpu-g-triton`) | DONE |
| ts sweep + diagnosis (D3+D1) | DONE |
| B+6 M1/M3 | FAIL |
| Next | B+7 4-socket drain in `ggml-rpc.cpp` |

`-ts` default for gate profiling is **`25,12,25,38`** (not legacy `36,24,24,16`). Legacy G2 row peaked 0.7% overlap @ n=128 only.