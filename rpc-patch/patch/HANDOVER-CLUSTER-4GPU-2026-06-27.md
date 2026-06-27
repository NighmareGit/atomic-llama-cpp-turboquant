# Handover: 4-GPU Config G Primary Cluster (2026-06-27)

**Purpose:** Resume Path B+ multi-GPU cluster work after shutdown.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Session transcript:** `C:\Users\nightmare\.grok\sessions\D%3A%5Cprojects%5Catomic-llama-cpp-5070ti%5Catomic-llama-cpp-turboquant\019f06be-da15-75d2-a3ba-ca266a57fcda\updates.jsonl`

| Location | Git commit | Notes |
|----------|------------|-------|
| Windows + gitea + GitHub | `6363d6f02` | Stable 4-GPU primary committed |
| Handover commit (this doc) | pending push | Adds snapshots + pull script |
| Romulus host | `833ad4429` | **One commit behind** -- sync on resume |

---

## 0. What we did this session (timeline)

1. **Investigated aborted 4-GPU hang** -- initial hypothesis was `load_tensors` fan-out failure across RPC workers.
2. **Parsed server logs** -- all 4-GPU variants (including legacy 6600) complete `load_tensors` in ~64-85s with correct buffer splits.
3. **Identified real stall** -- legacy topology with RX6600 (`192.168.8.176:50052`) stops at:
   ```
   srv load_model: initializing slots, n_slots = 1
   ```
   Last log line in `trace-g-4gpu-romulus-q8-pp1-server.log`; process eventually exits (`load_exit`) without ever reaching `server ready`.
4. **Reordered fan-out plan** -- Tier 0: 3-GPU (7900+5060+3060) flawless; Tier 1: swap 6600 for Windows 5070 (`192.168.8.21:50053`).
5. **Validated primary topology** -- 3/3 gate PASS, G 38-43 t/s, load ~85s.
6. **Captured trace** -- hotpath 20.7 ms/tok; 5060 is straggler (9.6 ms/tok per split).
7. **Committed production config** -- `6363d6f02` pushed to gitea + GitHub.
8. **Pulled bench snapshots** into git before shutdown; wrote node snapshots on romulus/remus home dirs.

---

## 1. Executive summary

We bisected a **4-GPU hang** that looked like load fan-out failure. Root cause: **RX6600 (`:50052`) topology stalls at `initializing slots`**, not during `load_tensors`. Replacing 6600 with **Windows 5070 Ti RPC (`192.168.8.21:50053`)** yields a **stable 4-GPU path**.

| Milestone | Status |
|-----------|--------|
| 4-GPU primary (7900+3060+5060+5070) | **3/3 PASS**, G 38-43 t/s, load ~85s |
| 3-GPU control (7900+5060+3060) | PASS (2/3; one curl flake) |
| 4-GPU + RX6600 | **HANG** at slot init -- parked |
| Trace hotpath (4-GPU primary) | 20.7 ms/tok serial splits; 5060 straggler |
| Docs + launchers | `6363d6f02` on remotes; handover in this commit |

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

SSH: `hunter` / `12345` (use `sshpass` or keys).  
Env: `PATHB_ROMULUS_SSH_PASS=12345`, `PATHB_REMUS_SSH_PASS=12345`.

**Remote shutdown snapshots (on host, not in git):**
- romulus: `~/CLUSTER-SHUTDOWN-SNAPSHOT-2026-06-27.txt`
- remus: `~/CLUSTER-SHUTDOWN-SNAPSHOT-2026-06-27.txt`

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
export PATHB_ROMULUS_SSH_PASS=12345
# Windows first:
#   powershell -File scripts/cuda-windows-5070ti/pathb-rpc-server.ps1
cd rpc-patch/scripts
BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_NCMOE= BENCH_TRACE=0 \
  ./pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary
```

**Legacy (do not use for production):**
```bash
BENCH_4GPU_PRESET=legacy-6600 ./pathb-romulus-4gpu-bench.sh trace-g-4gpu-legacy-6600
```

---

## 4. Investigation evidence (RX6600 vs 5070)

### Legacy 6600 (HANG) -- `trace-g-4gpu-romulus-q8-pp1`

- Endpoint: `192.168.8.176:50051,192.168.8.176:50052,127.0.0.1:50051`
- `ts=28,12,28,32`
- Load completes: `sched_reserve: graph splits = 5`, `sched copies = 4`
- **Stall:** `initializing slots, n_slots = 1` (no further progress)
- VRAM on 3060 drops to ~155 MiB after exit (model unloaded)
- Meta saved: `romulus-host/trace-g-4gpu-romulus-q8-pp1.meta`

### Primary 5070 (PASS) -- `trace-g-4gpu-primary`

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

## 5. Bench results (saved in git)

| Artifact | Location |
|----------|----------|
| Canonical doc | `docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md` |
| Summary | `rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md` |
| Node status | `rpc-patch/patch/bench-results/cluster-4gpu-primary/NODE-STATUS-2026-06-27.txt` |
| Romulus meta/results | `rpc-patch/patch/bench-results/cluster-4gpu-primary/romulus-host/` |
| Trace summary | `romulus-host/trace-g-4gpu-primary-trace-summary.txt` |
| Legacy 6600 fail | `romulus-host/trace-g-4gpu-romulus-q8-pp1.meta` |
| Tracking | `rpc-patch/docs/rpc-path-b-plus-tracking.md` |
| Pull script | `rpc-patch/scripts/pull-romulus-bench-snapshot.sh` |

### Gate run numbers

| Label | RESULT | Load | G (t/s) |
|-------|--------|------|---------|
| trace-g-4gpu-primary | PASS | 85s | 43.0 |
| trace-g-4gpu-primary-r2 | PASS | 85s | 38.2 |
| trace-g-4gpu-primary-r3 | PASS | 85s | 43.1 |
| trace-g-4gpu-primary-trace | FAIL* | 85s | ~37 |

*Run 3 `curl error (http=200)` -- inference OK; bench parser flake.

### Trace hotpath (per-token backend ms)

| Backend | ms/tok |
|---------|--------|
| Total (serial splits) | 20.71 |
| backend1 (5060) | 9.63 |
| backend2 (3060) | 5.51 |
| backend3 (5070) | 5.41 |
| SET_TENSOR_HASH (load) | 6478 ms / 280 calls |

### On romulus host only (NOT in git -- large)

Base: `/home/hunter/atomic-llama-cpp-turboquant/rpc-patch/patch/bench-results/rpc-server-bench/`

| Label | Notes |
|-------|-------|
| `trace-g-4gpu-primary*` | Gate runs r1/r2/r3 + trace |
| `trace-g-3gpu-primary-r*` | 3-GPU control |
| `trace-g-4gpu-romulus-q8-pp1*` | Legacy 6600 hang repro |
| `trace-g-4gpu-romulus-plus*` | Earlier 6600 attempts |
| `trace-g-4gpu-full-trace/` | 6600 trace while hung |

Pull before wiping romulus:
```bash
./rpc-patch/scripts/pull-romulus-bench-snapshot.sh
# Full telemetry (~2.5MB per trace):
# scp -r hunter@192.168.8.108:.../trace-g-4gpu-primary-trace/telemetry ./local-backup/
```

---

## 6. Node state at handover (2026-06-27 17:32 CEST)

### Windows (dev workstation)

| Item | State |
|------|-------|
| Git | `6363d6f02` + handover commit (local) |
| `rpc-server.exe` | **RUNNING** pid **79096**, started 17:15, `:50053` LISTENING on `0.0.0.0` |
| Portable build | `build-cuda-b-bin/portable/` |
| Model (local benches) | `D:\models\Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf` |

**Shutdown:**
```powershell
# Option A
.\scripts\cuda-windows-5070ti\pathb-rpc-server.ps1 -Stop
# Option B
Stop-Process -Id 79096 -Force
```

### Romulus (head + gitea + ROCm client)

| Item | State |
|------|-------|
| Git | `833ad4429` -- sync to tip on resume |
| `llama-server` | **none** (clean) |
| 3060 RPC | docker `pathb-rpc-romulus` Up 3h, host pid **466559** on :50051 |
| ROCm client binary | `build-rocm-docker/bin/llama-server` (built Jun 27) |
| Gitea | docker `gitea_beast` Up |
| Disk | `/home` 437G/696G used (64%) |
| Snapshot file | `~/CLUSTER-SHUTDOWN-SNAPSHOT-2026-06-27.txt` |

**Resume sync:**
```bash
ssh hunter@192.168.8.108
cd ~/atomic-llama-cpp-turboquant
./rpc-patch/scripts/pathb-node-git-sync.sh
```

### Remus (RPC workers)

| Item | State |
|------|-------|
| `pathb-rpc-remus` | Up 3h, 5060 Ti, :50051 |
| `rx6600-rpc` | Up 3h, :50052 (parked) |
| Image | `atomic-llama-remus-pathb-rpc:latest` |
| Snapshot file | `~/CLUSTER-SHUTDOWN-SNAPSHOT-2026-06-27.txt` |

---

## 7. Safe shutdown procedure

Run in this order:

```bash
# 1. Romulus -- ensure no hung client
ssh hunter@192.168.8.108 'pgrep -a llama-server || echo clean'

# 2. Windows -- stop 5070 worker
# powershell -File scripts/cuda-windows-5070ti/pathb-rpc-server.ps1 -Stop

# 3. Optional -- stop RPC docker (containers restart on boot if compose enabled)
ssh hunter@192.168.8.176 'docker stop pathb-rpc-remus rx6600-rpc'
ssh hunter@192.168.8.108 'docker stop pathb-rpc-romulus'

# 4. Power off nodes
```

No hung `llama-server` on romulus at handover. Safe to power off after stopping Windows `rpc-server`.

---

## 8. Startup checklist (next session)

1. Power on: romulus, remus, Windows.
2. **Remus:** verify `pathb-rpc-remus` Up (`pathb-remus-rpc.sh status`).
3. **Romulus:** `pathb-romulus-rpc.sh status`; **git sync** to `6363d6f02+`.
4. **Windows:** `.\scripts\cuda-windows-5070ti\pathb-rpc-server.ps1` (verify `netstat :50053`).
5. **Verify from romulus:**
   ```bash
   nc -zv 192.168.8.176 50051
   nc -zv 127.0.0.1 50051
   nc -zv 192.168.8.21 50053
   ```
6. **Smoke:** `./pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary`
7. Optional trace: `BENCH_TRACE=1` then `pathb-hotpath-summary.sh` on telemetry dir.

---

## 9. Known issues

| Issue | Impact | Workaround |
|-------|--------|------------|
| RX6600 in 4-GPU | Hang at `initializing slots` | Use primary topology (5070 :50053) |
| `pathb-rpc-server.ps1` hidden start | Worker exits silently sometimes | Run foreground or verify `netstat :50053` |
| Bench `curl error (http=200)` | RESULT=FAIL on fox run 3 | Inference OK; fix parser in `bench-5gpu.sh` |
| Romulus behind git tip | Missing launchers until sync | `pathb-node-git-sync.sh` after pull |
| `q4_0` + `n-cpu-moe 8` on ROCm 35B | Segfault (older repro) | Use `q8_0`, omit ncmoe |
| Windows portable stale vs Linux | proto mismatch risk | Rebuild `build.ps1` if RPC code changes |

---

## 10. Next work (prioritized)

| Priority | Task | Notes |
|----------|------|-------|
| P0 | Sync romulus/remus to git tip | `pathb-node-git-sync.sh` |
| P1 | 5-GPU without 6600 | Bring 3090+3070 node online |
| P2 | RX6600 slot-init debug | gdb `BENCH_GDB_PRESET=4gpu-legacy-6600` |
| P3 | Fix bench curl false FAIL | `bench-5gpu.sh` parser |
| P4 | Windows portable rebuild | If RPC code changes |
| P5 | Path C / server multi-GPU | Only if collapsing remus 5060+6600 |

---

## 11. Key scripts (this session)

| Script | Purpose |
|--------|---------|
| `pathb-romulus-3gpu-bench.sh` | 3-GPU presets (`primary-5060-3060`, etc.) |
| `pathb-romulus-4gpu-bench.sh` | 4-GPU primary default; `legacy-6600` preset |
| `pathb-romulus-gdb-repro.sh` | gdb repro presets |
| `pathb-node-git-sync.sh` | Remote git fetch + reset |
| `pull-romulus-bench-snapshot.sh` | Pull meta/results before shutdown |
| `pathb-rpc-server.ps1` | Windows :50053 worker (prefers LAN IP over VMware) |

---

## 12. Git remotes

| Remote | URL |
|--------|-----|
| gitea | `http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git` |
| origin | `https://github.com/NighmareGit/atomic-llama-cpp-turboquant.git` |

Fetch gitea on romulus before work. Handover commit should be on both remotes after push.

---

## 13. References

- [CLUSTER-4GPU-PRIMARY.md](../../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md)
- [RPC-BUG-HUNT.md](../../docs/cuda-windows-5070ti/RPC-BUG-HUNT.md) -- RX6600 slot-init section
- [rpc-path-b-plus-tracking.md](../docs/rpc-path-b-plus-tracking.md)
- [cluster-4gpu-primary/summary.md](bench-results/cluster-4gpu-primary/summary.md)
- [NODE-STATUS-2026-06-27.txt](bench-results/cluster-4gpu-primary/NODE-STATUS-2026-06-27.txt)

**Session end:** cluster safe to shut down. All critical bench meta/results are in git under `romulus-host/`.