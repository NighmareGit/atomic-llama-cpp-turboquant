# Handover: cluster ops (docker, build, folders)

**Purpose:** How to bring up Path B RPC workers, run benches, and find artifacts. Not tied to any single bug investigation.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus`  
**Updated:** 2026-07-04  
**Node layout (read first):** [CLUSTER-NODE-LAYOUT.md](CLUSTER-NODE-LAYOUT.md)

---

## 1. Quick start

From any machine with repo + `sshpass`:

```bash
cd /path/to/atomic-llama-cpp-turboquant
set -a && source .scratch/cluster-access.env && set +a

# Linux RPC workers (remus 5060 + romulus 3060)
./rpc-patch/scripts/pathb-cluster-up.sh start
./rpc-patch/scripts/pathb-cluster-up.sh status

# 3-GPU fox smoke (no Jupiter / Triton) -- romulus SSH bench
./rpc-patch/scripts/pathb-romulus-3gpu-bench.sh primary-5060-3060 trace-g-3gpu-smoke

# 4-GPU primary (needs Jupiter :50053)
# Start Windows RPC first (section 5), then:
BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_TRACE=1 \
  ./rpc-patch/scripts/pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary
```

Live probe snapshot: `.scratch/cluster-inventory.md`  
SSH/password env: `.scratch/cluster-access.env` (gitignored, `chmod 600`)

---

## 2. Cluster map

| Node | IP | GPU | Role | Port | Service |
|------|-----|-----|------|------|---------|
| **romulus** | 192.168.8.108 | RX 7900 XTX | ROCm **client** (native `llama-server`) | - | host binary |
| romulus | 127.0.0.1 | RTX 3060 Ti | CUDA RPC worker | **50051** | docker `pathb-rpc-romulus` |
| **remus** | 192.168.8.176 | RTX 5060 Ti | CUDA RPC worker | **50051** | docker `pathb-rpc-remus` |
| remus | 192.168.8.176 | RX 6600 | ROCm RPC (experimental) | **50052** | docker `rx6600-rpc` |
| **jupiter** | 192.168.8.21 | RTX 5070 Ti | CUDA RPC worker (Windows) | **50053** | `rpc-server.exe` portable |
| **triton** | 192.168.8.23 | 3090 + 3070 | CUDA RPC workers | **50054** / **50055** | docker (manual boot) |

**SSH users:** Linux `hunter`; Windows Jupiter `nightmare`. Passwords in `.scratch/cluster-access.env`.

**Device order for `-ts`:** matches `--rpc` endpoint order. Example 4-GPU primary:

```
RPC: 192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053
ts:  36,24,24,16   # ROCm0 7900, RPC0 5060, RPC1 5070, RPC2 3060
```

---

## 3. Repo and build locations

**Canonical path (all Linux nodes):** `~/projects/atomic-llama-cpp-turboquant`

| Host | Repo path | Client build | RPC build |
|------|-----------|--------------|-----------|
| romulus | `~/projects/atomic-llama-cpp-turboquant` | `build-rocm-docker/bin/llama-server` | via docker image build |
| remus | `~/projects/atomic-llama-cpp-turboquant` | usually N/A (worker only) | via docker image build |
| triton | `~/projects/atomic-llama-cpp-turboquant` | usually N/A (worker only) | via docker image build |
| jupiter | `D:\projects\atomic-llama-cpp-5070ti\...` (Windows) | `build-cuda-b-bin/portable/` | same portable dir |

Legacy `~/atomic-llama-cpp-turboquant` on romulus is deprecated — archive with `scripts/cluster-node-fresh-clone.sh`.

**CMake targets (repo root):**

```bash
cmake --build build-rocm-docker --target llama-server    # romulus client
cmake --build build-cuda-b-bin --target rpc-server       # Linux CUDA worker binary (staged into docker)
```

Gitea remote on romulus; bench wrappers SSH from laptop/remus and run `~/bench-5gpu.sh` on romulus.

---

## 4. Docker deploy folders

Synced from repo by lifecycle scripts. Do not edit only on remote -- change `rpc-patch/deploy/` then `deploy` action.

| Node | Remote dir | Repo source | Image | Container |
|------|------------|-------------|-------|-----------|
| remus | `~/docker/Atomic-Llama-Remus-PathB` | `rpc-patch/deploy/Atomic-Llama-Remus-PathB/` | `atomic-llama-remus-pathb-rpc:latest` | `pathb-rpc-remus` |
| romulus | `~/docker/Atomic-Llama-Romulus-PathB` | `rpc-patch/deploy/Atomic-Llama-Romulus-PathB/` | `atomic-llama-romulus-pathb-rpc:latest` | `pathb-rpc-romulus` |
| triton | `~/docker/Atomic-Llama-Triton-PathB` | `rpc-patch/deploy/Atomic-Llama-Triton-PathB/` | per-GPU tags | `:50054`, `:50055` |
| remus | `~/docker/Atomic-Llama-Remus-RX6600` | `rpc-patch/deploy/Atomic-Llama-Remus-RX6600/` | rx6600 images | experimental |

### Lifecycle scripts

| Script | Host | Actions |
|--------|------|---------|
| `pathb-remus-rpc.sh` | remus (SSH or local) | `start\|stop\|status\|logs\|deploy\|build\|rebuild` |
| `pathb-romulus-rpc.sh` | romulus (SSH) | same + git sync on rebuild |
| `pathb-cluster-up.sh` | orchestrator | `start\|stop\|status` both Linux workers |
| `pathb-triton-rpc.sh` | triton (SSH) | start/stop when node is routed |

**Start both Linux workers:**

```bash
source .scratch/cluster-access.env
./rpc-patch/scripts/pathb-cluster-up.sh start
```

**Rebuild after ggml-rpc changes:**

```bash
./rpc-patch/scripts/pathb-remus-rpc.sh rebuild
./rpc-patch/scripts/pathb-romulus-rpc.sh rebuild
```

Romulus build uses Squid proxy at `192.168.8.108:3128` when reachable. `build.sh` aborts if disk free < 25 GB.

**Romulus-local 3060** (when logged into romulus): `PATHB_CLUSTER_LOCAL=1 ./rpc-patch/scripts/pathb-cluster-up.sh start --local` uses `pathb-start-rpc.sh` instead of/in addition to docker on 108.

---

## 5. Jupiter (Windows 5070) RPC

Must listen on `0.0.0.0:50053` (not localhost-only) or romulus cannot connect.

```powershell
cd D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant\build-cuda-b-bin\portable
.\rpc-server.exe -H 0.0.0.0 -p 50053 -d CUDA0
```

Or scheduled task: `scripts\b6-gate-jupiter-start-rpc-task.ps1` (see `rpc-patch/patch/HANDOVER-SESSION-2026-06-29.md`).

**Gotcha:** `Start-Process -WindowStyle Hidden` without portable cwd exits silently.

Verify from romulus/remus:

```bash
nc -zv 192.168.8.21 50053
```

---

## 6. Triton (manual boot)

Not auto-started by `pathb-cluster-up.sh`. Node may be offline (`no route`) until powered/networked.

```bash
source .scratch/cluster-access.env
ssh user@192.168.8.23
cd ~/docker/Atomic-Llama-Triton-PathB
pkill -f 'rpc-server.*50054' || true
pkill -f 'rpc-server.*50055' || true
docker compose up -d
ss -tlnp | grep -E '50054|50055'
```

5-GPU client example (romulus):

```
--rpc 192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054,192.168.8.23:50055
-ts  19,10,30,10,31
```

Launcher: `pathb-romulus-5gpu-bench.sh`

---

## 7. Benchmark harness

| Goal | Script | Runs on |
|------|--------|---------|
| 4-GPU HTTP gate | `pathb-romulus-4gpu-bench.sh <label>` | SSH -> romulus `bench-5gpu.sh` |
| 3-GPU gate | `pathb-romulus-3gpu-bench.sh <preset> [label]` | SSH -> romulus |
| Single-node matrix | `rpc-server-bench.sh pathb <label>` | romulus native |
| Pull artifacts | `pull-romulus-bench-snapshot.sh` | from laptop/remus |

**Common env:**

| Var | Typical |
|-----|---------|
| `BENCH_MODEL` | `/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf` |
| `BENCH_CTK` / `BENCH_CTV` | `q8_0` / `q8_0` (gate); script default `q4_0` if unset |
| `BENCH_TS` | topology-specific (see section 2) |
| `BENCH_GEN_TOKENS` | `128` |
| `BENCH_TRACE` | `1` for telemetry jsonl |
| `GGML_PIPELINE_PLUS` | `1` (default in bench scripts) |

**Fox prompt:** default in `rpc-server-bench.sh` -- `The quick brown fox jumps over the lazy dog.`

**Artifacts:**

```
rpc-patch/patch/bench-results/rpc-server-bench/<label>.{meta,result,server.log}
rpc-patch/patch/bench-results/rpc-server-bench/<label>/telemetry/   # if BENCH_TRACE=1
rpc-patch/patch/bench-results/cluster-4gpu-primary/                 # pulled snapshots
```

Parse telemetry: `rpc-patch/scripts/pathb-hotpath-summary.sh <telemetry-dir>`

---

## 8. Health checks

```bash
# RPC ports
nc -zv 192.168.8.176 50051   # remus
nc -zv 192.168.8.108 50051   # romulus docker
nc -zv 192.168.8.21 50053    # jupiter

# Cluster status
./rpc-patch/scripts/pathb-cluster-up.sh status

# VRAM preflight (multi-topology)
python3 rpc-patch/scripts/pathb-rpc-vram-preflight.py
```

---

## 9. Related docs

| Doc | Content |
|-----|---------|
| [rpc-patch/README.md](../README.md) | Patch overview, Config A-D |
| [HANDOVER-CLUSTER-4GPU-2026-06-28.md](HANDOVER-CLUSTER-4GPU-2026-06-28.md) | 4-GPU gate session history |
| [docs/rpc-multi-backend-pipeline-plus/](../docs/rpc-multi-backend-pipeline-plus/) | Mission, CONFIGURATION, TRACKING |
| [BENCHMARKING.md](../../BENCHMARKING.md) | T0-T3 taxonomy |
| `.scratch/cluster-inventory.md` | Last probe snapshot |

---

## 10. Shutdown

```bash
./rpc-patch/scripts/pathb-cluster-up.sh stop
# Jupiter: stop rpc-server.exe or schtask
# Triton: docker compose down on host
```

`b6-gate-cluster-shutdown.ps1` stops Linux nodes only (not Jupiter).