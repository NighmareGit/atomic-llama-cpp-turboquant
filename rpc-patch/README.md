# RPC Path A/B patch collateral

Private fork work on **llama.cpp RPC cross-GPU performance**: batching and pipelining (Path A), then event-based pipeline parallelism (Path B). This folder lives **inside** the `atomic-llama-cpp-turboquant` git tree (branch `Path-B-Event-Support-Pipeline-Plus`, protocol v4.3) so docs, scripts, handover notes, and benchmark metadata are versioned with the feature branch.

**Git repo root:** parent of this folder (`atomic-llama-cpp-turboquant/`)  
**This folder:** `rpc-patch/` (docs, scripts, handover, bench artifacts)  
**Builds:** `../build-cuda-b-bin/`, `../build-rocm-docker/`  
**Status (2026-06-27):** Path B on Config A-D; **Path-B Plus B+1 PRODUCTION READY** (Phase 5). **4-GPU Config G primary STABLE** (Phase 10): romulus 7900 + 3060 + 5060 + 5070, G ~40 t/s, no RX6600. Windows 2-device default: `ts=50,50`, G=**48.9 t/s**. Start: [docs/rpc-path-b-plus-overview.md](docs/rpc-path-b-plus-overview.md), [CLUSTER-4GPU-PRIMARY.md](../docs/cuda-windows-5070ti/CLUSTER-4GPU-PRIMARY.md). Tracking: [docs/rpc-path-b-plus-tracking.md](docs/rpc-path-b-plus-tracking.md). RX6600 4-GPU: slot-init hang (parked).

---

## What this patch does

llama.cpp can split a model across a **local GPU** and a **remote RPC worker** (`rpc-server` over TCP). Default RPC is synchronous; large models spend time waiting on the network and device sync.

This project adds three incremental optimizations to `ggml-rpc`:

| Path | Protocol | What it changes | Throughput impact (measured) |
|------|----------|-----------------|------------------------------|
| **A1** | 4.1.0 | Batch `SET_TENSOR` into `SET_TENSOR_BATCH` | +3.6% prompt on 9B Config A |
| **A2** | 4.1.0 | Pipeline `GET_TENSOR` receive (defer recv, flush before sync) | ~0% vs A1 alone |
| **B** | 4.2.2 | `EVENT_RECORD`, `caps.async` + `caps.events`, scheduler pipeline copies | **+3% to +14%** gen vs A1 on 27B-36B server matrix; **+7.3%** on 9B |

Path B enables multi-backend **pipeline parallelism** (`sched copies = 4`) when RPC and local backends both support events. Correctness fixes (central drain in `send_rpc_cmd`, `event_wait` coherency) are required for A1+A2+B to run together.

**Recommendation:** Use Path B builds for cross-GPU **llama-server** workloads. Regress with `scripts/rpc-server-bench-matrix.sh`.

---

## Repository layout

```
atomic-llama-cpp-turboquant/            # git repo root (branch Path-B-Event-Support-Pipeline-Plus)
├── ggml/src/ggml-rpc/                  # RPC patch source code
├── build-cuda-b-bin/bin/rpc-server     # Linux Docker + Windows native (see below)
├── build-cuda-b-bin/portable/          # Windows portable bundle (CUDA DLLs included)
├── build-rocm-docker/bin/llama-server, llama-cli
├── docs/cuda-windows-5070ti/           # Windows RTX 5070 Ti build docs + benchmarks
├── scripts/cuda-windows-5070ti/        # Windows native build/smoke scripts
└── rpc-patch/                          # <-- you are here
    ├── README.md                       # this file
    ├── docs/                           # optimization reports, tracking, plans
    ├── scripts/                        # benchmark harnesses
    └── patch/                          # handover, implementation notes, bench-results/
        ├── HANDOVER.md
        ├── bench-results/
        └── PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md
```

Models are **not** in this repo. Benchmarks expect GGUF files under `/mnt/models/` on the host.

---

## Hardware setup (Config A)

Default benchmark topology:

| Role | GPU | VRAM | Docker image | Binary |
|------|-----|------|--------------|--------|
| Client (`llama-server` / `llama-cli`) | AMD RX 7900 XTX | ~24 GB | `llama-rocm-patched` | `build-rocm-docker/bin/` |
| Worker (`rpc-server`) | NVIDIA RTX 3060 Ti | ~8 GB | `llama-rpc-cuda-a2` | `build-cuda-b-bin/bin/` (Path B) |

- RPC listens on `127.0.0.1:50051` (`--network host`).
- **Never** pin client and worker to the same GPU.
- With `--rpc`, device index **0 = RPC0**, **1 = ROCm0**. Use percentage tensor-split, e.g. `-ts 10,90` (10% RPC, 90% ROCm).

**72B note:** On **Config C** (~45 GB, `ts=35,15,50`), dense 72B runs at ~5 t/s with `fit off` ngl=60 (RPC event-drain fix, 2026-06-26). MoE **coder-next-q4** up to 21 t/s. qwen-next-80b needs `fit on`. Matrix phases 1-4 complete. See `patch/bench-results/72b-matrix/README.md`, `scripts/pathb-72b-vram-calc.py`, `scripts/pathb-72b-matrix.sh`.

### Config B: Romulus + remus (remote 5060 Ti)

| Role | GPU | Host |
|------|-----|------|
| Client | 7900 XTX | Romulus |
| Worker | 5060 Ti 16 GB | remus.local |

```bash
export PATHB_REMUS_SSH_PASS=...   # or use SSH keys
./rpc-patch/scripts/pathb-remus-rpc.sh start
BENCH_CONFIG=remus ./rpc-patch/scripts/rpc-server-bench-matrix.sh
```

See [docs/rpc-multi-node-remus.md](docs/rpc-multi-node-remus.md). Deploy files: [deploy/Atomic-Llama-Remus-PathB/](deploy/Atomic-Llama-Remus-PathB/).

### Config D: Romulus + remus RX 6600 (ROCm RPC)

| Role | GPU | Host |
|------|-----|------|
| Client | 7900 XTX | Romulus |
| Worker | RX 6600 8 GB | remus.local |

Compartmentalized ROCm images (server + rpc separate). See [docs/rpc-remus-rx6600.md](docs/rpc-remus-rx6600.md). Deploy: [deploy/Atomic-Llama-Remus-RX6600/](deploy/Atomic-Llama-Remus-RX6600/).

```bash
export PATHB_REMUS_DOCKER_DIR=~/docker/Atomic-Llama-Remus-RX6600
export BENCH_RPC_MODE=remote BENCH_CTK=turbo3 BENCH_CTV=turbo3 BENCH_TS=1,1
./rpc-patch/scripts/rpc-server-bench.sh pathb remus-rx6600-9b-turbo3-ts11
```

9B on 8 GB RPC worker: **turbo3/turbo3** KV cache (`-ctk turbo3 -ctv turbo3`).

### Windows native: RTX 5070 Ti build host

| Role | GPU | Host | Binary |
|------|-----|------|--------|
| Standalone `llama-server` (phase 1) | RTX 5070 Ti 16 GB | Windows 11 | `build-cuda-b-bin/portable/` |
| Future `rpc-server` worker | same | same | `build-cuda-b-bin/portable/rpc-server.exe` |

Uses the **same** `build-cuda-b-bin` tree and Path B flags (`GGML_RPC=ON`, `GGML_SCHED_MAX_COPIES=4`) as Linux benches.
Portable binary also targets Ampere (`86-real`) for 3070/3090 nodes.

```powershell
.\scripts\cuda-windows-5070ti\build.ps1
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1
```

Docs: [docs/cuda-windows-5070ti/README.md](../docs/cuda-windows-5070ti/README.md).  
Multi-node benches (remus RPC from Windows): [docs/cuda-windows-5070ti/MULTI-NODE.md](../docs/cuda-windows-5070ti/MULTI-NODE.md).

| Config | Topology | ts default | VRAM planner |
|--------|----------|------------|--------------|
| E | remus 5060 Ti + Windows 5070 Ti | `50,50` | `pathb-72b-vram-calc.py --config config-e` |
| F | remus 5060 + RX6600 + Windows 5070 Ti | `30,12,58` | `pathb-72b-vram-calc.py --config config-f` |
| G | romulus 7900 + 3060 + 5060 + 5070 (no 6600) | `36,24,24,16` | `pathb-romulus-4gpu-bench.sh` |

Windows bench results (2026-06-27): Config E through 27B; Config F through 80B MoE.  
Romulus 4-GPU primary (2026-06-27): G 38-43 t/s, load ~85s. See `patch/bench-results/cluster-4gpu-primary/summary.md`.  
**Handover:** `patch/HANDOVER-CLUSTER-4GPU-2026-06-27.md`
Artifact index: [docs/cuda-windows-5070ti/benchmarks/README.md](../docs/cuda-windows-5070ti/benchmarks/README.md).

Linux Docker deploy (`rpc-patch/deploy/`) is separate; do not edit those files for Windows builds.

---

## Quick start

```bash
cd /path/to/atomic-llama-cpp-turboquant   # git repo root

# Clean stale containers (important)
docker rm -f bench-rpc bench-llama pathb-rpc 2>/dev/null

# Full server matrix: A1 vs A1+A2 vs Path B (27B, 31B, 35B, 36B)
./rpc-patch/scripts/rpc-server-bench-matrix.sh

# Single 9B Path B run
export BENCH_MODEL=/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf \
       BENCH_CTX=8192 BENCH_CTK=q8_0 BENCH_CTV=turbo3 BENCH_TS=1,1
./rpc-patch/scripts/rpc-server-bench.sh pathb my-9b-run

# Path B cli regression (4B)
./rpc-patch/scripts/pathb-start-rpc.sh
./rpc-patch/scripts/pathb-test.sh 4b

# 72B (server + auto-fit)
./rpc-patch/scripts/pathb-test.sh qwen72b
```

Rebuild after changing `ggml-rpc`:

```bash
cmake --build build-cuda-b-bin --target rpc-server -j$(nproc)
cmake --build build-rocm-docker --target llama-server llama-cli -j$(nproc)
```

---

## Scripts (`scripts/`)

All scripts resolve paths from `rpc-patch/` automatically; run from **git repo root** or any cwd. Logs default to `rpc-patch/patch/bench-results/` unless overridden.

| Script | Purpose |
|--------|---------|
| `pathb-start-rpc.sh` | Start named CUDA `rpc-server` container (`pathb-rpc`) on :50051 |
| `pathb-run-test.sh` | Low-level `llama-cli` cross-GPU test with log polling |
| `pathb-test.sh` | Presets: `4b`, `matrix-4b`, `gemma12b`, `qwen27b`, `qwen72b`, `kimi72b`, ... |
| `pathb-72b-server.sh` | 72B via `llama-server` + `--fit on` (called by `pathb-test.sh qwen72b`) |
| `pathb-72b-vram-calc.py` | Print viable `-ngl` / `-ts` combos for 72B on 24+8 GB VRAM |
| `pathb-72b-matrix.sh` | Phased 72B+ matrix (Config C, phases 1-4, eval prompts) |
| `rpc-server-bench.sh` | Single `llama-server` <-> `rpc-server` bench; variants `a1`, `a1a2`, `pathb` |
| `rpc-server-bench-matrix.sh` | Full comparison matrix across model sizes |
| `rpc-ts-fit-probe.sh` | Probe tensor-split / `--fit` with `rocm-smi` + `nvidia-smi` during load |

### Common environment variables

| Variable | Used by | Meaning |
|----------|---------|---------|
| `BENCH_MODEL`, `BENCH_CTX`, `BENCH_CTK`, `BENCH_CTV`, `BENCH_TS`, `BENCH_NGL` | server bench | Model and llama-server flags |
| `BENCH_GEN_TOKENS`, `BENCH_RUNS`, `BENCH_NO_WARMUP` | server bench | Generation length, repeats, `--no-warmup` |
| `BENCH_EXTRA` | server bench | Extra llama-server args (e.g. `--fit on`) |
| `BENCH_LOG_DIR` | server bench | Default: `patch/bench-results/rpc-server-bench` |
| `PATHB_LOG_DIR` | pathb scripts | Default: `patch/bench-results/pathb-runs` |
| `PROBE_*` | fit probe | Same pattern as `BENCH_*` for probe harness |
| `LLAMA_TURBOQUANT_ROOT` | all | Override path to `atomic-llama-cpp-turboquant/` |

### Tensor-split cheat sheet

| Model | `-ts` | `-ngl` | ctx | cache | extras |
|-------|-------|--------|-----|-------|--------|
| 9B | `1,1` | 99 | 8192 | q8_0 / turbo3 | Config A/C client |
| 9B (RX 6600 8GB) | `1,1` | 99 | 8192 | **turbo3 / turbo3** | Config D worker VRAM |
| 27B / 31B | `10,90` | 99 | 8192 | q8_0 / turbo3 | `--no-warmup -np 1` |
| 35B / 36B MoE | `10,90` | 99 | 4096 | q4_0 | `--n-cpu-moe 8` |
| 72B IQ4_XS | `10,90` | 0 + `--fit on` | 1024 | q4_0 | `pathb-72b-server.sh` |

**Avoid:** `ts=90,10` (OOM on 8 GB RPC), `ts=4,1` style ratios (80% on RPC), `-ngl 99` on 72B.

---

## Documentation guide

Read in this order if you are new:

1. **[patch/HANDOVER.md](patch/HANDOVER.md)** — executive summary, code changes, hardware, scripts, benchmark results.
2. **[docs/rpc-remus-rx6600.md](docs/rpc-remus-rx6600.md)** — Config D RX 6600 ROCm Docker (standalone + RPC).
3. **[docs/RPC_PATH_AB_OPTIMIZATION_REPORT.md](docs/RPC_PATH_AB_OPTIMIZATION_REPORT.md)** — detailed analysis, gaps, anti-patterns, future work.
4. **[docs/rpc-path-b-tracking.md](docs/rpc-path-b-tracking.md)** — Path B checklist, verification logs, issues log (actively maintained).
5. **[patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md](patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md)** — implementation snippets and insertion points.
6. **[docs/rpc-path-a-tracking.md](docs/rpc-path-a-tracking.md)** — Path A1/A2 history and benchmarks.

Supporting / historical:

- `docs/rpc-path-b-plan.md`, `docs/rpc-path-b-handover.md` — design and earlier handover
- `docs/rpc-path-c-plan.md` — planned Path C (not shipped)
- `patch/HANDOVER_B4_FINAL_VERIFICATION.md` — B4 close checklist (completed)
- `patch/bench-results/README.md` — log directory layout

---

## Code changes (where to look)

Path B touches RPC protocol and scheduler integration:

```
atomic-llama-cpp-turboquant/
  ggml/include/ggml-rpc.h              # RPC_PROTO_MINOR_VERSION, EVENT_RECORD
  ggml/src/ggml-rpc/transport.h        # message types
  ggml/src/ggml-rpc/ggml-rpc.cpp       # batch, pipeline, events, drain fixes
```

Upstream llama.cpp pipeline detection lives in `src/llama-context.cpp` (`pipeline parallelism enabled`, `sched copies = N`). No changes required there for this patch; Path B makes RPC backends event-capable so the existing scheduler path activates.

Docker images:

- `llama-rpc-cuda` — A1 rpc-server (image binary, v4.1.0)
- `llama-rpc-cuda-a2` — A1+A2 rpc-server image; also used as CUDA base for Path B mount
- `llama-rocm-patched` — ROCm client

---

## Benchmark artifacts

All run logs live under **`patch/bench-results/`** (not `/tmp`):

| Subdirectory | Harness | Typical files |
|--------------|---------|---------------|
| `rpc-server-bench/` | `rpc-server-bench.sh`, matrix | `<label>.meta`, `.result`, `-server.log` |
| `rpc-ts-probe/` | `rpc-ts-fit-probe.sh` | `<label>.meta`, `.gpu`, `-server.log` |
| `pathb-runs/` | `pathb-run-test.sh`, `pathb-test.sh` | `<label>.log`, `.meta`, `.raw` |
| `72b-matrix/` | `pathb-72b-matrix.sh` | 72B+ phased matrix; see `72b-matrix/README.md` |
| `remus-rx6600/` | standalone + Config D smoke | `summary.txt` |

Summary: `patch/bench-results/rpc-server-bench/matrix-summary.txt`

---

## Key results (Config A, 2026-06-25)

**llama-server matrix** (`ts=10,90`, `--no-warmup -np 1`), Path B vs A1 generation t/s:

| Model | A1 | Path B | Delta |
|-------|-----|--------|-------|
| 9B (ts=1,1) | 54.4 | 58.4 | +7.3% |
| 27B | 27.1 | 27.9 | +3% |
| 31B | 23.5 | 26.7 | +13.6% |
| 35B MoE | 58.9 | 64.2 | +9% |
| 36B MoE | 57.1 | 61.1 | +7% |

**llama-cli Path B regression:** Qwen3.5-4B Q4_K_M ~**110 t/s** gen (primary fast regression).

---

## Contributing / upstream

This is a **private fork patch**, not an upstream llama.cpp PR. If merging upstream, read `atomic-llama-cpp-turboquant/AGENTS.md` and `CONTRIBUTING.md`: human-authored changes only, disclose AI assist, no automated PR submission.

---

## Git remote

Feature branch `Path-B-Event-Support-Pipeline-Plus` is pushed to the local Gitea instance:

`http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git` (remote name: `gitea`)

Upstream GitHub remains `origin` for the base turboquant repo.