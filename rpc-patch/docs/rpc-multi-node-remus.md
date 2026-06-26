# Multi-Node Path B: Romulus + remus

**Date:** 2026-06-26  
**Branch:** `Path-B-Event-Support` (RPC v4.2.2)

## Topology

| Config | llama-server | RPC worker(s) | Notes |
|--------|--------------|---------------|-------|
| **A** (baseline) | Romulus 7900 XTX | Romulus 3060 Ti localhost | `matrix-summary.txt` |
| **B** | Romulus 7900 XTX | remus 5060 Ti 16GB remote | `-ts 15,85` |
| **C** | Romulus 7900 XTX | remus 5060 Ti + Romulus 3060 Ti | `-ts 12,8,80` |
| **D** | Romulus 7900 XTX | remus RX 6600 8GB ROCm | `-ts 1,1` (9B) |

- remus hostname: `remus.local` (192.168.8.176 from Romulus DNS; also 192.168.8.22 on remus)
- RPC port: **50051** (TCP, host network)
- Models: `/mnt/models` on Romulus only (workers do not mount GGUFs)

## remus Docker stack

Path: `~/docker/Atomic-Llama-Remus-PathB/` on remus  
Repo copy: [`rpc-patch/deploy/Atomic-Llama-Remus-PathB/`](../deploy/Atomic-Llama-Remus-PathB/)

```bash
# on remus
cd ~/docker/Atomic-Llama-Remus-PathB
./build.sh          # GPU compile via docker run + runtime image
docker compose up -d
```

Image: `atomic-llama-remus-pathb-rpc:latest`  
Build uses Squid proxy at `192.168.8.108:3128`.

## Romulus orchestration

```bash
export PATHB_REMUS_SSH_PASS=...   # or SSH key

# disk preflight + bench cleanup
./rpc-patch/scripts/pathb-disk-preflight.sh --cleanup --remote

# remus RPC lifecycle
./rpc-patch/scripts/pathb-remus-rpc.sh start|stop|status|logs

# Config B matrix
BENCH_CONFIG=remus ./rpc-patch/scripts/rpc-server-bench-matrix.sh

# Config C (dual RPC)
./rpc-patch/scripts/pathb-multi-rpc.sh start
BENCH_CONFIG=config-c ./rpc-patch/scripts/rpc-server-bench-matrix.sh
```

### Bench env vars

| Var | Config B | Config C |
|-----|----------|----------|
| `BENCH_RPC_MODE` | `remote` | `multi` |
| `BENCH_RPC_HOST` | `remus.local` | - |
| `BENCH_RPC_ENDPOINT` | - | `remus.local:50051,127.0.0.1:50051` |
| `BENCH_TS` | `15,85` | `12,8,80` |

## Disk / cleanup policy

- Run `pathb-disk-preflight.sh --cleanup` before matrix; never `docker system prune -a`
- Post-matrix: remove `bench-rpc`, `bench-llama` only; keep remus compose dir + image
- Abort if Romulus root free < 20 GB

## VRAM helpers

```bash
python3 rpc-patch/scripts/pathb-72b-vram-calc.py --config remus
python3 rpc-patch/scripts/pathb-72b-vram-calc.py --config config-c
./rpc-patch/scripts/pathb-model-fit-scan.sh --config config-c
```

## Results

### Smoke (Config B, 9B)

| Label | avg G (t/s) | Notes |
|-------|-------------|-------|
| remus-smoke-9b | 88.3 | load 10s, ts=15,85 |

### Matrix Config B (`ts=15,85`, remus 5060 Ti remote)

| Model | Config A pathb (local 3060 Ti) | Config B remus | Delta |
|-------|-------------------------------|----------------|-------|
| 27B Qwen | 27.9 t/s | **30.9** | +11% |
| 31B Gemma | 26.7 | **29.1** | +9% |
| 35B MoE | 64.2 | **81.1** | +26% |
| 36B MoE | 61.1 | **79.9** | +31% |

Full log: [`matrix-summary-remus.txt`](../patch/bench-results/rpc-server-bench/matrix-summary-remus.txt)

### Matrix Config C (`ts=12,8,80`, remus + local 3060 Ti)

| Model | Config B remus | Config C dual-RPC | Notes |
|-------|----------------|-------------------|-------|
| 27B | 30.9 | 27.4 | dual-RPC slower |
| 31B | 29.1 | 26.2 | dual-RPC slower |
| 35B MoE | 81.1 | 54.2 | MoE split across 2 RPC hurts |
| 36B MoE | 79.9 | 51.6 | same |

**Recommendation:** Use **Config B** (single remus 5060 Ti worker) for this hardware pair.

Full log: [`matrix-summary-config-c.txt`](../patch/bench-results/rpc-server-bench/matrix-summary-config-c.txt)

### Large model: Qwen3-72B IQ4_XS (Config B)

| Label | Load | G (t/s) | Notes |
|-------|------|---------|-------|
| remus-qwen72b-fit | 21s | 1.9 | `--fit on -ngl 0`; device_info showed ROCm+CPU only (RPC not engaged) |

72B fits in combined 40 GB GPU budget but generation is CPU-bound without RPC layer offload at `-ngl 0`.

### 72B+ matrix Config C (`ts=35,15,50`, 2026-06-26)

Three workers: remus 5060 Ti (35%) + Romulus 3060 Ti (15%) + 7900 XTX (50%) = ~44.5 GB. Phases 1-4 **complete** after RPC fix.

| Model | Production config | G (t/s) |
|-------|-------------------|---------|
| Llama-70B / Qwen-72B / Kimi-72B | fitoff ngl=60, -fa on | ~5 |
| Qwen-Next-80B Q5_K_M | fiton ngl=0 ncmoe=8 | ~16 |
| Coder-Next Q4_K_M MoE | fitoff ngl=60 ncmoe=8 | 16-21 |

Harness: `PATHB_CUDA_DISABLE_GRAPHS=1 ./rpc-patch/scripts/pathb-72b-matrix.sh --phase all`  
Cheatsheet: [`phase-summary.txt`](../patch/bench-results/72b-matrix/phase-summary.txt)  
Full write-up: [`patch/bench-results/72b-matrix/README.md`](../patch/bench-results/72b-matrix/README.md)

**RPC fix:** `ggml-rpc.cpp` -- drain deferred `EVENT_RECORD` before `GET_TENSOR`. Rebuild rpc-server + libggml-rpc; `docker cp` to remus.

## Config D: Romulus + remus RX 6600 (ROCm RPC)

**Status:** smoke PASS (2026-06-26). Full doc: [rpc-remus-rx6600.md](rpc-remus-rx6600.md)

Path on remus: `~/docker/Atomic-Llama-Remus-RX6600/`  
Repo: [`deploy/Atomic-Llama-Remus-RX6600/`](../deploy/Atomic-Llama-Remus-RX6600/)

| Image | Role |
|-------|------|
| `atomic-llama-remus-rx6600-server:latest` | Standalone llama-server on RX 6600 |
| `atomic-llama-remus-rx6600-rpc:latest` | RPC worker for Romulus client |

```bash
# on remus (stop pathb-rpc-remus first - same port 50051)
cd ~/docker/Atomic-Llama-Remus-RX6600
./build-rpc.sh
docker compose up -d rx6600-rpc
```

```bash
# from romulus
export PATHB_REMUS_DOCKER_DIR=~/docker/Atomic-Llama-Remus-RX6600
export BENCH_RPC_MODE=remote BENCH_RPC_HOST=remus.local BENCH_TS=1,1
export BENCH_CTK=turbo3 BENCH_CTV=turbo3
./rpc-patch/scripts/rpc-server-bench.sh pathb remus-rx6600-9b-turbo3-ts11
```

9B on 8 GB worker: use `turbo3/turbo3` KV cache. Smoke result: **G=26.5 t/s** avg (`ts=1,1`, ctx=8192).

Bench summary: [`patch/bench-results/remus-rx6600/summary.txt`](../patch/bench-results/remus-rx6600/summary.txt)

## Open roadmap

- Config D matrix (27B+) on RX 6600 8 GB worker
- Path C single-node multi-GPU aggregation (same physical rpc-server)