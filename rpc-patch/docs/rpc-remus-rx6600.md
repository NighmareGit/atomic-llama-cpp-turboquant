# remus RX 6600 ROCm Docker (Config D)

**Date:** 2026-06-26  
**Branch:** `Path-B-Event-Support` (RPC v4.2.2)  
**Host:** `remus.local` (192.168.8.176)

ROCm compartmentalized Docker stack for the AMD Radeon RX 6600 (gfx1032 physical, gfx1030 build target, 8 GB VRAM) on remus. Separate images for `llama-server` and `rpc-server`; never fused.

Deploy source: [`rpc-patch/deploy/Atomic-Llama-Remus-RX6600/`](../deploy/Atomic-Llama-Remus-RX6600/)  
Deployed on remus: `~/docker/Atomic-Llama-Remus-RX6600/`

---

## Topology (Config D)

| Role | GPU | Host | Port |
|------|-----|------|------|
| Client (`llama-server`) | RX 7900 XTX 24 GB | Romulus | 8081 (bench) |
| Worker (`rpc-server`) | RX 6600 8 GB | remus.local | 50051 |

| Config | Client | Worker | `-ts` (9B) | Notes |
|--------|--------|--------|------------|-------|
| **D** | Romulus 7900 XTX | remus RX 6600 ROCm | `1,1` | AMD client + AMD worker (ROCm RPC) |
| B (legacy) | Romulus 7900 XTX | remus 5060 Ti CUDA | `15,85` | See [rpc-multi-node-remus.md](rpc-multi-node-remus.md) |

Models stay on Romulus `/mnt/models`. RPC worker does not mount GGUFs.

---

## Images

| Image | Binary | When to use |
|-------|--------|-------------|
| `atomic-llama-remus-rx6600-server:latest` | llama-server | Standalone inference on remus RX 6600 |
| `atomic-llama-remus-rx6600-rpc:latest` | rpc-server | Romulus client + remus worker (Config D) |

Base: `rocm/dev-ubuntu-22.04:6.4-complete`. CMake: `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_HIP_ROCWMMA_FATTN=ON`.

### Runtime env (both images)

| Variable | Value | Why |
|----------|-------|-----|
| `HIP_VISIBLE_DEVICES` | `0` | Select RX 6600, not RTX 5060 Ti |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | ROCm 6.4 rocBLAS ships gfx1030 TensileLibrary only |

Device mounts: `/dev/kfd`, `/dev/dri`, groups `video` + `render`, `network_mode: host`.

### rpc-server device flag

Use `-d ROCm0`, not `HIP0`. The binary reports `ROCm0: AMD Radeon RX 6600`.

---

## Build (on remus)

```bash
cd ~/docker/Atomic-Llama-Remus-RX6600

# Standalone server image
./build-server.sh

# RPC worker image (Config D)
./build-rpc.sh
```

Preflight: `rocm-smi` on host, root free >= 25 GB. Squid proxy at `192.168.8.108:3128` when reachable.

---

## Standalone llama-server (phase 1)

Copy models from Romulus when needed (no permanent share yet):

```bash
# from romulus
sshpass -p ... scp /mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf hunter@remus.local:~/models/
```

Stop containers that also use `HIP_VISIBLE_DEVICES=0` (`llama_amd_atom_*`) to free full 8 GB VRAM.

```bash
# on remus
docker stop llama_amd_atom_tq_remus llama_amd_atom_rpc_tq_remus 2>/dev/null || true
cd ~/docker/Atomic-Llama-Remus-RX6600
docker compose up -d rx6600-llama-server
curl -s http://remus.local:8080/health
```

### Recommended flags

| Model | ngl | ctx | ctk / ctv | extras |
|-------|-----|-----|-----------|--------|
| 4B | 99 | 4096 | q8_0 / q8_0 | `--ctx-checkpoints 0` |
| 9B | 99 | 8192 | **turbo3 / turbo3** | `--ctx-checkpoints 0` |
| 27B | 24 | 2048 | q8_0 / q8_0 | partial offload; copy on demand |

9B uses `turbo3` for both KV cache dtypes (`-ctk` and `-ctv`) to fit ctx=8192 in 8 GB (~5.4 GB VRAM at load vs ~7 GB with q8_0/turbo3).

Remove large GGUFs from `~/models/` after testing.

---

## RPC worker (Config D)

Only one service on port 50051. Stop `pathb-rpc-remus` (5060 Ti CUDA stack) before starting RX 6600 RPC:

```bash
docker stop pathb-rpc-remus
cd ~/docker/Atomic-Llama-Remus-RX6600
docker compose up -d rx6600-rpc
docker compose logs -f rx6600-rpc
```

Do not run `rx6600-llama-server` and `rx6600-rpc` on the RX 6600 simultaneously.

---

## Romulus orchestration (Config D)

```bash
export PATHB_REMUS_SSH_PASS=...
export PATHB_REMUS_DOCKER_DIR=~/docker/Atomic-Llama-Remus-RX6600
export REMUS_RPC_IP=192.168.8.176
export BENCH_RPC_MODE=remote
export BENCH_RPC_HOST=remus.local
export BENCH_TS=1,1
export BENCH_NO_WARMUP=1
export BENCH_CTK=turbo3
export BENCH_CTV=turbo3

./rpc-patch/scripts/rpc-server-bench.sh pathb remus-rx6600-9b-turbo3-ts11
```

Logs only (start rpc service manually; `pathb-remus-rpc.sh start` would recreate all compose services):

```bash
PATHB_REMUS_DOCKER_DIR=~/docker/Atomic-Llama-Remus-RX6600 \
  ./rpc-patch/scripts/pathb-remus-rpc.sh logs
```

---

## Benchmark results (2026-06-26)

Full summary: [`patch/bench-results/remus-rx6600/summary.txt`](../patch/bench-results/remus-rx6600/summary.txt)

### Standalone llama-server

| Label | Model | Cache | G (t/s) |
|-------|-------|-------|---------|
| rx6600-4b-qwen | Qwen3.5-4B | q8_0/q8_0 | 51.7 |
| rx6600-9b-turbo3 | Qwen3.5-9B-MTP | turbo3/turbo3 | 32.4 |
| rx6600-27b-qwen | Qwen3.5-27B | q8_0/q8_0, ngl=24 | 5.1 |

### Cross-node RPC (7900 XTX + RX 6600)

| Label | Model | Cache | G (t/s) avg |
|-------|-------|-------|-------------|
| remus-rx6600-9b-turbo3-ts11 | Qwen3.5-9B-MTP | turbo3/turbo3, ts=1,1 | 26.5 |

Artifacts: `patch/bench-results/rpc-server-bench/remus-rx6600-9b-*.meta`

---

## Gotchas

1. **gfx1032 vs gfx1030:** Build for `gfx1030`; set `HSA_OVERRIDE_GFX_VERSION=10.3.0` at runtime. Without override, rocBLAS fails on first generation (`TensileLibrary.dat` missing for gfx1032).
2. **Context checkpoints:** Qwen3.5 needs `--ctx-checkpoints 0` on standalone server to avoid ROCm crash at slot init.
3. **VRAM sharing:** `llama_amd_atom_tq_remus` and `llama_amd_atom_rpc_tq_remus` also pin `HIP_VISIBLE_DEVICES=0` and reserve ~4 GB. Stop them before 9B+ standalone or large RPC loads.
4. **Port 50051:** `pathb-rpc-remus` (CUDA 5060 Ti) and `rx6600-rpc` conflict. Only one RPC worker on remus at a time unless using a different port.
5. **PathB folder untouched:** `~/docker/Atomic-Llama-Remus-PathB/` remains the CUDA 5060 Ti stack; Config D uses the separate RX6600 folder.

---

## Related docs

- [rpc-multi-node-remus.md](rpc-multi-node-remus.md) - Config B/C (5060 Ti CUDA)
- [deploy/Atomic-Llama-Remus-RX6600/README.md](../deploy/Atomic-Llama-Remus-RX6600/README.md) - deploy quick reference
- [patch/HANDOVER.md](../patch/HANDOVER.md) - full project handover