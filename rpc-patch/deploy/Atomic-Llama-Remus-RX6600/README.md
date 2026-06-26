# Atomic-Llama-Remus-RX6600

Compartmentalized ROCm Docker images for remus AMD Radeon RX 6600 (gfx1032, RDNA2, 8 GB).

Full documentation: [`rpc-patch/docs/rpc-remus-rx6600.md`](../../docs/rpc-remus-rx6600.md)

Two separate images (never fused):

| Image | Binary | Port | Phase |
|-------|--------|------|-------|
| `atomic-llama-remus-rx6600-server:latest` | llama-server | 8080 | **Now** |
| `atomic-llama-remus-rx6600-rpc:latest` | rpc-server | 50051 | Later |

`HIP_VISIBLE_DEVICES=0` selects the RX 6600 (not the RTX 5060 Ti). `HSA_OVERRIDE_GFX_VERSION=10.3.0` maps gfx1032 to gfx1030 rocBLAS kernels (required on ROCm 6.4). Containers mount `/dev/kfd` and `/dev/dri` with `video` + `render` groups.

Base image: `rocm/dev-ubuntu-22.04:6.4-complete`. Build target: `gfx1030` (physical card is gfx1032; ROCm 6.4 rocBLAS ships gfx1030 kernels).

## First phase: standalone llama-server on RX 6600

### 1. Copy models from romulus

Models are not on remus permanently. Copy what you need, then clean up after testing.

```bash
# from romulus
mkdir -p ~/models   # on remus, via ssh
sshpass -p 12345 ssh hunter@remus.local 'mkdir -p ~/models'
sshpass -p 12345 scp /mnt/models/Qwen3.5-4B-Q4_K_M.gguf hunter@remus.local:~/models/
sshpass -p 12345 scp /mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf hunter@remus.local:~/models/
```

For larger matrix runs (27B), copy on demand:

```bash
sshpass -p 12345 scp /mnt/models/Qwen3.5-27B-Q5_K_M.gguf hunter@remus.local:~/models/
```

### 2. Build server image (on remus)

```bash
cd ~/docker/Atomic-Llama-Remus-RX6600
./build-server.sh
```

Requires `rocm-smi` on host. Aborts if root free < 25 GB. Uses Squid proxy at `192.168.8.108:3128` when reachable.

### 3. Run llama-server

```bash
docker compose up -d rx6600-llama-server
docker compose logs -f rx6600-llama-server
```

Default compose command loads `Qwen3.5-4B-Q4_K_M.gguf` from `/home/hunter/models` with `-ngl 99 -c 4096`.

Override model or flags by editing `docker-compose.yml` or running directly:

```bash
docker run --rm -d --name rx6600-llama-server \
  --network host --ipc host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -e HIP_VISIBLE_DEVICES=0 \
  -v /home/hunter/models:/models:ro \
  atomic-llama-remus-rx6600-server:latest \
  -m /models/Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -c 4096 \
  -ctk q4_0 -ctv q4_0 \
  --host 0.0.0.0 --port 8080 --no-warmup -np 1
```

### 4. Verify

```bash
# health
curl -s http://remus.local:8080/health

# short generation
curl -s http://remus.local:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in one sentence."}],"max_tokens":32}'

# GPU usage (on remus host while loading)
rocm-smi --showuse --showmeminfo vram
```

Logs should list `AMD Radeon RX 6600`. The NVIDIA card must not be used.

### 5. Benchmark presets (8 GB VRAM)

Adjust `docker-compose.yml` command or use `docker compose run` with overrides.

| Model | File | ngl | ctx | cache |
|-------|------|-----|-----|-------|
| 4B | `Qwen3.5-4B-Q4_K_M.gguf` | 99 | 4096 | q4_0 / q4_0 |
| 9B | `Qwen3.5-9B-MTP-Q4_K_M.gguf` | 99 | 8192 | turbo3 / turbo3 |
| 27B | `Qwen3.5-27B-Q5_K_M.gguf` | 40-60 | 4096 | q4_0 / q4_0 |

27B may need lower `-ngl` and `-c` to fit 8 GB. Remove the GGUF from `~/models/` when done.

### 6. Cleanup models

```bash
rm -f ~/models/Qwen3.5-27B-Q5_K_M.gguf   # large files first
# keep 4B/9B only while actively testing
```

## RPC phase: rpc-server (romulus client + remus worker)

Stop any other service on port 50051 first (e.g. `docker stop pathb-rpc-remus`). Do not run `rx6600-llama-server` and `rx6600-rpc` on the RX 6600 at the same time.

```bash
./build-rpc.sh
docker compose up -d rx6600-rpc    # ROCm0 device, not HIP0
docker compose logs -f rx6600-rpc
```

Romulus llama-server connects with `--rpc remus.local:50051`. Models stay on romulus; the rpc worker does not mount GGUFs.

From romulus:

```bash
export PATHB_REMUS_SSH_PASS=...
export PATHB_REMUS_DOCKER_DIR=~/docker/Atomic-Llama-Remus-RX6600
export REMUS_RPC_IP=192.168.8.176
export BENCH_RPC_MODE=remote BENCH_RPC_HOST=remus.local
export BENCH_TS=1,1 BENCH_NO_WARMUP=1
export BENCH_CTK=turbo3 BENCH_CTV=turbo3   # 9B: compact KV cache for 8 GB worker

./rpc-patch/scripts/rpc-server-bench.sh pathb remus-rx6600-9b-turbo3-ts11
```

For lifecycle logs only (start only the rpc service manually; `pathb-remus-rpc.sh start` would bring up both compose services):

```bash
PATHB_REMUS_DOCKER_DIR=~/docker/Atomic-Llama-Remus-RX6600 ./rpc-patch/scripts/pathb-remus-rpc.sh logs
```

## Disk

- `build-server.sh` / `build-rpc.sh` abort if root free < 25 GB
- After build: `docker builder prune -f` is safe; keep this directory and built images
- Do not touch `~/docker/Atomic-Llama-Remus-PathB/` (CUDA 5060 Ti stack)