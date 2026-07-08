# Docker Blueprints

Deployable Docker images and start scripts for llama.cpp with CUDA + Path-B+ features.

## Layout

```
dockers/blueprints/
  atomic-llama-cuda-pathb/     # CUDA server + rpc-server (Blackwell-ready)
    Dockerfile.server
    Dockerfile.rpc
    build-server.sh
    build-rpc.sh
    run-server.sh
    run-rpc.sh
    docker-compose.yml
```

## Quick Start

```bash
cd atomic-llama-cuda-pathb

# Build (requires --gpus all on host)
./build-server.sh
./build-rpc.sh

# Run
./run-server.sh -m /path/to/model.gguf
./run-rpc.sh
```

## Images

| Image | Port | Purpose |
|-------|------|---------|
| `atomic-llama-cuda-pathb-server:latest` | 8080 | OpenAI-compatible HTTP server |
| `atomic-llama-cuda-pathb-rpc:latest` | 50051 | RPC backend device |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GIT_BRANCH` | `Path-B-Event-Support-Pipeline-Plus` | Git branch to build from |
| `GIT_COMMIT` | (empty) | Specific commit to checkout |
| `PROXY_IP` | `192.168.8.108` | Squid proxy for apt |
| `CTX` | `256000` | Total context length |
| `PARALLEL` | `2` | Number of concurrent slots |

## Run Defaults

| Parameter | Default |
|-----------|---------|
| context | 256000 (128k per slot x 2) |
| KV cache | turbo3 / turbo3 |
| MTP rounds | 1 |
| Draft tokens | 2 |
| Parallel slots | 2 |
| GPU layers | 99 |
| Batch size | 2048 |

## Related

- [Blackwell Build Guide](../../docs/blackwell/README.md)
- [Docker Docs](../../docs/docker.md)
- [CUDA Build Guide](../../docs/build.md#cuda)
