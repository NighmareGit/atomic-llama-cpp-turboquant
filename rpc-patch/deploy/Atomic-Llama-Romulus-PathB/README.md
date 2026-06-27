# Atomic-Llama-Romulus-PathB

Path B (v4.2.2) CUDA rpc-server for romulus RTX 3060 Ti 8GB.

## Build (on romulus, requires GPU)

```bash
cd ~/docker/Atomic-Llama-Romulus-PathB
./build.sh
```

Uses Squid proxy at 192.168.8.108:3128 when reachable.

## Run

```bash
docker compose up -d
docker compose logs -f
```

Listens on `0.0.0.0:50051`. Remote llama-server connects via `--rpc 192.168.8.108:50051`.

## Disk

- build.sh aborts if root free < 25 GB
- After build: `docker builder prune -f` safe; keep this directory and `atomic-llama-romulus-pathb-rpc:latest`