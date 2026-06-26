# Atomic-Llama-Remus-PathB

Path B (v4.2.2) CUDA rpc-server for remus RTX 5060 Ti 16GB.

## Build (on remus, requires GPU)

```bash
cd ~/docker/Atomic-Llama-Remus-PathB
./build.sh
```

Uses Squid proxy at 192.168.8.108:3128 when reachable.

## Run

```bash
docker compose up -d
docker compose logs -f
```

Listens on `0.0.0.0:50051`. Romulus llama-server connects via `--rpc remus.local:50051`.

## Disk

- build.sh aborts if root free < 25 GB
- After build: `docker builder prune -f` safe; keep this directory and `atomic-llama-remus-pathb-rpc:latest`