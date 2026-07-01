# Atomic-Llama-Triton-PathB

Isolated CUDA rpc-server containers for triton RTX 3090 (`:50054`) and RTX 3070 (`:50055`).

Each container binds one physical GPU as `CUDA0` (no multi-device RPC backend).

## Build (on triton)

```bash
cd ~/docker/Atomic-Llama-Triton-PathB
# after: cmake --build build-cuda-b-bin --target rpc-server
./build-from-bin.sh
```

## Run

```bash
docker compose up -d
docker compose ps
ss -tlnp | grep -E '50054|50055'
```

Stop native host rpc-server before starting compose:

```bash
pkill -f 'rpc-server.*50054' || true
pkill -f 'rpc-server.*50055' || true
```

## 5-GPU client (romulus)

```
-rpc 192.168.8.176:50051,127.0.0.1:50051,192.168.8.23:50054,192.168.8.23:50055
-ts  19,10,30,10,31
```