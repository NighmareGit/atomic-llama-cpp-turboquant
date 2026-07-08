# Blackwell GPU Build Guide (RTX 50-Series, Compute 12.0)

Build, run, and deploy llama.cpp with Blackwell GPUs (RTX 5060 Ti, 5070 Ti, 5080, 5090) — compute capability **12.0**, code-named `120a-real`.

## Quick Reference

| Item | Value |
|------|-------|
| GPU arch | Blackwell (compute 12.0) |
| nvcc arch flag | `120a-real` |
| CUDA Toolkit | 12.8+ recommended, 12.9 tested |
| CMake arg | `-DCMAKE_CUDA_ARCHITECTURES="120a-real"` |
| Native build fallback | `nvcc warning : Cannot find valid GPU for '-arch=native'` — use explicit `120a-real` |
| Blackwell FP4 | `BLACKWELL_NATIVE_FP4 = 1` in `system_info` output |

## Prerequisites

- NVIDIA driver >= 570 (supports Blackwell)
- CUDA Toolkit 12.8+ (12.9 tested with `nvidia/cuda:12.9.1-devel-ubuntu22.04`)
- CMake 3.24+, Ninja, build-essential

## CMake Build

```bash
cmake -B build -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="120a-real" \
    -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DGGML_CUDA_MMQ=ON
cmake --build build --config Release -- -j$(nproc)
```

**Key flags for Blackwell:**

| Flag | Purpose |
|------|---------|
| `120a-real` | Native Blackwell kernels (required — `native` auto-detect fails) |
| `GGML_CUDA_F16=ON` | FP16 compute support |
| `GGML_CUDA_K_QUANTS=ON` | K-quant kernels (turbo2/turbo3/turbo4 KV cache) |
| `GGML_CUDA_MMQ=ON` | Custom MMQ matmul kernels for quantized models |
| `GGML_SCHED_MAX_COPIES=4` | Pipeline overlap for speculative decoding |

**Never use `-DCMAKE_CUDA_ARCHITECTURES=native` on Blackwell.** nvcc cannot auto-detect compute 12.0 and will silently fall back to a generic arch, producing kernels that fail at runtime with `GRAPH_COMPUTE` errors.

## Docker Build (Host-Compiled)

For Docker images where compilation happens on the GPU host (not inside the container), use the two-stage approach:

```bash
# Step 1: Compile on GPU host, copy artifacts to staging/
./build-server.sh   # or build-rpc.sh

# Step 2: Pack into runtime image (no GPU needed)
docker build -f Dockerfile.server -t atomic-llama-cuda-pathb-server:latest .
```

See [../../docker/Atomic-Llama-CUDA-PathB/](../../docker/Atomic-Llama-CUDA-PathB/README.md) for full build/run scripts.

## Runtime

```bash
# Direct binary
./build/bin/llama-server -m model.gguf -c 128000 -ctk turbo3 -ctv turbo3 \
    -ngl 99 --parallel 2 --host 0.0.0.0

# Docker
docker run --rm --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=0 --env CUDA_VISIBLE_DEVICES=0 \
    --publish 8080:8080 \
    --volume /path/to/models:/models:ro \
    atomic-llama-cuda-pathb-server:latest \
    --model /models/model.gguf -c 128000 -ctk turbo3 -ctv turbo3 \
    -ngl 99 --parallel 2
```

## TurboQuant KV + Blackwell

Blackwell GPUs fully support TurboQuant KV cache types (`turbo2`, `turbo3`, `turbo4`) via the CUDA backend's `turbo3`/`turbo4` paths. The `turbo2` path falls back to the reference implementation.

Recommended: `-ctk turbo3 -ctv turbo3` for ~4.3x KV compression with dedicated CUDA kernels.

## MTP Speculative Decoding + Blackwell

MTP speculative decoding (`--spec-type draft-mtp --spec-draft-n-max 2`) works with Blackwell. The server logs `speculative decoding will use checkpoints` when enabled.

## Known Issues

### nvcc native auto-detect fails

```
nvcc warning : Cannot find valid GPU for '-arch=native', default arch is used
```

Always specify `-DCMAKE_CUDA_ARCHITECTURES="120a-real"` explicitly.

### rpc-server abort on 5070 Ti

If `rpc-server.exe` crashes with an abort dialog during `GRAPH_COMPUTE`, the build lacked `120a-real` kernels. Rebuild with the correct arch. See [../cuda-windows-5070ti/RPC-BUG-HUNT.md](../cuda-windows-5070ti/RPC-BUG-HUNT.md).

## Benchmarks

| GPU | Model | Mode | TPS | Context |
|-----|-------|------|----:|---------|
| RTX 5060 Ti (16GB) | Qwen3.5-9B-MTP | turbo3 + MTP | ~70 tok/s | 128k/slot |
| RTX 5070 Ti (16GB) | gemma-4-E4B | turbo3 + MTP | ~67.8 tok/s | 128 |
| RTX 5070 Ti (16GB) | gemma-4-E4B | turbo3 + MTP | ~64.5 tok/s | 512 |

See [../cuda-windows-5070ti/benchmarks/](../cuda-windows-5070ti/benchmarks/) for full benchmark results.

## Related Docs

- [CUDA Build Guide](../build.md#cuda) — general CUDA build instructions
- [Windows 5070 Ti](../cuda-windows-5070ti/README.md) — Windows build + multi-node RPC
- [Docker](../docker.md) — official llama.cpp Docker images
- [Docker Path-B+](../../docker/Atomic-Llama-CUDA-PathB/README.md) — host-compiled Docker images with RPC
