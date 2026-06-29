# Config G Primary: 4-GPU cluster (stable, no RX6600)

**Date:** 2026-06-29 (tensor-split + JUPITER arch ops updated)  
**Status:** **PRODUCTION STABLE** (3/3 gate PASS 2026-06-27); B+6 gate profiling in progress  
**Branch:** `Path-B-Event-Support-Pipeline-Plus` @ `cc4df4f87`+

## Topology

| Cluster | GPU | Role | Host | RPC endpoint |
|---------|-----|------|------|--------------|
| client | RX 7900 XTX | ROCm **client** (profiler / llama-server) | romulus | (local HIP0) |
| RPC0 | RTX 5060 Ti | CUDA RPC worker | remus | `192.168.8.176:50051` |
| RPC1 | RTX 3060 Ti | CUDA RPC worker | romulus docker | `127.0.0.1:50051` |
| RPC2 | RTX 5070 Ti | CUDA RPC worker | JUPITER (Windows) | `192.168.8.21:50053` |

**Do not use RX6600 (`:50052`) in this path.** Legacy 4-GPU with 6600 hangs at `initializing slots` (slot-init deadlock). See [RPC-BUG-HUNT.md](RPC-BUG-HUNT.md#rx6600-slot-init-hang-2026-06-27).

```text
romulus 7900 (ROCm client)
    |-- TCP --> remus 5060 (:50051)
    |-- TCP --> romulus 3060 (:50051)
    '-- TCP --> Windows 5070 (:50053)
```

## Recommended flags

| Setting | Value |
|---------|-------|
| Model (romulus) | `/mnt/models/Qwen3.6-35B-A3B-APEX-I-Quality.gguf` |
| `-rpc` | `192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053` |
| `-ts` | `25,12,25,38` (see tensor-split order below) |
| `-ctk` / `-ctv` | `q8_0` |
| `GGML_PIPELINE_PLUS` | `1` |
| `--n-cpu-moe` | omit (do not use `8` on ROCm 35B APEX) |
| `BENCH_TRACE` | `0` for smoke; `1` for hotpath |

### Tensor-split index order (`-ts`)

With tensor parallelism, **RPC endpoints are registered first**, then the local ROCm GPU (`llama.cpp` default device list). The `-ts` comma positions follow that order, **not** the cluster table above.

| `-ts` index | Device | VRAM (live validate-rpc) | Share |
|-------------|--------|--------------------------|-------|
| 0 | RPC0 remus 5060 Ti | 15849 MB | 25 |
| 1 | RPC1 romulus 3060 Ti | 7841 MB | 12 |
| 2 | RPC2 JUPITER 5070 Ti | 16302 MB | 25 |
| 3 | ROCm0 romulus 7900 XTX | 24560 MB | 38 |

Legacy `36,24,24,16` (documented as 7900/5060/5070/3060) **over-allocated the 8 GB 3060** (24%) and **under-allocated the 16 GB 5070** (16%). Recompute with `rpc-patch/scripts/pathb-72b-vram-calc.py --config config-g --ts 25,12,25,38`.

Profiler sched backends (romulus trace): `backend1` = 5060 straggler on old split; after rework watch `backend2` (5070) under higher 5070 share.

## Bench launcher (romulus client)

```bash
export PATHB_ROMULUS_SSH_PASS=...
# Windows rpc-server must be up first:
#   powershell -File scripts/cuda-windows-5070ti/pathb-rpc-server.ps1

cd rpc-patch/scripts
BENCH_CTK=q8_0 BENCH_CTV=q8_0 BENCH_NCMOE= BENCH_TRACE=0 GGML_PIPELINE_PLUS=1 \
  ./pathb-romulus-4gpu-bench.sh trace-g-4gpu-primary
```

Presets on `pathb-romulus-4gpu-bench.sh`:
- `BENCH_4GPU_PRESET=primary` (default) -- this topology
- `BENCH_4GPU_PRESET=legacy-6600` -- experimental; known hang

3-GPU subset (7900 + 5060 + 3060): `pathb-romulus-3gpu-bench.sh primary-5060-3060`

## Measured results (2026-06-27)

| Label | Load | G (t/s) | Result |
|-------|------|---------|--------|
| `trace-g-4gpu-primary` | ~85s | 43.0 | PASS |
| `trace-g-4gpu-primary-r2` | ~85s | 38.2 | PASS |
| `trace-g-4gpu-primary-r3` | ~85s | 43.1 | PASS |
| `trace-g-4gpu-primary-trace` | ~85s | ~37 | trace OK (curl flake run 3) |

Artifacts on romulus: `rpc-patch/patch/bench-results/rpc-server-bench/trace-g-4gpu-primary*/`  
Summary: [rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md](../../rpc-patch/patch/bench-results/cluster-4gpu-primary/summary.md)

### Trace hotpath (`trace-g-4gpu-primary-trace`)

| Metric | Value |
|--------|-------|
| Serial split sum | ~20.7 ms/tok |
| Straggler | remus 5060 ~9.6 ms/tok |
| `SET_TENSOR_HASH` | 280 calls / 6.5s (load fan-out, normal) |
| `assembly_overlap_count` | 1075 (B+1 pass) |
| Slot init | completes (no hang) |

## Ops checklist (every run)

1. **JUPITER (Windows)** `rpc-server` on `:50053` listening (`0.0.0.0:50053`):
   ```powershell
   # foreground (debug)
   .\scripts\cuda-windows-5070ti\pathb-rpc-server.ps1

   # persistent schtask (B+6 gate)
   .\scripts\b6-gate-jupiter-start-rpc-task.ps1
   ```
   Foreground start is more reliable than hidden `Start-Process` if the worker exits silently.
   **Before every deploy:** confirm portable was built with `120a-real` (see [BUILD.md](BUILD.md#5070-ti-cuda-arch)). Do **not** copy Triton-only (`-Profile triton`, `86-real` only) portable to JUPITER.
2. **remus** `pathb-rpc-remus` docker on `:50051`
3. **romulus** PathB 3060 docker on `127.0.0.1:50051`
4. **romulus** no hung profiler / `llama-server` from prior bench (`pgrep -a llama-pipeline-profiler`)

Verify from romulus:
```bash
nc -zv 192.168.8.176 50051
nc -zv 127.0.0.1 50051
nc -zv 192.168.8.21 50053

# R5 preflight (preferred)
LD_LIBRARY_PATH=build-rocm-docker/bin:/opt/rocm/lib \
  build-rocm-docker/bin/llama-pipeline-profiler \
  --validate-rpc -rpc 192.168.8.176:50051,127.0.0.1:50051,192.168.8.21:50053 \
  -ts 25,12,25,38
```

### JUPITER `abort()` dialog (2026-06-29)

| Symptom | Cause |
|---------|-------|
| Windows `abort()` from `rpc-server.exe` (Ignore / Exit / Continue) | Server-side `GGML_ASSERT` in `ggml-rpc.cpp` when `ggml_backend_graph_compute` fails |
| `:50053` not listening; romulus `recv failed` / `rpc_finish_event_response` | Process died after assert during prefill or decode |

**Root cause:** `ggml-cuda.dll` built with **`CMAKE_CUDA_ARCHITECTURES=86-real` only** (Triton/Ampere portable). RTX 5070 Ti needs **`120a-real`** kernels. Ampere-only builds can pass HELLO / memory probes but fail on real graph ops.

**Fix:**
```powershell
.\scripts\cuda-windows-5070ti\build.ps1 -Reconfigure   # sets 86-real;120a-real
scripts\b6-gate-jupiter-rebuild-rpc.cmd                  # rpc-server + ggml-cuda.dll -> portable
.\scripts\b6-gate-jupiter-start-rpc-task.ps1             # restart schtask
```

Confirm: `findstr CMAKE_CUDA_ARCHITECTURES build-cuda-b-bin\CMakeCache.txt` must include `120a-real`.

Details: [RPC-BUG-HUNT.md#5070-ti-rpc-server-abort-2026-06-29](RPC-BUG-HUNT.md#5070-ti-rpc-server-abort-2026-06-29)

## B+6 profiler preset (romulus)

```bash
# scripts/b6-gate-profiler-romulus.sh / b6-gate-run-remote.sh
b6-4gpu-g          # JUPITER :50053, -ts 25,12,25,38
b6-4gpu-g-triton   # triton 3090 :50054 swap, -ts 22,11,34,33
b6-5gpu-g          # + triton 3070 :50055, -ts 22,11,22,11,34

bash scripts/b6-gate-ts-sweep-4gpu.sh --phase grid --tokens 128
bash scripts/b6-gate-diagnose-runs.sh b6-4gpu-g-triton b6-4gpu-ts-sweep/G2-confirm
```

### B+6 gate results (2026-06-29, n=384 unless noted)

| Label | overlap | drain | straggler | G |
|-------|---------|-------|-----------|---|
| `b6-4gpu-g` canonical | 0.1% | 50.4s | backend3 5070 @ 11.6 ms/tok | 77.2 |
| `b6-4gpu-g-triton` | 0.1% | 5.9s | backend3 3090 @ 9.0 ms/tok | 63.1 |
| ts G2 n=128 | 0.7% | 2.3s | backend1 5060 | 60.2 |
| ts G4-confirm n=384 | 0.2% | 4.8s | backend1 5060 | 75.8 |

Mission next: B+7 4-RPC drain bisect; see [b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md).

## Parked / experimental

| Item | Status |
|------|--------|
| RX6600 `:50052` in 4-GPU | **HANG** at slot init -- dedicated ROCm debug session |
| 5-GPU with 6600 | blocked; use triton 3070 `:50055` instead |
| 6-GPU (3090 `:50054` + 3070 `:50055` + 4-GPU) | needs 6-value `-ts`; not in b6 presets yet |
| Windows as CUDA client 4-GPU + 6600 (`trace-g-4gpu-plus`) | HANG ~45min then RPC recv failed |

## Related docs

- [MULTI-NODE.md](MULTI-NODE.md) -- Windows + remus wiring
- [RPC-BUG-HUNT.md](RPC-BUG-HUNT.md) -- stall taxonomy + 6600 hang
- [PROFILING.md](PROFILING.md) -- 6600 straggler on Windows 3-GPU
- [rpc-patch/docs/rpc-path-b-plus-tracking.md](../../rpc-patch/docs/rpc-path-b-plus-tracking.md) -- implementation log