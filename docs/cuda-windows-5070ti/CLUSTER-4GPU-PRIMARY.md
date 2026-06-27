# Config G Primary: 4-GPU cluster (stable, no RX6600)

**Date:** 2026-06-27  
**Status:** **PRODUCTION STABLE** (3/3 gate PASS + trace captured)  
**Branch:** `Path-B-Event-Support-Pipeline-Plus` @ `833ad4429`+

## Topology

| # | GPU | Role | Host | RPC endpoint |
|---|-----|------|------|--------------|
| 0 | RX 7900 XTX | ROCm **client** (`llama-server`) | romulus | (local HIP0) |
| 1 | RTX 5060 Ti | CUDA RPC worker | remus | `192.168.8.176:50051` |
| 2 | RTX 3060 Ti | CUDA RPC worker | romulus docker | `127.0.0.1:50051` |
| 3 | RTX 5070 Ti | CUDA RPC worker | Windows | `192.168.8.21:50053` |

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
| `-ts` | `36,24,24,16` (7900 / 5060 / 5070 / 3060 by VRAM share) |
| `-ctk` / `-ctv` | `q8_0` |
| `GGML_PIPELINE_PLUS` | `1` |
| `--n-cpu-moe` | omit (do not use `8` on ROCm 35B APEX) |
| `BENCH_TRACE` | `0` for smoke; `1` for hotpath |

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

1. **Windows** `rpc-server` on `:50053` listening (`0.0.0.0:50053`):
   ```powershell
   .\scripts\cuda-windows-5070ti\pathb-rpc-server.ps1
   ```
   Foreground start is more reliable than hidden `Start-Process` if the worker exits silently.
2. **remus** `pathb-rpc-remus` docker on `:50051`
3. **romulus** PathB 3060 docker on `127.0.0.1:50051`
4. **romulus** no hung `llama-server` from prior bench (`pgrep -a llama-server`)

Verify from romulus:
```bash
nc -zv 192.168.8.176 50051
nc -zv 127.0.0.1 50051
nc -zv 192.168.8.21 50053
```

## Parked / experimental

| Item | Status |
|------|--------|
| RX6600 `:50052` in 4-GPU | **HANG** at slot init -- dedicated ROCm debug session |
| 5-GPU with 6600 | blocked; use 3090+3070 node variant when online |
| Windows as CUDA client 4-GPU + 6600 (`trace-g-4gpu-plus`) | HANG ~45min then RPC recv failed |

## Related docs

- [MULTI-NODE.md](MULTI-NODE.md) -- Windows + remus wiring
- [RPC-BUG-HUNT.md](RPC-BUG-HUNT.md) -- stall taxonomy + 6600 hang
- [PROFILING.md](PROFILING.md) -- 6600 straggler on Windows 3-GPU
- [rpc-patch/docs/rpc-path-b-plus-tracking.md](../../rpc-patch/docs/rpc-path-b-plus-tracking.md) -- implementation log