# EXPERIMENT: GRAPH_COMPUTE_ALL Multi-Device Bring-Up (NW4 Lever)

**Branch:** `experiment/graph-compute-all-bringup` (off `good-prototype` @ `54e530caf`)  
**Worktree:** `worktrees/experiment-graph-compute-all-bringup/`  
**Date:** 2026-08-01  
**Author:** graph-compute-all-experiment sub-agent  
**Status:** **GATE 1 PASS (build) / GATE 1 FAIL (runtime) → GATE 2 BLOCKED → experiment INFEASIBLE for heterogeneous pair**

---

## Abstract

The `RPC_CMD_GRAPH_COMPUTE_ALL` path (cmd 21, "Path C") is implemented in `ggml-rpc.cpp` but was never used. It promises the **NW4 lever**: collapse N per-token blocking RPC round-trips to ~1 for same-host GPU pairs by running a single rpc-server process that owns multiple GPUs, eliminating loopback TCP overhead (~200 µs/round-trip).

**Verdict: The NW4 lever via GRAPH_COMPUTE_ALL is NOT realizable on the co-located romulus pair (7900 XTX ROCm + 3060 Ti CUDA).** A single rpc-server process cannot enumerate both a ROCm device and a CUDA device because the HIP and CUDA backends export **identical C symbols** (`ggml_backend_cuda_reg`, `ggml_backend_cuda_init`, etc.) — at runtime, dynamic loading with `RTLD_GLOBAL` resolves only ONE backend's registration; the other's devices become invisible. This is a fundamental ggml architecture constraint, not a build-system bug.

---

## Hypothesis

If one rpc-server process can own both the 7900 XTX (ROCm) and 3060 Ti (CUDA), then:
- `GGML_RPC_MULTIDEVICE=1` + multi-device server → `GRAPH_COMPUTE_ALL` fires
- Server-side multi-GPU scheduling collapses per-token RPC round-trips
- Throughput on the co-located pair improves vs. separate per-GPU servers (TCP loopback baseline)

---

## Hardware State (2026-08-01 ~15:10)

| GPU | Backend | Driver | VRAM Used | Status |
|-----|---------|--------|-----------|--------|
| Radeon RX 7900 XTX | ROCm (HIP) | ROCm 7.2.3 / gfx1100 | 0 MiB (FREE) | No llama-server; lease expired |
| RTX 3060 Ti | CUDA (native) | 580.173.02 / CUDA 13.0 | 195 MiB | Native rpc-server `PID 1534915` on `0.0.0.0:50051` (`-d CUDA0`), started 14:23 |

**Key operational finding:** The 3060 Ti now works **natively** on romulus (driver 580.173.02). Previously it required Docker (BUG-010 driver mismatch). The native rpc-server is alive and port 50051 is open. **Docker is no longer needed for the 3060 Ti.**

---

## GATE 1 — Multi-Backend Build Feasibility

**Question:** Can ONE rpc-server binary hold BOTH the ROCm (HIP) backend and the CUDA backend (`GGML_HIP=ON` + `GGML_CUDA=ON`)?

### Build: PASS ✅

CMake configure **succeeded** with both backends enabled:

```bash
cmake -S . -B build-hip-cuda-multi \
  -DGGML_CUDA=ON -DGGML_HIP=ON -DGGML_RPC=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

Output:
```
-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include (found version "12.8.93")
-- CUDA Toolkit found
-- Including CUDA backend
-- HIP and hipBLAS found
-- Including HIP backend
-- Using RPC backend
-- Including RPC backend
```

The full `rpc-server` binary **linked successfully** with `BUILD_SHARED_LIBS=ON`:
```
[100%] Linking CXX executable ../../bin/rpc-server
[100%] Built target rpc-server
```

`ldd` confirms both backend shared objects are linked:
```
libggml-cuda.so.0  => .../bin/libggml-cuda.so.0
libggml-hip.so.0   => .../bin/libggml-hip.so.0
```

### Runtime: FAIL ❌ (the blocker)

Despite building and linking, the multi-backend binary **cannot enumerate both devices**. Requesting both devices:

```bash
./build-hip-cuda-multi/bin/rpc-server -d CUDA0,ROCm0 -p 50098
```

Output:
```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 24560 MiB):
  Device 0: Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
error: unknown device: CUDA0        ← CUDA device INVISIBLE
available devices:
  ROCm0: Radeon RX 7900 XTX (24560 MiB, 24524 MiB free)
  CPU: Intel(R) Core(TM) i7-14700K (80253 MiB, 80253 MiB free)
```

Even requesting CUDA0 **alone** fails (only ROCm0 is ever visible):
```bash
./build-hip-cuda-multi/bin/rpc-server -d CUDA0 -p 50098
# → error: unknown device: CUDA0
# → available: ROCm0, CPU only
```

### Root Cause: Identical Symbol Collision

Both backends export the **same C symbols** because the HIP backend is compiled from the **same source files** as CUDA:

- `ggml/src/ggml-hip/CMakeLists.txt:60`: `file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")` — HIP globs CUDA sources
- `ggml/include/ggml-cuda.h`: defines `ggml_backend_cuda_reg()`, `ggml_backend_cuda_init()`, etc. — used by BOTH backends
- `ggml/src/ggml-hip/CMakeLists.txt:115`: `target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)` — HIP even defines `GGML_USE_CUDA`

Symbol comparison (via `nm -D`):
```
libggml-cuda.so:  T ggml_backend_cuda_reg  (0x27d9b0)
libggml-hip.so:   T ggml_backend_cuda_reg  (0x4122770)   ← SAME symbol name
```

Both `.so` files export `ggml_backend_cuda_reg`, `ggml_backend_cuda_init`, `ggml_backend_cuda_buffer_type`, etc.

**Mechanism:** `ggml-backend-dl.cpp:38` loads backends with `RTLD_NOW | RTLD_GLOBAL`. The first-loaded backend's symbols become the global definition. `ggml-backend-reg.cpp:569-570` calls `ggml_backend_load_best("cuda", ...)` then `ggml_backend_load_best("hip", ...)`. Because both register via the identical `ggml_backend_cuda_reg` symbol, only ONE backend's device registration survives — the other's devices never appear in the global registry.

`LD_DEBUG=files` confirmed load order: `libggml-hip.so.0` loads first (direct NEEDED dep of rpc-server), so **ROCm wins and CUDA is shadowed**.

### GATE 1 Verdict

| Aspect | Result |
|--------|--------|
| CMake configure (HIP+CUDA) | ✅ PASS |
| Compile + link rpc-server | ✅ PASS |
| Runtime: enumerate CUDA0 | ❌ FAIL (invisible) |
| Runtime: enumerate ROCm0 | ✅ works |
| Runtime: enumerate BOTH | ❌ FAIL — fundamental symbol collision |

**A single process can host only ONE GPU backend type.** This is a ggml architectural constraint.

---

## GATE 2 — Multi-GPU Server Bring-Up

**Question:** Can a single rpc-server process own BOTH GPUs (7900 via ROCm + 3060 Ti via CUDA)?

### Verdict: BLOCKED ❌

GATE 2 is blocked by GATE 1's runtime failure. A single rpc-server process can enumerate devices from only ONE backend:
- Multi-backend binary → sees only ROCm0 (7900), CUDA0 invisible
- ROCm-only binary → sees only ROCm0
- CUDA-only binary → sees only CUDA0

**There is no configuration that makes ONE server process see both devices.**

The task's fallback ("server inside t2e-fixed-cuda Docker container with CUDA 3060 Ti, PLUS the ROCm backend") does NOT resolve this: the symbol collision is in the ggml backend layer, independent of Docker vs. native. A Docker container running the same multi-backend binary would still only see one backend's devices.

### Design Intent Confirmed

The code comment at `ggml-rpc.cpp:2944-2945` reveals GRAPH_COMPUTE_ALL was designed for **same-backend** multi-GPU:
```cpp
// Fires when n_devices_on_endpoint > 1, i.e. a single RPC backend
// has 2+ GPUs (e.g. rpc-server -d CUDA0,CUDA1).
```

The example is `rpc-server -d CUDA0,CUDA1` — **two CUDA devices**, not a CUDA+ROCm mix. The path requires `n_devices_on_endpoint > 1`, meaning one backend type with multiple devices. Heterogeneous backends (ROCm + CUDA) in one process were never supported because of the symbol collision.

---

## A/B Test — SKIPPED (blocked by GATE 2)

The A/B comparison (multi-device server + `GGML_RPC_MULTIDEVICE=1` vs. separate per-GPU servers) cannot be performed because the multi-device server cannot be brought up. No benchmark numbers.

---

## Feasibility Verdict

| Question | Answer |
|----------|--------|
| Does GRAPH_COMPUTE_ALL code exist? | ✅ Yes (cmd 21, ggml-rpc.cpp:192, 2641, 2848-2867) |
| Does it build? | ✅ Yes |
| Does it work at runtime? | ❌ No — single process can't host both backends |
| Is NW4 lever realizable on 7900+3060Ti? | ❌ **NO** — heterogeneous backends collide |
| Would it work for same-backend pair (e.g. 2× CUDA)? | ⚠️ Unknown — not tested (would need 2 CUDA GPUs in one process) |

### The Real Reason "Implemented, Never Used"

The GRAPH_COMPUTE_ALL path is "implemented, never used" because:
1. **Heterogeneous multi-GPU in one process is impossible** in current ggml (symbol collision between HIP and CUDA backends — they share source and symbol names).
2. The path was designed for **same-backend** multi-GPU (`-d CUDA0,CUDA1`), which requires multiple GPUs of the same type behind one rpc-server — a configuration the current fleet lacks (each node has one GPU, except triton which has 3090+3070 but they're separate servers).
3. Even for same-backend, the path requires the multi-GPU server to report `RPC_CAP_MULTI_DEVICE` and the client to set `GGML_RPC_MULTIDEVICE=1` — a combination never exercised.

---

## Native CUDA Finding (Bonus)

The 3060 Ti now works **natively** on romulus host:
- Driver 580.173.02, CUDA 13.0
- Native rpc-server running: `/usr/local/bin/rpc-server -H 0.0.0.0 -p 50051 -d CUDA0` (PID 1534915)
- Port 50051 open and responsive
- **Docker no longer required** for the 3060 Ti (previously needed due to BUG-010 driver mismatch + missing 8cc38c2a6)

---

## Artifacts

| Artifact | Path |
|----------|------|
| Worktree | `worktrees/experiment-graph-compute-all-bringup/` |
| Branch | `experiment/graph-compute-all-bringup` |
| Multi-backend build | `build-hip-cuda-multi/` (configured, rpc-server built) |
| This report | `.scratch/benchmarks/experiment-graph-compute-all.md` |

---

## Recommendations

1. **Do NOT pursue NW4 via GRAPH_COMPUTE_ALL for heterogeneous pairs.** It's architecturally blocked.
2. **For same-backend multi-GPU** (if fleet ever has 2+ same-type GPUs in one node): the path is worth testing. Triton's 3090+3070 are different VRAM sizes but both CUDA — a same-backend test candidate (though 3070's 8 GiB limits layer-split).
3. **The symbol collision is fixable in principle** (rename HIP backend symbols, e.g. `ggml_backend_hip_reg`), but that's a deep ggml change touching `ggml-cuda.h`, `ggml-hip/CMakeLists.txt`, and `ggml-backend-reg.cpp` — high effort, and would only enable same-process heterogeneous multi-GPU, which the current fleet topology doesn't need.
4. **Keep native 3060 Ti server** — it's now the simplest path (no Docker). Update AGENTS.md to reflect Docker is no longer needed for 3060 Ti.
