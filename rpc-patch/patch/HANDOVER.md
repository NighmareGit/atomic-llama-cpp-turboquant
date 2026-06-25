# Path A + Path B RPC Optimization — Handover

**Date:** 2026-06-25  
**Git repo:** `atomic-llama-cpp-turboquant/` (branch `Path-B-Event-Support`)  
**Patch collateral:** `rpc-patch/` (this handover, docs, scripts, bench-results)  
**Status:** Path B (v4.2.2) implemented, benchmarked, documented. Ready for review / merge planning.

---

## 1. Executive summary

This branch adds **Path B: event-based pipeline parallelism** on top of existing **Path A1** (batch `SET_TENSOR`) and **Path A2** (pipelined `GET_TENSOR` receive). Controlled **llama-server <-> rpc-server** benchmarks on Config A (AMD client + NVIDIA RPC worker) show:

| Comparison | Result |
|------------|--------|
| Path B vs A1 (9B server) | **+7.3%** generation (58.4 vs 54.4 t/s) |
| Path B vs A1 (27B–36B matrix, ts=10,90) | **+3% to +14%** generation |
| A1+A2 vs A1 (all configs) | **~0%** (within variance) |

Path B is the recommended RPC stack for cross-GPU **server** workloads. Path A2 remains for shared drain infrastructure but does not improve throughput alone.

---

## 2. What changed (code)

### Protocol

| Version | `RPC_PROTO` | Key additions |
|---------|-------------|---------------|
| A1 | 4.1.0 | `SET_TENSOR_BATCH` (17) |
| A2 | 4.1.0 | Client pipelined `GET_TENSOR` |
| **B** | **4.2.2** | `EVENT_RECORD` (18), `caps.async/events=true` |

### Modified files (git, in `atomic-llama-cpp-turboquant/`)

```
atomic-llama-cpp-turboquant/ggml/include/ggml-rpc.h
atomic-llama-cpp-turboquant/ggml/src/ggml-rpc/transport.h
atomic-llama-cpp-turboquant/ggml/src/ggml-rpc/ggml-rpc.cpp
```

### Correctness fixes (required for A1+A2+B together)

Without these, Path B aborts, hangs, or returns `recv failed (bytes_recv=0)`:

1. **Central drain** in `send_rpc_cmd` (with response): `drain_pending_event_response()` + `flush_pending_get_tensor()` before sync reads.
2. **`event_wait` coherency:** only `recv_rpc_cmd_deferred` when `tls_pending_event.pending`; else clear `ev->response_pending`.
3. **Struct ordering:** drain helpers below `rpc_msg_event_record_rsp`.

See `patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md` for line-level implementation notes.

---

## 3. Hardware topology (Config A)

| Role | GPU | Docker image | Device env |
|------|-----|--------------|------------|
| **Client** (llama-server / llama-cli) | AMD RX 7900 XTX 24 GB | `llama-rocm-patched` | `HIP_VISIBLE_DEVICES=0` |
| **Worker** (rpc-server) | NVIDIA RTX 3060 Ti 8 GB | `llama-rpc-cuda-a2` (Path B) or `llama-rpc-cuda` (A1) | `CUDA_VISIBLE_DEVICES=0` |
| **Network** | localhost TCP | port **50051** | `--network host` |
| **Models** | `/mnt/models` | read-only mount | |

**Rule:** Never run llama-server and rpc-server on the same GPU.

---

## 4. Build artifacts

Path B workspace builds under `atomic-llama-cpp-turboquant/` (volume-mounted into Docker):

```
atomic-llama-cpp-turboquant/build-cuda-b-bin/bin/rpc-server
atomic-llama-cpp-turboquant/build-rocm-docker/bin/llama-server
atomic-llama-cpp-turboquant/build-rocm-docker/bin/llama-cli
```

Override with env `LLAMA_TURBOQUANT_ROOT` if the checkout path differs.

Image binaries (no mount):

```
llama-rpc-cuda       # A1 rpc-server (v4.1.0)
llama-rpc-cuda-a2    # A1+A2 rpc-server (v4.1.0)
llama-rocm-patched   # ROCm llama-server/cli
```

Rebuild Path B after code changes:

```bash
cd atomic-llama-cpp-turboquant
cmake --build build-cuda-b-bin --target rpc-server -j$(nproc)
cmake --build build-rocm-docker --target llama-server llama-cli -j$(nproc)
```

---

## 5. Tensor-split and memory (critical for >27B)

### Device index order

With `--rpc 127.0.0.1:50051`, logs list **RPC0 first, ROCm0 second**:

```
-ts 10,90  ->  RPC0 gets 10%, ROCm0 gets 90%
-ts 90,10  ->  RPC0 gets 90%  -> OOM on 8 GB worker
```

Verify with `load_tensors:` lines in server logs or `scripts/rpc-ts-fit-probe.sh` (polls `rocm-smi` / `nvidia-smi`).

### Recommended presets

| Model class | ts | ngl | ctx | cache | extras |
|-------------|-----|-----|-----|-------|--------|
| 9B | `1,1` | 99 | 8192 | q8_0 / turbo3 | — |
| 27B / 31B dense | `10,90` | 99 | 8192 | q8_0 / turbo3 | `--no-warmup -np 1` |
| 35B / 36B MoE | `10,90` | 99 | 4096 | q4_0 / q4_0 | `--n-cpu-moe 8 --no-warmup -np 1` |

### Fit flags (llama-server supports)

```
--fit on|off              default on
--fit-target 512,1024     margin MiB per device (RPC, ROCm)
--n-cpu-moe N             offload MoE experts for first N layers to CPU
```

With explicit `-ts` and `-ngl`, `--fit on` often makes no changes. Use `--fit off` for manual tuning.

### Anti-patterns (learned the hard way)

- `ts=2,1` or `ts=3,1` at ctx 8192 -> RPC OOM (~12 GB alloc).
- `ts=18,1` with `ngl=25` -> most of 25 GPU layers on RPC, rest CPU mmap -> **~5 t/s** (not 18% on RPC).
- Path B default 4-slot warmup on >27B -> rpc-server crash; use **`--no-warmup -np 1`**.

---

## 6. Scripts

All under `rpc-patch/scripts/`. Default log output: `rpc-patch/patch/bench-results/`.

| Script | Purpose |
|--------|---------|
| `pathb-start-rpc.sh` | Start single CUDA rpc-server on :50051 |
| `pathb-run-test.sh` | llama-cli cross-GPU test with health/log polling |
| `pathb-test.sh` | Presets: `4b`, `matrix-4b`, `gemma12b`, `qwen27b`, `qwen72b`, `kimi72b`, ... |
| `pathb-72b-server.sh` | 72B via llama-server + `--fit on` (IQ4_XS, CPU offload) |
| `pathb-72b-vram-calc.py` | VRAM budget calculator for 72B on 24+8 GB |
| `rpc-server-bench.sh` | Single llama-server <-> rpc-server bench (A1 / A1a2 / pathb) |
| `rpc-server-bench-matrix.sh` | Full A1 vs A1a2 vs Path B matrix (27b, 31b, 35b, 36b) |
| `rpc-ts-fit-probe.sh` | Tensor-split / fit probe with GPU VRAM monitoring |

### Quick start — server matrix

```bash
cd atomic-llama-cpp-turboquant   # git repo root
docker rm -f bench-rpc bench-llama pathb-rpc 2>/dev/null

./rpc-patch/scripts/rpc-server-bench-matrix.sh          # all models
./rpc-patch/scripts/rpc-server-bench-matrix.sh 27b 31b  # subset
```

### Quick start — 9B regression

```bash
export BENCH_MODEL=/mnt/models/Qwen3.5-9B-MTP-Q4_K_M.gguf \
       BENCH_CTX=8192 BENCH_CTK=q8_0 BENCH_CTV=turbo3 BENCH_TS=1,1
./scripts/rpc-server-bench.sh a1 9b-q4km-a1
./scripts/rpc-server-bench.sh pathb 9b-q4km-pathb
```

### Container hygiene

Scripts use fixed names `bench-rpc` / `bench-llama` (or `pathb-rpc`) and `cleanup()` before each run. Stale containers cause port 50051/8081 conflicts and compound GPU memory use.

```bash
docker rm -f bench-rpc bench-llama pathb-rpc pathb-* 2>/dev/null
```

---

## 7. Documentation map

| Path | Content |
|------|---------|
| `README.md` | **Project overview** (start here for layout + scripts map) |
| `patch/HANDOVER.md` | This file |
| `docs/RPC_PATH_AB_OPTIMIZATION_REPORT.md` | Full optimization report |
| `patch/bench-results/` | Archived benchmark logs |
| `patch/PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md` | Line-level Path B code |
| `docs/rpc-path-b-tracking.md` | Tracking notes |
| `atomic-llama-cpp-turboquant/` | llama.cpp git tree (Path B code changes) |

---

## 8. Benchmark results (archived)

Artifacts live in `patch/bench-results/`. Summary in `rpc-server-bench/matrix-summary.txt`.

### Controlled server matrix (2026-06-25, ts=10,90, --no-warmup -np 1)

| Model | A1 | A1+A2 | Path B | B vs A1 |
|-------|-----|-------|--------|---------|
| Qwen 27B Q5_K_M | 27.1 | 27.1 | **27.9** | +3.0% |
| Gemma 31B Q4_K_M | 23.5 | 23.2 | **26.7** | +13.6% |
| Qwen 35B A3B MoE | 58.9 | 58.1 | **64.2** | +9.0% |
| Qwen 3.6 35B APEX MoE | 57.1 | 58.2 | **61.1** | +7.0% |
| Qwen 9B MTP Q4_K_M (ts=1,1) | 54.4 | — | **58.4** | +7.3% |

### llama-cli regression (Path B, Config A)

| Model | Gen t/s | Notes |
|-------|---------|-------|
| Qwen 4B Q4_K_M | ~110 | Primary cli regression target |
| Qwen 9B (cli) | ~70 | q4_0 cache |
| gemma 12B | ~51 | spot-check |

Per-run files: `patch/bench-results/rpc-server-bench/<label>.{meta,result,server.log,rpc.log}`

---

## 9. Docker operational notes

### Apparent "hangs"

Model load uses in-place spinner (`\r`); `docker logs` looks empty. Scripts poll `/health` or `docker exec ... tail server.log`.

### Path B on large models

Requires `--no-warmup -np 1` (now default in `rpc-server-bench.sh` via `BENCH_NO_WARMUP=1`). Without it, 4-slot warmup can crash workspace `rpc-server` on >27B.

### Log locations during runs

| Harness | Directory |
|---------|-----------|
| Server bench | `patch/bench-results/rpc-server-bench/` |
| TS/fit probe | `patch/bench-results/rpc-ts-probe/` |
| CLI tests | `patch/bench-results/pathb-runs/` |

Override with `BENCH_LOG_DIR`, `PROBE_LOG_DIR`, `PATHB_LOG_DIR` env vars.

---

## 10. Models used (under `/mnt/models`)

| Label | File | Use |
|-------|------|-----|
| 9B | `Qwen3.5-9B-MTP-Q4_K_M.gguf` | Server regression (only Q4_K 9B found) |
| 27B | `Qwen3.5-27B-Q5_K_M.gguf` | Dense large-model matrix |
| 31B | `gemma-4-31B-it-Q4_K_M.gguf` | Dense large-model matrix |
| 35B | `Qwen3.5-35B-A3B.i1-Q4_K_M.gguf` | MoE matrix |
| 36B* | `Qwen3.6-35B-A3B-APEX-I-Quality.gguf` | MoE (*no true 36B dense; A3B family) |
| 4B cli | `Qwen3.5-4B-Q4_K_M.gguf` | Fast cli regression |

**Avoid:** `Qwen_Qwen3.5-9B-*` Unsloth-tagged builds (reported damaged).

---

## 11. Known issues / open work

1. ~~**Pipeline debug:**~~ Done — `b4-sched-debug-v.raw` shows `pipeline parallelism enabled` and `sched copies = 4` (`GGML_SCHED_DEBUG=1` + `-v`).
2. **Config B:** NV client / AMD worker matrix not run (optional).
3. **72B:** B4 attempted; IQ4_XS ~40GB exceeds 24+8GB Config A. Presets fixed to `ts=15,85`. See `docs/rpc-path-b-tracking.md`.
4. **MoE at ctx 8192 + turbo3:** Not tested; 35B/36B matrix used ctx 4096 q4_0 + ncmoe 8.
5. **Upstream merge:** Branch is a private fork patch; follow `AGENTS.md` if contributing upstream (human-authored, disclose AI assist).

---

## 12. Suggested next steps for the next engineer

1. Read `docs/RPC_PATH_AB_OPTIMIZATION_REPORT.md` for full analysis.
2. Re-run matrix after any rpc.cpp change: `./scripts/rpc-server-bench-matrix.sh`
3. For new models, probe split first: `PROBE_TS=10,90 ./scripts/rpc-ts-fit-probe.sh myprobe`
4. Confirm Path B pipeline in logs: `GGML_SCHED_DEBUG=1` on llama-server.
5. If merging: single commit series A1 -> A2 -> B + drain fixes; keep `RPC_PROTO_MINOR_VERSION` bump documented.

---

## 13. File tree (project root)

```
atomic-llama-cpp-turboquant/        # git repo root
  rpc-patch/
    README.md                       # project overview (start here)
    patch/
    HANDOVER.md
    PATH_B_EVENT_SUPPORT_IMPLEMENTATION.md
    bench-results/
      rpc-server-bench/
      rpc-ts-probe/
      pathb-runs/
  docs/
    RPC_PATH_AB_OPTIMIZATION_REPORT.md
    rpc-path-b-tracking.md
  scripts/
    pathb-start-rpc.sh
    pathb-run-test.sh
    pathb-test.sh
    pathb-72b-server.sh
    pathb-72b-vram-calc.py
    rpc-server-bench.sh
    rpc-server-bench-matrix.sh
    rpc-ts-fit-probe.sh
  atomic-llama-cpp-turboquant/    # llama.cpp source + builds (Path-B-Event-Support)
    ggml/src/ggml-rpc/...
    build-cuda-b-bin/
    build-rocm-docker/
```

---

*Handover prepared 2026-06-25. Benchmark artifacts archived under `patch/bench-results/`.*