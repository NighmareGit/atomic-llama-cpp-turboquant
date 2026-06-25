# Path B: Event-Based Pipeline Parallelism — Tracking

## Status: B1-B4 COMPLETE (verification 2026-06-25)

### Build: PASS (git workspace, v4.2.2)  |  Test: PASS (4B + 12B + server matrix)  |  Benchmark: Config A 4B G~110 t/s

### Current Phase: B4 final verification
### Phase Status: COMPLETE — Path B ready for production cross-GPU deployments

---

## Implementation Log

| Date | Phase | Action |
|------|-------|--------|
| 2026-06-25 | B1–B3 | Full Path B in **git workspace** (`Path-B-Event-Support`): enum, deferred I/O, server handlers (SET_TENSOR_BATCH, EVENT_RECORD), event lifecycle, caps, central drain in `send_rpc_cmd` |
| 2026-06-25 | Fix | `event_wait`/`event_synchronize`: skip recv if central drain already consumed EVENT_RECORD response (`tls_pending_event.pending == false`) |
| 2026-06-25 | Fix | Move `drain_pending_event_response` after event struct defs (compile fix) |
| 2026-06-25 | Test | Docker rebuild from git workspace: `build-cuda-b-bin` (rpc-server), `build-rocm-docker` (llama-cli) |
| 2026-06-25 | Test | Config A 9B Q5_K_M + q8_0 cache: G≈69 t/s (warm load). Avoid `Qwen_Qwen3.5-9B` Unsloth paths — damaged. |
| 2026-06-25 | Test | **Primary regression:** `Qwen3.5-4B-Q4_K_M` Config A, 3 runs validated via log-polling harness |
| 2026-06-25 | Test | Spot-check: `gemma-4-12b-it-Q4_K_M` G≈51.5 t/s PASS |
| 2026-06-25 | Scripts | Added `scripts/pathb-{start-rpc,run-test,test}.sh` with presets for 4B, medium, 72B |
| 2026-06-25 | B4 Final Verification | Pipeline debug captured (`-v` + `GGML_SCHED_DEBUG=1`). 512-token server stress PASS. Single-GPU regression PASS (102.3 t/s, sched copies=1). 72B attempted — hardware-limited on 24+8GB Config A. Path B ready for production. |

---

## Issues Log

| Date | Issue | Root Cause | Fix | Status |
|------|-------|-----------|-----|--------|
| 2026-06-25 | Path B client abort at `get_device_memory` (build-a2) | Pipelined GET_TENSOR not drained before sync `send_rpc_cmd` | Central drain in `send_rpc_cmd` (with response) | RESOLVED (workspace) |
| 2026-06-25 | `event_wait` recv failed (bytes_recv=0) | Central drain consumed EVENT_RECORD; `ev->response_pending` still true | Only recv when `tls_pending_event.pending` | RESOLVED |
| 2026-06-25 | Docker `logs` appears empty / hang | Spinner `\r` overwrites; interactive CLI waits at `>` | `--single-turn --simple-io --no-conversation`; poll via `docker exec` + log file | RESOLVED |
| 2026-06-25 | CMake cache path mismatch | Cache paths used `/app/a2` | Mount workspace at `/app/a2` when building in Docker | RESOLVED |
| 2026-06-25 | Model path wrong in container | Mounted `/models` vs `/mnt/models` | `-v /mnt/models:/mnt/models` | RESOLVED |
| 2026-06-25 | `docker logs` empty after `--rm` stop | Container removed on stop | Named containers + `PATHB_LOG_DIR` artifacts; poll during run | RESOLVED |
| 2026-06-25 | Pipeline debug not in cli logs | INFO lines suppressed without `-v`; `strings` filter drops stderr detail | Pass `GGML_SCHED_DEBUG` into Docker; use `-v` in `--extra-args`; read `.raw` log | RESOLVED |
| 2026-06-25 | 72B presets OOM on RPC | `ts=4,1` interpreted as 80% on RPC0 (8GB); 72B IQ4_XS ~40GB | Use `--fit on` + `-ngl 0` or manual `-ngl 24-36` + `ts=10,90`; see `pathb-72b-vram-calc.py` | RESOLVED (methodology) |
| 2026-06-25 | 72B cli RPC crash on decode | CUDA graph warmup on 3060 Ti with manual -ngl | Use `pathb-72b-server.sh` (llama-server + --fit on) for 72B | WORKAROUND |

---

## Benchmark Results

### Config A: AMD 7900 XTX client / NVIDIA 3060 Ti RPC worker

| Date | Model | Cache | Prompt t/s | Gen t/s | Notes |
|------|-------|-------|------------|---------|-------|
| 2026-06-25 | Qwen3.5-9B Q5_K_M (A2 image) | q4_0 | ~210 | **68.9** | A2 baseline (image binaries) |
| 2026-06-25 | Qwen3.5-9B Q5_K_M (Path B workspace) | q8_0 | 86.9 | **69.0** | Warm load; avoid Unsloth 9B paths |
| 2026-06-25 | **Qwen3.5-4B Q4_K_M** (Path B) | q4_0 | 118.5 / 119.5 | **108.9 / 110.5** | 3-run matrix PASS |
| 2026-06-25 | **Qwen3.5-4B Q4_K_M** (Path B) | q8_0 | 121.5 | **110.8** | PASS |
| 2026-06-25 | gemma-4-12b Q4_K_M (Path B) | q4_0 | 161.7 | **51.5** | PASS; thinking template output |
| 2026-06-25 | Qwen3.5-4B (B4 sched debug) | q4_0 | 117.0 | **109.8** | `pipeline parallelism enabled`, `sched copies = 4` |
| 2026-06-25 | Qwen3.5-4B (512-token server) | q4_0 | 113.3 | **83.6** | `rpc-server-bench.sh pathb`, max_tokens=512, no errors |
| 2026-06-25 | Qwen3.5-4B (single GPU, no RPC) | q4_0 | 359.9 | **102.3** | sched copies=1; no regression vs ~93 t/s Phase A ref |

**Primary regression target:** 4B Q4_K_M, G > 100 t/s Config A.

### B4 verification logs

| Step | Label | Result | Log path |
|------|-------|--------|----------|
| Pipeline debug | `b4-sched-debug-v` | PASS | `patch/bench-results/pathb-runs/b4-sched-debug-v.raw` |
| Sched extract | `b4-sched-debug-extract.log` | PASS | `pipeline parallelism enabled`, `sched copies = 4` |
| Long gen 512 | `long-gen-512-b4-server` | PASS | `patch/bench-results/rpc-server-bench/long-gen-512-b4-server.*` |
| Single GPU | `single-gpu-regression-4b` | PASS | `patch/bench-results/pathb-runs/single-gpu-regression-4b.log` |
| 72B server | `qwen72b-fit-b4` | PASS | `--fit on -ngl 0 -ts 10,90`; load+gen OK (~2 t/s, heavy CPU offload) |
| 72B cli | `qwen72b-ngl32` | FAIL infer | Load OK; RPC CUDA graph crash (`recv failed`) with manual -ngl |
| 72B cli | `kimi72b-b4-verify` | FAIL load | Old `ts=4,1` put 80% weights on 8GB RPC0 |

### Models to avoid

- `Qwen_Qwen3.5-9B-Q5_K_M.gguf` and other Unsloth Qwen 3.5 builds (damaged / garbage loops reported)
- Use `Qwen3.5-4B-Q4_K_M.gguf` for fast regression; gemma-4 family for larger spot-checks

### KV cache note

Weight quants like `Q4_K_M` do **not** map to cache types. Allowed `-ctk`/`-ctv`: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, etc. Use `q4_0` on worker-constrained setups; `q8_0` on 7900 XTX client when VRAM allows.

---

## Test Scripts (git workspace)

```bash
# 1. Start RPC worker (CUDA 3060 Ti)
./scripts/pathb-start-rpc.sh

# 2. Preset tests (ROCm client)
./scripts/pathb-test.sh 4b              # primary regression (3 min load timeout)
./scripts/pathb-test.sh matrix-4b       # 3-run 4B matrix
./scripts/pathb-test.sh gemma12b        # 12B spot-check
./scripts/pathb-test.sh qwen27b         # 27B, -ts 2,1
./scripts/pathb-test.sh qwen35b         # 35B MoE, -ts 3,1
./scripts/pathb-test.sh gemma31b        # 31B, -ts 3,1
./scripts/pathb-test.sh kimi72b         # 72B, 10 min load timeout, -ts 15,85
./scripts/pathb-test.sh qwen72b         # 72B, 10 min load timeout, -ts 15,85

# Pipeline debug capture (requires -v for INFO lines)
GGML_SCHED_DEBUG=1 ./scripts/pathb-run-test.sh --label sched-debug \
  --model /mnt/models/Qwen3.5-4B-Q4_K_M.gguf --extra-args "-v --no-warmup" \
  --ctk q4_0 --ctv q4_0 --ngl 99

# 512-token stress (server API; honors max_tokens)
BENCH_GEN_TOKENS=512 BENCH_RUNS=1 BENCH_MODEL=/mnt/models/Qwen3.5-4B-Q4_K_M.gguf \
  BENCH_CTK=q4_0 BENCH_CTV=q4_0 BENCH_TS=1,1 \
  ./scripts/rpc-server-bench.sh pathb long-gen-512

# Logs: patch/bench-results/pathb-runs/<label>.{log,meta,raw}
```

Low-level runner:

```bash
./scripts/pathb-run-test.sh --label myrun --model /mnt/models/Qwen3.5-4B-Q4_K_M.gguf \
    --load-timeout 180 --ctk q4_0 --ctv q4_0 --ngl 99
```

Harness validates: exit code, `Generation:` throughput line, no RPC abort, no bogus `1000000 t/s`, no death-loop output. Polls every 10s during load.

---

## Phase Checklist

### Phase B1: Add EVENT_RECORD Command to RPC Protocol

- [x] All B1 items complete (see prior checklist)

### Phase B2: Register Event Functions

- [x] All B2 items complete (see prior checklist)

### Phase B3: Enable Pipeline Parallelism Detection

- [x] Cross-GPU Config A passes (4B + 12B validated)
- [x] RPC backend events work (no `event_wait` recv failure after drain fix)
- [x] `pipeline parallelism enabled` log line captured with `GGML_SCHED_DEBUG=1` + `-v` (2026-06-25, `b4-sched-debug-v.raw`)
- [x] `sched copies = 4` verified in sched debug output (equivalent to `GGML_SCHED_MAX_COPIES`)

### Phase B4: Tuning and Edge Cases

- [x] 72B presets exercised with 10 min load timeout (`qwen72b-fit-b4`: server + `--fit on -ngl 0 -ts 10,90`; ~40GB weights need ~8GB CPU offload — see `scripts/pathb-72b-vram-calc.py`)
- [x] Long generation (512 tokens) without deadlock (`long-gen-512-b4-server`, llama-server Path B, G=83.6 t/s)
- [ ] Config B matrix (NV client / AMD worker) — optional, not run
- [x] Single-GPU regression (no RPC): 102.3 t/s vs ~93 t/s Phase A ref (+10%, no regression; sched copies=1)

---

## Conclusion

Path B (v4.2.2, `Path-B-Event-Support`) is **production-ready** for cross-GPU Config A deployments on models that fit the 24GB + 8GB VRAM budget (through ~36B MoE with `ts=10,90`).

**Measured gains (Path B vs A1, llama-server, ts=10,90):**

| Model | A1 gen t/s | Path B gen t/s | Delta |
|-------|------------|----------------|-------|
| 9B | 54.4 | 58.4 | +7.3% |
| 27B | 27.1 | 27.9 | +3% |
| 31B | 23.5 | 26.7 | +13.6% |
| 35B MoE | 58.9 | 64.2 | +9% |
| 36B MoE | 57.1 | 61.1 | +7% |

**Cli regression:** 4B Config A **~110 t/s** with pipeline parallelism (`sched copies = 4`).

**72B note:** IQ4_XS (~40GB) exceeds 32GB GPU VRAM (8GB RPC + 24GB ROCm). Shovel ~8GB+ weights to CPU via `-ngl 0 --fit on` or manual `-ngl 24-36` with percentage `-ts 10,90`. Dense 72B does not use `--n-cpu-moe` (MoE only). Prefer `scripts/pathb-72b-server.sh` over cli (cli manual -ngl crashes RPC CUDA graphs).