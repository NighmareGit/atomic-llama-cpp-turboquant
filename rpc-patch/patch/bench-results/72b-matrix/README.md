# 72B+ matrix (Config C, ts=35,15,50)

**Date:** 2026-06-26  
**Script:** `rpc-patch/scripts/pathb-72b-matrix.sh`  
**Topology:** RPC0=remus 5060 Ti (`192.168.8.176`), RPC1=romulus 3060 Ti docker (`127.0.0.1:50051`), ROCm0=7900 XTX (~44.5 GB combined)  
**Endpoint:** `192.168.8.176:50051,127.0.0.1:50051` (remus-first listing; 3060 is on romulus only)

## Phase 1 summary (load feasibility)

**Rerun 2026-06-26T12:18Z** (post RPC event-drain fix, `GGML_CUDA_DISABLE_GRAPHS=1`): `matrix-summary.txt`, log `phase1-rerun-20260626T1218Z.log` (~24 min)

| Preset | fit off (manual -ngl) | fit on fallback | Winner | Notes |
|--------|----------------------|-----------------|--------|-------|
| llama70b | **PASS** G=4.9 ngl=60, 60 GPU layers | - | fitoff ngl=60 | Q4_K_M |
| qwen72b | **PASS** G=4.8 ngl=60, 60 GPU layers | - | fitoff ngl=60 | IQ4_XS |
| kimi72b | **PASS** G=5.3 ngl=60, 60 GPU layers | - | fitoff ngl=60 | IQ4_XS |
| qwen-next-80b | load FAIL (all fitoff ngl/ncmoe combos) | **PASS** G=16.9 ncmoe=8, 0 GPU | fiton ngl=0 | 53 GB Q5_K_M MoE; expected |
| coder-next | **PASS** G=0.4 ngl=60 ncmoe=8, 49 GPU | - | fitoff | APEX MoE; slow but coherent |
| coder-next-q4 | **PASS** G=16.4 ngl=60 ncmoe=8, 49 GPU | - | fitoff | best MoE end-to-end |

Pre-fix run (2026-06-26 morning): dense 72B fitoff loaded but gen failed (RPC crash). See `matrix-summary.pre-fix-20260626T1218Z.txt`.

Summary: `matrix-summary.txt` | `phase-summary.txt` (production cheatsheet)  
Speed ranking: `load-ranking.txt` | Winners: `phase-winners.txt` + `phase-winners-notes.txt`  
Phase logs: `phase2-20260626T1250Z.log`, `phase3-20260626T1331Z.log`, `phase4-20260626T1356Z.log`

### Validated load split (dense 72B, fit off, ngl=56)

```
offloaded 56/81 layers to GPU
RPC0[192.168.8.176:50051]  ~9554 MiB  (5060, ~35%)
RPC0[127.0.0.1:50051]      ~3857 MiB  (3060, ~15%)
ROCm0                      ~14185 MiB (7900, ~50%)
CPU_Mapped                 ~12947 MiB (~30% layers)
```

VRAM planner: `python3 rpc-patch/scripts/pathb-72b-vram-calc.py --config config-c --gguf <path>`

## RPC inference fix (2026-06-26, RESOLVED)

Pre-fix: dense 72B `fit off` loaded but first prefill failed at `ggml-rpc.cpp` GET_TENSOR (`cmd_child_to_router:error`).

**Root cause:** `graph_compute` sent deferred `EVENT_RECORD` after `GRAPH_RECOMPUTE` without setting `tls_pending_event.pending`; the next `GET_TENSOR` skipped draining the event response and desynced TCP.

**Fix:** `ggml/src/ggml-rpc/ggml-rpc.cpp` -- set `tls_pending_event` after deferred event; central drain in all `send_rpc_cmd` paths; `SET_TENSOR_BATCH` flush to owning socket.

**Ops:** rebuild `build-cuda-b-bin-sync/rpc-server` + `build-rocm-docker/libggml-rpc.so`; sync remus via `docker cp`; set `GGML_CUDA_DISABLE_GRAPHS=1` on workers. Debug harness: `pathb-72b-debug.sh`.

Pre-fix archive: `matrix-summary.pre-fix-20260626T1218Z.txt`

## Phases 2-4 (2026-06-26, post-fix)

Logs: `phase2-20260626T1250Z.log`, `phase3-20260626T1331Z.log`, `phase4-20260626T1356Z.log`

### Phase 2: kv-unified x flash-attn (`-fa on/off`, 3 runs x 32 tok)

| Preset | kvu=off fa=on | kvu=off fa=off | kvu=on fa=on | kvu=on fa=off |
|--------|---------------|----------------|--------------|---------------|
| llama70b | PASS G=5.1 | FAIL (q4_0 V needs FA) | PASS G=4.9 | FAIL |
| qwen72b | PASS G=5.1 | FAIL | PASS G=5.1 | FAIL |
| kimi72b | PASS G=5.4 | FAIL | PASS G=5.4 | FAIL |
| qwen-next-80b | PASS G=16.4 | FAIL | PASS G=15.9 | FAIL |
| coder-next | FAIL load | FAIL | FAIL load | FAIL |
| coder-next-q4 | PASS G=18.9 | FAIL | PASS G=21.1 | FAIL |

**Takeaway:** keep `-fa on` with q4_0 KV. kv-unified on/off is neutral for dense 72B (~5 t/s). coder-next APEX unstable on reload in p2+ (p1 single-run OK).

### Phase 3: context / KV quant

| Preset | 8k q4_0 | 8k q8/turbo3 | 32k q8/turbo3 |
|--------|---------|--------------|---------------|
| llama70b | PASS | PASS | PASS |
| qwen72b | PASS | SKIP turbo3 | SKIP turbo3 |
| kimi72b | PASS | SKIP turbo3 | SKIP turbo3 |
| qwen-next-80b | PASS | PASS | PASS |
| coder-next | FAIL all | FAIL | FAIL |
| coder-next-q4 | PASS | PASS | PASS |

32k ctx loads for dense 72B and MoE Q4 on ~45 GB pool.

### Phase 4: eval prompts (256 tok, 6 prompts from `large-model-eval.json`)

| Preset | Result |
|--------|--------|
| llama70b, qwen72b, kimi72b, qwen-next-80b, coder-next-q4 | **PASS** |
| coder-next (APEX) | FAIL (load exit in p2-p4; p1 fox prompt only) |

## Key artifacts

| Pattern | Content |
|---------|---------|
| `<label>.meta` | Bench config, VRAM excerpt |
| `<label>-server.log` | llama-server verbose load + inference |
| `<label>.result` | Throughput lines (when gen succeeds) |
| `<label>-rpc.log` | Local/remus RPC logs (when captured) |
| `<preset>-vram-calc.txt` | VRAM planner output per model |