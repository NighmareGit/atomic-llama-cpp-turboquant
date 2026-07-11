# Blackwell sm120 Server Crash Investigation

**Status: CLOSED (2026-07-09)** - resolved by CUDA `-INFINITY` fix. This folder is archived reference only.

## Resolution

| Item | Detail |
|------|--------|
| Symptom | `llama-server` silent crash on RTX 5070 Ti during/after generation |
| Root cause | `-INFINITY` -> `-((float)(1e+300))` in CUDA kernels (nvcc #221-D on MSVC 12.9) |
| Fix | `neg_inf_f32()` bit-cast helpers in `ggml/src/ggml-cuda/common.cuh` + `.cu` updates |
| Also required | Clean rebuild (`ggml-cuda.dll` stale artifacts caused BEX64) |

## Documents

| File | Purpose |
|------|---------|
| [INVESTIGATION.md](INVESTIGATION.md) | Full report, test matrix, closure summary |
| [CODE-CHANGES.md](CODE-CHANGES.md) | CUDA fix, CMake/RPC linker changes |
| [TEST-RESULTS.md](TEST-RESULTS.md) | Pre-fix failures + post-fix verification |
| [RESEARCH.md](RESEARCH.md) | Research log (hypotheses, now closed) |
| [SCRIPTS-LIST.md](SCRIPTS-LIST.md) | Build/test scripts inventory |

## Post-Fix Verification (RTX 5070 Ti)

| Test | Config | Result |
|------|--------|--------|
| Smoke | `smoke-llama-server.ps1`, Qwen3.5-9B-MTP | SMOKE_OK |
| Default np4 | `-c 4096/16384`, fit on, FA on/off | PASS |
| Long gen | 500 tokens, np4 | PASS (~121 t/s) |
| MTP 20k | `-c 20480 -np 1 -ctk turbo3 -ctv q8_0 --spec-type draft-mtp` | PASS (~128 t/s, 91% draft acceptance) |

Logs: `cfg-*.err`, `gen-*.err`, `smoke-test.err`, `mtp-20k-test/` in this folder.

## Quick Commands

```powershell
# Build (see blackwell-windows-build-guide/README.md)
.\blackwell-windows-build-guide\build-blackwell.ps1

# Smoke test
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1 -ModelPath D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf

# Recommended production flags (5070 Ti, single slot, MTP)
cd build-cuda-b-bin\bin\Release
.\llama-server.exe -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf `
  -c 20480 -ngl 99 -np 1 --no-kv-unified `
  -ctk turbo3 -ctv q8_0 -ctkd f16 -ctvd f16 `
  --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.6 `
  -b 2048 -ub 1024 --flash-attn on --host 0.0.0.0 --port 8080
```

## Related Docs

- [../blackwell-windows-build-guide/README.md](../blackwell-windows-build-guide/README.md) - Windows sm_120 build
- [../docs/blackwell/README.md](../docs/blackwell/README.md) - Cross-platform Blackwell guide
- [../docs/cuda-windows-5070ti/README.md](../docs/cuda-windows-5070ti/README.md) - 5070 Ti ops + RPC