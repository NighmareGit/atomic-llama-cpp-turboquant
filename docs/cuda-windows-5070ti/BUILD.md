# Build: cuda-windows-5070ti

Canonical build documentation: [../cuda-windows/BUILD.md](../cuda-windows/BUILD.md)

```powershell
.\scripts\cuda-windows-5070ti\build.ps1
# equivalent:
.\scripts\cuda-windows\build.ps1 -Profile all
```

5070 Ti uses `-Profile all` (`86-real;120a-real`) for portable binaries that also run on Ampere nodes.

## Smoke test

```powershell
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1
.\scripts\cuda-windows-5070ti\smoke-llama-server.ps1 -ModelPath "D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf"
```

Logs: `docs/cuda-windows-5070ti/benchmarks/<timestamp>/`

Monitor VRAM: `nvidia-smi -l 2`