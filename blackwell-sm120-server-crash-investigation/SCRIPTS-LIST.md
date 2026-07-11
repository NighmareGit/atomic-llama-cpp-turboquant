# Scripts Inventory

**Investigation status: CLOSED (2026-07-09).** Use canonical scripts below for ongoing ops.

## Build Scripts

| Script | Path | Purpose | Status |
|--------|------|---------|--------|
| Blackwell build | `blackwell-windows-build-guide/build-blackwell.ps1` | Clean sm_120 Windows build | **Canonical** |
| Path B build | `scripts/cuda-windows-5070ti/build.ps1` | 5070 Ti + RPC portable (`120a-real`) | **Canonical** |
| CUDA install | `scripts/install-cuda-1292.ps1` | Install CUDA 12.9.2 from local package | Optional |
| Manual cmake | See `blackwell-windows-build-guide/README.md` | Bypass VS CUDA 13.3 integration | Documented |

## Test Scripts

| Script | Path | Purpose | Status |
|--------|------|---------|--------|
| Server smoke | `scripts/cuda-windows-5070ti/smoke-llama-server.ps1` | Health check + chat completion | **Canonical** |
| Server test | `scripts/test-server.ps1` | Start llama-server + send requests | Legacy |
| CLI test | `scripts/test-cli.ps1` | Batch llama-cli testing | Legacy |
| GPU monitor | `scripts/gpu-monitor.ps1` | Monitor GPU utilization during inference | Optional |

## Cluster Scripts (Legacy)

| Script | Path | Purpose | Status |
|--------|------|---------|--------|
| Romulus local up | `scripts/romulus-local-up.sh` | Start local 2-GPU on romulus cluster | Not used locally |
| Romulus local build | `scripts/romulus-local-build.sh` | Build for romulus cluster | Not used locally |
| Legacy inventory | `scripts/cluster-legacy-inventory.sh` | List legacy cluster dirs | Not used locally |
| Legacy salvage | `scripts/cluster-legacy-salvage.sh` | Salvage legacy cluster data | Not used locally |
| Legacy archive | `scripts/cluster-legacy-archive.sh` | Archive legacy cluster data | Not used locally |
| Git remotes | `rpc-patch/patch/pathb-cluster-git-remotes.sh` | Pick newest reachable git tip | Not used locally |

## Model Conversion Scripts

| Script | Path | Purpose |
|--------|------|---------|
| HuggingFace conversion | `convert_hf_to_gguf.py` | Convert HF models to GGUF |
| GGUF update | `convert_hf_to_gguf_update.py` | Update GGUF metadata |
| Lora conversion | `convert_lora_to_gguf.py` | Convert LoRA to GGUF |
| Legacy conversion | `convert_llama_gguf_to_gguf.py` | Legacy llama format conversion |

## Conversion Backend Scripts (Python)

Located in `conversion/` directory - 70+ model-specific converters:
- `llama.py`, `gemma.py`, `qwen.py`, `mistral.py`, `deepseek.py`, `grok.py`, etc.

## CI Scripts

| Script | Path | Purpose |
|--------|------|---------|
| CI run | `ci/run.sh` | Main CI pipeline runner |
| CI README | `ci/README.md` | CI documentation |
| CI MUSA README | `ci/README-MUSA.md` | MUSA (Moore) backend docs |

## Docker Scripts

Located in `dockers/blueprints/`:
- 8 files: 4 shell scripts, 1 markdown, 1 RPC config, etc.

---

## Manual Commands Used (Not in Scripts)

### Build Command (Current)
```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 && cmake -B build-cuda-b-bin -G "Ninja Multi-Config" -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=ON -DGGML_CUDA_MMQ=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-cuda-b-bin --config Release --parallel'
```

### llama-cli Test Command
```powershell
cd build-cuda-b-bin\bin\Release
.\llama-cli.exe -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -n 512 -ngl 99 -c 16384 -t 4 --flash-attn off --no-warmup -p "Test prompt" -e | Out-File -FilePath models\test-output.txt
```

### Server Start Command
```powershell
cd build-cuda-b-bin\bin\Release
.\llama-server.exe --host 0.0.0.0 --port 8080 -m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -ngl 99 -c 4096 -t 4 --flash-attn off
```

### Curl Test Command
```powershell
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Log Files Generated

| Log File | Size | Description |
|----------|------|-------------|
| `build-all.log` | - | Full build output |
| `build-cuda-full.log` | - | CUDA backend build output |
| `build-vs-cuda/` | - | VS CUDA build logs |
| `gpu-debug.log` | - | GPU debug output |
| `gpu-debug2.log` | - | GPU debug output #2 |
| `server-start.log` | - | Server startup log |
| `server-help.txt` | - | Server --help output |
| `models/server-*.log` | Various | Individual server crash logs (13+ files) |
| `models/test-qwen-mtp-stdout.txt` | 20 MB | llama-cli test output |
