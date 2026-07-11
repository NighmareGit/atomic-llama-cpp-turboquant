# Advanced Model & Draft Settings Benchmark Report (RTX 5070 Ti)

This report investigates the performance impact of batching sizes (`-b`/`-ub`), unified KV caches (`--kv-unified`), and draft cache quantization (`-ctkd`/`-ctvd`) on Qwen 3.5 9B speculative decoding.

---

## 1. Performance Matrix

| Metric Section | Configuration | Prompt Eval (TPS) | Gen Speed (TPS) | Draft Acceptance | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Batch Size** | `MTP-Base (b=1024, ub=512)` | 576.96 | 109.14 | 88.2% | Standard baseline batch sizes. |
| | `MTP-SmallBatch (b=128, ub=128)` | 618.99 | 111.14 | 86.1% | Small batches reduce prefill concurrency but save VRAM. |
| | **`MTP-LargeBatch (b=2048, ub=1024)`** | **635.79** | **122.07** | **91.7%** | **Best for Prefill & Gen.** Large batches maximize CUDA core occupancy. |
| **KV-Unified** | `MTP-KV-Unified-On` | 640.37 | 114.50 | 82.5% | Unified slot buffer (helpful for multi-sequence setups). |
| | **`MTP-KV-Unified-Off`** | **641.99** | **122.83** | **85.4%** | **Best for Single-Slot.** Disabling unified buffer saves lookup overhead. |
| **Draft Cache** | **`MTP-DraftCache-F16`** | **656.76** | **119.75** | **91.9%** | **Best Draft Cache.** Unquantized cache has no noise and lowest compute overhead. |
| | `MTP-DraftCache-Q8` | 647.37 | 100.60 | 83.9% | Quantization overhead and noise degrade throughput. |
| | `MTP-DraftCache-Turbo3` | 614.74 | 111.87 | 84.6% | 3-bit quantization noise degrades draft accuracy. |

---

## 2. Analysis & Recommendations

### 2.1. Batch Size (`-b` / `-ub`)
- **Impact**: Setting larger batch sizes (**`-b 2048 -ub 1024`**) maximizes prompt processing speed (**635.79 TPS**) by fully saturating the CUDA cores.
- **Recommendation**: Keep `-b 2048 -ub 1024` unless VRAM limits require scaling down to `-b 1024 -ub 512`.

### 2.2. Unified KV Cache (`--kv-unified` / `-kvu`)
- **Impact**: For single-slot execution (`-np 1`), disabling the unified cache (**`--no-kv-unified`**) increases generation speed from 114.50 TPS to **122.83 TPS**. Unified caching introduces lookup overhead that is only beneficial when multiple slots are running in parallel.
- **Recommendation**: Use `--no-kv-unified` for single-sequence/single-slot tasks; use `--kv-unified` for multi-sequence servers.

### 2.3. Draft KV Cache Quantization (`-ctkd` / `-ctvd`)
- **Impact**: Quantizing the draft model's KV cache to `q8_0` or `turbo3` drops the draft acceptance rate (from 91.9% down to 83.9%) and reduces generation speed. 
- **Reasoning**: The draft KV cache is extremely small (holding only 2-3 draft tokens). Quantizing it saves no meaningful memory bandwidth but introduces dequantization noise which impairs the draft model's predictions, causing more tokens to be rejected. Additionally, dequantization calculations add GPU kernel launch latencies.
- **Recommendation**: Always leave the draft cache unquantized (**`-ctkd f16 -ctvd f16`**), while keeping the target model's cache quantized (e.g., `-ctk q8_0 -ctv turbo3`).

---

## 3. Recommended Run Command (Single Slot, High-Speed)
```powershell
llama-server.exe `
  -m D:\Models\Qwen3.5-9B-MTP-Q4_K_M.gguf `
  -c 180000 `
  --host 0.0.0.0 `
  --port 8080 `
  -fa on `
  -ctk q8_0 `
  -ctv turbo3 `
  -ctkd f16 `
  -ctvd f16 `
  -ngl 99 `
  -b 2048 `
  -ub 1024 `
  -np 1 `
  --no-kv-unified `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  --spec-draft-n-min 1 `
  --draft-p-min 0.6
```
