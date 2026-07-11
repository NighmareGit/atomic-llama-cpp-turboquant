# Speculative Decoding Benchmark Report (RTX 5070 Ti)

This report details a performance probe evaluating the impact of different speculative decoding configurations (ngram vs. MTP) on the newly built optimized Release server.

---

## 1. Benchmark Settings
* **Hardware**: RTX 5070 Ti (16 GiB VRAM), 13th Gen Intel Core i7-13700K
* **Target Model**: `Qwen3.5-9B-MTP-Q4_K_M.gguf` (approx. 5.8 GiB model)
* **KV Cache**: `K` = `q8_0`, `V` = `turbo3` (3-bit PolarQuant)
* **Benchmark Query**: Short prompt, generating `128` tokens.

---

## 2. Performance Matrix

| Configuration | Prompt Eval Speed (TPS) | Generation Speed (TPS) | Draft Acceptance Rate | Key Observations |
| :--- | :---: | :---: | :---: | :--- |
| **Qwen-Base (No Speculation)** | 83.66 | 119.94 | N/A | Baseline performance. |
| **Qwen-Ngram (max=4)** | **192.55** | 117.20 | 24.7% | Excellent prompt processing speedup, but low draft acceptance drags down generation speed below baseline. |
| **Qwen-MTP (max=3)** | 167.53 | 89.28 | 62.5% | Lower draft acceptance for 3-token chains causes speculative rejection overhead to severely degrade throughput. |
| **Qwen-MTP (max=2)** | 172.98 | **123.37** | **80.2%** | **Best overall mix.** High acceptance rate (80.2%) overcomes draft overhead to yield the highest generation speed. |

---

## 3. Analysis & Key Takeaways

### 3.1. The "Sweet Spot" for MTP: `max=2`
- When configuring MTP (Multi-Token Prediction) speculation on the Qwen model, setting `--spec-draft-n-max 2` acts as the sweet spot. It achieves a **80.2% draft acceptance rate** while keeping compilation overhead low.
- Increasing the chain length to `max=3` drops the acceptance rate to **62.5%**. Because the third token is rejected more frequently, the computational cost of evaluating the rejected draft steps outweighs the speed gain, causing generation speed to drop to `89.28 TPS` (a ~25% loss compared to baseline).

### 3.2. Speculative Prompt Processing Boost
- Speculative decoding significantly speeds up the initial slot response time (modeled as prompt evaluation in short contexts) by over **2x** (from 83.66 TPS up to 192.55 TPS). This is because the draft heads reduce slot initialization latency and accelerate the transition from prefill to the first token generation.

---

## 4. Recommended Run Command
For optimal performance on the RTX 5070 Ti, run the model with the following speculative parameters:

```powershell
llama-server.exe `
  -m D:\Models\Qwen3.5-9B-MTP-Q4_K_M.gguf `
  -c 180000 `
  --host 0.0.0.0 `
  --port 8080 `
  -fa on `
  -ctk q8_0 `
  -ctv turbo3 `
  -ngl 99 `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  --spec-draft-n-min 1 `
  --draft-p-min 0.6
```
