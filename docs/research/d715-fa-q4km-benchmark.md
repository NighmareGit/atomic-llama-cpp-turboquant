# D7.15: FA + Q4_K_M Benchmark Results

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3.5-9B-MTP-Q4_K_M.gguf (5.87 GB) |
| Hardware | Romulus 2-GPU: 7900XTX (ROCm) + 3060Ti (RPC CUDA) |
| Binary | build-hip/bin/llama-cli (b10144, FA enabled) |
| Flags | `-n 64 -ngl 999 --temp 0.0 --rpc 127.0.0.1:50051 -st` |
| Context | 64 tokens generated |

## Results

### FA ON (Run 1)
- **Prompt Processing (PP):** 112.3 t/s
- **Generation (TG):** 76.2 t/s

### FA ON (Run 2)
- **Prompt Processing (PP):** 114.7 t/s
- **Generation (TG):** 77.1 t/s

### FA ON Average
- **PP:** ~113.5 t/s
- **TG:** ~76.7 t/s

### FA OFF Baseline
- **Status:** FAILED - OOM on 3060Ti (segfault)
- Without FA, attention buffers exceed 3060Ti VRAM (needs ~8.5GB+)
- FA ON is **required** for this model on the 2-GPU Romulus config

## Comparison Table

| Configuration | TG (t/s) | Notes |
|---------------|----------|-------|
| D7.3 Q6_K + FA ON | 143.0 | Baseline from D7.3 |
| D7.15 Q4_K_M + FA ON | 76.7 | This benchmark |
| Delta | -46.4% | Significant regression |

## Analysis

### Unexpected Regression

The Q4_K_M result (76.7 t/s) is **46% slower** than the Q6_K baseline (143.0 t/s). This is counter-intuitive because:

1. Q4_K_M is smaller (4-bit vs 6-bit) - should require less memory bandwidth
2. Q4_K_M should allow more layers on the 3060Ti
3. Smaller quantization typically yields higher TG throughput

### Possible Explanations

1. **3060Ti Bottleneck:** The 3060Ti (8GB) may be the straggler. With Q6_K, the 7900XTX handles more layers. With Q4_K_M, more layers fit on the 3060Ti, making it the bottleneck.

2. **RPC Overhead:** The Q4_K_M model may have different layer distribution across RPC, exposing communication overhead.

3. **Different Compute Profile:** Q4_K_M uses different dequantization kernels that may be slower on the 3060Ti's GA104 architecture.

## Verdict

**Q4_K_M does NOT deliver the expected 15-25% combined gain.** Instead, it shows a significant -46% regression compared to Q6_K.

This suggests:
- The 3060Ti is the performance bottleneck when more layers fit on it
- Q6_K's larger size may actually help by keeping more compute on the faster 7900XTX
- The expected "smaller model = faster" assumption does not hold for this 2-GPU split configuration

## Recommendations

1. Investigate layer distribution between 7900XTX and 3060Ti for both quantizations
2. Consider forcing more layers onto the 7900XTX with Q4_K_M (if VRAM allows)
3. Profile whether the 3060Ti is the bottleneck with Q4_K_M
