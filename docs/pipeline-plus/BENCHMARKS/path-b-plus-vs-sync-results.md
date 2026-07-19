# Path-B+ vs SYNC — generated results

Rows: 36 ok / 36 total | last campaign: `20260701-210320`

Regenerate:
```bash
python3 scripts/b6-gate-sync-vs-plus-comparison.py
```

## Workload: n384 (n=384 gate depth)

### b6-3gpu-g-triton — workload `n384`

| Model ID | Model | SYNC G | PLUS G | Delta% | Status |
|----------|-------|--------|--------|--------|--------|
| M1 | Qwen3.6-35B-A3B | 46.7 | 69.1 | +48.1% | S:ok P:ok |
| M3 | Gemma-4-26B-A4B | 53.7 | 81.0 | +50.8% | S:ok P:ok |
| M5 | Qwen3.5-27B | 22.8 | 35.4 | +55.7% | S:ok P:ok |

### b6-5gpu-g-prod — workload `n384`

| Model ID | Model | SYNC G | PLUS G | Delta% | Status |
|----------|-------|--------|--------|--------|--------|
| M1 | Qwen3.6-35B-A3B | 43.3 | 59.6 | +37.7% | S:ok P:ok |
| M3 | Gemma-4-26B-A4B | 46.1 | 69.0 | +49.9% | S:ok P:ok |
| M5 | Qwen3.5-27B | 20.0 | 28.1 | +40.0% | S:ok P:ok |
| M6 | Llama-3-70B Q4 | 11.0 | 15.2 | +38.3% | S:ok P:ok |
| M7 | Qwen3-Next-80B | 36.1 | 47.5 | +31.4% | S:ok P:ok |
| M8 | Kimi-Dev-72B | 12.3 | 16.0 | +30.7% | S:ok P:ok |

## Workload: n2048-mt (n=2048 multi-turn)

### b6-3gpu-g-triton — workload `n2048-mt`

| Model ID | Model | SYNC G | PLUS G | Delta% | Status |
|----------|-------|--------|--------|--------|--------|
| M1 | Qwen3.6-35B-A3B | 47.6 | 69.3 | +45.6% | S:ok P:ok |
| M3 | Gemma-4-26B-A4B | 51.4 | 82.6 | +60.6% | S:ok P:ok |
| M5 | Qwen3.5-27B | 22.5 | 34.9 | +54.9% | S:ok P:ok |

### b6-5gpu-g-prod — workload `n2048-mt`

| Model ID | Model | SYNC G | PLUS G | Delta% | Status |
|----------|-------|--------|--------|--------|--------|
| M1 | Qwen3.6-35B-A3B | 43.1 | 60.1 | +39.6% | S:ok P:ok |
| M3 | Gemma-4-26B-A4B | 46.7 | 70.8 | +51.6% | S:ok P:ok |
| M5 | Qwen3.5-27B | 20.0 | 27.8 | +39.0% | S:ok P:ok |
| M6 | Llama-3-70B Q4 | 10.7 | 14.8 | +38.4% | S:ok P:ok |
| M7 | Qwen3-Next-80B | 36.1 | 49.5 | +37.2% | S:ok P:ok |
| M8 | Kimi-Dev-72B | 11.8 | 15.5 | +31.9% | S:ok P:ok |

