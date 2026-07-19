# ROCm-max split hunt
base=`/tmp/rocm-max-split-20260719165956`

| Run | TG t/s | layers | exit |
|-----|-------:|--------|------|
| cur-1gpu-rocm | 103.70 | {'ROCm0': 294} | exit=0 |
| cur-ts5-95 | 83.88 | {'RPC0': 21, 'ROCm0': 273} | exit=0 |
| cur-ts95-5 | - | {'RPC0': 40, 'ROCm0': 2} | exit=134 |
| fast-1gpu-rocm | 104.32 | {'ROCm0': 294} | exit=0 |
| fast-ts10-90 | 92.76 | {'ROCm0': 264, 'RPC0': 30} | exit=0 |
| fast-ts2-98 | 97.98 | {'ROCm0': 288, 'RPC0': 6} | exit=0 |
| fast-ts24-76 | 87.63 | {'ROCm0': 228, 'RPC0': 66} | exit=0 |
| fast-ts5-95 | 95.96 | {'ROCm0': 276, 'RPC0': 18} | exit=0 |
| fast-ts90-10 | - | {'ROCm0': 46, 'RPC0': 38} | exit=1 |
| fast-ts95-5 | - | {'ROCm0': 44, 'RPC0': 40} | exit=1 |

**Best live:** fast-1gpu-rocm @ 104.32 t/s

## Historical
- Peer 148: TS **24,76** (Qwen 35B), Plus=1, q8_0, dual ROCm+RPC
- With RPC-first enum, 24,76 ~= 24% RPC + 76% ROCm (already ROCm-majority)
- D7 docs also cite **30,70**
