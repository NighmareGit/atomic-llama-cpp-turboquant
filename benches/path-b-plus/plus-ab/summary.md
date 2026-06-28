# T1: Path-B Plus A/B -- published rows

Model unless noted: `Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf`, ctx=4096, q4_0 KV, ngl=99,
fox prompt, 128 gen tokens, single slot (`-np 1`).

## Matrix B -- scheduler pipeline (Layer B)

| Label | Plus | Topology | ts | G (t/s) | overlap_count | overlap_pct | Source |
|-------|------|----------|-----|---------|---------------|-------------|--------|
| trace-f-3gpu-legacy | 0 | 5070+5060+6600 | 30,12,58 | 39.3 | 127 | 0.1% | spikes S3 |
| trace-f-3gpu-plus | 1 | 5070+5060+6600 | 30,12,58 | 42.8 | 289 | 0.2% | spikes S5 |
| trace-f-2gpu-plus | 1 | 5070+5060 | 50,50 | **48.9** | 297 | 0.6% | production default; diagnose 2026-06-28 |
| trace-g-4gpu-primary | 1 | 7900+3060+5060+5070 | 36,24,24,16 | 38-43 | 1075 | 0.1% | cluster 2026-06-27 |
| trace-g-4gpu-primary-resume | 1 | same | 36,24,24,16 | 23-29 | TBD | TBD | smoke 2026-06-28 |

Full artifact paths:

- Windows 2/3-GPU: `docs/cuda-windows-5070ti/benchmarks/trace-f-*`
- Romulus 4-GPU: `rpc-patch/patch/bench-results/rpc-server-bench/trace-g-4gpu-primary*`
- Hotpath parse: `rpc-patch/scripts/pathb-hotpath-summary.sh`

## Drain comparison (128-token window, S3)

| Run | Plus | drain_flush_ms | blocking_ms |
|-----|------|----------------|-------------|
| trace-f-3gpu-legacy | 0 | 2523 | 32375 |
| trace-f-3gpu-plus | 1 | 2599 | 31144 |
| trace-f-2gpu-plus | 1 | **1734** | 29812 |

## Regression log

Paired profiler runs append to [../regression.jsonl](../regression.jsonl). Seed offline rows:

```bash
./scripts/llama-pipeline-import-baseline.sh
```

## TBD (run `scripts/bench-pipeline-plus-ab.sh`)

Paired on/off cells on the **same** host + topology in one session:

| Cell | `GGML_PIPELINE_PLUS` | Status |
|------|----------------------|--------|
| `*-plus-on` | 1 | use stub |
| `*-plus-off` | 0 | use stub |