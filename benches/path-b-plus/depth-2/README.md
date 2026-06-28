# T0: depth-2 A/B results

Runs comparing `LLAMA_PIPELINE_DEPTH2` unset (on) vs `=0` (off) against the same
`llama-server` MTP or NextN configuration.

## Publish target (Matrix A)

| Cell | depth-2 | Spec | KV | Host | short G (n=128) | Status |
|------|---------|------|-----|------|-----------------|--------|
| gemma-26B-mtp-d2-on | on | mtp | turbo3 | M4 Max | 80.5 (matrix) | [MTP.md](../../../MTP.md) |
| gemma-26B-mtp-d2-off | off | mtp | turbo3 | TBD | TBD | run stub |

## How to populate

```bash
HOST=127.0.0.1 PORT=8080 N_PREDICT=128 RUNS=3 ./scripts/bench-pipeline-depth2.sh
```

Output: `benches/path-b-plus/depth-2/<timestamp>-depth2-ab/summary.md`