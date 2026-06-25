# Benchmark artifacts

Project root: `/home/hunter/projects/atomic-llama-cpp-patch/patch/bench-results/`

Logs archived from RPC Path A/B testing. Default output for scripts in `../../scripts/` (override via `BENCH_LOG_DIR`, `PROBE_LOG_DIR`, `PATHB_LOG_DIR`).

## Layout

| Subdirectory | Scripts | Contents |
|--------------|---------|----------|
| `rpc-server-bench/` | `rpc-server-bench.sh`, `rpc-server-bench-matrix.sh` | `<label>.meta`, `.result`, `-server.log`, `-rpc.log` |
| `rpc-ts-probe/` | `rpc-ts-fit-probe.sh` | `<label>.meta`, `.gpu` (rocm-smi/nvidia-smi poll), `-server.log` |
| `pathb-runs/` | `pathb-run-test.sh`, `pathb-test.sh` | `<label>.log`, `.meta`, `.log.raw` |

## Key summary file

`rpc-server-bench/matrix-summary.txt` — consolidated generation averages across matrix runs.

## Reproduce

```bash
./scripts/rpc-server-bench-matrix.sh
```

Logs append here by default; no `/tmp` paths required.