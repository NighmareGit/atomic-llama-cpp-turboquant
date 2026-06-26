# Benchmark artifacts

Project root: `/home/hunter/projects/atomic-llama-cpp-patch/patch/bench-results/`

Logs archived from RPC Path A/B testing. Default output for scripts in `../../scripts/` (override via `BENCH_LOG_DIR`, `PROBE_LOG_DIR`, `PATHB_LOG_DIR`).

## Layout

| Subdirectory | Scripts | Contents |
|--------------|---------|----------|
| `rpc-server-bench/` | `rpc-server-bench.sh`, `rpc-server-bench-matrix.sh` | `<label>.meta`, `.result`, `-server.log`, `-rpc.log` |
| `rpc-ts-probe/` | `rpc-ts-fit-probe.sh` | `<label>.meta`, `.gpu` (rocm-smi/nvidia-smi poll), `-server.log` |
| `pathb-runs/` | `pathb-run-test.sh`, `pathb-test.sh` | `<label>.log`, `.meta`, `.log.raw` |
| `72b-matrix/` | `pathb-72b-matrix.sh` | Phase 1-4 large-model matrix; see `72b-matrix/README.md` |

## Key summary files

| File | Config |
|------|--------|
| `rpc-server-bench/matrix-summary.txt` | A (local 3060 Ti) |
| `rpc-server-bench/matrix-summary-remus.txt` | B (remus 5060 Ti remote) |
| `rpc-server-bench/matrix-summary-config-c.txt` | C (dual RPC workers) |
| `72b-matrix/matrix-summary.txt` | 72B+ phased matrix full log (phases 1-4) |
| `72b-matrix/phase-summary.txt` | Production preset cheatsheet |
| `72b-matrix/load-ranking.txt` | Phase 1 gen speed ranking |
| `72b-matrix/phase-winners.txt` | Seeds for phase 2+ reruns |

Remote bench `.meta` files include `rpc_mode`, `endpoint`, and `df -h` before/after.

### 72B+ matrix (2026-06-26, phases 1-4 complete)

Config C `ts=35,15,50`, ~45 GB pool. Post RPC fix: dense 72B **fitoff ngl=60** PASS (~5 t/s). MoE **coder-next-q4** G=16-21 t/s. qwen-next-80b needs **fiton**. Details: [`72b-matrix/README.md`](72b-matrix/README.md).

## Reproduce

```bash
./scripts/rpc-server-bench-matrix.sh
```

Logs append here by default; no `/tmp` paths required.