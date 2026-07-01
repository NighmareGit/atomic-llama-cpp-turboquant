# validate-rpc investigation (PR 8)

**Date:** 2026-07-01  
**SHA:** 0b6542e1d (pre-RPC-only preflight helper)

## Root cause (historical hang)

Remus docker CUDA client + triton `:50054` could hang during `--validate-rpc` when HELLO bytes were not sent promptly (TCP buffering + early drain on init commands). Fixed in `ggml-rpc.cpp`:

- Combined cmd+size+data send buffer
- Skip drain for HELLO + DEVICE_COUNT
- Post-hello `tls_pending_*` reset

Hang was **configuration-dependent**, not remus-permanent.

## Matrix results (2026-07-01)

| Cell | Client | Endpoint | Status | Elapsed |
|------|--------|----------|--------|---------|
| A | romulus ROCm | triton :50054 | PASS | 1s |
| B | remus docker CUDA | triton :50054 | PASS | 18s |
| C | remus docker CUDA | remus :50051 | PASS | 18s |
| D | romulus ROCm | 4-GPU triple RPC | PASS | 1s |

Logs: `A.log` .. `D.log` in this directory. TSV: `results.tsv`.

## Code hardening

`pipeline_rpc_validate_prepare()` loads `libggml-rpc.so` only for `--validate-rpc`, avoiding full CUDA/ROCm init when the plugin is on `LD_LIBRARY_PATH`.

## Policy

- 4-GPU gate runs: may use RPC preflight (all cells pass).
- 2-GPU bisects: may keep `PROFILER_SKIP_VALIDATE=1` for speed.

Re-run: `bash scripts/b6-gate-validate-rpc-matrix.sh --run`