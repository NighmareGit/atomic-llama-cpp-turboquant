# Handover: B+8–B+13 + Triton Linux pivot (2026-06-30)

**Purpose:** Resume Path-B-Plus lateral work (B+8–B+13), 2-GPU gate bisect on **Triton** after Jupiter LAN failure.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus` (local **ahead 8**, not pushed)  
**Mission root:** [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/)  
**Prior handover:** [HANDOVER-SESSION-2026-06-29.md](HANDOVER-SESSION-2026-06-29.md)

| Location | Path | Notes |
|----------|------|-------|
| **remus (this host)** | `/home/hunter/projects/atomic-llama-cpp-turboquant/atomic-llama-cpp-turboquant` | edit/commit here |
| **triton worker** | `user@192.168.8.23` → `~/projects/atomic-llama-cpp-turboquant` | Ubuntu **24.04** (was Windows in old docs) |
| **JUPITER** | `192.168.8.21:50053` | ping OK; **TCP connect times out** from remus — blocked |

**SSH (triton):** `user@192.168.8.23`, password <redacted> (use `pexpect` on remus — no `sshpass`).

---

## Locked mission decisions (from grilling)

| ID | Rule |
|----|------|
| **M3** | `overlap_pct >= 5%` is hard gate (**FAIL** everywhere so far) |
| **G1** | 2-GPU bisect first (`b6-2gpu-f` style, **n=384** canonical; **n=128** for fast bisect) |
| **B+8** | Primary bisect lever for overlap; on FAIL → **R2-then-R1** (flag bisect, then revert) |
| **Q5** | S5 PASS + (M1≥1% OR `input_wait_copy` −20% AND overlap≥0.5%) |

**Env rollback:** `GGML_PIPELINE_PLUS=0` or per-flag `=0` (`GGML_RPC_EVENT_DEFER_BARRIER`, `GGML_PIPELINE_BARRIER_PARTIAL`, etc.).

---

## Session summary (done)

### Code shipped locally (commit stack, oldest → newest)

| SHA | Summary |
|-----|---------|
| `bfc3b253f` | B+8 partial `pipeline_barrier` + B+10/B+13 sched copy path |
| `8340a1e7f` | B+9 EVENT defer-to-barrier + B+7a′ multi-socket drain |
| `aba1f00e5` | docs B+8–B+13 shipped; trace `barrier_mask` |
| `f1baf8a04` | **fix:** move `rpc_drain_all_endpoints_pending()` below `tls_pending_*` (compile) |
| `ed171e6c2` | `b6-gate-remus-docker.sh`, presets `b6-2gpu-jupiter`, `b6-2gpu-jupiter-plus0` |
| `b3acd9f1d` | **B+9 topology guard:** defer only when `ggml_backend_rpc_server_count() >= 2` (default `GGML_RPC_EVENT_DEFER_MIN_SERVERS=2`) |
| `032bba5b8` | **B+8b:** strict partial barrier — `barrier_slot_pending[]`, intersect with `src_mask` when `GGML_PIPELINE_BARRIER_PARTIAL_STRICT=1`; trace emits `pending_mask` |

### remus client build

- Docker CUDA 12.8, arch **`120a-real`** (RTX 5060 Ti)
- Binaries: `build-cuda-b-bin/bin/llama-pipeline-profiler`, `llama-cli`
- Run via `scripts/b6-gate-remus-docker.sh` (mounts repo at `/src`, models at `/models`)

### Jupiter attempt (blocked mid-session)

- Brief window ~17:29–17:35 UTC: smoke + bisects ran
- **Blocker:** `nc -zv 192.168.8.21 50053` **times out** from remus (firewall / bind localhost)
- B+9 guard verification on Jupiter **never completed** post-`b3acd9f1d`

| Label | n | G (t/s) | overlap% | stall | Notes |
|-------|---|---------|----------|-------|-------|
| smoke (Plus=1) | 128 | 84.4 | 0.8 | 0.976 | defer hurting |
| bisect `DEFER=0` | 128 | **102.5** | 0.6 | 0.857 | target for B+9 guard |
| Plus all on | 384 | 115.2 | 0.2 | 0.98 | |
| Plus=0 | 384 | **119.8** | 0.2 | 0.98 | best G |
| `DEFER=0` n=384 | 384 | 114.1 | 0.2 | 0.90 | |

**B+8 finding:** `pipeline_barrier_mask` always `wait_mask=7`, `src_mask=7` — motivated B+8b.

Artifacts: `benches/path-b-plus/b6-2gpu-jupiter*`, matrix row in `benches/path-b-plus/b6-diagnosis-matrix.tsv`.

### Triton pivot (Ubuntu 24.04 @ 192.168.8.23)

**Hardware:** RTX **3090** (24 GB, `CUDA0`, `:50054`) + RTX **3070** (8 GB, parked `:50055`).

**Done on triton:**

1. Repo synced via tarball → `~/projects/atomic-llama-cpp-turboquant` (no `.git` on worker)
2. Installed build deps: `cmake`, `ninja-build`, `libopenblas-dev`, `libomp-dev` (`sudo apt`, pw <redacted>)
3. Native CUDA build **sm_86-real** (~2.3 min): `build-cuda-b-bin/bin/rpc-server` (196 KB, Jun 30 22:50)
4. RPC worker running:
   ```bash
   export LD_LIBRARY_PATH=~/projects/atomic-llama-cpp-turboquant/build-cuda-b-bin/bin:/usr/local/cuda/lib64
   nohup ~/projects/atomic-llama-cpp-turboquant/build-cuda-b-bin/bin/rpc-server \
     -H 0.0.0.0 -p 50054 -d CUDA0 > /tmp/triton-rpc-50054.log 2>&1 &
   ```
5. **Connectivity:** `nc -zv 192.168.8.23 50054` **OK** from remus

**Not done:** triton git clone/sync; `:50055` 3070 worker; systemd/supervisor for rpc-server; update stale Windows docs under `docs/cuda-windows-triton/`.

### remus docker script fix (uncommitted)

`scripts/b6-gate-remus-docker.sh` — pass env into container (`PROFILER_BIN`, `PROFILER_SKIP_VALIDATE`, `PROFILER_OUT_DIR`, `BENCH_*`). Without this, container fell back to `build-rocm-docker/bin/llama-pipeline-profiler` (missing).

---

## Cluster map (2-GPU gate focus)

| Node | IP | GPU | Port | Role |
|------|-----|-----|------|------|
| **remus** | 192.168.8.176 | RTX 5060 Ti | local CUDA | profiler **client** (docker) |
| **triton** | 192.168.8.23 | RTX 3090 | **:50054** | RPC worker (`b6-2gpu-f-triton`) |
| triton | 192.168.8.23 | RTX 3070 | :50055 | spare (5-GPU preset) |
| JUPITER | 192.168.8.21 | 5070 Ti | :50053 | **unreachable** from remus |

**Preset `b6-2gpu-f-triton`:** `-rpc 192.168.8.23:50054`, `-ts 50,50`, model `Qwen3.5-9B-MTP-Q4_K_M.gguf`.

**Historical ref (old Windows triton):** G≈**187** t/s, overlap 0.3% — much faster straggler than Jupiter path.

**Baseline `b6-2gpu-f` (remus :50051, 35B):** G≈75.6, overlap 0.2%, drain 4514 ms.

---

## Known blockers

### 1. `--validate-rpc` hangs (HELLO)

- Symptom: profiler prints `ggml_cuda_init`, then blocks **minutes+**; strace shows `connect(192.168.8.23:50054)` OK but no HELLO `sendto` before timeout
- Triton log: `Accepted client connection` then `recv returned 0 (peer closed?)` — server waits for HELLO byte, client never completes handshake (or dies on timeout)
- **Workaround for benches:** `PROFILER_SKIP_VALIDATE=1` + `--skip-rpc-validate`
- **Do not assume R5 preflight passed** until this is root-caused (protocol mismatch? `send_rpc_cmd` drain path? docker DNS?)

### 2. Docker `PROFILER_OUT_DIR` must be under `/src`

Wrong (writes inside container FS, lost on exit):
```bash
PROFILER_OUT_DIR=/home/hunter/projects/.../benches/path-b-plus/foo
```

Correct:
```bash
PROFILER_OUT_DIR=/src/benches/path-b-plus/b6-2gpu-f-triton-guard-n128
```

### 3. Jupiter `:50053` LAN timeout

Use triton for 2-GPU RPC worker until firewall/bind fixed on JUPITER.

### 4. Docs drift

`docs/cuda-windows-triton/README.md` still describes **Windows** + PowerShell. Triton is now **Ubuntu 24.04** — need Linux ops doc or update README.

---

## In-flight at handoff

**Running:** `b6-2gpu-f-triton` n=128, `GGML_PIPELINE_PLUS=1`, `--skip-rpc-validate`

```bash
cd /home/hunter/projects/atomic-llama-cpp-turboquant/atomic-llama-cpp-turboquant
PROFILER_SKIP_VALIDATE=1 BENCH_GEN_TOKENS=128 \
  PROFILER_OUT_DIR=/home/hunter/projects/.../b6-2gpu-f-triton-guard-n128   # ← WRONG PATH (container-local)
bash scripts/b6-gate-remus-docker.sh b6-2gpu-f-triton --skip-rpc-validate
```

- Docker container was up ~17+ min; profiler PID inside container on `-rpc 192.168.8.23:50054`
- **On resume:** check `docker ps`; if still running, `docker logs -f <id>` or wait; if done, **copy results from container** before `docker rm`:
  ```bash
  CID=$(docker ps -q --filter ancestor=nvidia/cuda:12.8.0-devel-ubuntu22.04 | head -1)
  docker cp "$CID:/home/hunter/projects/atomic-llama-cpp-turboquant/atomic-llama-cpp-turboquant/benches/path-b-plus/b6-2gpu-f-triton-guard-n128" \
    ./benches/path-b-plus/
  bash scripts/b6-gate-diagnose-runs.sh b6-2gpu-f-triton-guard-n128
  ```

---

## Next session (priority order)

1. **Finish / re-run** `b6-2gpu-f-triton` n=128 with correct `PROFILER_OUT_DIR=/src/benches/...` — confirm G, overlap, stall vs Jupiter bisect (`DEFER=0` → ~102 t/s target for guard sanity)
2. **Canonical n=384** `b6-2gpu-f-triton` (Plus=1, then bisect flags: `GGML_RPC_EVENT_DEFER_BARRIER=0`, `GGML_PIPELINE_BARRIER_PARTIAL_STRICT=0`, etc.)
3. **B+8b trace check:** `pending_mask` in `pipeline_barrier_mask`; `wait_mask` should **not** always be `7`
4. **B+9 guard verify:** with single RPC server, defer should auto-off; compare stall vs Jupiter smoke 0.976
5. **Fix `--validate-rpc` hang** (blocking for automated preflight)
6. **Triton ops:** git sync instead of tarball; `scripts/b6-gate-triton-linux-start-rpc.sh` (start/stop/restart); optional `:50055` 3070
7. **Push** 8 local commits + uncommitted `b6-gate-remus-docker.sh` fix
8. Later: B+7a′ 4-GPU drain on `b6-4gpu-g-triton` once 2-GPU gate understood

---

## Command cheat sheet

### Triton — build rpc-server (native, already done once)

```bash
ssh user@192.168.8.23
cd ~/projects/atomic-llama-cpp-turboquant
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build-cuda-b-bin -G Ninja \
  -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86-real \
  -DGGML_CUDA_F16=ON -DGGML_CUDA_K_QUANTS=ON -DGGML_CUDA_MMQ=ON \
  -DGGML_SCHED_MAX_COPIES=4
cmake --build build-cuda-b-bin --target rpc-server -j$(nproc)
```

### Triton — start / check RPC

```bash
pkill -f 'rpc-server.*50054' || true
export LD_LIBRARY_PATH=~/projects/atomic-llama-cpp-turboquant/build-cuda-b-bin/bin:/usr/local/cuda/lib64
nohup ~/projects/atomic-llama-cpp-turboquant/build-cuda-b-bin/bin/rpc-server \
  -H 0.0.0.0 -p 50054 -d CUDA0 > /tmp/triton-rpc-50054.log 2>&1 &
ss -tlnp | grep 50054
tail -20 /tmp/triton-rpc-50054.log
```

### remus — 2-GPU gate (recommended invocation)

```bash
cd /home/hunter/projects/atomic-llama-cpp-turboquant/atomic-llama-cpp-turboquant

# smoke n=128
PROFILER_SKIP_VALIDATE=1 BENCH_GEN_TOKENS=128 \
  PROFILER_OUT_DIR=/src/benches/path-b-plus/b6-2gpu-f-triton-guard-n128 \
  bash scripts/b6-gate-remus-docker.sh b6-2gpu-f-triton --skip-rpc-validate

# canonical n=384
PROFILER_SKIP_VALIDATE=1 BENCH_GEN_TOKENS=384 \
  PROFILER_OUT_DIR=/src/benches/path-b-plus/b6-2gpu-f-triton-n384 \
  bash scripts/b6-gate-remus-docker.sh b6-2gpu-f-triton --skip-rpc-validate

# bisect example
PROFILER_SKIP_VALIDATE=1 BENCH_GEN_TOKENS=128 GGML_RPC_EVENT_DEFER_BARRIER=0 \
  PROFILER_OUT_DIR=/src/benches/path-b-plus/b6-2gpu-f-triton-no-defer \
  bash scripts/b6-gate-remus-docker.sh b6-2gpu-f-triton --skip-rpc-validate

bash scripts/b6-gate-diagnose-runs.sh b6-2gpu-f-triton-guard-n128
```

### remus — preflight (currently broken — debug first)

```bash
# Hangs after ggml_cuda_init — do not gate on this until fixed
docker run --rm --gpus=all -v "$PWD:/src" -w /src -e LD_LIBRARY_PATH=/src/build-cuda-b-bin/bin \
  nvidia/cuda:12.8.0-devel-ubuntu22.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq libgomp1 libopenblas0 >/dev/null && \
    /src/build-cuda-b-bin/bin/llama-pipeline-profiler --validate-rpc -rpc 192.168.8.23:50054 -ts 50,50'
```

### Flag reference (B+8–B+13, default on with `GGML_PIPELINE_PLUS=1`)

| Flag | Effect |
|------|--------|
| `GGML_PIPELINE_PLUS=0` | Master rollback |
| `GGML_RPC_EVENT_DEFER_BARRIER=0` | B+9 off |
| `GGML_RPC_EVENT_DEFER_MIN_SERVERS=2` | B+9 guard (defer only if ≥2 RPC servers) |
| `GGML_PIPELINE_BARRIER_PARTIAL=0` | B+8 off |
| `GGML_PIPELINE_BARRIER_PARTIAL_STRICT=0` | B+8b off |
| `GGML_RPC_MULTI_SOCKET_FLUSH=0` | B+7a′ off |

---

## Uncommitted local changes

```
 M scripts/b6-gate-remus-docker.sh    # docker -e pass-through (needed)
 M benches/path-b-plus/regression.jsonl
?? benches/path-b-plus/b6-2gpu-jupiter*/  # Jupiter run artifacts
?? benches/path-b-plus/b6-diagnosis-matrix.tsv
```

---

## Files to read first in new session

1. This handover
2. [docs/rpc-multi-backend-pipeline-plus/PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) Phase 2
3. [docs/rpc-multi-backend-pipeline-plus/TRACKING.md](../../docs/rpc-multi-backend-pipeline-plus/TRACKING.md)
4. `ggml/src/ggml-backend.cpp` — B+8/B+8b barrier_mask trace
5. `ggml/src/ggml-rpc/ggml-rpc.cpp` — B+9 defer + `ggml_backend_rpc_event_defer_barrier()`
6. `scripts/b6-gate-remus-docker.sh`, `scripts/b6-gate-profiler-romulus.sh`

---

*Generated 2026-06-30 — remus session, Triton Linux pivot, Jupiter blocked.*