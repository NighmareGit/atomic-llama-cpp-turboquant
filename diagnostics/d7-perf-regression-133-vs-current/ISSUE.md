# Restore dual-GPU full-throttle TG + fix GPipe/server MTP

Type: bugfix + performance regression  
Status: resolved
Triage: ready for agent  
Blocked by: none (kernel gate must pass; see Phase 0)

## Resolution (2026-07-19)

1. **Dual TG regression**: Already resolved at current tip. Bisect identified `251f1a157` as first-bad (100.7 t/s + hang), but `f68e17b9b` shows GOOD performance (108.3 t/s). No code changes needed.

2. **MTP/GPipe server**: Works correctly when `--device` flag is NOT used. The `--device ROCm0,RPC0` + MTP combination causes OOM because MTP context inherits the device list and attempts to allocate KV cache on the RPC device (3060 Ti, 8GB) instead of respecting the model's existing ROCm-heavy layer assignment.

**Artifacts**: `runs/agent-fix/SUMMARY.md` contains full results.

**Recommended action**: Document that `--device` is incompatible with MTP on multi-GPU. Use `-ts 2,98` without `--device` for dual-GPU MTP.

## Parent / forensic pack (required reading)

Read in this order before changing code:

| Doc | Why |
|-----|-----|
| [README.md](README.md) | Index of measurements in this pack |
| [GARBLE-VS-PERF-TIMELINE.md](GARBLE-VS-PERF-TIMELINE.md) | Separates **garble first-bad** from **dual TG drop** |
| [PRE-PLACEMENT-PIPELINE-STORY.md](PRE-PLACEMENT-PIPELINE-STORY.md) | Target util signature (7900 ~100%, 3060 ~40%+) |
| [KERNEL-GATE.md](KERNEL-GATE.md) | Ubuntu **7.0.0-28** ROCm regression; LKG **6.17.0-40** |
| [ROCM-MAX-AND-KERNEL.md](ROCM-MAX-AND-KERNEL.md) | Split hunt + rocprof kernel mix |
| [runs/bisect-util/SUMMARY.md](runs/bisect-util/SUMMARY.md) | Coarse dual TG bisect table |
| Related **FIXED** garble pack: [../pipeline-plus-dual-gpu-garble/README.md](../pipeline-plus-dual-gpu-garble/README.md) | Target PPLUS copy-slot fix `f68e17b9b` |
| Related **FIXED** RPC garble: [../rpc-weak-symbol-bug/](../rpc-weak-symbol-bug/) | Weak-symbol serialize `72c2aa3cd` |
| Process skill: diagnosing-bugs | Red loop first; no theory without a failing command |

**These issues are related but not the same commit:**

| Issue | First-bad / window | Fix status |
|-------|-------------------|------------|
| Target multi-GPU PPLUS **KV garble** | `87357519e` (2026-07-13) | FIXED `f68e17b9b` |
| MTP draft vs target sched race | same week | partial `470798451` (draft PP off) |
| RPC weak-symbol garble | fixed `72c2aa3cd` | FIXED |
| Dual TG **full-throttle loss** (~98 → ~85, dual &lt; 1-GPU) | after `863c10e39`, by `251f1a157`..`f68e17b9b` | **THIS TICKET** |
| Server **GPipe + draft-mtp** abort | open (profiler GPipe+MTP often OK) | **THIS TICKET** |

Do not re-open `f68e17b9b` as "the perf fix" without measuring: that commit **correctly** stops barrier rotation on graph reuse (D8 documents dual-buffer cost).

## Problem (two sentences)

1. **Perf:** On romulus dual-GPU (7900 XTX ROCm + 3060 Ti CUDA RPC), sustained TG with Plus=1 and ROCm-heavy split is ~**85 t/s** on current tip vs ~**98 t/s** at `19db22abb`/`863c10e39` and ~**104 t/s** single-GPU; dual no longer beats single-GPU and does not show the historical "7900 saturated + 3060 clearly busy" pipeline signature. Historical peer/D7 claims of **133-148** are unreproduced on this host/kernel/RPC stack; primary restore target is **dual full-throttle relative to 1-GPU and pre-drop tips**, not necessarily 148.
2. **MTP:** `llama-server` with `GGML_SCHED_GPIPE=1` + `--spec-type draft-mtp` has aborted (RPC remote crash / malformed response) during load or init; profiler GPipe+MTP often loads. Need a green dual-GPU path for draft-mtp (server and/or matched profiler) with coherent text and non-zero draft accept on long gen.

## Scope

**In scope**

- Git bisect / commit scan of dual TG regression window (see Phase 2)
- Root-cause and **fix** for dual-GPU TG regression (sched/RPC/sync/event path preferred over "turn Plus off")
- Debug and **fix** GPipe + draft-mtp failure on `llama-server` (and any remaining draft/target interaction under Plus after `f68e17b9b` / `470798451`)
- Fixed harness: kernel gate, no `--placement` unless testing placement, pinned ROCm 7.2.3, pinned `-ts`
- Util sampling (ROCm + NVIDIA, including docker RPC GPU if needed)
- Update this pack's SUMMARY / status when done; cross-link garble pack if fix touches copy-slots again
- Optional D8-safe path to restore rotation **only if** it is the measured dual cliff and correctness holds

**Out of scope**

- Re-litigating Plus=0 as the only solution for garble (already fixed)
- Shipping experimental WMMA vec_dot / D7.13 nwarps hacks
- Forcing Ubuntu kernel 7.0.0-28 "to match CI" (forbidden; see gate)
- Pure docs-only "148 was a different universe" without a dual>1-GPU restore attempt
- Placement control plane feature work (except proving plans are off, or a plan path is the regression)

## Non-goals

- Leaving dual permanently slower than 1-GPU as "expected"
- Enabling placement plans by default to "fix" TG
- Claiming success from short chat (2-11 tokens) only
- AI-generated PR spam to upstream llama.cpp (this is a private ops fork; follow AGENTS.md)

## Known facts (do not re-litigate)

| Fact | Source |
|------|--------|
| Running kernel for all recent benches: **6.17.0-40-generic** | `uname -r`; KERNEL-GATE.md |
| Bad kernel **7.0.0-28** installed; `/boot/vmlinuz` may default to it | dpkg / boot symlinks |
| Peer historical split: **TS 24,76**; ROCm-max dual: **TS 2,98** (RPC-first enum) | ROCM-MAX-AND-KERNEL.md |
| Dual TG ~**98** at `19db22abb` and `863c10e39` with `-ts 2,98` | runs/bisect-util/SUMMARY.md |
| Dual TG ~**90** at `251f1a157`, ~**85** at `f68e17b9b`/current | same |
| Current 1-GPU ~**104**; dual &lt; 1-GPU | same |
| LDS ON vs OFF ~flat on dual | same |
| Placement **not** applied in profiler unless `--placement PATH` | llama-gpipe-profiler.cpp |
| Garble first-bad for target Plus: **`87357519e`**; fix **`f68e17b9b`** | pipeline-plus-dual-gpu-garble |
| 1-GPU rocprof kernel mix fast tip ≈ current (matvec-dominated) | ROCM-MAX-AND-KERNEL.md |
| `5af38bf52` tip dual load **RPC crash** on retest | bisect-util logs |

## Environment (canonical)

**Node:** romulus (primary)

```text
Kernel:   MUST pass: bash scripts/gate-rocm-kernel.sh check
          LKG: 6.17.0-40-generic
          FORBIDDEN for ROCm: 7.0.0-28*
ROCm:     7.2.3 (/opt/rocm-7.2.3)
RPC:      docker pathb-rpc-romulus :50051 (CUDA 3060 Ti)
Client:   HIP build-hip or tip worktree builds
Model:    /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf
Split:    -ts 2,98   # ROCm-max under RPC-first device order; confirm layers via load log
          (peer historical was 24,76; use 2,98 for dual>1gpu restore)
Flags:    GGML_PIPELINE_PLUS=1
          no --placement unless intentionally testing
          -ctk q8_0 -ctv q8_0 -c 4096 -ngl 99 -sm layer
CMake:    -DGGML_HIP=ON (not HIPBLAS)
          -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++
          -DCMAKE_HIP_ARCHITECTURES=gfx1100
          -DGGML_HIP_ROCWMMA_FATTN=ON
          -DGGML_HIP_MMVQ_LDS_PROTOTYPE=OFF  # default for A/B unless testing LDS
          -DGGML_HIP_WMMA_VECDOT_EXPERIMENTAL=OFF
```

**Worktrees already built (may still be present):**

```text
/tmp/llama-tip-19db22abb
/tmp/llama-tip-863c10e39
/tmp/llama-tip-5af38bf52   # dual load may crash RPC
/tmp/llama-tip-614928615
/tmp/llama-tip-251f1a157
/tmp/llama-tip-f68e17b9b
repo build-hip / build-hip-lds-off
```

## Success criteria

### Perf (all required)

1. **Dual TG &gt; 1-GPU TG** on same tip, same model, same ctx, n_gen=128, multi-rep (or dual within ~3% of 1-GPU with clear dual benefit on PP / concurrent work). Preferred: dual **&gt;= 98 t/s** at `-ts 2,98` matching pre-drop tips, or better.
2. **Util signature** during TG: ROCm high (target ~80-100% mean busy), NVIDIA/RPC GPU **clearly non-zero** (historical memory ~40%+; measure via host `nvidia-smi` **and** `docker exec` into RPC container if host reads ~0).
3. Coherent generation (garble detector CLEAN) with Plus=1 multi-GPU.
4. Every result file records: `uname -r`, ROCm version, git sha, cmdline, placement off/on.

### MTP (all required)

1. `llama-server` loads with dual-GPU + Plus=1 + `--spec-type draft-mtp` (n_max=2) **without** RPC abort.
2. Long gen (n&gt;=64 or 128): draft accept **&gt;&gt; 6%** (broken-era); prefer &gt;= ~50% on easy prompts; no garble.
3. Optional: GPipe stages=3 + draft-mtp green on server **or** document why GPipe+MTP must stay profiler-only and ship non-GPipe dual MTP as production.

### Process

- Root cause named with file/function and failing vs fixed command.
- No upstream pure-AI PR; private fork ops OK. Follow AGENTS.md for commits if human asks to commit.

## Agent workflow

### Phase 0 - Gate and claim

1. Set this ticket `Status: in_progress`.
2. Run:

```bash
bash scripts/gate-rocm-kernel.sh check
# exit 2 = STOP. Eject to LKG 6.17.0-40 (eject-help). Do not bench on 7.0.0-28.
bash scripts/gate-rocm-kernel.sh status
# if OK_RUNNING_BUT_BAD_INSTALLED: warn human to pin GRUB before reboot; continue only on 6.17*
```

3. Confirm RPC: `nc -z 127.0.0.1 50051`. Prefer **matched** rpc-server binary/era when bisecting mid tips that crash remote.
4. Read forensic pack above. Do not "fix" by Plus=0 only.

### Phase 1 - Reproduce red loops

**Perf red (dual loses / below pre-drop):**

```bash
export GGML_PIPELINE_PLUS=1
export LD_LIBRARY_PATH=$PWD/build-hip/bin:/opt/rocm/lib
# 1-GPU
./build-hip/bin/llama-gpipe-profiler -m /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  -ngl 99 -sm layer --ctx-size 4096 -ctk q8_0 -ctv q8_0 --tasks tg -n 128 -r 5 --warmup \
  -o /tmp/agent-1gpu.json --out-dir /tmp/agent-1gpu
# Dual ROCm-max (RPC-first: small,large)
./build-hip/bin/llama-gpipe-profiler -m /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  -rpc 127.0.0.1:50051 -ngl 99 -sm layer -ts 2,98 \
  --ctx-size 4096 -ctk q8_0 -ctv q8_0 --tasks tg -n 128 -r 5 --warmup \
  -o /tmp/agent-dual.json --out-dir /tmp/agent-dual
# RED if dual tps < 1gpu tps (or dual << 98 on pre-drop tip rebuild)
```

**MTP red (server):**

```bash
# After gate check; matched client+rpc preferred
GGML_PIPELINE_PLUS=1 GGML_SCHED_GPIPE=1 \
./build-hip/bin/llama-server -m /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  --rpc 127.0.0.1:50051 -ngl 99 -sm layer -ts 2,98 --device ROCm0,RPC0 \
  -c 4096 -ctk q8_0 -ctv turbo3 -fa on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 \
  --host 127.0.0.1 --port 18090 --metrics
# RED: abort / RPC crash during load
# Also test GPIPE=0 + draft-mtp (may already be green) vs GPIPE=1
```

**Garble control (must stay green):** Plus=1 multi-GPU short + long prompts CLEAN via `rpc-patch/patch/loop-check-garble.sh` (use `reasoning_content` for thinking models).

### Phase 2 - Bisect dual TG (dense)

Coarse window already measured:

```text
GOOD dual ~98:  19db22abb , 863c10e39
FAIL load:      5af38bf52 (RPC crash) — skip or fix RPC match first
DROP:           by 251f1a157 (~90) then f68e17b9b/current (~85)
```

1. `git bisect` (or manual midpoints) on **profiler dual `-ts 2,98`** between:
   - good: `863c10e39` (or `19db22abb`)
   - bad: `251f1a157` first, then `251f1a157`..`f68e17b9b` if needed
2. Same cmake flags every tip; **LDS OFF**; no placement file.
3. If a tip crashes RPC: rebuild/restart **matched** rpc-server for that sha, or mark UNTESTABLE and bisect around it.
4. For each midpoint record TG + util + whether `placement plan active` appears (must not).
5. Name **first-bad commit** for dual TG drop with one-line mechanism hypothesis.

Scan commits in window for:

- `ggml-backend.cpp` (events, barrier, compute_splits, gpipe wait/record)
- `ggml-rpc.cpp` (serialize, flush, defer, weak symbols, GRAPH_COMPUTE*)
- `llama-context.cpp` (gpipe, MTP sched_reserve, pipeline_parallel)
- allreduce / placement only if default-on path is proven

### Phase 3 - Fix dual TG

1. Prefer minimal fix that restores dual &gt;= pre-drop without reintroducing garble.
2. If first-bad is `f68e17b9b`-class: implement **D8-style** safe rotation (update `node->src` on rotate) rather than reverting correctness.
3. If first-bad is RPC flush/serialize: restore overlap without stale tensor reads; add test or loop.
4. Re-run Phase 1 perf red until green.
5. Keep garble loop green.

### Phase 4 - Fix MTP (server GPipe + draft-mtp)

1. Minimal server cmdline matrix:

| GPipe | draft-mtp | Expected today |
|-------|-----------|----------------|
| 0 | 1 | often OK (verify) |
| 1 | 0 | ? |
| 1 | 1 | RED (abort) |

2. Capture RPC server logs + client stderr at abort.
3. Likely seams: dual `llama_context` (target + MTP), gpipe stage signal to RPC, draft sched `pipeline_parallel` off, matched protocol with rpc-server.
4. Fix so GPipe+MTP loads **or** document production config = dual Plus + MTP **without** GPipe if GPipe cannot be made safe; still must meet draft accept + no garble.
5. Long-gen accept rate + CLEAN text required.

### Phase 5 - Verify and close

1. Table: 1-GPU vs dual TG; util; MTP accept; garble CLEAN.
2. Update [runs/bisect-util/SUMMARY.md](runs/bisect-util/SUMMARY.md) or add `runs/agent-fix/SUMMARY.md`.
3. Update this file Status: resolved; link fix commits.
4. If copy-slot touched: add note on [../pipeline-plus-dual-gpu-garble/README.md](../pipeline-plus-dual-gpu-garble/README.md).
5. Do not `git push` / open upstream PR unless human explicitly accepts AGENTS.md ban risk.

## Commit / change discipline

- Prefer small commits with `Assisted-by:` if human asks you to commit (not Co-authored-by).
- ASCII only in messages (`-` not emdash).
- No pure-AI PR descriptions to public llama.cpp.

## Artifacts to leave

```text
diagnostics/d7-perf-regression-133-vs-current/
  ISSUE.md              # this ticket
  runs/agent-fix/      # heatmaps, util, server logs, SUMMARY.md
/tmp/agent-*            # scratch OK if referenced from SUMMARY
```

## Quick command index

```bash
# Kernel gate
bash scripts/gate-rocm-kernel.sh check
bash scripts/gate-rocm-kernel.sh eject-help

# Garble loop (related pack)
bash rpc-patch/patch/loop-check-garble.sh <port> 'What is the capital of France?' 64 42

# Prior measurements
ls /tmp/bisect-util-20260719172713
ls /tmp/rocm-max-split-* /tmp/d7-a2a-profiler-* /tmp/tip-hunt-*
```
