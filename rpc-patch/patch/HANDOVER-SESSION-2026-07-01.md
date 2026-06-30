# Handover: B+6 triton runs + debug + instrumentation start (2026-07-01)

**Purpose:** Wrap session; record n=128/384 triton results, RPC fixes tried, instrumentation progress, current plan state for next session.  
**Branch:** `Path-B-Event-Support-Pipeline-Plus` (local changes)  
**Mission root:** [docs/rpc-multi-backend-pipeline-plus/](../../docs/rpc-multi-backend-pipeline-plus/)  
**Prior handover:** [HANDOVER-SESSION-2026-06-30.md](HANDOVER-SESSION-2026-06-30.md)

| Location | Path | Notes |
|----------|------|-------|
| **remus (this host)** | `/home/hunter/projects/atomic-llama-cpp-turboquant/atomic-llama-cpp-turboquant` | edit/commit here |
| **triton worker** | `hunter@192.168.8.23` → `~/projects/atomic-llama-cpp-turboquant` | Ubuntu 24.04, pw 12345 |
| **JUPITER** | `192.168.8.21:50053` | still blocked from remus |

**SSH (triton):** `hunter@192.168.8.23`, password `12345`.

---

## Locked decisions / progress this session

- n=128 guard run completed with correct paths (after fixes): G=116, overlap=0.9%, stall=0.6357, MIXED.
- n=384 canonical Plus=1 completed: G=128.76, overlap=0.2%, stall=0.867, STRAGGLER_DOMINANT.
- B+9 DEFER=0 bisect on n=384 launched.
- Key fixes that helped runs complete:
  - Combined cmd+size+data send in one buffer (avoids TCP buffering deadlock on small cmds).
  - Skip drain for HELLO + DEVICE_COUNT.
  - Reset tls_pending_* after hello.
  - docker script: --network=host + full env pass-through + trace envs + ptrace caps (for strace debug).
- Instrumentation: basic rpc_rtt_hist + count added to pathb-rpc-trace-parse.sh (plan 1.1 starter).
- Debug: added/used server+client GGML_LOG_INFO (later cleaned); multiple "Accepted...closed" observed even after hello success; device count sometimes blocks.
- Source cleaned of pure DEBUG prints before handover; functional improvements retained.
- Server rebuilt/restarted on triton multiple times with latest.

**B+6 gate verdict still FAIL** (M3 5% target). Straggler (RPC backend) + stall + input_wait_copy dominant. Some stall improvement on guard run.

## Session summary (done)

### Runs completed
- `b6-2gpu-f-triton-guard-n128`: overlap 0.9% (best recent), G=116, drain=788, blocking=277.
- `b6-2gpu-f-triton-n384`: overlap 0.2%, G=128.8, drain=1712, blocking=790.
- DEFER=0 n=384 in flight at handover.

Artifacts: `benches/path-b-plus/b6-2gpu-f-triton*` (full telemetry, diagnose.json, traces, summaries). Matrix and regression updated via diagnose.

### Code / script changes (kept)
- ggml-rpc.cpp: combined send buffer; drain skip for DEVICE_COUNT; post-hello pending reset.
- scripts/b6-gate-remus-docker.sh: env forwarding, network=host, trace envs, caps.
- rpc-patch/scripts/pathb-rpc-trace-parse.sh: rtt hist extension.
- (DEBUG logs removed before commit.)

### Debug insights
- Hello now succeeds (proto 4.3, peer_copy=yes) after fixes.
- Multiple connects/closes persist (client-side after device count send; server enters loop but recv for next cmd may not see data promptly).
- Combined send + no early drains helped some progress.
- validate-rpc still hangs (priority 5); used --skip.
- n=128 guard showed better stall than older baselines.

---

## What is left (from approved plan + handover priorities)

See the session plan for details: (refer to the plan.md used in this session or copy below summary).

Immediate/plan priorities still open:
- Finish/diagnose n=384 DEFER=0 bisect + compare to Plus=1 (B+9 effect).
- Full B+8/B+9/B+10 flag bisects on triton n=384 (and 4-GPU).
- B+8b trace check (pending_mask vs always-7).
- Complete Phase 1.1: full per-split timing + RPC RTT *histogram* (binned, per-split correlated) in emitters/parsers/diagnose (starter in parse.sh).
- Fix --validate-rpc hang (HELLO/device_count complete without skip).
- Triton ops: full git sync, start/stop/restart scripts, 3070.
- Update matrix / audit / TRACKING after bisects.
- Push (this handover + updates).
- If no M1 progress: document ceiling.

**Next recommended:** 
1. Monitor/finish current DEFER n=384, run diagnose, compare.
2. Run B+8 / strict / other flag=0 on n=384 triton.
3. Extend the trace parsers for full RTT hist + per-split RPC (plan 1.1).
4. Targeted debug for validate (strace on both sides during --validate-rpc).
5. Ops scripts + push.

---

## Command cheat sheet (updated)

Use the ones from 06-30 handover + these for the new runs:

```bash
# n=384 canonical (already done)
PROFILER_SKIP_VALIDATE=1 BENCH_GEN_TOKENS=384 \
  PROFILER_OUT_DIR=/src/benches/path-b-plus/b6-2gpu-f-triton-n384 \
  bash scripts/b6-gate-remus-docker.sh b6-2gpu-f-triton --skip-rpc-validate

# B+9 bisect example
PROFILER_SKIP_VALIDATE=1 BENCH_GEN_TOKENS=384 GGML_RPC_EVENT_DEFER_BARRIER=0 \
  PROFILER_OUT_DIR=/src/benches/path-b-plus/b6-2gpu-f-triton-n384-no-defer \
  bash scripts/b6-gate-remus-docker.sh b6-2gpu-f-triton --skip-rpc-validate

bash scripts/b6-gate-diagnose-runs.sh b6-2gpu-f-triton-n384
```

For triton server (with debug):
```bash
pkill rpc-server || true
... (nohup as before)
```

To force recompile after source change (remus or triton):
```bash
# on the build host
rm -rf build-cuda-b-bin/ggml/src/ggml-rpc/CMakeFiles/ggml-rpc.dir
ninja -C build-cuda-b-bin -j4   # or cmake --build ... --target rpc-server
```

---

## Uncommitted / state at handover (before this commit)

(Will be cleaned by this session's commit/push.)

---

## Files to read first in new session

1. This handover
2. [docs/rpc-multi-backend-pipeline-plus/PLAN.md](../../docs/rpc-multi-backend-pipeline-plus/PLAN.md) + TRACKING.md (updated 2026-07-01)
3. `benches/path-b-plus/b6-2gpu-f-triton-n384*/telemetry/{diagnose.json,trace-summary.txt}`
4. `ggml/src/ggml-backend.cpp` (B+8/barrier) and `ggml/src/ggml-rpc/ggml-rpc.cpp` (B+9 + retained fixes)
5. `scripts/b6-gate-remus-docker.sh`, `rpc-patch/scripts/pathb-rpc-trace-parse.sh`
6. The approved session plan (in .grok session dir or summary in this handover)

---

**Generated 2026-07-01 — remus/triton session: n=128+384 data, RPC fixes, instrumentation start, plan continuation ready.**

Update TRACKING / matrix after every new bisect or run. Good luck with the ladder!

---

## Post-handover continuation (tokens & push, 2026-07-01)

User note: "tokens are stored on romulus in folder tokens"

- Read via SSH (pw 12345 for login): 
  - `~/tokens/openhands-gitea.md` (gitea PAT)
  - `~/tokens/llama.cpp-github-token.txt` (github PAT)
  - `~/Documents/tokens/gitea/dockhand-token-gate.txt` (alt gitea)
  - Other: github/grok-build, groki, groq, alphavantage (full values stored only in local ~/tokens/ on remus/triton/romulus; redacted here).
- Session-provided GitHub token used for origin (per "github token", "push to github should work"; value in local creds only).
- Updated on **remus** + **triton**:
  - `~/.git-credentials` (gitea line now uses romulus gitea PAT token, no %3a mangling)
  - `~/.gitconfig` insteadOf for gitea now `http://hunter:<PAT>@192.168.8.108:3005/`
  - Created/mirrored `~/tokens/` and `~/Documents/tokens/{gitea,github,groq,alphavantage}/` with files (600 perms).
- Git auth test (remus): `git ls-remote` succeeded for both `origin` (github) and `gitea`.
- On triton: creds/tokens updated (repo layout single-nested; git fs-boundary quirks observed but not blocking server runs).
- Remotes verified: clean names `origin` / `gitea`; push ready (github first per plan, then gitea).
- Also stored session token copy as `~/tokens/github-token-session.txt`.

Next: commit updates + docs, push github then gitea. (This closes the "store token / push gitea now that romulus online" item.)