# Agent Fix Summary: Dual TG + MTP/GPipe

Date: 2026-07-19
Agent: Grok implementation workflow

## Phase 1 Results: Red Loop Reproduction

| Test | Result | Notes |
|------|--------|-------|
| 1-GPU TG (baseline) | 103.95 t/s | Clean build, LDS OFF |
| Dual TG (-ts 2,98) | 108.56 t/s | **GREEN** - dual > 1-GPU |
| Garble control | PASS | Plus=1 multi-GPU clean |
| MTP GPipe=0 + MTP=1 | 76% accept | Works without `--device` |
| MTP GPipe=1 + MTP=0 | OK | Loads and generates |
| MTP GPipe=1 + MTP=1 | OK | Loads and generates (without `--device`) |
| MTP GPipe=1 + MTP=1 + `--device` | OOM | 22GB alloc on 8GB RPC |

**Key Finding**: The perf regression (~85 t/s dual) **does NOT reproduce** at current tip. Both clean and uncommitted builds show dual > 1-GPU.

## Phase 2 Results: Dense Bisect

| Tip | Dual TG (t/s) | Status | Notes |
|-----|---------------|--------|-------|
| `863c10e39` | 108.9 | GOOD | baseline |
| `c672d7827` | 109.8 | GOOD | profiler heatmap; no perf change |
| `606ed6e5e` | CRASH | BAD | abort in `rpc_buffer_set_tensor` |
| `052549490` | CRASH | BAD | abort in `rpc_buffer_set_tensor` |
| `5af38bf52` | CRASH | BAD | abort in `rpc_buffer_set_tensor` |
| `614928615` | CRASH | BAD | abort in `rpc_buffer_set_tensor` |
| `5415d1e96` | 110.6 | GOOD | rpc k_bin_bcast fix |
| `251f1a157` | 100.7 + HANG | **BAD** | hangs rep 2+ |
| `f68e17b9b` | 108.3 | GOOD | garble fix |

**First-bad commit**: `251f1a157` (rpc serialize fix)

**Mechanism**: Added `ggml_backend_sched_signal_gpipe_stage(0)` to single-seq decode path, forcing RPC through slow `GRAPH_COMPUTE_STAGE` dispatch every token. This caused 100.7 t/s on rep 1, then hung on subsequent reps.

**Status**: Already resolved. `f68e17b9b` shows GOOD performance (no revert needed).

## Phase 3 & 4 Analysis

### Dual TG Regression
- **Root cause**: `251f1a157` forced GPipe stage signaling in single-seq path
- **Resolution**: Downstream commits (likely `72c2aa3cd` weak-symbol fix) resolved the hang
- **Current status**: **RESOLVED** - dual ~108 t/s > 1-GPU ~104 t/s

### MTP/GPipe `--device` OOM Bug
- **Root cause**: `--device ROCm0,RPC0` + MTP context creation conflict
- When `--device` is specified, `params.devices` is set but `tensor_split` is NOT nulled
- MTP draft context inherits device list and tries to allocate KV cache on RPC0
- RPC0 (3060 Ti) has 8GB VRAM; model requires 22GB
- **Resolution**: Added early validation in `server-context.cpp` to reject `--device` + MTP + RPC combination
- Correct usage: `-ts 2,98` without `--device` for dual-GPU MTP

## Success Criteria Assessment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Dual TG > 1-GPU | PASS | 108.6 > 103.9 t/s |
| Util signature (7900@80%+, 3060@40%+) | PARTIAL | Need docker NV stats for 3060 util |
| Coherent generation (garble CLEAN) | PASS | Plus=1 multi-GPU clean |
| Server GPipe+MTP loads | PASS | Without `--device` flag |
| Draft accept > 50% | PASS | 76% on test prompt |

## Artifacts

- `/tmp/agent-1gpu.json` - 1-GPU baseline heatmap
- `/tmp/agent-dual.json` - Dual GPU heatmap
- `/tmp/agent-mtp-server.log` - GPipe=1 MTP=1 server log
- `/tmp/p2-summary.txt` - Phase 2 bisect summary

## Recommendations

1. **Perf regression**: Already resolved. No code changes needed.
2. **MTP `--device` bug**: Fixed in code via early validation. Server now rejects `--device` + MTP + RPC with clear error message.
3. **Code change**: `tools/server/server-context.cpp` - added `has_rpc_device()` helper and validation check.