# Plan: validate historical 133+ tip, then dual profiler

**Status:** in progress  
**Agreed approach:** user + agent (2026-07-19)

## Why this is a good idea

1. **Reproduce first** - if tip X no longer hits ~133 on this box, the "regression" may be env/RPC/hardware, not code.
2. **Quality gate** - 133 with garbled MTP is not a success target; measure draft accept + sample text.
3. **Same harness** - `llama-gpipe-profiler` + same ctk/ctv **q8_0** (already used on slow run).
4. **Kernel + sched** - rocprofv3 / sched-trace on **both** tips to smell network hop / sync stalls.
5. **Then bisect** - only after both ends of the range are measured.

## Tip selection

| Role | SHA | Why |
|------|-----|-----|
| **Fast tip (candidate)** | `b145d6fce` | 2026-07-16: post D6.10.1 + FA cmake era; D7.1 table measured same day |
| Alt FA baseline | `f2c0e1192` | first `GGML_HIP_ROCWMMA_FATTN=ON` |
| Slow tip (current) | `802ccb4c6` / worktree HEAD binary | already measured ~74.5 profiler |

## Isolation

- `git worktree add /tmp/llama-d7-133-ref b145d6fce`
- Build under worktree `build-hip/` (do not clobber main tree build)
- Prefer **matched** rpc-server from same tip if protocol issues; else try current docker RPC first

## Validation matrix (per tip)

| Step | Command class | Pass criteria |
|------|---------------|---------------|
| A | profiler D7 shape (GPipe3, MTP n_max=2, n=128, r=5, ctk/ctv q8_0) | TG band vs 131-133 / quality |
| B | short sample / draft accept (profiler --sample or server smoke) | not garble; draft accept >> 6% |
| C | rocprofv3 kernel-trace (n_gen smaller, e.g. 32-64) | compare gap/wait vs compute |
| D | optional GGML_SCHED_TRACE=1 decode | hop/sync stalls |

## After both tips

- If fast tip still ~133 + clean: **bisect** fast..slow on profiler TG.
- If fast tip ~75: look at RPC image, docker, drivers, not only git.
- If fast tip ~133 but garbled: document; success metric becomes clean+speed.
