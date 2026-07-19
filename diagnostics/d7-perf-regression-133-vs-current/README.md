# Dual-GPU TG regression + MTP/GPipe (forensics + agent ticket)

**Status:** RESOLVED (2026-07-19)

## Resolution Summary

1. **Dual TG regression**: Already resolved. Current tip shows dual ~108 t/s > 1-GPU ~104 t/s. See `runs/agent-fix/SUMMARY.md`.

2. **MTP/GPipe server**: Works correctly without `--device` flag. The `--device` flag is incompatible with MTP on multi-GPU (OOM on RPC device). Use `-ts 2,98` for dual-GPU MTP.  
**Node:** romulus (7900 XTX + 3060 Ti RPC)  
**Related FIXED packs:** [pipeline-plus-dual-gpu-garble](../pipeline-plus-dual-gpu-garble/), [rpc-weak-symbol-bug](../rpc-weak-symbol-bug/)

## Agent entry point

**[ISSUE.md](ISSUE.md)** — full ticket: bisect dual TG drop, scan commits, fix regression, debug/fix server GPipe+MTP.  
Includes kernel gate, success criteria, and red/green commands.

## Pack index

| File | Content |
|------|---------|
| [ISSUE.md](ISSUE.md) | **Agent ticket** (start here) |
| [GARBLE-VS-PERF-TIMELINE.md](GARBLE-VS-PERF-TIMELINE.md) | Garble first-bad vs dual TG drop vs WMMA/placement |
| [PRE-PLACEMENT-PIPELINE-STORY.md](PRE-PLACEMENT-PIPELINE-STORY.md) | Full-throttle util memory (7900/3060) |
| [KERNEL-GATE.md](KERNEL-GATE.md) | Ubuntu 7.0.0-28 ROCm regression; LKG 6.17.0-40 |
| [ROCM-MAX-AND-KERNEL.md](ROCM-MAX-AND-KERNEL.md) | Split hunt + rocprof |
| [TIP-HUNT.md](TIP-HUNT.md) | Tip worktrees / live tip hunt |
| [ASSESSMENT.md](ASSESSMENT.md) | Early assessment vs historical 133-148 |
| [PLAN-VALIDATE-133-TIP.md](PLAN-VALIDATE-133-TIP.md) | Validate historical tip plan |
| [runs/bisect-util/SUMMARY.md](runs/bisect-util/SUMMARY.md) | Coarse bisect dual TG |
| [runs/tip-hunt/](runs/tip-hunt/) | Peer vs D7 mode tip matrix |
| [runs/rocm-max-split/](runs/rocm-max-split/) | ROCm-max split results |
| [runs/kernel-prof/](runs/kernel-prof/) | rocprofv3 notes |

## Kernel gate (mandatory)

```bash
bash scripts/gate-rocm-kernel.sh check
```

Do not ROCm-bench on **7.0.0-28\***. Last known good: **6.17.0-40-generic**.

## One-line summary

Correctness garble (Plus multi-GPU copy-slot) is **fixed** (`f68e17b9b`); dual-GPU **throughput and pipeline overlap** and **server GPipe+MTP** remain **open** — see ISSUE.md.
