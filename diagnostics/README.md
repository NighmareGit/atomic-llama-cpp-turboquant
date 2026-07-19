# Bug Diagnostics

This directory collects forensic records of hard bugs — the full investigation
chain from symptom to root cause to fix to verification. Each subdirectory
covers one issue.

Keeping these in-tree means future maintainers can discover related issues
without re-deriving hours of debugging work.

## Index

| Directory | Bug | Status | Fixed in |
|-----------|-----|--------|----------|
| [rpc-weak-symbol-bug/](rpc-weak-symbol-bug/) | RPC output garbling from weak symbol resolution in shared libraries | FIXED | `72c2aa3cd` |
| [pipeline-plus-dual-gpu-garble/](pipeline-plus-dual-gpu-garble/) | KV-cache garbling on multi-GPU with GGML_PIPELINE_PLUS=1 (target + draft contexts) | FIXED | `f68e17b9b` |
| [d7-perf-regression-133-vs-current/](d7-perf-regression-133-vs-current/) | Dual-GPU TG full-throttle loss (~98 to ~85; dual &lt; 1-GPU) + server GPipe/MTP | OPEN | - |

## Agent tickets

| Ticket | Title | Status |
|--------|-------|--------|
| [pipeline-plus-dual-gpu-garble/ISSUE.md](pipeline-plus-dual-gpu-garble/ISSUE.md) | Fix multi-GPU target-context KV garble under GGML_PIPELINE_PLUS (debug+fix loop) | resolved |
| [d7-perf-regression-133-vs-current/ISSUE.md](d7-perf-regression-133-vs-current/ISSUE.md) | Bisect/fix dual-GPU TG regression + debug/fix GPipe/server MTP (related to garble pack) | open |

## Kernel gate (ROCm)

Before any ROCm dual-GPU work: `bash scripts/gate-rocm-kernel.sh check`  
See [d7-perf-regression-133-vs-current/KERNEL-GATE.md](d7-perf-regression-133-vs-current/KERNEL-GATE.md).
