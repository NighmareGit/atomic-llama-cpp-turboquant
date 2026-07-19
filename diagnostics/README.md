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

## Agent tickets

| Ticket | Title | Status |
|--------|-------|--------|
| [pipeline-plus-dual-gpu-garble/ISSUE.md](pipeline-plus-dual-gpu-garble/ISSUE.md) | Fix multi-GPU target-context KV garble under GGML_PIPELINE_PLUS (debug+fix loop) | resolved |
