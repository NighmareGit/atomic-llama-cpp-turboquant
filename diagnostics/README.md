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
