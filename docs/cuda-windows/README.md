# Windows CUDA cluster collateral (shared)

Path B / Path-B-Plus native Windows builds share one portable tree and one build script.

| Document | Scope |
|----------|-------|
| [BUILD.md](BUILD.md) | Canonical build (all Windows CUDA nodes) |
| [../cuda-windows-5070ti/README.md](../cuda-windows-5070ti/README.md) | 5070 Ti host, Config E/F/G |
| [../cuda-windows-triton/README.md](../cuda-windows-triton/README.md) | Triton 3090 RPC spike (`:50054`) |

| Script | Scope |
|--------|-------|
| [../../scripts/cuda-windows/build.ps1](../../scripts/cuda-windows/build.ps1) | Canonical build |
| [../../scripts/cuda-windows-5070ti/](../../scripts/cuda-windows-5070ti/) | 5070 Ti ops scripts |
| [../../scripts/cuda-windows-triton/](../../scripts/cuda-windows-triton/) | Triton ops scripts |

B+6 gate mission: [rpc-patch/docs/b6-gate/PLAN.md](../../rpc-patch/docs/b6-gate/PLAN.md)