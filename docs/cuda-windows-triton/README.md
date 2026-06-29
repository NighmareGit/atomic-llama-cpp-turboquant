# Windows CUDA -- Triton (3090 + 3070)

**Host:** triton @ `192.168.8.23`  
**Role:** B+6 spike RPC worker (3090 on `:50054`) -- remus 5060 replacement for A/B profiling.  
**Mission:** [rpc-patch/docs/b6-gate/PLAN.md](../../rpc-patch/docs/b6-gate/PLAN.md)

## Hardware

| GPU | VRAM | Role in spike |
|-----|------|---------------|
| RTX 3090 | 24 GB | RPC worker `CUDA0`, port `:50054` |
| RTX 3070 | 8 GB | **Parked** (insufficient for 35B MoE RPC share) |

## Build

Shared script: [docs/cuda-windows/BUILD.md](../cuda-windows/BUILD.md)

```powershell
cd C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus
.\scripts\cuda-windows\build.ps1 -Profile triton
```

## RPC worker (spike)

```powershell
.\scripts\cuda-windows-triton\pathb-rpc-server.ps1
.\scripts\cuda-windows-triton\pathb-rpc-server.ps1 -Restart
.\scripts\cuda-windows-triton\pathb-rpc-server.ps1 -Stop
```

Verify from romulus:

```bash
nc -zv 192.168.8.23 50054
```

## Profiler A/B (romulus client, same machine)

| Label | `--rpc` | `-ts` |
|-------|---------|-------|
| `b6-2gpu-f` | `192.168.8.176:50051` | `50,50` |
| `b6-2gpu-f-triton` | `192.168.8.23:50054` | `50,50` |

## Local tree

Primary clone on triton: `C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus`

## Remote ops (from JUPITER / worktree)

Triton is Windows. Use **cmd + git** on triton, not nested PowerShell/bash over SSH (quoting breaks easily).

On triton (local cmd):

```cmd
cd /d C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus
scripts\b6-gate-triton-git-sync.cmd
```

From orchestrator (JUPITER):

```powershell
.\scripts\b6-gate-triton-remote.ps1 sync
.\scripts\b6-gate-triton-remote.ps1 status
.\scripts\b6-gate-triton-remote.ps1 rpc
.\scripts\b6-gate-triton-remote.ps1 audit
```

Portable deploy (no VS on triton): build on JUPITER, copy to `\\192.168.8.23\C$\backup\pathb-portable`, then `b6-gate-triton-remote.ps1 rpc`.

## Related

- [cuda-windows-5070ti](../cuda-windows-5070ti/README.md) -- 5070 Ti `:50053`
- [b6-gate/TRACKING.md](../../rpc-patch/docs/b6-gate/TRACKING.md) -- step checklist