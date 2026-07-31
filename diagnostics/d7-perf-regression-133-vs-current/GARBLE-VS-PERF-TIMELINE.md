# Garble vs dual-TG drop vs WMMA/placement (unified timeline)

## Dual TG bisect (kernel 6.17.0-40, `-ts 2,98`, no placement)

Artifacts: `/tmp/bisect-util-20260719172713`

| Tip | Role | Dual TG | Notes |
|-----|------|--------:|-------|
| `19db22abb` | pre-placement D7 | **98.5** | best dual in matrix |
| `863c10e39` | placement P0 | **98.2** | placement code in tree, plan not applied |
| `5af38bf52` | attn-local placement | **FAIL** | RPC remote crash on load |
| `614928615` | allreduce plumbing | **FAIL** | same class / incomplete |
| `251f1a157` | rpc serialize fix | **89.7** | first measured dual drop |
| `f68e17b9b` | PPLUS garble fix | **85.3** | correctness; rotation off |
| current LDS on/off | HEAD | **~85** | LDS not the cliff |
| current 1-GPU | HEAD | **103.9** | dual still loses to single |

## Garble first-bad (correctness)

| Commit | Date | Role |
|--------|------|------|
| **`87357519e`** | 2026-07-13 | **First-bad** target multi-GPU PPLUS garble (`prev_copy` / barrier rotation) |
| `470798451` | 2026-07-13 | MTP draft PP off (partial) |
| `72c2aa3cd` | 2026-07-19 | RPC weak-symbol garble (separate) |
| **`f68e17b9b`** | 2026-07-19 | Target PPLUS fix (no rotate on reuse) |

**Not first-bad:** placement (07-17+), allreduce (07-18), WMMA FA (07-16).

## WMMA / HIP foolery

| Date | Change | Garble root? | Perf? |
|------|--------|--------------|-------|
| 07-16 | FA rocWMMA ON | No | +TG when working |
| 07-16–17 | LDS / WMMA vec_dot prototypes | No | noise; LDS A/B flat |
| 07-16 | Wrong cmake `HIPBLAS` / hipcc path | Build confusion | silent CPU if HIP off |

## Kernel gate

- LKG: **6.17.0-40-generic**
- Bad: **7.0.0-28** (installed as default boot) — eject before reboot
- Script: `scripts/gate-rocm-kernel.sh`

## Causal split

1. **Correctness hell (garble):** 07-13 copy-slot path → fixed 07-19. Placement/AR did not invent it.
2. **Dual TG drop (~98 → ~85):** after P0, by rpc week (`251f1a157`) and garble fix (`f68e17b9b`); dual still **< 1-GPU** → pipeline not full throttle.
3. **Full-throttle util (7900@100% / 3060@40%):** not recovered; need better NV util sampling + bisect `863c10e39..251f1a157` denser (skip crashy 5af if client/RPC mismatch).
