# Bisect dual TG + util

- **Kernel:** 6.17.0-40-generic (LKG). Gate: `scripts/gate-rocm-kernel.sh`
- **Fixed cmdline:** `-ts 2,98`, dual RPC, no `--placement`, ctk/ctv q8_0, n=128 r=3
- **ROCm:** 7.2.3

| Tip | Role | TG t/s |
|-----|------|-------:|
| `19db22abb` | pre-placement D7 tip | 98.45 |
| `863c10e39` | placement P0 | 98.20 |
| `5af38bf52` | attn-local placement | FAIL |
| `614928615` | allreduce plumbing | FAIL |
| `251f1a157` | rpc serialize fix | 89.73 |
| `f68e17b9b` | PPLUS garble fix | 85.33 |
| `802ccb4c6-lds-on` | current LDS ON | 85.17 |
| `802ccb4c6-lds-off` | current LDS OFF | 84.68 |
| `802ccb4c6-1gpu` | current 1-GPU baseline | 103.87 |

## Verdict
1. **Placement P0 alone does not kill dual TG** (863c10e39 still ~98).
2. **Drop appears after mid placement / allreduce / rpc window** (by 251f1a157 ~90, then f68e17b9b/current ~85).
3. **LDS ON vs OFF** not the dual cliff (~same TG).
4. **Util sampling** did not recover 7900@100%/3060@40% signature (NV mean ~0-2%; improve with docker nvidia stats).
5. **Kernel:** running OK; **eject bad 7.0.0-28 from default boot** before any reboot.

Artifacts: `/tmp/bisect-util-20260719172713`
