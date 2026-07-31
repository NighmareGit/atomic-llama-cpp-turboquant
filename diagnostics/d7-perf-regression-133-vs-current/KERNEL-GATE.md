# Kernel gate: Ubuntu 7.0.0-28 AMDGPU / ROCm regression

**Last known good (romulus, 2026-07-19):** `6.17.0-40-generic`  
**Bad package:** `linux-image-7.0.0-28-generic` (`7.0.0-28.28~24.04.1`)  
**Script:** [`scripts/gate-rocm-kernel.sh`](../../scripts/gate-rocm-kernel.sh)

## What

Ubuntu HWE shipped kernel **7.0.0-28.28** with a severe **AMDGPU HMM** performance regression for ROCm compute (up to **~42×** slower). Fixed upstream in **Linux 7.0.13**. Canonical still shipped 7.0.0-28 for security and told ROCm users to **avoid / roll back**.

- Phoronix: https://www.phoronix.com/news/Ubuntu-7.0-AMDGPU-Regress  
- ROCm: https://github.com/ROCm/ROCm/issues/6358  

## Romulus inventory (2026-07-19)

| Item | Value |
|------|--------|
| Running | **6.17.0-40-generic** (OK) |
| Installed bad | **7.0.0-28-generic** (iF / partial install) |
| HWE meta | `linux-image-generic-hwe-24.04` → **7.0.0-28.28** |
| `/boot/vmlinuz` symlink | points at **7.0.0-28** (next default boot is unsafe) |
| `/boot/vmlinuz.old` | **6.17.0-40** (LKG) |

**Gate policy:** any ROCm / dual-GPU / llama-gpipe bench **must** run on LKG (or a confirmed fixed post-7.0.13 Ubuntu package). If `uname -r` matches `7.0.0-28*`, **eject** and reboot to LKG.

## Commands

```bash
# Before every ROCm session
bash scripts/gate-rocm-kernel.sh check   # exit 2 = refuse work

# Print status
bash scripts/gate-rocm-kernel.sh status

# Root eject / pin LKG (print-only; needs sudo)
bash scripts/gate-rocm-kernel.sh eject-help
```

## Eject to last known good (root)

```bash
# After reboot into 6.17.0-40 if needed:
sudo grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.17.0-40-generic"
sudo update-grub
sudo apt-mark hold linux-image-7.0.0-28-generic linux-headers-7.0.0-28-generic 2>/dev/null || true
# optional until fixed HWE exists:
# sudo apt-mark hold linux-image-generic-hwe-24.04
uname -r   # must be 6.17.0-40-generic
```

## Bench integration

Every new result under `diagnostics/d7-perf-regression-133-vs-current/` must record:

```
kernel=$(uname -r)
rocm=$(cat /opt/rocm/.info/version 2>/dev/null)
gate=$(bash scripts/gate-rocm-kernel.sh status)
```
