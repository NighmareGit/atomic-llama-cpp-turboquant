#!/usr/bin/env bash
# Gate ROCm / llama benches against the Ubuntu AMDGPU compute regression kernel.
#
# Bad package (do not use for ROCm):
#   linux-image-7.0.0-28-generic  (7.0.0-28.28~24.04.1)
#   Ubuntu HWE meta currently points here; up to ~42x ROCm slowdown (HMM retry).
#   Fixed upstream in Linux 7.0.13; Ubuntu follow-up package not yet assumed present.
#
# Last known good on romulus (2026-07-19):
#   6.17.0-40-generic
#
# Usage:
#   source scripts/gate-rocm-kernel.sh          # exports helpers; runs check
#   bash scripts/gate-rocm-kernel.sh check      # exit 0 ok, 2 bad kernel
#   bash scripts/gate-rocm-kernel.sh status     # print status only
#   bash scripts/gate-rocm-kernel.sh eject-help # print root commands to pin LKG
#
# Env:
#   ROCM_KERNEL_LKG   default 6.17.0-40-generic
#   ROCM_KERNEL_BAD_REGEX  default '7\.0\.0-28'

set -euo pipefail

ROCM_KERNEL_LKG="${ROCM_KERNEL_LKG:-6.17.0-40-generic}"
ROCM_KERNEL_BAD_REGEX="${ROCM_KERNEL_BAD_REGEX:-7\\.0\\.0-28}"

rocm_kernel_status() {
  local k
  k="$(uname -r)"
  echo "running_kernel=$k"
  echo "lkg=$ROCM_KERNEL_LKG"
  if [[ "$k" =~ $ROCM_KERNEL_BAD_REGEX ]]; then
    echo "status=BAD_KERNEL"
    echo "reason=Ubuntu/AMDGPU ROCm performance regression (HMM); up to ~42x slower compute"
    echo "action=eject: reboot into $ROCM_KERNEL_LKG (or any non-matching fixed kernel)"
    return 2
  fi
  if [[ -e /boot/vmlinuz-7.0.0-28-generic ]] || [[ -L /boot/vmlinuz && "$(readlink -f /boot/vmlinuz 2>/dev/null || true)" == *7.0.0-28* ]]; then
    echo "status=OK_RUNNING_BUT_BAD_INSTALLED"
    echo "warn=/boot default vmlinuz may point at 7.0.0-28; next reboot is unsafe for ROCm"
    echo "action=pin GRUB to $ROCM_KERNEL_LKG; hold/remove 7.0.0-28 package"
    return 1
  fi
  echo "status=OK"
  return 0
}

rocm_kernel_require_good() {
  local st
  set +e
  rocm_kernel_status
  st=$?
  set -e
  if [[ $st -eq 2 ]]; then
    echo "FATAL: refusing ROCm/llama work on bad kernel $(uname -r)" >&2
    echo "Reboot to last known good: $ROCM_KERNEL_LKG" >&2
    echo "See: scripts/gate-rocm-kernel.sh eject-help" >&2
    return 2
  fi
  if [[ $st -eq 1 ]]; then
    echo "WARN: running kernel OK but bad 7.0.0-28 is installed as boot default" >&2
  fi
  return 0
}

rocm_kernel_eject_help() {
  cat <<EOF
# Eject bad Ubuntu 7.0.0-28 kernel / pin last known good (needs root)
#
# Last known good: $ROCM_KERNEL_LKG
# Bad:             7.0.0-28-generic (package 7.0.0-28.28~24.04.1)
#
# 1) Confirm you are NOT on the bad kernel:
uname -r
# If it matches 7.0.0-28*, reboot into advanced GRUB -> $ROCM_KERNEL_LKG first.
#
# 2) Pin GRUB default to LKG (Ubuntu):
sudo grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux ${ROCM_KERNEL_LKG}"
# or edit /etc/default/grub:
#   GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux ${ROCM_KERNEL_LKG}"
sudo update-grub
#
# 3) Hold / remove the bad image so HWE does not keep preferring it:
sudo apt-mark hold linux-image-7.0.0-28-generic linux-headers-7.0.0-28-generic 2>/dev/null || true
# Prefer hold over purge until a fixed 7.0.x package exists; then:
#   sudo apt-mark unhold ... && sudo apt install linux-image-generic-hwe-24.04
#
# 4) Optional: hold HWE meta if it keeps pulling 7.0.0-28:
#   sudo apt-mark hold linux-image-generic-hwe-24.04
#
# 5) Reboot and verify:
#   uname -r   # expect ${ROCM_KERNEL_LKG}
#   bash scripts/gate-rocm-kernel.sh check
#
# References:
#   https://www.phoronix.com/news/Ubuntu-7.0-AMDGPU-Regress
#   https://github.com/ROCm/ROCm/issues/6358
#   diagnostics/d7-perf-regression-133-vs-current/KERNEL-GATE.md
EOF
}

cmd="${1:-check}"
case "$cmd" in
  check)
    rocm_kernel_require_good
    ;;
  status)
    rocm_kernel_status || true
    ;;
  eject-help)
    rocm_kernel_eject_help
    ;;
  *)
    echo "usage: $0 check|status|eject-help" >&2
    exit 1
    ;;
esac
