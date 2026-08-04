#!/usr/bin/env bash
# Bootstrap Heretic on a Kaggle TPU v5e-8 VM.
# Idempotent: safe to run multiple times.
#
# Usage:  bash tpu/bootstrap.sh [repo_dir]
#   repo_dir  - where to clone the repo (default: ~/heretic)
#
# Rules:
#  - NEVER reinstall torch / torchvision / torch_xla (VM image has a matched
#    XLA build; PyPI torch would silently break TPU access).
#  - Use whatever Python 3.10+ interpreter is available.
set -euo pipefail

REPO_DIR="${1:-$HOME/heretic}"
REPO_URL="${HERETIC_REPO_URL:-https://github.com/instax-dutta/heretic.git}"

say() { printf '\n=== %s ===\n' "$*"; }

say "System"
nproc || true
python3 --version
python3 - <<'PY' || true
import shutil, psutil, os
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB total, "
      f"{psutil.virtual_memory().available / 1e9:.1f} GB available")
print(f"Disk: {shutil.disk_usage('/').free / 1e9:.1f} GB free")
print(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE', 'NOT SET')}")
PY

say "TPU detection (before any install)"
python3 - <<'PY' || true
try:
    import torch, torch_xla
    import torch_xla.core.xla_model as xm
    print(f"PyTorch:   {torch.__version__}")
    print(f"torch_xla: {torch_xla.__version__}")
    device = xm.xla_device()
    print(f"Device:    {device}")
    print(f"HW:        {xm.xla_device_hw(device)}")
    print(f"Cores:     {xm.xla_device_count()}")
except Exception as e:
    print(f"torch_xla check failed: {e}")
    print("Continuing - may need the Kaggle TPU runtime enabled.")
PY

say "Clone / update repo"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout master
    git -C "$REPO_DIR" pull --ff-only origin master
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

say "Install package (torch/torchvision/torch_xla untouched)"
cd "$REPO_DIR"
# torch is deliberately unversioned in pyproject.toml, so pip sees the VM's
# preinstalled build as satisfying the requirement and leaves it alone.
pip install -e ".[tpu]"

say "Verify heretic + TPU integration"
python3 - <<'PY'
from heretic.system import (
    detect_tpu, get_xla_device_count, get_accelerator_info, setup_tpu_environment,
)
setup_tpu_environment()
assert detect_tpu(), "detect_tpu() returned False - TPU not usable"
print(f"detect_tpu()        : {detect_tpu()}")
print(f"XLA device count    : {get_xla_device_count()}")
print(get_accelerator_info())
print("BOOTSTRAP OK - TPU detected, heretic installed")
PY
