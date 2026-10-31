#!/usr/bin/env bash
# Install Isaac Lab 3.0.0 with Isaac Sim 6.2.0.0 (PhysX backend) for the PhysX baseline of the benchmark:
#
#   PHYSX_BENCH_PREFIX=<dir> scripts/physx_bench/install.sh [--dry-run]
#
# PHYSX_BENCH_SKIP_GPU_CHECK=1 skips the final CUDA check, e.g. to install while the GPU is busy with other work;
# apart from that check, the install only reads the driver version (nvidia-smi) and never uses the GPU.
#
# Follows Isaac Lab's documented pip route (uv, Python 3.12, the isaacsim extra with the v3.0.0 uv-overrides file),
# with everything under $PHYSX_BENCH_PREFIX (see common.sh). Idempotent: uv, Python and the venv are created only when
# missing, and the package install is skipped when the stamp file for the same specification exists.
#
# Torch build: Isaac Lab 3.0 defaults to PyTorch cu130, which needs NVIDIA driver >= 580.65 (Isaac Lab's documented
# minimum). On an older driver the cu130 build cannot use CUDA, so the cu128 build of the same torch 2.11.0 is
# installed instead (PHYSX_BENCH_TORCH_BACKEND=cu130|cu128 overrides the automatic choice). Physics settings are not
# affected by this choice, but a driver below 580.65 is outside Isaac Lab's support range; run.sh records it.
#
# Size and time: 35-40 GB in total (venv ~29 GB, of which isaacsim ~17 GB; the largest wheel is
# isaacsim-extscache-kit, 5.1 GiB); 275 packages, ~20 min at ~15 MB/s. pypi.nvidia.com can be slow, hence
# UV_HTTP_TIMEOUT=900 (s) by default.
#
# The install does not accept the Isaac Sim EULA and does not import isaacsim (importing it triggers the EULA prompt).
set -euo pipefail
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

UV_VERSION=0.13.6
ISAACLAB_VERSION=3.0.0
OVERRIDES_URL="https://raw.githubusercontent.com/isaac-sim/IsaacLab/v${ISAACLAB_VERSION}/tools/wheel_builder/uv-overrides.txt"
START_S=$(date +%s)
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

# platform checks
[ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] || { echo "install.sh: Linux x86_64 only (Isaac Sim wheels)." >&2; exit 1; }
GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)
if [ -z "$GLIBC" ] || [ "$(printf '%s\n2.35\n' "$GLIBC" | sort -V | head -1)" != "2.35" ]; then
  echo "install.sh: Isaac Sim wheels need glibc >= 2.35 (found '${GLIBC:-unknown}'); use an Ubuntu 22.04/24.04 container." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "install.sh: nvidia-smi not found (NVIDIA driver required)." >&2; exit 1; }
DRV=$(physx_bench_driver_int)
DRV_STR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ -n "${PHYSX_BENCH_TORCH_BACKEND:-}" ]; then
  BACKEND=$PHYSX_BENCH_TORCH_BACKEND
elif [ "${DRV:-0}" -ge 580065 ]; then
  BACKEND=cu130
else
  BACKEND=cu128
  echo "install.sh: driver $DRV_STR < 580.65 (Isaac Lab 3.0 minimum): installing the cu128 torch build; results are feasibility-only." >&2
fi
echo "install.sh: prefix $PHYSX_BENCH_PREFIX, driver $DRV_STR, torch backend $BACKEND, glibc $GLIBC"

# uv (pinned) and Python 3.12
if [ ! -x "$PHYSX_BENCH_PREFIX/uv/bin/uv" ]; then
  if [ "$DRY" = 1 ]; then
    echo "+ curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | UV_INSTALL_DIR=$PHYSX_BENCH_PREFIX/uv/bin sh"
  else
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$PHYSX_BENCH_PREFIX/uv/bin" UV_NO_MODIFY_PATH=1 sh
  fi
fi
# uv-managed CPython 3.12 (not a system interpreter), so every host runs the same Python build
[ -d "$PHYSX_BENCH_VENV" ] || run uv venv --python 3.12 --python-preference only-managed --seed "$PHYSX_BENCH_VENV"
if [ "$DRY" = 0 ]; then
  # shellcheck disable=SC1091
  source "$PHYSX_BENCH_VENV/bin/activate"
fi

SPEC="isaaclab[isaacsim]==$ISAACLAB_VERSION torch-backend=$BACKEND"
STAMP="$PHYSX_BENCH_PREFIX/.install_stamp"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$SPEC" ]; then
  echo "install.sh: already installed ($SPEC); delete $STAMP to force a reinstall."
else
  run curl -fsSL -o "$PHYSX_BENCH_PREFIX/uv-overrides.txt" "$OVERRIDES_URL"
  run uv pip install "isaaclab[isaacsim]==$ISAACLAB_VERSION" \
    --overrides "$PHYSX_BENCH_PREFIX/uv-overrides.txt" \
    --extra-index-url https://pypi.nvidia.com \
    --torch-backend "$BACKEND" \
    --index-strategy unsafe-best-match
  [ "$DRY" = 1 ] || echo "$SPEC" > "$STAMP"
fi

if [ "$DRY" = 0 ]; then
  # sanity: Isaac Lab importable (no isaacsim import, so no EULA prompt) and, unless PHYSX_BENCH_SKIP_GPU_CHECK=1,
  # torch can use CUDA on one GPU only (PHYSX_BENCH_GPU / CUDA_VISIBLE_DEVICES; the others are not touched)
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${PHYSX_BENCH_GPU:-0}}" \
    PHYSX_BENCH_SKIP_GPU_CHECK="${PHYSX_BENCH_SKIP_GPU_CHECK:-0}" python -I - <<'EOF'
import importlib.metadata as md
import os
import torch
import warp as wp
import isaaclab  # noqa: F401
gpu = "not checked (PHYSX_BENCH_SKIP_GPU_CHECK=1)"
if os.environ["PHYSX_BENCH_SKIP_GPU_CHECK"] != "1":
    assert torch.cuda.is_available(), "torch cannot use CUDA (driver too old for this torch build?)"
    gpu = torch.cuda.get_device_name(0)
print("install.sh: OK  torch", torch.__version__, f"(CUDA {torch.version.cuda}) | warp", wp.__version__,
      "| isaaclab", md.version("isaaclab"), "| isaacsim", md.version("isaacsim"), "| GPU", gpu)
EOF
  echo "install.sh: size $(du -sh "$PHYSX_BENCH_PREFIX" 2>/dev/null | cut -f1), $(( ($(date +%s) - START_S) / 60 )) min"
  echo "install.sh: next, accept the Isaac Sim EULA yourself if you agree (see README), then run scripts/physx_bench/run.sh"
fi
