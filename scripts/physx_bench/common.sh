# Shared environment for scripts/physx_bench (sourced by install.sh and run.sh; not executed directly).
#
# Everything the Isaac Sim / Isaac Lab stack writes goes under $PHYSX_BENCH_PREFIX: the uv binary and its Python, the
# venv, the uv/pip/torch/warp caches, and a private HOME / XDG / TMPDIR (Kit writes to ~/.nvidia-omniverse,
# ~/.cache/ov and ~/.local/share/ov; Isaac Lab caches downloaded USD assets under TMPDIR). The caller's HOME is kept
# in PHYSX_BENCH_CALLER_HOME.
#
# Nothing here sets OMNI_KIT_ACCEPT_EULA or writes Isaac Sim's EULA marker; physx_bench_require_eula only checks that
# the operator has accepted the license themselves.

if [ -z "${PHYSX_BENCH_PREFIX:-}" ]; then
  echo "physx_bench: set PHYSX_BENCH_PREFIX to the install directory (35-40 GB, e.g. node-local scratch)." >&2
  return 2 2>/dev/null || exit 2
fi
mkdir -p "$PHYSX_BENCH_PREFIX"
PHYSX_BENCH_PREFIX=$(cd "$PHYSX_BENCH_PREFIX" && pwd)
export PHYSX_BENCH_PREFIX
PHYSX_BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PHYSX_BENCH_DIR
export PHYSX_BENCH_CALLER_HOME="${PHYSX_BENCH_CALLER_HOME:-$HOME}"

export HOME="$PHYSX_BENCH_PREFIX/home"
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share"
export TMPDIR="$PHYSX_BENCH_PREFIX/tmp"
export UV_CACHE_DIR="$PHYSX_BENCH_PREFIX/cache/uv" UV_PYTHON_INSTALL_DIR="$PHYSX_BENCH_PREFIX/uv/python"
export UV_TOOL_DIR="$PHYSX_BENCH_PREFIX/uv/tools" UV_NO_MODIFY_PATH=1
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"
export PIP_CACHE_DIR="$PHYSX_BENCH_PREFIX/cache/pip" TORCH_HOME="$PHYSX_BENCH_PREFIX/cache/torch"
export WARP_CACHE_PATH="$PHYSX_BENCH_PREFIX/cache/warp"
export PYTHONNOUSERSITE=1
export PATH="$PHYSX_BENCH_PREFIX/uv/bin:$PATH"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

PHYSX_BENCH_VENV="$PHYSX_BENCH_PREFIX/env_isaaclab"
export PHYSX_BENCH_VENV
if [ -f "$PHYSX_BENCH_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PHYSX_BENCH_VENV/bin/activate"
fi

PHYSX_BENCH_EULA_URL="https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"

# Driver version as an integer MMMmmm (580.65.06 -> 580065), or empty without nvidia-smi.
physx_bench_driver_int() {
  local v
  v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) || return 0
  [ -n "$v" ] || return 0
  awk -F. '{printf "%d%03d\n", $1, $2}' <<<"$v"
}

# Exit unless the operator has accepted the Isaac Sim / Omniverse EULA (Isaac Sim's own rule, isaacsim/kit/kit_app.py:
# OMNI_KIT_ACCEPT_EULA in y/yes/1, or the marker file isaacsim/kit/EULA_ACCEPTED whose first line is y/yes/1).
physx_bench_require_eula() {
  local kit_dir marker val
  case "$(printf '%s' "${OMNI_KIT_ACCEPT_EULA:-}" | tr '[:upper:]' '[:lower:]')" in
    y|yes|1) echo "physx_bench: EULA accepted by the operator (OMNI_KIT_ACCEPT_EULA set in the caller's environment)."; return 0 ;;
  esac
  kit_dir=$(python -I -c "import importlib.util, os; s = importlib.util.find_spec('isaacsim'); print(os.path.join(os.path.dirname(s.origin), 'kit'))" 2>/dev/null) || kit_dir=""
  marker="$kit_dir/EULA_ACCEPTED"
  if [ -n "$kit_dir" ] && [ -f "$marker" ]; then
    val=$(head -1 "$marker" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    case "$val" in y|yes|1) echo "physx_bench: EULA accepted by the operator (marker file present)."; return 0 ;; esac
  fi
  cat >&2 <<EOF
physx_bench: the NVIDIA Isaac Sim / Omniverse license (EULA) has not been accepted in this installation.
These scripts never accept it for you. Read the license:
  $PHYSX_BENCH_EULA_URL
and, if you agree, accept it yourself in one of the two ways Isaac Sim supports:
  - interactively, once per installation (prints the notice and asks "Do you accept the EULA? (Yes/No)"):
      PHYSX_BENCH_PREFIX=$PHYSX_BENCH_PREFIX bash -c 'source $PHYSX_BENCH_DIR/common.sh && python -c "import isaacsim"'
  - or by exporting OMNI_KIT_ACCEPT_EULA=YES in your own shell / job script before calling run.sh.
EOF
  exit 2
}
