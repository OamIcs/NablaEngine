#!/usr/bin/env bash
# Set up the NablaEngine environment on a new Linux x86_64 machine with an NVIDIA driver for CUDA 12.x:
#
#   scripts/setup_env.sh [--env DIR] [--python PY] [--steps venv,deps,mjw,nvcc,activate] [--cuda X.Y.Z] [--dry-run]
#
# Steps (in this order; each is skipped when its result is already there):
#   venv      a Python 3.12 venv in DIR/venv (DIR: --env, default <repo>/.dk_env; the locked versions need 3.12)
#   deps      requirements-lock.txt (the tested freeze: numpy, mujoco 3.17.0, JAX 0.12 with its CUDA 12 plugins, scipy,
#             ...) and Warp 1.18.0+cu12 (the CUDA 12 build: CUDA 13 dropped Volta), the wheel checked by its sha256
#   mjw       MuJoCo Warp 3.17.0 from PyPI (mujoco-warp==3.17.0, --no-deps: its dependencies are in the lock), only
#             for the comparison tools (scripts/make_mjw_reference.py, nablaengine.reference.warp_physics); the engine,
#             the JAX interface and the checks against the committed references do not need it. Skip it with
#             --steps venv,deps,nvcc,activate. The benchmark harness has its own MuJoCo Warp environments
#             (benchmarks/envs/setup.sh).
#   nvcc      a CUDA 12.x nvcc for the engine's builds: $NVCC, nvcc on PATH, $CUDA_HOME/bin, /usr/local/cuda/bin,
#             DIR/cuda/bin; else conda's cuda-nvcc into DIR/cuda (conda, mamba or micromamba on PATH). --cuda X.Y.Z
#             picks the release; default 12.4.1 (the tested one, driver 550) unless the driver supports CUDA 12.9 or
#             newer, then 12.9.1, or exactly 12.8 (R570, the first Blackwell driver), then 12.8.1: sm_120 needs 12.8+.
#             The pip wheel nvidia-cuda-nvcc-cu12 has no nvcc (only ptxas).
#   activate  DIR/activate.sh: PATH (the venv), NVCC, PYTHONPATH (a checkout's src, see below) and PYTHONNOUSERSITE,
#             with the paths computed from the script's own location (no machine-specific paths for the repository or
#             the environment; NVCC is absolute only when it lies outside DIR). It does not set TMPDIR or DK_FIXTURES.
#             The checkout is the one the current directory is in when activate.sh is sourced (the nearest directory at
#             or above it with src/nablaengine: a clone, a git worktree or a copy), so one environment serves several
#             checkouts; outside any checkout, the one that ran setup_env.sh (recorded relative to DIR, so moving both
#             together keeps it), with a note on stderr. If neither exists (e.g. that checkout was moved and the
#             current directory is not in one), sourcing fails: a message, return status 1, nothing changed.
# --dry-run prints the commands instead of running them (nothing is created). Afterwards:
#   cd <checkout> && source DIR/activate.sh && python scripts/env_check.py
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$REPO/.dk_env"
PY=""
STEPS="venv,deps,mjw,nvcc,activate"
CUDA_REL=""
DRY=0
MJW_SPEC="mujoco-warp==3.17.0"
WARP_WHEEL="https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
WARP_SHA256="6dda3c163dde4edaa7d46ade9fff3ddb1126ee7b147d9d890cd38cc70dafd656"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "setup_env.sh: $*" >&2; exit 1; }
say() { echo "== $*"; }
run() {  # a command, or with --dry-run its printout
  if [ "$DRY" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_DIR="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --cuda) CUDA_REL="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "setup_env.sh: unknown argument $1" >&2; usage 1 ;;
  esac
done
case "$ENV_DIR" in /*) ;; *) ENV_DIR="$PWD/$ENV_DIR" ;; esac
has_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

[ "$(uname -s)" = Linux ] || die "Linux only"
say "repository $REPO, environment $ENV_DIR, steps $STEPS$([ "$DRY" = 1 ] && echo ' (dry run)')"
VPY="$ENV_DIR/venv/bin/python"

# ---------------------------------------------------------------- venv
if has_step venv; then
  if [ -x "$VPY" ]; then
    say "venv: $ENV_DIR/venv exists ($("$VPY" --version 2>&1))"
  else
    if [ -z "$PY" ]; then
      for c in python3.12 python3; do
        if command -v "$c" > /dev/null && "$c" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
          PY="$(command -v "$c")"; break
        fi
      done
    fi
    [ -n "$PY" ] || die "no Python 3.12 found (the locked versions need it): pass --python /path/to/python3.12"
    "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' || die "$PY is not Python 3.12"
    say "venv: $PY -> $ENV_DIR/venv"
    run mkdir -p "$ENV_DIR"
    run "$PY" -m venv "$ENV_DIR/venv"
  fi
fi

# ---------------------------------------------------------------- deps
if has_step deps; then
  [ -x "$VPY" ] || [ "$DRY" = 1 ] || die "no venv at $ENV_DIR/venv (run the venv step)"
  say "deps: requirements-lock.txt and Warp 1.18.0+cu12"
  run "$VPY" -m pip install -r "$REPO/requirements-lock.txt"
  run "$VPY" -m pip install "warp-lang @ $WARP_WHEEL#sha256=$WARP_SHA256"
fi

# ---------------------------------------------------------------- mjw
if has_step mjw; then
  [ -x "$VPY" ] || [ "$DRY" = 1 ] || die "no venv at $ENV_DIR/venv (run the venv step)"
  say "mjw: $MJW_SPEC (PyPI; only the comparison tools need it)"
  run "$VPY" -m pip install --no-deps "$MJW_SPEC"
fi

# ---------------------------------------------------------------- nvcc
NVCC_FOUND=""
nvcc_release() { { "$1" --version 2>/dev/null || true; } | grep -Eo 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 || true; }
if has_step nvcc || has_step activate; then
  for c in "${NVCC:-}" "$(command -v nvcc || true)" "${CUDA_HOME:+$CUDA_HOME/bin/nvcc}" /usr/local/cuda/bin/nvcc \
           "$ENV_DIR/cuda/bin/nvcc"; do
    if [ -n "$c" ] && [ -x "$c" ]; then NVCC_FOUND="$c"; break; fi
  done
fi
if has_step nvcc; then
  if [ -n "$NVCC_FOUND" ]; then
    rel="$(nvcc_release "$NVCC_FOUND")"
    say "nvcc: $NVCC_FOUND (release $rel)"
    case "$rel" in 12.*) ;; *) echo "warning: $NVCC_FOUND is not a CUDA 12 nvcc (CUDA 13 dropped sm_70)" >&2 ;; esac
  else
    if [ -z "$CUDA_REL" ]; then
      drv="$(nvidia-smi 2>/dev/null | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | cut -d' ' -f3 || true)"
      case "$drv" in 12.9|1[3-9].*) CUDA_REL=12.9.1 ;; 12.8) CUDA_REL=12.8.1 ;; *) CUDA_REL=12.4.1 ;; esac
      say "nvcc: none found; the driver supports CUDA ${drv:-?}: conda's CUDA $CUDA_REL (--cuda overrides)"
    fi
    case "$CUDA_REL" in  # the tested pins (12.8.1: the label's versions, not tested here); others: the label's own
      12.4.1) pkgs=(cuda-nvcc=12.4.131 cuda-cudart-dev=12.4.127 cuda-cudart-static=12.4.127) ;;
      12.8.1) pkgs=(cuda-nvcc=12.8.93 cuda-cudart-dev=12.8.90 cuda-cudart-static=12.8.90) ;;
      12.9.1) pkgs=(cuda-nvcc=12.9.86 cuda-cudart-dev=12.9.79 cuda-cudart-static=12.9.79) ;;
      *) pkgs=(cuda-nvcc cuda-cudart-dev cuda-cudart-static) ;;
    esac
    conda=""
    for c in conda mamba micromamba; do if command -v "$c" > /dev/null; then conda="$c"; break; fi; done
    [ -n "$conda" ] || [ "$DRY" = 1 ] || die "no nvcc and no conda / mamba / micromamba: install a CUDA 12.x toolkit and set NVCC"
    run "${conda:-conda}" create -y -q -p "$ENV_DIR/cuda" -c "nvidia/label/cuda-$CUDA_REL" "${pkgs[@]}"
    NVCC_FOUND="$ENV_DIR/cuda/bin/nvcc"
  fi
fi

# ---------------------------------------------------------------- activate
if has_step activate; then
  repo_rel="$(realpath -m --relative-to="$ENV_DIR" "$REPO")"  # the checkout seen from the environment
  case "$NVCC_FOUND" in
    "$ENV_DIR"/*) nvcc_line="export NVCC=\"\$DK_ENV/${NVCC_FOUND#"$ENV_DIR"/}\"" ;;
    "") nvcc_line="# (no nvcc found when this was written: set NVCC to a CUDA 12.x nvcc)" ;;
    *) nvcc_line="export NVCC=\"$NVCC_FOUND\"  # outside the environment" ;;
  esac
  say "activate: $ENV_DIR/activate.sh"
  if [ "$DRY" = 1 ]; then
    echo "+ (write $ENV_DIR/activate.sh: DK_ENV from its location, DK_REPO the checkout of the current directory, else \$DK_ENV/$repo_rel; $nvcc_line)"
  else
    mkdir -p "$ENV_DIR"
    cat > "$ENV_DIR/activate.sh" <<EOF
# The NablaEngine environment, written by scripts/setup_env.sh. Source it (bash or zsh), best from the checkout:
#   source <this directory>/activate.sh
# PYTHONPATH gets the src/ of the checkout the current directory is in (the nearest directory at or above it with
# src/nablaengine); outside any checkout, of the one that wrote this file (\$DK_ENV/$repo_rel), with a note. Neither:
# an error, return status 1 and nothing changed. Paths follow this file's location; TMPDIR and DK_FIXTURES are left
# alone (see README.md and docs/configuration.md).
_dk_env="\$(cd "\$(dirname "\${BASH_SOURCE[0]:-\$0}")" && pwd)"
_dk_repo="\$PWD"
while [ "\$_dk_repo" != / ] && [ ! -f "\$_dk_repo/src/nablaengine/__init__.py" ]; do _dk_repo="\$(dirname "\$_dk_repo")"; done
if [ ! -f "\$_dk_repo/src/nablaengine/__init__.py" ]; then
  _dk_repo="\$(cd "\$_dk_env/$repo_rel" 2>/dev/null && pwd || true)"
  if [ -z "\$_dk_repo" ] || [ ! -f "\$_dk_repo/src/nablaengine/__init__.py" ]; then
    echo "activate.sh: no NablaEngine checkout at or above \$PWD, nor at \$_dk_env/$repo_rel (the checkout that wrote this file): cd into a checkout and source it again; nothing was changed" >&2
    unset _dk_env _dk_repo
    return 1 2>/dev/null || exit 1
  fi
  echo "activate.sh: \$PWD is not in a NablaEngine checkout; using \$_dk_repo" >&2
fi
DK_ENV="\$_dk_env"
DK_REPO="\$_dk_repo"
export DK_ENV DK_REPO
unset _dk_env _dk_repo
export PATH="\$DK_ENV/venv/bin:\$PATH"
$nvcc_line
export PYTHONPATH="\$DK_REPO/src"
export PYTHONNOUSERSITE=1
EOF
  fi
fi
say "done$([ "$DRY" = 1 ] && echo ' (dry run: nothing was changed)'). Next: cd $REPO && source $ENV_DIR/activate.sh && python scripts/env_check.py"
