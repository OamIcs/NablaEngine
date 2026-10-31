#!/usr/bin/env bash
# Create the per-engine Python environments of the benchmark (benchmarks/envs/envs.toml maps engines to them):
#
#   benchmarks/envs/setup.sh [--prefix DIR] [--envs dk,mjw-native,mjw-pr] [--python PY] [--warp auto|default|cu12]
#                            [--dry-run]
#
#   --prefix  where the environments go (default $DK_BENCH_ENV_PREFIX, else <repo>/runs/bench/envs; the driver reads
#             the same variable). Relative paths are relative to the current directory.
#   --envs    which ones (default all three):
#     dk          the NablaEngine environment of scripts/setup_env.sh (nablaengine, mujoco / mujoco-mjx 3.17.0, JAX,
#                 warp-lang 1.18.0). An existing one ($DK_ENV with venv/bin/python, as its activate.sh sets) is used
#                 as it is; else scripts/setup_env.sh --env <prefix>/dk creates one.
#     mjw-native  <prefix>/mjw-native/venv: requirements-mjw-native.txt (PyPI mujoco-warp 3.17.0, mujoco 3.17.0,
#                 warp-lang 1.18.0)
#     mjw-pr      <prefix>/mjw-pr/venv: requirements-mjw-pr.txt (mujoco 3.17.0, warp-lang 1.18.0, no mujoco_warp), and
#                 the source trees each engine puts on PYTHONPATH, in <prefix>/mjw-pr/src/:
#                   pr1535      etaoxing/mujoco_warp @ 357a75d, as published (google-deepmind/mujoco_warp#1535's head)
#                   pr1535_fix  357a75d + etaoxing/mujoco_warp#1 (4af093f), as published
#                 (one clone, the other as a git worktree of it; the adapters check each tree's source hash)
#     genesis     <prefix>/genesis/venv (not in the default set; --envs genesis): PyTorch 2.14.1 from PyTorch's index
#                 (cu126 on CUDA 12 drivers and on Volta, else cu128), then requirements-genesis.txt (genesis-world
#                 1.4.5, quadrants 1.3.3, mujoco 3.17.0)
#   --python  a Python 3.12 interpreter for the venvs (default: python3.12 or python3 if that is 3.12)
#   --warp    auto (default): the default warp-lang 1.18.0 wheel (CUDA 13) when the driver supports CUDA 13
#             (nvidia-smi's "CUDA Version" >= 13.0), else NVIDIA's 1.18.0+cu12 wheel of the same release (CUDA 12
#             drivers, Volta: the default wheel falls back to the CPU there without an error); default / cu12 force one.
#             The choice is written to <prefix>/<env>/warp_wheel.txt; the results record the installed version tag.
#   --dry-run print the commands instead of running them.
# Existing venvs and trees are kept (only checked). Afterwards: python -m benchmarks.bench env-check
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${DK_BENCH_ENV_PREFIX:-$REPO/runs/bench/envs}"
ENVS="dk,mjw-native,mjw-pr"
PY=""
WARP="auto"
DRY=0
MJW_URL="https://github.com/etaoxing/mujoco_warp.git"
MJW_COMMIT="357a75d60a56d67d476942a1b6e54b3045ee8e87"
PR1_COMMIT="4af093f02a3f48f0009ca7c916674b928136f310"
WARP_CU12="https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
WARP_CU12_SHA256="6dda3c163dde4edaa7d46ade9fff3ddb1126ee7b147d9d890cd38cc70dafd656"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "setup.sh: $*" >&2; exit 1; }
say() { echo "== $*"; }
run() { if [ "$DRY" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --envs) ENVS="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --warp) WARP="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "setup.sh: unknown argument $1" >&2; usage 1 ;;
  esac
done
case "$PREFIX" in /*) ;; *) PREFIX="$PWD/$PREFIX" ;; esac
case "$WARP" in auto|default|cu12) ;; *) die "--warp must be auto, default or cu12" ;; esac
has_env() { case ",$ENVS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for e in ${ENVS//,/ }; do case "$e" in dk|mjw-native|mjw-pr|genesis) ;; *) die "unknown environment $e" ;; esac; done

if [ "$WARP" = auto ]; then
  drv="$(nvidia-smi 2>/dev/null | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | cut -d' ' -f3 || true)"
  case "$drv" in 1[3-9].*|[2-9][0-9].*) WARP=default ;; *) WARP=cu12 ;; esac
  say "driver CUDA ${drv:-?}: warp-lang 1.18.0 $([ "$WARP" = default ] && echo 'default (CUDA 13) wheel' || echo '+cu12 wheel')"
fi

find_python() {
  if [ -n "$PY" ]; then echo "$PY"; return; fi
  for c in python3.12 python3; do
    if command -v "$c" > /dev/null && "$c" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
      command -v "$c"; return
    fi
  done
  die "no Python 3.12 found: pass --python /path/to/python3.12"
}

make_venv() {  # $1 = environment directory, $2 = requirements file
  local dir="$1" req="$2" vpy="$1/venv/bin/python"
  if [ -x "$vpy" ]; then
    say "$(basename "$dir"): $dir/venv exists ($("$vpy" --version 2>&1)); not modified"
    return
  fi
  local py; py="$(find_python)"
  say "$(basename "$dir"): $py -> $dir/venv"
  run mkdir -p "$dir"
  run "$py" -m venv "$dir/venv"
  run "$vpy" -m pip install --upgrade pip
  run "$vpy" -m pip install -r "$req"
  if [ "$WARP" = cu12 ]; then
    run "$vpy" -m pip install --force-reinstall --no-deps "warp-lang @ $WARP_CU12#sha256=$WARP_CU12_SHA256"
  fi
  if [ "$DRY" = 1 ]; then echo "+ (write $dir/warp_wheel.txt: $WARP)"; else echo "$WARP" > "$dir/warp_wheel.txt"; fi
}

# ---------------------------------------------------------------- dk
if has_env dk; then
  if [ -n "${DK_ENV:-}" ] && [ -x "$DK_ENV/venv/bin/python" ]; then
    say "dk: the existing NablaEngine environment \$DK_ENV ($("$DK_ENV/venv/bin/python" --version 2>&1)); not modified"
  elif [ -x "$PREFIX/dk/venv/bin/python" ]; then
    say "dk: $PREFIX/dk exists; not modified"
  else
    say "dk: scripts/setup_env.sh --env $PREFIX/dk"
    args=(--env "$PREFIX/dk"); [ -n "$PY" ] && args+=(--python "$PY"); [ "$DRY" = 1 ] && args+=(--dry-run)
    "$REPO/scripts/setup_env.sh" "${args[@]}"
  fi
fi

# ---------------------------------------------------------------- mjw-native
if has_env mjw-native; then
  make_venv "$PREFIX/mjw-native" "$REPO/benchmarks/envs/requirements-mjw-native.txt"
fi

# ---------------------------------------------------------------- mjw-pr
if has_env mjw-pr; then
  make_venv "$PREFIX/mjw-pr" "$REPO/benchmarks/envs/requirements-mjw-pr.txt"
  SRC="$PREFIX/mjw-pr/src"
  command -v git > /dev/null || die "git is needed for the mjw-pr source trees"
  head_is() { [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" = "$2" ]; }
  if [ -d "$SRC/pr1535/.git" ] || [ -f "$SRC/pr1535/.git" ]; then
    head_is "$SRC/pr1535" "$MJW_COMMIT" || [ "$DRY" = 1 ] || die "$SRC/pr1535 is not at ${MJW_COMMIT:0:7}"
    say "mjw-pr: $SRC/pr1535 at ${MJW_COMMIT:0:7}"
  else
    say "mjw-pr: $MJW_URL @ ${MJW_COMMIT:0:7} -> $SRC/pr1535"
    run mkdir -p "$SRC"
    run git clone "$MJW_URL" "$SRC/pr1535"
    run git -C "$SRC/pr1535" checkout --detach "$MJW_COMMIT"
  fi
  if [ -e "$SRC/pr1535_fix/.git" ]; then
    head_is "$SRC/pr1535_fix" "$PR1_COMMIT" || [ "$DRY" = 1 ] || die "$SRC/pr1535_fix is not at ${PR1_COMMIT:0:7}"
    say "mjw-pr: $SRC/pr1535_fix at ${PR1_COMMIT:0:7}"
  else
    say "mjw-pr: + etaoxing/mujoco_warp#1 (${PR1_COMMIT:0:7}) -> $SRC/pr1535_fix"
    run git -C "$SRC/pr1535" fetch origin pull/1/head
    run git -C "$SRC/pr1535" merge-base --is-ancestor "$MJW_COMMIT" "$PR1_COMMIT"
    run git -C "$SRC/pr1535" worktree add --detach "$SRC/pr1535_fix" "$PR1_COMMIT"
  fi
fi
# ---------------------------------------------------------------- genesis
if has_env genesis; then
  dir="$PREFIX/genesis"; vpy="$dir/venv/bin/python"
  if [ -x "$vpy" ]; then
    say "genesis: $dir/venv exists ($("$vpy" --version 2>&1)); not modified"
  else
    py="$(find_python)"
    drv="${drv:-$(nvidia-smi 2>/dev/null | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | cut -d' ' -f3 || true)}"
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
    case "$drv" in 1[3-9].*|[2-9][0-9].*) tcu=cu128 ;; *) tcu=cu126 ;; esac
    case "$cc" in 7.*) tcu=cu126 ;; esac  # Volta: the cu128 builds have no sm_70 kernels
    say "genesis: $py -> $dir/venv (torch 2.14.1+$tcu)"
    run mkdir -p "$dir"
    run "$py" -m venv "$dir/venv"
    run "$vpy" -m pip install --upgrade pip
    run "$vpy" -m pip install "torch==2.14.1" --index-url "https://download.pytorch.org/whl/$tcu"
    run "$vpy" -m pip install -r "$REPO/benchmarks/envs/requirements-genesis.txt"
    if [ "$DRY" = 1 ]; then echo "+ (write $dir/torch_wheel.txt: $tcu)"; else echo "$tcu" > "$dir/torch_wheel.txt"; fi
  fi
fi
say "done$([ "$DRY" = 1 ] && echo ' (dry run: nothing was changed)'). Prefix $PREFIX; next: python -m benchmarks.bench env-check"
