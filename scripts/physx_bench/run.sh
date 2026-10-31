#!/usr/bin/env bash
# Run the PhysX baseline suite (one command; see README.md):
#
#   PHYSX_BENCH_PREFIX=<install dir> scripts/physx_bench/run.sh [--quick] [--out-dir DIR] [--no-idle-check]
#                                                                [--budget-min M]
#
#   --quick          validation mode (N=1024 only, short rounds; ~3 min including Kit start-ups)
#   --out-dir DIR    output directory (must not exist or be empty); default runs/physx_bench/<UTC time>_<mode>
#                    in the repository (git-ignored)
#   --no-idle-check  skip the check that the GPU has no other compute process before each configuration
#                    (use it on an exclusive GPU; also PHYSX_BENCH_IDLE_CHECK=0)
#   --budget-min M   wall-time budget for the suite excluding install (default 55, or PHYSX_BENCH_BUDGET_MIN);
#                    optional configurations are skipped when they would not fit
#
# GPU: the first device of CUDA_VISIBLE_DEVICES (as set by Slurm or the caller); when unset, PHYSX_BENCH_GPU
# (default 0) is exported as CUDA_VISIBLE_DEVICES. Requires install.sh to have completed and the Isaac Sim EULA to
# have been accepted by the operator (this script only checks; it never accepts it).
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
REPO=$(cd "$PHYSX_BENCH_DIR/../.." && pwd)

MODE=full
OUT=""
IDLE=${PHYSX_BENCH_IDLE_CHECK:-1}
BUDGET=${PHYSX_BENCH_BUDGET_MIN:-55}
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE=quick ;;
    --out-dir) OUT=$2; shift ;;
    --no-idle-check) IDLE=0 ;;
    --budget-min) BUDGET=$2; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "run.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

[ -x "$PHYSX_BENCH_VENV/bin/python" ] && [ -f "$PHYSX_BENCH_PREFIX/.install_stamp" ] || {
  echo "run.sh: no completed install under PHYSX_BENCH_PREFIX; run scripts/physx_bench/install.sh first." >&2; exit 2; }
physx_bench_require_eula
if [ -n "${OMNI_KIT_ACCEPT_EULA:-}" ]; then
  export PHYSX_BENCH_EULA_HOW="operator set OMNI_KIT_ACCEPT_EULA"
else
  export PHYSX_BENCH_EULA_HOW="operator-written marker file"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${PHYSX_BENCH_GPU:-0}"
  echo "run.sh: CUDA_VISIBLE_DEVICES unset; using GPU ${CUDA_VISIBLE_DEVICES} (set PHYSX_BENCH_GPU to change)."
fi
DRV=$(physx_bench_driver_int)
if [ "${DRV:-0}" -lt 580065 ]; then
  export PHYSX_BENCH_DRIVER_BELOW_MIN=1
  echo "run.sh: driver below Isaac Lab 3.0's minimum 580.65.06: results are feasibility-only." >&2
fi
PHYSX_BENCH_TORCH_BACKEND_USED=$(sed -n 's/.*torch-backend=//p' "$PHYSX_BENCH_PREFIX/.install_stamp")
export PHYSX_BENCH_TORCH_BACKEND_USED
export HEADLESS=1

[ -n "$OUT" ] || OUT="$REPO/runs/physx_bench/$(date -u +%Y%m%dT%H%M%SZ)_$MODE"
ARGS=(--mode "$MODE" --out-dir "$OUT" --budget-min "$BUDGET")
[ "$IDLE" = 1 ] || ARGS+=(--no-idle-check)
echo "run.sh: mode $MODE, budget $BUDGET min, output $OUT"
cd "$REPO"
python "$PHYSX_BENCH_DIR/run_suite.py" "${ARGS[@]}"
