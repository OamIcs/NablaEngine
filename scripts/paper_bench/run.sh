#!/usr/bin/env bash
# The paper timing kit in one command (scripts/paper_bench/README.md):
#
#   scripts/paper_bench/run.sh <out_dir>
#
# 1. scripts/paper_bench/setup.sh (idempotent; native MuJoCo Warp venvs, the PR #1535 trees, Menagerie under
#    $PAPER_BENCH_PREFIX); PAPER_BENCH_SKIP_SETUP=1 skips it. A failed setup is recorded and the parts that need what
#    it would have installed are skipped; the rest still run. The setup gets at most PAPER_BENCH_SETUP_TIMEOUT_MIN
#    minutes (default 15), and the job budget counts from the start of run.sh, the setup included.
# 2. scripts/paper_bench/orchestrate.py in the main environment ($PAPER_BENCH_DEV_PYTHON, default python3: the
#    interpreter with NablaEngine's dependencies, JAX and mujoco-mjx): every part in its own process, per-part
#    timeouts, a whole-job budget ($PAPER_BENCH_BUDGET_MIN, default 53 min).
# 3. scripts/paper_bench/summarize.py: <out_dir>/paper_bench.md.
# Exit status: 0 if every part and the setup succeeded, 1 otherwise (2: nothing ran); the JSON is written either way.
set -uo pipefail

if [ $# -ne 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,14p' "$0"; exit 2
fi
# the whole-job budget counts from here (the setup included)
PAPER_BENCH_JOB_START="$(date +%s)"
export PAPER_BENCH_JOB_START
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$1/logs" || exit 2
OUT="$(cd "$1" && pwd)"
DEV_PY="${PAPER_BENCH_DEV_PYTHON:-python3}"
command -v "$DEV_PY" > /dev/null 2>&1 || { echo "[paper_bench] PAPER_BENCH_DEV_PYTHON=$DEV_PY not found" >&2; exit 2; }
export PAPER_BENCH_PREFIX="${PAPER_BENCH_PREFIX:-$REPO/runs/paper_bench_env}"
# the native venvs are created from the development interpreter unless PAPER_BENCH_PYTHON says otherwise
export PAPER_BENCH_PYTHON="${PAPER_BENCH_PYTHON:-$DEV_PY}"
cd "$REPO"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[paper_bench] CUDA_VISIBLE_DEVICES is not set: every part uses CUDA device 0 (nvidia-smi index 0)" >&2
fi

SETUP=ok
if [ "${PAPER_BENCH_SKIP_SETUP:-0}" = 1 ]; then
  SETUP=skipped
else
  # at most PAPER_BENCH_SETUP_TIMEOUT_MIN minutes (default 15): a hung download must not eat the job
  timeout -k 30 "$(( ${PAPER_BENCH_SETUP_TIMEOUT_MIN:-15} * 60 ))" scripts/paper_bench/setup.sh > "$OUT/logs/setup.log" 2>&1
  rc=$?
  [ $rc -eq 124 ] && echo "[paper_bench] setup timed out" >> "$OUT/logs/setup.log"
  if [ $rc -ne 0 ]; then
    SETUP="failed:$rc"
    echo "[paper_bench] setup failed (exit $rc), see $OUT/logs/setup.log; continuing with what is installed" >&2
    tail -5 "$OUT/logs/setup.log" >&2
  fi
fi

PYTHONPATH="$REPO/src" "$DEV_PY" scripts/paper_bench/orchestrate.py --out "$OUT" --setup-status "$SETUP"
rc=$?
PYTHONPATH="$REPO/src" "$DEV_PY" scripts/paper_bench/summarize.py "$OUT/paper_bench.json" --out "$OUT/paper_bench.md" \
  > "$OUT/logs/summarize.log" 2>&1 || { echo "[paper_bench] summarize failed, see $OUT/logs/summarize.log" >&2; rc=1; }
echo "[paper_bench] wrote $OUT/paper_bench.json and $OUT/paper_bench.md (exit $rc)" >&2
exit $rc
