#!/usr/bin/env bash
# Idempotent setup of the native engine environments of the paper timing kit (scripts/paper_bench/README.md).
#
#   scripts/paper_bench/setup.sh            # from anywhere; installs under $PAPER_BENCH_PREFIX
#
# Creates under $PAPER_BENCH_PREFIX (default: <repo>/runs/paper_bench_env):
#   venv_mjw/   PyPI mujoco==3.17.0, mujoco-warp==3.17.0, warp-lang==1.18.0 (the default PyPI wheel, built for CUDA 13,
#               when the driver is >= 580 or no driver is found; NVIDIA's +cu12 wheel of the same release for an
#               older driver; override with PAPER_BENCH_WARP_WHEEL=default|cu12)
#   venv_pr/    mujoco==3.17.0, warp-lang==1.18.0 (same wheel rule), absl-py, etils[epath], numpy; NO mujoco_warp
#               installed: the two source trees below are put on PYTHONPATH by the kit
#   src/mujoco_warp_357a75d/  etaoxing/mujoco_warp @ 357a75d (the head of google-deepmind/mujoco_warp PR #1535)
#   src/mujoco_warp_4af093f/  etaoxing/mujoco_warp @ 4af093f (head of etaoxing/mujoco_warp#1: two public fixes)
#   menagerie/  google-deepmind/mujoco_menagerie @ f3b6013 (unitree_g1/ only; the XML is parsed, nothing executed)
#   setup.json  what was installed (versions, warp wheel tag, commits)
# Each component is skipped when its stamp file says it is complete with the same specification. A component that
# must be installed without network access fails with a clear message (exit status 2). Python for the venvs:
# $PAPER_BENCH_PYTHON (default python3; >= 3.10).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${PAPER_BENCH_PREFIX:-$REPO/runs/paper_bench_env}"
PY="${PAPER_BENCH_PYTHON:-python3}"
MJ_VER=3.17.0
MJW_VER=3.17.0
WARP_VER=1.18.0
# NVIDIA's CUDA 12 build of warp-lang 1.18.0 (GitHub release asset; PyPI only has the CUDA 13 build)
WARP_CU12_URL="https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl#sha256=6dda3c163dde4edaa7d46ade9fff3ddb1126ee7b147d9d890cd38cc70dafd656"
PR_REPO="https://github.com/etaoxing/mujoco_warp.git"
PR_COMMITS="357a75d60a56d67d476942a1b6e54b3045ee8e87 4af093f02a3f48f0009ca7c916674b928136f310"
MEN_REPO="https://github.com/google-deepmind/mujoco_menagerie.git"
MEN_COMMIT="f3b6013629c7b94f8ee93fd977b161fa8350f9bc"

log() { echo "[paper_bench setup $(date +%H:%M:%S)] $*" >&2; }
die() { echo "[paper_bench setup] ERROR: $1" >&2; exit "${2:-1}"; }

mkdir -p "$PREFIX/src"
LOCK="$PREFIX/.setup.lock"
exec 9>"$LOCK"
flock -w 600 9 || die "another setup holds $LOCK"

NET=""
need_net() {  # fails (exit 2) when PyPI / GitHub cannot be reached
  if [ -z "$NET" ]; then
    if "$PY" - <<'EOF' >/dev/null 2>&1
import urllib.request
for u in ("https://pypi.org/simple/mujoco/", "https://github.com/"):
    urllib.request.urlopen(u, timeout=15).read(64)
EOF
    then NET=ok; else NET=none; fi
  fi
  [ "$NET" = ok ] || die "no network access (PyPI / GitHub unreachable) and '$1' is not set up under $PREFIX. Run this setup once on a node with network access (same PAPER_BENCH_PREFIX), or skip the parts that need it." 2
}

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' 2>/dev/null \
  || die "PAPER_BENCH_PYTHON=$PY must be Python >= 3.10"

# warp-lang wheel: the PyPI default needs a CUDA 13 driver (>= 580)
WHEEL="${PAPER_BENCH_WARP_WHEEL:-auto}"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
if [ "$WHEEL" = auto ]; then
  major="${DRIVER%%.*}"
  if [ -z "$major" ]; then  # no GPU here: assume the GPU machines' driver supports CUDA 13 (>= 580)
    WHEEL=default
    log "no NVIDIA driver found: warp wheel 'default' (CUDA 13 build); set PAPER_BENCH_WARP_WHEEL=cu12 for a driver < 580"
  elif [ "$major" -ge 580 ] 2>/dev/null; then WHEEL=default; else WHEEL=cu12; fi
fi
case "$WHEEL" in
  default) WARP_SPEC="warp-lang==$WARP_VER" ;;
  cu12) WARP_SPEC="warp-lang @ $WARP_CU12_URL" ;;
  *) die "PAPER_BENCH_WARP_WHEEL must be auto, default or cu12 (got $WHEEL)" ;;
esac
log "prefix $PREFIX, python $("$PY" -c 'import platform; print(platform.python_version())'), driver ${DRIVER:-none}, warp wheel $WHEEL"

make_venv() {  # name, stamp spec, pip args...
  local name="$1" spec="$2"; shift 2
  local v="$PREFIX/$name"
  if [ -f "$v/.paper_bench_ok" ] && [ "$(cat "$v/.paper_bench_ok")" = "$spec" ]; then
    log "$name: present ($spec)"; return 0
  fi
  need_net "$name"
  log "$name: creating ($spec)"
  rm -rf "$v"
  if ! "$PY" -m venv "$v"; then  # an interpreter without ensurepip: a venv without pip, then pip's own bootstrap
    log "$name: '$PY -m venv' failed; retrying with --without-pip and pip's get-pip.py"
    rm -rf "$v"
    "$PY" -m venv --without-pip "$v"
    "$PY" -c 'import sys, urllib.request; open(sys.argv[1], "wb").write(urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py", timeout=120).read())' "$v/get-pip.py"
    PYTHONNOUSERSITE=1 "$v/bin/python" "$v/get-pip.py" -q
    rm -f "$v/get-pip.py"
  fi
  PYTHONNOUSERSITE=1 "$v/bin/python" -m pip install -q --upgrade pip
  PYTHONNOUSERSITE=1 "$v/bin/python" -m pip install -q "$@"
  # the default PyPI warp wheel is pulled by mujoco-warp; pin the chosen wheel last (no dependency changes)
  PYTHONNOUSERSITE=1 "$v/bin/python" -m pip install -q --force-reinstall --no-deps "$WARP_SPEC"
  PYTHONNOUSERSITE=1 "$v/bin/python" -m pip freeze > "$v/pip_freeze.txt"
  echo "$spec" > "$v/.paper_bench_ok"
}

make_venv venv_mjw "mujoco==$MJ_VER mujoco-warp==$MJW_VER warp=$WHEEL" "mujoco==$MJ_VER" "mujoco-warp==$MJW_VER" "$WARP_SPEC"
make_venv venv_pr "mujoco==$MJ_VER warp=$WHEEL absl-py etils[epath] numpy" "mujoco==$MJ_VER" absl-py "etils[epath]" numpy "$WARP_SPEC"

# the PR #1535 trees (plain exports of the commits, no .git: nothing to modify by accident)
CLONE="$PREFIX/src/.mujoco_warp_git"
for c in $PR_COMMITS; do
  d="$PREFIX/src/mujoco_warp_${c:0:7}"
  if [ -f "$d/.paper_bench_ok" ] && [ "$(cat "$d/.paper_bench_ok")" = "$c" ]; then log "tree ${c:0:7}: present"; continue; fi
  need_net "src/mujoco_warp_${c:0:7}"
  if [ ! -d "$CLONE" ]; then
    git init -q "$CLONE"
    git -C "$CLONE" remote add origin "$PR_REPO"
  fi
  log "tree ${c:0:7}: fetching"
  git -C "$CLONE" fetch -q --depth 1 origin "$c"
  rm -rf "$d"; mkdir -p "$d"
  git -C "$CLONE" archive "$c" | tar -x -C "$d"
  [ -f "$d/mujoco_warp/__init__.py" ] || die "tree $c: mujoco_warp/__init__.py missing after export"
  echo "$c" > "$d/.paper_bench_ok"
done
rm -rf "$CLONE"

# Menagerie G1 (MJX's own G1 model), sparse
MD="$PREFIX/menagerie"
if [ -f "$MD/.paper_bench_ok" ] && [ "$(cat "$MD/.paper_bench_ok")" = "$MEN_COMMIT" ]; then
  log "menagerie: present"
else
  need_net menagerie
  log "menagerie: fetching unitree_g1 at ${MEN_COMMIT:0:7}"
  rm -rf "$MD"
  git init -q "$MD"
  git -C "$MD" remote add origin "$MEN_REPO"
  git -C "$MD" sparse-checkout set unitree_g1
  git -C "$MD" fetch -q --depth 1 --filter=blob:none origin "$MEN_COMMIT"
  git -C "$MD" -c advice.detachedHead=false checkout -q FETCH_HEAD
  [ -f "$MD/unitree_g1/scene_mjx.xml" ] || die "menagerie: unitree_g1/scene_mjx.xml missing"
  echo "$MEN_COMMIT" > "$MD/.paper_bench_ok"
fi

# record what is installed
"$PY" - "$PREFIX" "$WHEEL" "${DRIVER:-}" <<'EOF'
import json, re, sys
from pathlib import Path
prefix, wheel, driver = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def freeze(v):
    p = prefix / v / "pip_freeze.txt"
    out = {}
    for line in p.read_text().splitlines() if p.exists() else []:
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|@)\s*(.+)$", line.strip())
        if m:
            val = m.group(3)
            if "/" in val:  # a URL: keep the wheel file name only
                val = val.split("/")[-1].split("#")[0]
            out[m.group(1).lower()] = val
    return out
rec = dict(warp_wheel=wheel, driver_at_setup=driver or None,
           venvs={v: freeze(v) for v in ("venv_mjw", "venv_pr")},
           trees={p.name: (p / ".paper_bench_ok").read_text().strip() for p in sorted((prefix / "src").glob("mujoco_warp_*"))},
           menagerie=(prefix / "menagerie" / ".paper_bench_ok").read_text().strip())
(prefix / "setup.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
EOF
log "done"
