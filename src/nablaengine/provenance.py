"""Where a result came from: the NablaEngine source revision, and for the scripts' result JSONs the GPU, driver,
toolkit, package versions and the reference physics (MuJoCo Warp's version, source tree hash and switches) it ran
with.

  from nablaengine import provenance
  res["provenance"] = provenance.provenance()   # before writing a result JSON
  provenance.keep_sections(res, [(old, kept)])  # --update: the provenance of the sections kept from the old JSON

Paths are recorded relative to the repository (or to ~), never host names, user names or job ids. Every probe is
best effort: a missing tool or package gives None, never an exception.
"""

import datetime
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # the checkout when nablaengine is imported from <checkout>/src
PACKAGES = ("numpy", "mujoco", "mujoco-mjx", "jax", "jaxlib", "jax-cuda12-plugin", "warp-lang", "mujoco-warp", "scipy")
# an environment switch (MJW_<NAME>) that a MuJoCo Warp source tree reads: mujoco_warp() records the ones its sources
# name (a stock MuJoCo Warp has none)
MJW_SWITCH = re.compile(r"""["'](MJW_[A-Z0-9_]+)["']""")
# a checkout may pin a MuJoCo Warp source tree: <checkout>/third_party/<name>/tree.sha256 holding its tree_sha256
TREE_FILES = "third_party/*/tree.sha256"
ENV_PREFIXES = ("DK_", "MJW_")
ENV_NAMES = ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "CUDA_DEVICE_ORDER")


def _run(cmd, cwd=None, timeout=20):
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def rel(path):
    """path relative to the repository when inside it, else with the home directory as ~, else only its last part
    (.../name: no absolute paths of the machine in results; str, None stays None)."""
    if path is None:
        return None
    p = Path(path).expanduser().absolute()  # a symlinked runs/ stays runs/
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    for q in (p, resolved):
        for base, prefix in ((REPO, ""), (Path.home(), "~/")):
            try:
                return prefix + str(q.relative_to(base))
            except ValueError:
                continue
    return f".../{p.name}"


def _installed_commit():
    """The commit pip recorded for an install from a git URL (PEP 610 direct_url.json), None otherwise."""
    import json
    from importlib import metadata
    try:
        text = metadata.distribution("nablaengine").read_text("direct_url.json")
        commit = (json.loads(text or "{}").get("vcs_info") or {}).get("commit_id")
    except (metadata.PackageNotFoundError, ValueError, OSError):
        return None
    return commit[:7] if commit else None


def source_commit():
    """(short commit, dirty) of the git checkout this package is imported from; for a pip install from a git URL, the
    commit pip recorded and dirty None; (None, None) otherwise (a wheel, an rsync copy). dirty: tracked files differ
    from the commit (untracked files do not count)."""
    if not ((REPO / ".git").exists() and (REPO / "src" / "nablaengine").is_dir()):
        return _installed_commit(), None
    commit = _run(["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"])
    if commit is None:
        return None, None
    status = _run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    return commit, (None if status is None else bool(status))


def gpu():
    """{name, compute_cap, driver} of the first visible GPU (nvidia-smi), None without one."""
    from nablaengine import cuda
    q = ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
    out = (_run([*q, "-i", cuda.visible_gpu()]) if cuda.visible_gpu() else None) or _run(q)
    if not out:
        return None
    name, cc, driver = [x.strip() for x in out.splitlines()[0].split(",")][:3]
    return dict(name=name, compute_cap=cc, driver=driver)


def nvcc_version(nvcc=None):
    """The nvcc release, e.g. '12.4.131' (from 'V12.4.131' of nvcc --version), None without nvcc."""
    from nablaengine import cuda
    try:
        nvcc = nvcc or cuda.nvcc()
    except FileNotFoundError:
        return None
    m = re.search(r"\bV(\d+\.\d+\.\d+)", _run([nvcc, "--version"]) or "")
    return m.group(1) if m else None


def packages(names=PACKAGES):
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def tree_sha256(pkg_dir):
    """sha256 over 'sha256  mujoco_warp/<path>' lines of the package's .py files in byte order: the same as
    `find mujoco_warp -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum` in the tree root."""
    pkg_dir = Path(pkg_dir)
    files = sorted(f"{pkg_dir.name}/{p.relative_to(pkg_dir).as_posix()}" for p in pkg_dir.rglob("*.py") if p.is_file())
    lines = "".join(f"{hashlib.sha256((pkg_dir.parent / f).read_bytes()).hexdigest()}  {f}\n" for f in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def tested_tree_sha256():
    """The tree_sha256 of the MuJoCo Warp source tree the checkout pins (TREE_FILES), None when it pins none (a
    released MuJoCo Warp is identified by its version)."""
    for f in sorted(REPO.glob(TREE_FILES)):
        try:
            return next(x for x in f.read_text().split() if re.fullmatch(r"[0-9a-f]{64}", x))
        except (OSError, StopIteration):
            continue
    return None


def mujoco_warp():
    """The installed MuJoCo Warp without importing it (no Warp start-up): version, the environment switches its
    sources name (MJW_SWITCH, sorted; none in a stock MuJoCo Warp), its tree_sha256 and whether that is the pinned
    tree (None when the checkout pins none). None if it is not installed."""
    try:
        spec = importlib.util.find_spec("mujoco_warp")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg = Path(list(spec.submodule_search_locations)[0])
    try:
        text = "".join(p.read_text(errors="replace") for p in (pkg / "_src").glob("*.py"))
        tree = tree_sha256(pkg)
    except OSError:
        text, tree = "", None
    tested = tested_tree_sha256()
    return dict(version=packages(("mujoco-warp",))["mujoco-warp"], path=rel(pkg),
                switches=sorted(set(MJW_SWITCH.findall(text))), tree_sha256=tree,
                tested_tree=None if tested is None or tree is None else tree == tested)


def environment():
    """The DK_* / MJW_* switches and a few JAX / CUDA variables that are set (paths relative, see rel)."""
    out = {}
    for k, v in sorted(os.environ.items()):
        if k.startswith(ENV_PREFIXES) or k in ENV_NAMES:
            if k == "XLA_FLAGS":  # flags such as --xla_dump_to=<path>: each path relative
                out[k] = " ".join(f"{f.split('=', 1)[0]}={rel(f.split('=', 1)[1])}" if "=" in f and "/" in f else f
                                  for f in v.split())
            else:
                out[k] = rel(v) if "/" in v else v
    return out


def keep_sections(res, sources):
    """Record where the sections that a run keeps from earlier result JSONs (--update, --merge) came from, as
    res["provenance_kept_sections"] = [{sections, provenance}] (one entry per distinct provenance; none kept: no key).
    sources: [(earlier JSON, [names of the sections taken from it]), ...], later ones win. A section gets the earlier
    JSON's own record of it (its provenance_kept_sections) when there is one, else that JSON's provenance (None for a
    JSON without one). res["provenance"] stays the provenance of this run's sections."""
    src = {}
    for old, names in sources:
        rec = {s: e.get("provenance") for e in old.get("provenance_kept_sections") or [] for s in e.get("sections", [])}
        src.update({s: rec.get(s, old.get("provenance")) for s in names})
    out = []
    for s, p in src.items():
        e = next((e for e in out if e["provenance"] == p), None)
        if e is None:
            out.append(dict(sections=[s], provenance=p))
        else:
            e["sections"].append(s)
    res.pop("provenance_kept_sections", None)
    if out:
        res["provenance_kept_sections"] = out


def provenance():
    """The provenance block of a result JSON (see the module docstring)."""
    from nablaengine import cuda, fixtures
    import nablaengine
    commit, dirty = source_commit()
    try:
        arch = cuda.arch()
    except Exception:  # noqa: BLE001  (no GPU and no DK_ARCH)
        arch = None
    return dict(date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                nablaengine=dict(version=nablaengine.__version__, commit=commit, dirty=dirty),
                gpu=gpu(), arch=arch, nvcc=nvcc_version(), python=platform.python_version(), packages=packages(),
                mujoco_warp=mujoco_warp(), environment=environment(), fixtures=rel(fixtures.fixtures_dir()))
