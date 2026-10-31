"""Fetch the Track B model, MuJoCo Menagerie's Unitree G1 MJX scene, at its pinned commit and verify every file.

pins.json (next to this file) pins each model (top-level keys without a leading underscore): the repository, the
commit and why it was chosen, the entry file, a link to the model's licence, and the sha256 of every file the entry
needs (the scene, the robot XML it includes, the meshes, and the licence text, which travels with the files).
download() fetches exactly those files from https://raw.githubusercontent.com/<repo>/<commit>/<path> into a
temporary directory under the models root, checks every sha256 and that MuJoCo compiles the entry, and only then moves
the directory into place as <root>/<repository name>/ (for the G1: <root>/mujoco_menagerie/unitree_g1/scene_mjx.xml),
with the pin it was verified against in <root>/<repository name>/.pin.json. Nothing is vendored into the repository.
The downloaded files are untrusted data: only MuJoCo parses them, and nothing from there is run or imported.

Models root: $DK_BENCH_MODELS when set and not empty (relative paths from the current directory), else
runs/bench/models of the checkout (ignored by git); --root overrides both. Results do not depend on it.

  python -m benchmarks.bench.models.fetch            # fetch if needed, verify, print the scene path (exit 1 on failure)
  python -m benchmarks.bench.models.fetch --check    # verify an existing copy only, without network access
  python -m benchmarks.bench.models.fetch --repin <commit> --write-pins <file>   # maintainers: the same files at
                                                     # another commit, hashed and compile-checked, as a new pins file

  from benchmarks.bench.models import fetch
  fetch.scene_path()              # the verified unitree_g1/scene_mjx.xml (FileNotFoundError when not fetched)
  fetch.scene_path(fetch=True)    # fetch first when missing
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

PINS = Path(__file__).with_name("pins.json")
REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "runs" / "bench" / "models"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
MAX_FILE_BYTES = 64 * 2**20
DEFAULT_MODEL = "menagerie_g1"
PIN_RECORD = ".pin.json"


def models_root(root=None):
    """--root, else $DK_BENCH_MODELS, else runs/bench/models of the checkout."""
    return Path(root or os.environ.get("DK_BENCH_MODELS") or DEFAULT_ROOT).expanduser().absolute()


def pins():
    """{model name: pin entry} of pins.json (keys starting with '_' are notes)."""
    return {k: v for k, v in json.loads(PINS.read_text()).items() if not k.startswith("_")}


def pin(name=DEFAULT_MODEL):
    """The pins.json entry of a model."""
    p = pins()
    if name not in p:
        raise KeyError(f"no model {name!r} in {PINS.name} ({sorted(p)})")
    return p[name]


def _safe(path):
    """A pinned path must be relative and stay inside the model directory."""
    p = PurePosixPath(path)
    if p.is_absolute() or not p.parts or any(x in ("", ".", "..") for x in p.parts):
        raise ValueError(f"unsafe pinned path {path!r}")
    return p


def model_dir(entry, root=None):
    return models_root(root) / entry["repo"].split("/")[-1]


def files_digest(entry):
    """sha256 over the 'sha256  path' lines of the pinned files in path order (as `sha256sum | sort -k2`)."""
    lines = "".join(f"{entry['files'][p]}  {p}\n" for p in sorted(entry["files"]))
    return hashlib.sha256(lines.encode()).hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(entry, d):
    """Problems of the copy in directory d (missing files, sha256 mismatches, another pinned commit); [] when it
    matches the pins."""
    bad = []
    try:
        rec = json.loads((Path(d) / PIN_RECORD).read_text())
        if rec.get("commit") != entry["commit"]:
            bad.append(f"fetched at commit {rec.get('commit')}, pinned {entry['commit']}")
    except (OSError, ValueError):
        bad.append(f"no {PIN_RECORD}")
    for p, want in sorted(entry["files"].items()):
        f = Path(d) / _safe(p)
        if not f.is_file():
            bad.append(f"missing {p}")
        elif _sha256(f) != want:
            bad.append(f"sha256 mismatch {p}")
    return bad


def compile_check(path):
    """MuJoCo compiles the entry (also proves the pinned file list is complete); returns (nq, nv, nu)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    return m.nq, m.nv, m.nu


def _download(url, dest, retries=3, timeout=60):
    """Download url to dest; returns its sha256."""
    for attempt in range(retries):
        try:
            h = hashlib.sha256()
            n = 0
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:  # noqa: S310 (https)
                for block in iter(lambda: r.read(1 << 20), b""):
                    n += len(block)
                    if n > MAX_FILE_BYTES:
                        raise ValueError(f"{url}: larger than {MAX_FILE_BYTES} bytes")
                    h.update(block)
                    f.write(block)
            return h.hexdigest()
        except (OSError, ValueError) as e:
            if attempt == retries - 1 or isinstance(e, ValueError):
                raise RuntimeError(f"download failed: {url}: {e}") from e
            time.sleep(2 * (attempt + 1))


def _fetch_into(entry, d, commit=None):
    """Download the pinned files (at commit, default the pinned one) into the empty directory d; {path: sha256}."""
    commit = commit or entry["commit"]
    out = {}
    for p in sorted(entry["files"]):
        dest = Path(d) / _safe(p)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out[p] = _download(RAW_URL.format(repo=entry["repo"], commit=commit, path=p), dest)
    return out


def download(name=DEFAULT_MODEL, root=None, force=False, log=print):
    """Fetch and verify a model; returns its directory. A verified copy is kept; a copy that does not match the pins
    (another commit, a changed file) is an error unless force, which fetches it again."""
    entry = pin(name)
    final = model_dir(entry, root)
    if final.exists() and not force:
        bad = verify(entry, final)
        if not bad:
            return final
        raise RuntimeError(f"{final} does not match {PINS.name}: {bad[:5]} (use --force to fetch again)")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=final.parent))
    try:
        log(f"fetching {len(entry['files'])} files of {entry['repo']}@{entry['commit'][:12]}")
        got = _fetch_into(entry, tmp)
        bad = [p for p, h in got.items() if h != entry["files"][p]]
        if bad:
            raise RuntimeError(f"sha256 mismatch against {PINS.name}: {bad}")
        compile_check(tmp / _safe(entry["entry"]))
        (tmp / PIN_RECORD).write_text(json.dumps(dict(model=name, repo=entry["repo"], commit=entry["commit"],
                                                      files_sha256=files_digest(entry)), indent=1) + "\n")
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o777 & ~umask)  # mkdtemp makes it private
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return final


def scene_path(name=DEFAULT_MODEL, root=None, fetch=False):
    """The entry file (unitree_g1/scene_mjx.xml) of a verified copy; with fetch=True it is fetched when missing."""
    entry = pin(name)
    d = model_dir(entry, root)
    if not d.exists():
        if not fetch:
            raise FileNotFoundError(f"{d} not found: run python -m benchmarks.bench.models.fetch")
        d = download(name, root)
    bad = verify(entry, d)
    if bad:
        raise RuntimeError(f"{d} does not match {PINS.name}: {bad[:5]}")
    return d / _safe(entry["entry"])


def repin(name, commit, out, root=None):
    """Hash the pinned file list at another commit and write a new pins file to out (never pins.json implicitly)."""
    all_pins = json.loads(PINS.read_text())
    old = all_pins[name]
    entry = copy.deepcopy(old)
    parent = models_root(root)
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".repin-", dir=parent))
    try:
        got = _fetch_into(entry, tmp, commit)
        compile_check(tmp / _safe(entry["entry"]))
    finally:
        shutil.rmtree(tmp)
    entry.update(commit=commit, files=got)
    entry["license"]["url"] = entry["license"]["url"].replace(old["commit"], commit)
    entry.pop("commit_date", None)  # unknown here: fill in by hand, with the reason for the new pin
    all_pins[name] = entry
    Path(out).write_text(json.dumps(all_pins, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"pins.json entry (default {DEFAULT_MODEL})")
    ap.add_argument("--root", default=None, help="models directory (default $DK_BENCH_MODELS, else runs/bench/models)")
    ap.add_argument("--check", action="store_true", help="verify an existing copy only (no network)")
    ap.add_argument("--force", action="store_true", help="fetch again even if a copy exists")
    ap.add_argument("--repin", metavar="COMMIT", default=None, help="maintainers: hash the files at COMMIT")
    ap.add_argument("--write-pins", metavar="FILE", default=None, help="with --repin: where to write the new pins")
    args = ap.parse_args(argv)
    try:
        if args.repin:
            if not args.write_pins:
                ap.error("--repin needs --write-pins")
            repin(args.model, args.repin, args.write_pins, args.root)
            print(f"wrote {args.write_pins}")
            return 0
        entry = pin(args.model)
        if args.check:
            d = model_dir(entry, args.root)
            bad = verify(entry, d) if d.exists() else [f"{d} not found"]
            if bad:
                print(f"FAIL {args.model}: {bad}", file=sys.stderr)
                return 1
            nq, nv, nu = compile_check(d / _safe(entry["entry"]))
            print(f"ok {args.model}: {len(entry['files'])} files match {PINS.name}; nq {nq} nv {nv} nu {nu}")
            print(d / _safe(entry["entry"]))
            return 0
        d = download(args.model, args.root, args.force)
        nq, nv, nu = compile_check(d / _safe(entry["entry"]))
        print(f"ok {args.model}: {entry['repo']}@{entry['commit'][:12]}, {len(entry['files'])} files verified; "
              f"nq {nq} nv {nv} nu {nu}; licence {entry['license']['url']}")
        print(d / _safe(entry["entry"]))
        return 0
    except Exception as e:  # noqa: BLE001  (a gate: any failure is exit 1 with its message)
        print(f"FAIL {args.model}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
