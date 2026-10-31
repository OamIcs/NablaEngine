"""Command line of the benchmark truth (python -m benchmarks.bench truth <command>, from the repository root, in the dk
environment; MuJoCo C on the CPU, the model check's native MuJoCo Warp run on the first visible GPU in the mjw-native
environment):

  truth make --corpora mf_s0,mf_s1,mf_s2,mf_s3,menagerie_s0 [--jobs N]     # truth files benchmarks/data/truth/<c>.npz
  truth forward --corpora mf_s0                                            # forward truth + labels only (cached)
  truth verify --corpora mf_s0,...                                         # truth file vs corpus, regenerated truth
  truth index --corpora mf_s0,mf_s1,mf_s2,mf_s3,menagerie_s0               # rebuild truth.json from the truth files

make writes one truth file per corpus (truth/store.py's format; never over a committed file without --force), a
summary JSON (--summary, default <cache-dir>/make_summary.json) and the index <truth dir>/truth.json (benchmark
corpora; truth files of other corpora, e.g. fixture-format ones, are indexed in truth.other.json). index rebuilds
truth.json from the given truth files alone (their meta records, sha256 and size), keeping each entry's run
provenance when truth.json has one. Caches (forward truth and labels, native MuJoCo Warp
labels, FD Jacobians) live in --cache-dir (default runs/bench/truth) and are keyed by the inputs, the model, the
options and the MuJoCo version; --recompute ignores them. --jobs MuJoCo C worker processes (default: the CPUs this
process may run on, os.sched_getaffinity). Truth files go to <data dir>/truth/, the data dir being $DK_BENCH_DATA
(default benchmarks/data; problem.py documents it). Every command exits non-zero on a failure.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmarks.bench import prov
from benchmarks.bench.truth import common


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _parser():
    ap = argparse.ArgumentParser(prog="python -m benchmarks.bench truth", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_args(p):
        p.add_argument("--corpora", type=_csv, required=True, help="comma-separated corpus names")
        p.add_argument("--cache-dir", default=common.CACHE_DEFAULT, help="caches (default runs/bench/truth)")
        p.add_argument("--jobs", type=int, default=None, help="MuJoCo C worker processes (default: the CPUs this "
                                                              "process may run on)")
        p.add_argument("--recompute", action="store_true", help="ignore the caches (they are rewritten)")
        p.add_argument("--model-path", help="model file of the corpora's problem (default: the problem's)")

    mk = sub.add_parser("make", help="produce truth files")
    common_args(mk)
    mk.add_argument("--no-gradient", action="store_true", help="forward truth, labels and model check only")
    mk.add_argument("--no-mjw", action="store_true", help="skip the native MuJoCo Warp model check (model_flags -1)")
    mk.add_argument("--jacobian-groups", type=_csv, default=None, help="groups whose states keep the full Jacobian "
                                                                       "(default G0)")
    mk.add_argument("--sample", type=int, default=None, help="states per group (default 64)")
    mk.add_argument("--seed", type=int, default=None, help="sampling seed (default 0)")
    mk.add_argument("--out-dir", help="directory of the truth files (default <data dir>/truth)")
    mk.add_argument("--force", action="store_true", help="replace a committed truth file")
    mk.add_argument("--summary", help="summary JSON (default <cache-dir>/make_summary.json)")
    fw = sub.add_parser("forward", help="forward truth and labels (cached)")
    common_args(fw)
    vf = sub.add_parser("verify", help="check truth files against their corpora and a regenerated forward truth")
    common_args(vf)
    ix = sub.add_parser("index", help="rebuild truth.json from the truth files")
    ix.add_argument("--corpora", type=_csv, required=True, help="comma-separated corpus names")
    ix.add_argument("--made-by", default=None, help="the command that made the files (default: as recorded, else "
                                                    "truth make --corpora <the corpora>)")
    for mod in EXTRA.values():  # subcommands of optional modules (present only where the module is)
        mod.add_parser(sub, common_args)
    return ap


CORE = ("__init__", "cli", "common", "mjc", "mjw_labels", "store")


def _extra_commands():
    """Additional subcommands: {name: module} of the other modules benchmarks/bench/truth/<module>.py that define, at
    top level, add_parser(sub, common_args) and cli(args, cache) -> int (found from the source, imported only then;
    the subcommand is the module name with '-' for '_')."""
    import ast
    import importlib
    out = {}
    for f in sorted(Path(__file__).resolve().parent.glob("*.py")):
        if f.stem in CORE or f.stem.startswith("_"):
            continue
        try:
            names = {n.name for n in ast.parse(f.read_text()).body if isinstance(n, ast.FunctionDef)}
        except (OSError, SyntaxError):
            continue
        if {"add_parser", "cli"} <= names:
            out[f.stem.replace("_", "-")] = importlib.import_module(f"benchmarks.bench.truth.{f.stem}")
    return out


EXTRA = {}
INDEX = "truth.json"
INDEX_OTHER = "truth.other.json"


def _index_entry(s, command):
    return dict(file=s["file"], sha256=s["sha256"], bytes=s["bytes"], labels=s["labels"], groups=s["groups"],
                model_check=s["model_check"], model_ok=s["model_ok"], gradient=s["gradient"], command=command,
                provenance=s["provenance"])


def _write_index(path, entries, replace=False):
    """Merge {corpus: entry} into an index file (replace: the given entries only)."""
    try:
        idx = {} if replace else json.loads(path.read_text())
    except (OSError, ValueError):
        idx = {}
    idx["format"] = "dk-bench-truth-index/1"
    idx["note"] = ("Truth files of the benchmark corpora (benchmarks/bench/truth/store.py documents the format; "
                   "python -m benchmarks.bench truth verify checks them)")
    idx.setdefault("corpora", {}).update(entries)
    idx["corpora"] = dict(sorted(idx["corpora"].items()))
    path.write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n")


def write_index(d, summaries, argv):
    """Merge the make summaries into <truth dir>/truth.json (benchmark corpora) or truth.other.json (other corpora):
    per corpus its truth file's sha256, size, counts, the command and the run's provenance (the truth files
    themselves hold only what their numbers depend on)."""
    cmd = prov.scrub("python -m benchmarks.bench truth " + " ".join(argv if argv is not None else sys.argv[1:]))
    for name, keep in ((INDEX, True), (INDEX_OTHER, False)):
        entries = {s["corpus"]: _index_entry(s, cmd) for s in summaries if (s.get("kind", "bench") == "bench") == keep}
        if entries:
            _write_index(Path(d) / name, entries)


def rebuild_index(d, corpora, made_by=None):
    """truth.json of the given corpora from their truth files alone (store.summary_of), keeping each entry's recorded
    run provenance (date, Python, CPU) when the current truth.json has one; the harness commit is not kept."""
    from benchmarks.bench.truth import store
    path = Path(d) / INDEX
    try:
        old = json.loads(path.read_text()).get("corpora") or {}
    except (OSError, ValueError):
        old = {}
    cmd = made_by or ("python -m benchmarks.bench truth make --corpora " + ",".join(corpora))
    entries = {}
    for name in corpora:
        s = store.summary_of(store.load(name))
        if s.get("kind") != "bench":
            raise ValueError(f"{name}: not a benchmark corpus (kind {s.get('kind')!r}); truth.json lists benchmark "
                             f"corpora only")
        pv = dict((old.get(name) or {}).get("provenance") or {})
        pv.pop("harness", None)
        s["provenance"] = pv or None
        entries[name] = _index_entry(s, cmd)
    _write_index(path, entries, replace=True)
    return entries


def main(argv=None):
    EXTRA.update(_extra_commands())
    a = _parser().parse_args(argv)
    from benchmarks.bench.truth import store
    cache = common.cache_dir(getattr(a, "cache_dir", None))
    if a.cmd == "make":
        pools, summ = {}, []
        try:
            for name in a.corpora:
                out = Path(a.out_dir) / f"{name}.npz" if a.out_dir else None
                s = store.make(name, cache=cache, jobs=a.jobs, pool_cache=pools, gradient=not a.no_gradient,
                               mjw=not a.no_mjw, jacobian_groups=tuple(a.jacobian_groups or store.JACOBIAN_GROUPS),
                               sample_cap=a.sample, seed=a.seed, recompute=a.recompute, out=out, force=a.force,
                               model_path=a.model_path)
                print(json.dumps(s), flush=True)
                summ.append(s)
        finally:
            for p in pools.values():
                p.close()
        path = Path(a.summary) if a.summary else cache / "make_summary.json"
        path.write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d"), harness=prov.harness_commit(),
                                        corpora=summ), indent=1))
        write_index(Path(a.out_dir) if a.out_dir else common.truth_dir(), summ, argv)
        return 0
    if a.cmd == "forward":
        from benchmarks.bench import problem as P
        for name in a.corpora:
            c = P.load_corpus(name)
            prob = common.problem_of(c, a.model_path)
            desc = common.check_model(c, prob)
            lab, info = store.forward_labels(c, prob, desc, cache, jobs=a.jobs or len(os.sched_getaffinity(0)),
                                             recompute=a.recompute)
            print(json.dumps(dict(corpus=name, states=len(lab["out"]), **info)), flush=True)
        return 0
    if a.cmd == "verify":
        bad = 0
        for name in a.corpora:
            try:
                tr = store.load(name)
                fw = tr.forward(cache=cache, jobs=a.jobs, recompute=a.recompute)
                res = dict(corpus=name, ok=True, forward=fw["_info"], file=prov.rel(tr.path),
                           bytes=tr.path.stat().st_size)
                if tr.has_gradient:
                    g = tr.gradient()
                    same = bool((fw["out"][g["index"]] == g["out"]).all())
                    res["fd_base_bitwise"] = same
            except Exception as e:  # noqa: BLE001
                bad += 1
                res = dict(corpus=name, ok=False, error=prov.scrub(f"{type(e).__name__}: {e}"))
            print(json.dumps(res), flush=True)
        return 1 if bad else 0
    if a.cmd == "index":
        entries = rebuild_index(common.truth_dir(), a.corpora, a.made_by)
        for k, e in entries.items():
            print(json.dumps(dict(corpus=k, file=e["file"], sha256=e["sha256"], bytes=e["bytes"],
                                  model_ok=e["model_ok"])), flush=True)
        return 0
    if a.cmd in EXTRA:
        return int(EXTRA[a.cmd].cli(a, cache) or 0)
    return 2


if __name__ == "__main__":
    sys.exit(main())
