"""Parts `genesis`, `rep<r>_genesis`: the native Genesis 1.4.5 forward (genesis-world from PyPI, quadrants 1.3.3,
unmodified) per control step on the mjlab G1, from corpus states.

The problem and the scene are those of the benchmark harness's Genesis adapter (benchmarks/bench/adapters/genesis.py,
which documents them and every deviation it records): the problem's model written as an MJCF with the 502 explicit
contact pairs declared through contype / conaffinity (Genesis's importer reads no <pair>; its default collision filter
gives the same 502 pairs, checked), Newton, 10 iterations, ls 8, tolerance 1e-8, implicitfast, constraint_timeconst =
the model's solref 0.02, SimOptions(dt = 0.02, substeps = 4), PD targets held (control_dofs_position), per-world foot
friction (per-world friction ratios) and torso COM (set_links_COM). On sm_70 the adapter selects quadrants' monolith
constraint-solver kernel (QD_PERFDISPATCH_FORCE), because the other one does not compile there; recorded.
  - timed call: the adapter's fwd op = [set_qpos(corpus qpos), set_dofs_velocity(corpus qvel),
    control_dofs_position(corpus ctrl), scene.step()] with the corpus inputs resident on the device, i.e. the state
    load and one control step of 4 substeps (as the other engines' timed calls, which copy the corpus inputs into
    their state first); sync = quadrants' sync + torch.cuda.synchronize(); common.timeit rounds of N calls and one
    sync, plus the latency of single synchronized calls
  - build (MJCF conversion, scene build with its first step and kernel compile or cache load) and the first call are
    timed separately
Two configurations, each in one fresh process per corpus (this script re-runs itself): gs.init's defaults (the
configuration of the accuracy tables; the headline row) and gs.init(performance_mode=True) (Genesis's documented
static-array mode, labelled); it runs in the Genesis environment
($PAPER_BENCH_GENESIS_PYTHON) with the repository root on PYTHONPATH.

  python scripts/paper_bench/genesis_native.py --out <dir>/genesis.json --corpora mf_s0,mf_s1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PKGS = ("genesis-world", "quadrants", "torch", "mujoco", "numpy")


def child(a):
    """One corpus in this process: {cell key: cell} and the adapter's provenance."""
    sys.path.insert(0, str(C.REPO))
    from benchmarks.bench import problem as P
    from benchmarks.bench.adapters.genesis import GenesisAdapter
    prob = P.get("mjlab")
    st = P.select(P.load_corpus(a.corpus), nworld=a.nworld, subset=0)
    ad = GenesisAdapter()
    t0 = time.perf_counter()
    setup = ad.setup(prob, st, a.nworld, a.iterations, performance_mode=a.mode == "performance")
    setup_s = time.perf_counter() - t0
    fwd = ad.ops()["fwd"]
    t0 = time.perf_counter()
    fwd()
    ad.sync()
    first_s = time.perf_counter() - t0
    r = C.timeit(fwd, ad.sync, latency=0 if C.QUICK else 10)
    outs = ad.outputs()
    import numpy as np
    finite = int(np.isfinite(np.concatenate([outs["qpos"], outs["qvel"]], 1)).all(1).sum())
    cell = dict(gs_init="gs.init(performance_mode=True)" if a.mode == "performance" else "gs.init defaults",
                corpus=dict(name=a.corpus, states=int(st.n) if hasattr(st, "n") else a.nworld, nworld=a.nworld,
                            selection="benchmarks.bench.problem.select subset 0 (every corpus state)"),
                forward=r, setup_s=round(setup_s, 3), setup_parts={k: round(v, 3) for k, v in setup.items()},
                first_call_s=round(first_s, 3), finite_worlds=finite,
                timed_call=("set_qpos + set_dofs_velocity + control_dofs_position + scene.step() (4 substeps), "
                            "inputs resident on the device"))
    try:
        import torch
        cell["torch_max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    prov = ad.provenance()
    ad.close()
    Path(a.out).write_text(C.scrub_text(json.dumps(dict(cell=cell, provenance=prov), default=C._default)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default=None)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--label", default="genesis")
    ap.add_argument("--modes", default="default,performance",
                    help="default: gs.init's defaults (the accuracy tables' configuration, the headline row); "
                         "performance: gs.init(performance_mode=True), Genesis's documented static-array mode")
    ap.add_argument("--child-corpus", dest="corpus", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-mode", dest="mode", default="default", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.corpus:
        return child(a)
    res = C.Result(a.out, part=a.label, nworld=a.nworld, iterations=a.iterations, substeps=4,
                   engine="Genesis 1.4.5 (native)", env=C.basic_env(PKGS),
                   processes="one fresh process per corpus")
    out = Path(a.out)
    cps = C.corpora(a.corpora)
    for name, mode in [(n, m) for n in cps for m in a.modes.split(",")]:
        key = f"{name}/nw{a.nworld}" if mode == "default" else f"{name}/{mode}/nw{a.nworld}"
        if Path(cps[name]).resolve().parent != C.MF_DIR.resolve():
            res.cell(key, dict(skipped="not a benchmark corpus (the Genesis adapter reads benchmarks/data corpora)"))
            continue
        tmp = out.with_name(f"{out.stem}.{name}.{mode}.child.json")
        cmd = [sys.executable, __file__, "--out", str(tmp), "--nworld", str(a.nworld), "--iterations",
               str(a.iterations), "--child-corpus", name, "--child-mode", mode]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, env=os.environ.copy())
        try:
            if p.returncode != 0 or not tmp.exists():
                raise RuntimeError(f"the corpus process exited with {p.returncode} (see the part's log)")
            d = json.loads(tmp.read_text())
            d["cell"]["process_s"] = round(time.perf_counter() - t0, 1)
            res.data.setdefault("provenance", d["provenance"])
            res.cell(key, d["cell"])
        except Exception as e:  # noqa: BLE001
            if C.is_oom(e):
                C.record_oom(res, key, e, a.nworld)
            else:
                res.error(key, e)
        finally:
            if tmp.exists():
                tmp.unlink()
    return 1 if res.data["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
