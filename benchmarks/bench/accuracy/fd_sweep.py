"""MuJoCo C finite differences across step sizes: convergence against the float64 truth, and cost. CPU only (numpy,
mujoco); the FD constructions are the mjc adapter's (benchmarks/bench/adapters/mjc_fd.py), so the sweep measures the
code the benchmark times.

Grid: eps in EPS (1e-3 ... 1e-9) x forward / centered differences x construction
  chain    MuJoCo C's native derivative: mjd_transitionFD per substep, chained over the control step
  direct   FD of the whole control step through mujoco.batch.Batch rollouts (the truth's construction; reference)
x options (OPTIONS): headline (10 iterations, tolerance 1e-8, ls 8: the headline problem) and tight (2000, 1e-13, 100:
the truth's options; direct centered tight at 1e-6 is the truth itself, reference_relation "self-reference").
States: the truth file's sampled states of --groups (default G0, G4, G5, G6), all of them or the model-ok ones
(--subset). Each Jacobian is scored with accuracy/gradient.py (the VJPs of the corpus's stored cotangents; the
floored and the raw relative error per block). Cost per configuration: wall seconds for the batch of states at
--threads, ms per world, and mj_step evaluations per world without stage skipping (forward S (1 + 2 nv + nu), centered
S (1 + 2 (2 nv + nu)) for S substeps; the chain's stage skipping makes its real cost lower, hence the measured time).

  python -m benchmarks.bench.accuracy.fd_sweep --corpora mf_s0,mf_s1 --threads 24 --out runs/bench/fd_sweep.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov
from benchmarks.bench.accuracy import gradient as AG

EPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
OPTIONS = dict(headline=dict(iterations=10, tolerance=1e-8, ls_iterations=8),
               tight=dict(iterations=2000, tolerance=1e-13, ls_iterations=100))
MODES = ("chain_forward", "chain_centered", "direct_forward", "direct_centered")
DIRECT_CHUNK = 96  # worlds per Batch of the direct construction


def subset_truth(GT, mask):
    """The gradient truth GT (accuracy/gradient.py) restricted to the sampled states mask (N,) bool."""
    out = dict(GT)
    pos = np.flatnonzero(mask)
    for k in ("index", "fd", "floor", "bad", "model_ok", "clamped", "axis_lt_1mm", "cot_qpos", "cot_qvel"):
        if GT.get(k) is not None:
            out[k] = GT[k][pos]
    out["groups"] = {g: m[pos] for g, m in GT["groups"].items()}
    if GT.get("jacobian") is not None and len(GT["jac_index"]):
        new = -np.ones(len(mask), np.int64)
        new[pos] = np.arange(len(pos))
        keep = mask[GT["jac_index"]]
        out["jac_index"] = new[GT["jac_index"][keep]]
        out["jacobian"] = GT["jacobian"][keep]
        out["jacobian_dcol"] = GT["jacobian_dcol"][keep]
    return out


def evaluations(nsub, nv, nu, centered):
    ncol = 2 * nv + nu
    return nsub * (1 + (2 if centered else 1) * ncol)


def sweep(GT, prob, threads, eps=EPS, options=OPTIONS, modes=MODES, subsets=("all", "model_ok"), log=print):
    """{f"{option}/{mode}/{eps:g}": result} on the states of GT (all of them), each with sec, ms_per_world,
    evaluations_per_world, nonfinite, reference_relation and the gradient scorer's entries (compact: per group,
    subset and block the floored statistics and the raw ones)."""
    from benchmarks.bench.adapters import mjc_fd
    corpus = P.load_corpus(GT["name"])
    W = GT["index"]
    s = {k: np.asarray(v[W], np.float64) for k, v in corpus.arrays.items()}
    out = {}
    for oname, opt in options.items():
        m = P.load_model(prob, iterations=int(opt["iterations"]))
        m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations = (int(opt["iterations"]), float(opt["tolerance"]),
                                                                   int(opt["ls_iterations"]))
        dr = P.DR(m, prob)
        chain = mjc_fd.ChainFD(m, dr, s, prob.substeps, threads)
        direct = {c: mjc_fd.DirectFD(m, dr, s, prob.substeps, threads, c, DIRECT_CHUNK) for c in (False, True)}
        try:
            for mode in modes:
                construction, kind = mode.split("_")
                centered = kind == "centered"
                for h in eps:
                    t0 = time.perf_counter()
                    if construction == "chain":
                        J, _ = chain.jacobians(h, centered)
                    else:
                        J, _ = direct[centered].jacobians(h)
                    dt = time.perf_counter() - t0
                    fd = dict(construction=construction, centered=centered, eps=h, options=oname)
                    src = dict(engine="mjc", variant=f"fd_{mode}", index=W, jacobian=J, fd=fd)
                    ent, _ = AG.score(src, GT, subsets=subsets)
                    res = dict(sec=round(dt, 4), ms_per_world=round(1e3 * dt / len(W), 4), threads=int(threads),
                               states=int(len(W)), evaluations_per_world=evaluations(prob.substeps, m.nv, m.nu,
                                                                                     centered),
                               nonfinite=int((~np.isfinite(J)).sum()), reference_relation=AG.reference_relation(fd),
                               groups={})
                    for e in ent:
                        if e.get("status") or e["object"] != "vjp":
                            continue
                        res["groups"].setdefault(e["group"], {}).setdefault(e["subset"], {})[e["block"]] = dict(
                            {k: e.get(k) for k in ("n", "median", "p90", "p99", "max", "cos_min")},
                            raw={k: e["raw"].get(k) for k in ("median", "p90", "p99", "max")},
                            states=e["usable_states"])
                    key = f"{oname}/{mode}/{h:g}"
                    out[key] = res
                    g0 = res["groups"].get("G0", {}).get(subsets[0], {})
                    raw = {b: g0[b]["raw"] for b in AG.BLOCKS if b in g0}
                    nan = float("nan")
                    log(f"[fd_sweep] {GT['name']} {key}: {dt:.2f} s  G0 " + "  ".join(
                        f"{b} {r.get('median', nan):.1e}/{r.get('p90', nan):.1e}" for b, r in raw.items()))
        finally:
            chain.close()
            for d in direct.values():
                d.close()
    return out


def parser():
    """The command line (a caller with its own truth format extends it and passes it to main)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpora", required=True, help="comma-separated corpus names with a gradient truth")
    ap.add_argument("--groups", default="G0,G4,G5,G6", help="truth groups whose sampled states are swept")
    ap.add_argument("--subset", default="all", choices=("all", "model_ok"), help="all sampled states or model-ok ones")
    ap.add_argument("--threads", type=int, default=None, help="threads (default: the CPUs this process may run on)")
    ap.add_argument("--eps", default=",".join(f"{e:g}" for e in EPS))
    ap.add_argument("--options", default=",".join(OPTIONS), help="option sets: headline, tight")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--max-states", type=int, default=None, help="at most this many states per corpus (the first)")
    ap.add_argument("--out", required=True, help="JSON of the sweep")
    return ap


def store_truth(name, args=None):
    """(gradient truth, problem) of corpus `name` from its truth file (truth/store.py)."""
    from benchmarks.bench.truth import store
    tr = store.load(name)
    return AG.truth_from_store(tr), tr.problem()


def main(argv=None, ap=None, load_truth=store_truth):
    """ap: the command line (default parser()); load_truth(name, args) -> (gradient truth as accuracy/gradient.py's,
    problem): the truth of each corpus (default its truth file)."""
    a = (ap or parser()).parse_args(argv)
    threads = a.threads or len(os.sched_getaffinity(0))
    eps = tuple(float(x) for x in a.eps.split(","))
    opts = {k: OPTIONS[k] for k in a.options.split(",")}
    modes = tuple(a.modes.split(","))
    groups = a.groups.split(",")
    res = dict(threads=threads, eps=list(eps), options=opts, modes=list(modes), groups=groups, subset=a.subset,
               corpora={}, construction="benchmarks/bench/adapters/mjc_fd.py (ChainFD, DirectFD)")
    t_all = time.perf_counter()
    for name in a.corpora.split(","):
        GT, prob = load_truth(name, a)
        mask = np.zeros(len(GT["index"]), bool)
        for g in groups:
            mask |= GT["groups"][g]
        if a.subset == "model_ok":
            mask &= GT["model_ok"]
        if a.max_states is not None:
            mask &= np.cumsum(mask) <= a.max_states
        sub = subset_truth(GT, mask)
        sub["groups"] = {g: m for g, m in sub["groups"].items() if g in groups}
        r = sweep(sub, prob, threads, eps, opts, modes, subsets=(a.subset,))
        res["corpora"][name] = dict(truth=GT["path"], states=int(mask.sum()),
                                    group_states={g: int(m.sum()) for g, m in sub["groups"].items()}, results=r)
    res["seconds"] = round(time.perf_counter() - t_all, 1)
    import mujoco
    res["provenance"] = dict(mujoco=mujoco.__version__, numpy=np.__version__, harness=prov.harness_commit(),
                             cpu=(prov.cpu() or {}).get("model"), affinity=len(os.sched_getaffinity(0)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
