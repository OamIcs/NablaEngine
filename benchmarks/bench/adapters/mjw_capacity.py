"""The documented capacity procedure of the MuJoCo Warp adapters, measurement half: how many contact-pool
entries and constraint rows MuJoCo Warp itself needs on every corpus of a problem, stored in
benchmarks/data/mjw_capacity.json, from which mjw_common.capacity() sizes each unit:

  naconmax = ceil(nworld * mu * (1 + margin)) + c_max,  nconmax = ceil(mu * (1 + margin)),
  njmax    = the smallest of MuJoCo Warp's valid sizes (16, 24, 32, 48, 64, 96, 128, 192, ...) >= r_max

mu: the largest mean number of pool entries per world over every corpus, iteration count and substep, where the pool
(naconmax entries, shared by all worlds) holds both the broadphase's candidate pairs (d.ncollision) and the contacts
(d.nacon), so an entry count is max(ncollision, nacon); c_max: the most contacts of one world in one substep (MuJoCo
Warp has no public per-world candidate count); r_max: the most constraint rows of one world in one substep; margin 0.25.

Measurement: for each corpus of the problem and each iteration count, all its states in one batch (the adapter's own
setup and forward: per-world DR, the problem's options), one control step with a generous capacity (--nconmax /
--njmax per world, default 128 / 512), after every substep: ncollision, nacon, contacts per world (contact.worldid),
rows (nefc). A measurement that overflows its own capacity fails (raise --nconmax / --njmax).
Verification (--verify, default on): the procedure's capacity on every state subset the harness runs
(benchmarks.bench.protocol: nworld in SWEEP, its subsets per nworld, tiled beyond the corpus) of every corpus at the
first iteration count; any overflow fails (raise --margin). Recorded per problem as `verified`.

Run it in the environment of the engine it measures (default mjw, the PyPI release, environment mjw-native), from the
repository root:

  CUDA_VISIBLE_DEVICES=<one GPU> PYTHONPATH=. runs/bench/envs/mjw-native/venv/bin/python \
      -m benchmarks.bench.adapters.mjw_capacity --out benchmarks/data/mjw_capacity.json [--force]

Defaults: problem mjlab with every Track A corpus of benchmarks/data/corpora.json (--corpora to choose others),
iterations 10 and 50; problem menagerie_g1 (if its model is fetched) with the
Track B corpora, the model's own iterations and 10. --out refuses to overwrite a file tracked by git without --force.
Exit status 1 when a measurement or the verification fails (nothing is written then).
"""

import argparse
import datetime
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from benchmarks.bench import adapters, prov
from benchmarks.bench import problem as P
from benchmarks.bench.adapters import mjw_common

SCHEMA = "dk-bench-mjw-capacity/1"
ITERATIONS = {"mjlab": (10, 50), "menagerie_g1": (None, 10)}


def default_corpora(prob):
    """The corpora of a problem's track: benchmarks/data/corpora.json's entries of that track, those that exist."""
    cj = P.data_dir() / "corpora.json"
    try:
        doc = json.loads(cj.read_text())
        names = [k for k, v in (doc.get("corpora") or {}).items() if v.get("track") == prob.track]
    except (OSError, ValueError):
        names = [f"mf_s{i}" for i in range(4)] if prob.track == "A" else ["menagerie_s0"]
    return [c for c in dict.fromkeys(names) if P.corpus_exists(c)]


def _opts(cls, **kw):
    if "tape_evidence" in getattr(cls, "OPTIONS", {}):
        kw["tape_evidence"] = False
    return kw


def _run(cls, prob, states, nworld, iterations, **opts):
    """setup + one diagnostics forward; (adapter stats, diagnostics, provenance)."""
    a = cls()
    a.setup(prob, states, nworld, iterations, **_opts(cls, **opts))
    a.outputs()
    diag = a.diagnostics()
    out = a.stats, diag, a.provenance()
    a.close()
    return out


def measure(engine, prob, corpus, iterations, nconmax, njmax):
    """The measurement of one corpus at one iteration count (module docstring) with the adapter of `engine`."""
    cls = adapters.load(engine)
    c = P.load_corpus(corpus)
    s, diag, pv = _run(cls, prob, P.select(c, c.n, 0), c.n, iterations, capacity="explicit", nconmax=nconmax,
                       njmax=njmax)
    n = c.n
    cand = [None if x is None else x / n for x in s["ncollision"]]
    out = dict(n=n, sha256=c.files[0]["sha256"], contacts_mean_per_substep=[x / n for x in s["nacon"]],
               candidates_mean_per_substep=cand,
               pool_mean_per_substep=[max(a / n, b or 0.0) for a, b in zip(s["nacon"], cand)],
               contacts_world_max=int(s["ncon"].max()), rows_max=int(s["nefc"].max()),
               rows_mean=float(s["nefc"].mean()), overflow_worlds=int(diag["overflow"]),
               overflow_bits=diag["extra"]["overflow_bits"], capacity=dict(nconmax=nconmax, njmax=njmax))
    if out["overflow_worlds"]:
        raise RuntimeError(f"{corpus} at {iterations} iterations overflows the measurement capacity {nconmax} / "
                           f"{njmax} in {out['overflow_worlds']} worlds ({out['overflow_bits']}): raise --nconmax / "
                           f"--njmax")
    return out, pv


def summarize(per_corpus, margin):
    """The problem's mu, c_max, r_max and the derived per-world sizes from the per-corpus measurements."""
    rows = [m for c in per_corpus.values() for m in c["iterations"].values()]
    mu = max(max(m["pool_mean_per_substep"]) for m in rows)
    cmax = max(m["contacts_world_max"] for m in rows)
    rmax = max(m["rows_max"] for m in rows)
    return dict(pool_mean_max=mu, contacts_mean_max=max(max(m["contacts_mean_per_substep"]) for m in rows),
                candidates_mean_max=max(max(x or 0.0 for x in m["candidates_mean_per_substep"]) for m in rows),
                contacts_world_max=cmax, rows_max=rmax, margin=margin,
                nconmax_per_world=int(math.ceil(mu * (1 + margin))), njmax=mjw_common.valid_size(rmax),
                naconmax_4096=int(math.ceil(4096 * mu * (1 + margin))) + cmax)


def harness_units(n_states):
    """[(nworld, subset)] the harness runs on a corpus of n_states states (protocol.SWEEP and its subset policy)."""
    from benchmarks.bench import protocol
    out = []
    for nw in protocol.SWEEP:
        k = protocol._subsets(n_states, nw, None)
        out += [(nw, s) for s in range(k)]
    return out


def verify(engine, prob, corpora, iterations, doc):
    """The procedure's capacity (from doc, written to a temporary file) on every harness subset of every corpus:
    {corpus: {"<nworld>.s<subset>": {overflow worlds, pool peak share, rows max}}}, and the total overflow."""
    cls = adapters.load(engine)
    res, bad = {}, 0
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="mjw_capacity_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f)
        for corpus in corpora:
            c = P.load_corpus(corpus)
            res[corpus] = {}
            for nw, sub in harness_units(c.n):
                s, diag, pv = _run(cls, prob, P.select(c, nw, sub), nw, iterations, capacity="procedure",
                                   capacity_file=tmp)
                r = dict(overflow_worlds=int(diag["overflow"]), bits=diag["extra"]["overflow_bits"],
                         pool_peak_share=round(diag["extra"]["pool_peak_share"], 4),
                         rows_max=int(diag["nefc_max"]), naconmax=pv["capacity"]["naconmax"],
                         njmax=pv["capacity"]["njmax"])
                bad += r["overflow_worlds"]
                res[corpus][f"{nw}.s{sub}"] = r
                print(f"[mjw_capacity] verify {prob.id} {corpus} nworld {nw} subset {sub}: overflow "
                      f"{r['overflow_worlds']}, pool peak {r['pool_peak_share']:.3f} of {r['naconmax']}, rows max "
                      f"{r['rows_max']} of {r['njmax']}", file=sys.stderr)
    finally:
        os.unlink(tmp)
    return res, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", default="mjw", help="the adapter that measures (default mjw: PyPI 3.17.0)")
    ap.add_argument("--problems", default="mjlab,menagerie_g1",
                    help="problems (a problem whose model is missing is skipped with a note)")
    ap.add_argument("--corpora", action="append", default=[],
                    help="PROBLEM=C1,C2: the corpora of a problem (default: default_corpora)")
    ap.add_argument("--iterations", action="append", default=[],
                    help="PROBLEM=10,50: iteration counts ('native' = the model's own)")
    ap.add_argument("--model-path", action="append", default=[], help="PROBLEM=PATH")
    ap.add_argument("--nconmax", type=int, default=128, help="measurement capacity: contacts per world")
    ap.add_argument("--njmax", type=int, default=512, help="measurement capacity: rows per world")
    ap.add_argument("--margin", type=float, default=mjw_common.MARGIN)
    ap.add_argument("--no-verify", action="store_true", help="skip the verification on the harness's subsets")
    ap.add_argument("--out", default=mjw_common.CAPACITY_FILE)
    ap.add_argument("--force", action="store_true", help="overwrite a file tracked by git")
    a = ap.parse_args(argv)

    def kv(items):
        return dict(it.partition("=")[::2] for it in items)

    corpora, iters, paths = kv(a.corpora), kv(a.iterations), kv(a.model_path)
    out_p = Path(a.out) if Path(a.out).is_absolute() else mjw_common.REPO / a.out
    if prov.is_committed(out_p) and not a.force:
        print(f"mjw_capacity: {prov.rel(out_p)} is tracked by git: pass --force to overwrite it", file=sys.stderr)
        return 2
    doc = dict(schema=SCHEMA, rule="naconmax = ceil(nworld mu (1 + margin)) + c_max; nconmax = ceil(mu (1 + margin)); "
                                   "njmax = the smallest MuJoCo Warp valid size >= r_max (mu: the largest mean "
                                   "max(broadphase candidates, contacts) per world over corpora, iterations and "
                                   "substeps; c_max: most contacts of one world; r_max: most rows of one world)",
               margin=a.margin, problems={}, skipped={},
               date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               command=" ".join(["python -m benchmarks.bench.adapters.mjw_capacity"] + list(
                   sys.argv[1:] if argv is None else argv)))
    measured_with = None
    ok = True
    for pid in [p for p in a.problems.split(",") if p]:
        prob = P.get(pid, model_path=paths.get(pid))
        if not prob.path().exists():
            doc["skipped"][pid] = f"model {prov.rel(prob.path())} not found (python -m benchmarks.bench fetch-models)"
            print(f"[mjw_capacity] {pid}: {doc['skipped'][pid]}", file=sys.stderr)
            continue
        names = corpora[pid].split(",") if pid in corpora else default_corpora(prob)
        its = ([None if x == "native" else int(x) for x in iters[pid].split(",")] if pid in iters
               else list(ITERATIONS.get(pid, (10,))))
        per = {}
        for c in names:
            per[c] = dict(iterations={})
            for it in its:
                try:
                    m, pv = measure(a.engine, prob, c, it, a.nconmax, a.njmax)
                except Exception as e:  # noqa: BLE001
                    ok = False
                    print(f"[mjw_capacity] {pid} {c} it {it}: {type(e).__name__}: {e}", file=sys.stderr)
                    continue
                measured_with = measured_with or dict(engine=a.engine, mujoco_warp=pv["mujoco_warp"],
                                                      versions=pv["versions"], device=pv["device"])
                per[c]["sha256"], per[c]["n"] = m.pop("sha256"), m.pop("n")
                per[c]["iterations"]["native" if it is None else str(it)] = m
                print(f"[mjw_capacity] {pid} {c} it {it}: pool entries / world "
                      f"{[round(x, 3) for x in m['pool_mean_per_substep']]} (contacts "
                      f"{[round(x, 3) for x in m['contacts_mean_per_substep']]}), contacts max "
                      f"{m['contacts_world_max']}, rows max {m['rows_max']} mean {m['rows_mean']:.1f}", file=sys.stderr)
        per = {k: v for k, v in per.items() if v["iterations"]}
        if not per:
            continue
        doc["problems"][pid] = dict(summarize(per, a.margin), problem=prob.describe(), corpora=per)
        if not a.no_verify and ok:
            try:
                res, bad = verify(a.engine, prob, list(per), its[0], doc)
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[mjw_capacity] verify {pid}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            doc["problems"][pid]["verified"] = dict(iterations="native" if its[0] is None else its[0],
                                                    overflow_worlds=bad, units=res)
            if bad:
                ok = False
                print(f"[mjw_capacity] verify {pid}: {bad} overflowing worlds: raise --margin", file=sys.stderr)
    doc["measured_with"] = measured_with
    if not ok:
        print("[mjw_capacity] a measurement or the verification failed: nothing written", file=sys.stderr)
        return 1
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("problem", "corpora", "verified")}
                      for k, v in doc["problems"].items()}, indent=1))
    print(f"[mjw_capacity] wrote {prov.rel(out_p)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
