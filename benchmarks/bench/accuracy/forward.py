"""Forward accuracy scorer: an engine's control-step outputs (qpos', qvel') on corpus states against the MuJoCo C
float64 forward truth (truth/store.py) or against another run, per state subset, with MuJoCo C's solver-truncation
floors. numpy and mujoco (mj_differentiatePos); the truth's forward arrays come from its cache or are regenerated.

Per world (rows of the source matched to corpus states by its `index`):
  qpos_rad   |dq[3:]|, dq = mj_differentiatePos(model, qpos_ref, qpos, 1): the base rotation and the hinge angles in
             the tangent space, in radians (without the base translation, whose unit differs)
  qpos_m     |dq[:3]|: the base translation, in metres
  qvel       |qvel - qvel_ref| / max(|qvel_ref|, 1e-9)
  qpos_rel   |qpos - qpos_ref| / max(|qpos_ref|, 1e-9) in qpos form (kept for comparison)
A world with a non-finite output is counted (nonfinite) and left out of the statistics. Statistics (stats.summary):
median, p90, p99, max, share above 1e-3, n.

Subsets (DEFAULT_SUBSETS): all; model_ok (the truth's model check against native MuJoCo Warp 3.17.0; the excluded
count is recorded); the groups G0, G4, G5, G6, G1 of the truth's MuJoCo C-only labels over all their states;
"<G>&model_ok".
References: "truth" (the forward truth), or any other source (e.g. the same engine's second run: run-to-run noise, or
native MuJoCo Warp at the same iterations). Floors (floor_sources): MuJoCo C float64 at 10 and 50 iterations
(tolerance 1e-8, ls 8) and, on Track B, at the model's own options, against the truth: pure solver truncation.

A source is a dict {index (B,), qpos (B, nq), qvel (B, nv)} (source_from_dump reads the harness's raw dump: index,
out_qpos, out_qvel) plus optional metadata engine, variant, label, iterations.

  from benchmarks.bench.accuracy import forward as AF
  from benchmarks.bench.truth import store
  tr = store.load("mf_s0")
  src = AF.source_from_dump("runs/bench/x/raw/<cell>.npz", engine="dk", iterations=10)
  entries, per_world = AF.score(src, tr)
  floors = [AF.score(s, tr)[0] for s in AF.floor_sources(tr).values()]

  python -m benchmarks.bench.accuracy.forward --truth mf_s0 --raw DUMP --engine dk --iterations 10 --floors --out X.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov
from benchmarks.bench.accuracy import stats

METRICS = ("qpos_rad", "qpos_m", "qvel", "qpos_rel")
DEFAULT_SUBSETS = ("all", "model_ok", "G0", "G4", "G5", "G6", "G1", "G0&model_ok")
FLOOR_LABELS = dict(it10="MuJoCo C float64, 10 iterations (solver truncation floor)",
                    it50="MuJoCo C float64, 50 iterations (solver truncation floor)",
                    native="MuJoCo C float64, the model's own solver options (solver truncation floor)")


def source_from_dump(path_or_dict, **meta):
    """A source from a raw dump (npz path or dict with index, out_qpos, out_qvel); meta: engine, variant, label,
    iterations, ..."""
    d = path_or_dict
    if not isinstance(d, dict):
        p = Path(d)
        p = p if p.is_absolute() else P.REPO / p
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta.setdefault("raw", prov.rel(p))
    if "out_qpos" not in d:
        raise KeyError("the dump has no outputs (out_qpos / out_qvel)")
    return dict(meta, index=np.asarray(d["index"], np.int64), qpos=np.asarray(d["out_qpos"], np.float64),
                qvel=np.asarray(d["out_qvel"], np.float64))


def truth_reference(tr, **kw):
    """The truth's forward outputs as a reference source over all corpus states."""
    fw = tr.forward(**kw)
    nq = tr.corpus().arrays["qpos"].shape[1]
    return dict(engine="truth", label="MuJoCo C float64 truth (2000 iterations, tolerance 1e-13, ls 100)",
                index=np.arange(len(fw["out"])), qpos=fw["out"][:, :nq], qvel=fw["out"][:, nq:], name="truth")


def floor_sources(tr, **kw):
    """{it10, it50[, native]: source}: MuJoCo C float64 at the floor options (the truth file's options.floors:
    truth/mjc.py floor_options), all corpus states."""
    fw = tr.forward(**kw)
    nq = tr.corpus().arrays["qpos"].shape[1]
    out = {}
    for k, opt in tr.meta["options"]["floors"].items():
        o = fw[f"out_{k}"]
        it = int(opt["iterations"]) if k != "native" else "native"
        out[k] = dict(engine="mjc", variant=f"float64_{k}", label=FLOOR_LABELS.get(k, k), iterations=it,
                      index=np.arange(len(o)), qpos=o[:, :nq], qvel=o[:, nq:], floor=True, options=opt)
    return out


def _model(tr):
    prob = tr.problem()
    return P.load_model(prob, iterations=1 if prob.explicit_iterations else None)


def errors(m, qpos, qvel, ref_qpos, ref_qvel):
    """Per-world METRICS (module docstring) and the non-finite mask; nan where the source is not finite."""
    qpos, qvel = np.asarray(qpos, np.float64), np.asarray(qvel, np.float64)
    ref_qpos, ref_qvel = np.asarray(ref_qpos, np.float64), np.asarray(ref_qvel, np.float64)
    import mujoco
    B = len(qpos)
    bad = ~(np.isfinite(qpos).all(1) & np.isfinite(qvel).all(1))
    dq = np.full((B, m.nv), np.nan)
    buf = np.zeros(m.nv)
    for i in np.flatnonzero(~bad):
        mujoco.mj_differentiatePos(m, buf, 1.0, ref_qpos[i], qpos[i])
        dq[i] = buf
    out = dict(qpos_rad=np.linalg.norm(dq[:, 3:], axis=1), qpos_m=np.linalg.norm(dq[:, :3], axis=1),
               qvel=np.linalg.norm(qvel - ref_qvel, axis=1) / np.maximum(np.linalg.norm(ref_qvel, axis=1), 1e-9),
               qpos_rel=np.linalg.norm(qpos - ref_qpos, axis=1) / np.maximum(np.linalg.norm(ref_qpos, axis=1), 1e-9))
    for k in out:
        out[k] = np.where(bad, np.nan, out[k])
    return out, bad


def subset_masks(tr, names=DEFAULT_SUBSETS, model_ok=None):
    """{subset: ((S,) bool or None when not available, the same mask without its model_ok condition)} over all corpus
    states; model_ok overrides the truth's model check."""
    S = tr.n
    mo = tr.model_ok() if model_ok is None else np.asarray(model_ok, bool)
    groups = tr.groups()
    out = {}
    for name in names:
        m, whole = np.ones(S, bool), np.ones(S, bool)
        for p in name.split("&"):
            if p == "all":
                continue
            if p == "model_ok":
                m = None if (mo is None or m is None) else m & mo
            elif p in groups:
                whole &= groups[p]
                m = None if m is None else m & groups[p]
            else:
                raise KeyError(f"unknown subset {p!r} (all, model_ok, {', '.join(groups)})")
        out[name] = (m, whole)
    return out


def score(source, tr, reference=None, subsets=DEFAULT_SUBSETS, model_ok=None, model=None, **fw_kw):
    """(entries, per_world) of a source against the truth (reference None) or a reference source, on the states
    both have. entries: one dict per subset (module docstring); per_world: {index, metric arrays, nonfinite}."""
    ref = reference or truth_reference(tr, **fw_kw)
    m = model or _model(tr)
    si, ri = np.asarray(source["index"], np.int64), np.asarray(ref["index"], np.int64)
    common, a, b = np.intersect1d(si, ri, assume_unique=False, return_indices=True)
    if len(common) != len(si):
        dup = len(si) - len(np.unique(si))
        if dup:
            raise ValueError(f"the source repeats {dup} states (a tiled nworld > corpus size): pass one copy")
    err, bad = errors(m, source["qpos"][a], source["qvel"][a], ref["qpos"][b], ref["qvel"][b])
    masks = subset_masks(tr, subsets, model_ok)
    entries = []
    base = {k: source.get(k) for k in ("engine", "variant", "label", "iterations", "raw") if source.get(k) is not None}
    for name, (mk, whole) in masks.items():
        e = dict(kind="forward", **base, corpus=tr.name, reference=ref.get("name", ref.get("engine", "truth")),
                 subset=name)
        if mk is None:
            entries.append(dict(e, status="not_available", note="the truth file has no model check"))
            continue
        sel = mk[common]
        e.update(states=int(sel.sum()), corpus_states=int(mk.sum()), missing=int(mk.sum() - sel.sum()),
                 nonfinite=int((bad & sel).sum()))
        if "model_ok" in name.split("&"):
            e["excluded"] = int(whole[common].sum() - sel.sum())
        for k in METRICS:
            s = stats.summary(err[k][sel])
            s.pop("nonfinite", None)
            e[k] = s
        if source.get("floor"):
            e["floor"] = True
        entries.append(e)
    per_world = dict(index=common, nonfinite=bad, **err)
    return entries, per_world


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", required=True, help="corpus name or truth file")
    ap.add_argument("--raw", action="append", default=[], help="raw dump npz (repeat; with --engine etc. per dump)")
    ap.add_argument("--engine", action="append", default=[])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--iterations", action="append", type=int, default=[])
    ap.add_argument("--reference-raw", help="score against this dump instead of the truth (e.g. run-to-run)")
    ap.add_argument("--floors", action="store_true", help="add MuJoCo C float64 at 10 / 50 iterations")
    ap.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    ap.add_argument("--cache-dir", default=None, help="the truth cache (default runs/bench/truth)")
    ap.add_argument("--jobs", type=int, default=None, help="MuJoCo C processes if the forward truth is regenerated")
    ap.add_argument("--out", help="JSON of the entries (default: stdout)")
    ap.add_argument("--per-world", help="npz of the per-world errors (one group of arrays per source)")
    a = ap.parse_args(argv)
    from benchmarks.bench.truth import store
    tr = store.load(a.truth)
    kw = dict(cache=a.cache_dir, jobs=a.jobs)
    ref = source_from_dump(a.reference_raw, name="reference") if a.reference_raw else None
    subsets = tuple(x for x in a.subsets.split(",") if x)
    srcs = []
    for i, r in enumerate(a.raw):
        pick = lambda lst, d=None: lst[i] if i < len(lst) else d  # noqa: E731
        srcs.append(source_from_dump(r, engine=pick(a.engine, f"source{i}"), variant=pick(a.variant, "default"),
                                     iterations=pick(a.iterations)))
    if a.floors:
        srcs += list(floor_sources(tr, **kw).values())
    if not srcs:
        ap.error("nothing to score: give --raw and / or --floors")
    m = _model(tr)
    out, pw = [], {}
    for k, s in enumerate(srcs):
        ent, w = score(s, tr, reference=ref, subsets=subsets, model=m, **kw)
        out += ent
        pw.update({f"s{k}__{n}": v for n, v in w.items()})
    text = json.dumps(dict(truth=prov.rel(tr.path), entries=out), indent=1)
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)
    if a.per_world:
        np.savez_compressed(a.per_world, **pw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
