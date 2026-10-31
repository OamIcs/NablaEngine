"""One representative state of the gradient accuracy tables, as vectors: the VJP of one input block (default qvel) of
one stored cotangent on one sampled state of a group, from the MuJoCo C float64 finite-difference truth and from each
source, component by component, with the dof names. For a figure that draws the truth and each engine side by side.

The (state, cotangent) is chosen by rule, not by hand, over the pooled corpora and the stored cotangents (the pooled
rows of the tables): among the group's sampled pairs where every source is finite and the block has valid truth
columns, the one whose floored errors (accuracy/gradient.py) are closest to every source's own pooled median, i.e.
the pair minimising max over sources of |log10(e_source) - log10(median_source)|. The JSON records each source's error
on that pair, its pooled median, and the share of the pooled pairs with a smaller error.

Reads finished harness jobs (dump cells, process 0); numpy and mujoco, no GPU.

  python -m benchmarks.bench.report.gradient_example --jobs runs/bench/<dir>/ --corpora mf_s0 mf_s1 mf_s2 mf_s3 \\
      --group G0 --sources dk mjx genesis mjc --out runs/bench/<dir>/gradient_example_G0.json
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov, schema
from benchmarks.bench.accuracy import gradient as AG
from benchmarks.bench.accuracy import job as AJ

LABELS = {"dk": "NablaEngine (exact gradient mode)", "mjx": "MJX 3.17.0 (forward-mode Jacobian)",
          "genesis": "Genesis 1.4.5 (native reverse mode)", "mjc": "MuJoCo C 3.17.0 finite differences",
          "mjw_pr1535": "MuJoCo Warp + PR #1535 @357a75d", "mjw_pr1535_fix": "MuJoCo Warp + PR #1535 @4af093f"}


def _docs(paths):
    out = []
    for p in paths:
        p = Path(p)
        for f in (sorted(p.glob("*.json")) if p.is_dir() else [p]):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and d.get("cells"):
                out.append((f, d))
    return out


def _cell(docs, engine, corpus, iterations, variant="default"):
    hits = []
    for path, doc in docs:
        for c in doc["cells"]:
            if c.get("kind") != "dump" or schema.parse_status(c.get("status", "error"))[0] != "ok":
                continue
            if (c["engine"], c["variant"], c["corpus"], c["iterations"], c.get("process", 0)) != \
                    (engine, variant, corpus, iterations, 0):
                continue
            if "gradient" not in (c.get("contents") or []) or (c.get("opts") or {}).get("eps"):
                continue  # the FD step sweeps carry their step in opts: only the default cell
            hits.append((path, doc, c))
    if len(hits) != 1:
        raise SystemExit(f"{engine}~{variant} {corpus} it{iterations}: {len(hits)} dump cells (need exactly one)")
    return hits[0]


def dof_names(m):
    """Names of the nv dofs: joint name, with _x/_y/_z for the translations and _rx/_ry/_rz for the rotations of free
    and ball joints."""
    import mujoco
    names = [""] * m.nv
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        a, t = m.jnt_dofadr[j], m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            sub = ["_x", "_y", "_z", "_rx", "_ry", "_rz"]
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            sub = ["_rx", "_ry", "_rz"]
        else:
            sub = [""]
        for k, s in enumerate(sub):
            names[a + k] = n + s
    return names


def _corpus_sources(docs, corpus, a):
    """(truth, GT, block slice, {source: (floored errors (N, C), block VJPs (N, C, ncol_block))}, info, usable (N, C))."""
    from benchmarks.bench.truth import store
    tr = store.load(corpus)
    GT = AG.truth_from_store(tr)
    sl = GT["blocks"][a.block]
    gmask = np.asarray(GT["groups"][a.group], bool)
    valid = (GT["bad"] == 0)[:, sl]
    per_src, info = {}, {}
    for eng in a.sources:
        path, doc, cell = _cell(docs, eng, corpus, a.iterations)
        sem, _ = AJ._semantics(doc, eng)
        src = AG.source_from_dump(P.REPO / cell["raw"], engine=eng, variant=cell["variant"], semantics=sem,
                                  fd=AJ._fd_meta(doc, cell), not_supported=cell.get("gradient_not_supported"))
        if src.get("blocks_supported") is not None and a.block not in src["blocks_supported"]:
            raise SystemExit(f"{eng}: block {a.block} not supported natively")
        _, per = AG.score(src, GT, semantics=sem)
        g, _, how = AG.vjp_at_truth(src, GT)
        per_src[eng] = (per[f"ef_{a.block}"], g[:, :, sl])
        info[eng] = dict(label=LABELS.get(eng, eng), job=prov.rel(path), raw=cell["raw"],
                         raw_sha256=prov.sha256(P.REPO / cell["raw"]), how=how, semantics=sem,
                         fd=AJ._fd_meta(doc, cell))
    ok = np.broadcast_to((gmask & valid.any(1))[:, None], per_src[a.sources[0]][0].shape).copy()
    for eng in a.sources:
        e, g = per_src[eng]
        ok &= np.isfinite(e) & np.isfinite(g).all(2)
    return tr, GT, sl, per_src, info, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", nargs="+", required=True, help="job documents or directories of them")
    ap.add_argument("--corpora", nargs="+", default=["mf_s0", "mf_s1", "mf_s2", "mf_s3"])
    ap.add_argument("--group", default="G0")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--block", default="qvel", choices=AG.BLOCKS)
    ap.add_argument("--sources", nargs="+", default=["dk", "mjx", "genesis", "mjc"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if prov.is_committed(Path(a.out)):
        print(f"{a.out} is committed: choose another path", file=sys.stderr)
        return 2

    docs = _docs(a.jobs)
    per_corpus = {corpus: _corpus_sources(docs, corpus, a) for corpus in a.corpora}
    pooled = {eng: np.concatenate([v[3][eng][0][v[5]] for v in per_corpus.values()]) for eng in a.sources}
    med = {eng: float(np.median(pooled[eng])) for eng in a.sources}
    best = None
    for corpus, (_, _, _, per_src, _, ok) in per_corpus.items():
        dev = np.max([np.abs(np.log10(np.maximum(per_src[eng][0], 1e-30)) - np.log10(med[eng]))
                      for eng in a.sources], 0)
        dev = np.where(ok, dev, np.inf)
        w, c = np.unravel_index(int(np.argmin(dev)), dev.shape)
        if best is None or dev[w, c] < best[0]:
            best = (float(dev[w, c]), corpus, int(w), int(c))
    dmin, corpus, w, c = best
    tr, GT, sl, per_src, info, _ = per_corpus[corpus]
    m = P.load_model(P.get("mjlab"), iterations=a.iterations)
    names = dof_names(m) if a.block == "qvel" else None  # the qvel columns are the nv dofs in order
    flo = AG.block_errors(GT["floor"], GT["fd"], GT["bad"] == 0, {a.block: sl})[a.block][0][w, c]
    out = dict(
        kind="gradient_example", created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        problem="mjlab G1, one control step (4 x 5 ms, ctrl held), Newton, iterations %d" % a.iterations,
        corpus=corpus, corpus_state=int(GT["index"][w]), group=a.group, block=a.block,
        block_columns=[sl.start, sl.stop], dof_names=names,
        cotangent=dict(index=c, qpos=GT["cot_qpos"][w, c].tolist(), qvel=GT["cot_qvel"][w, c].tolist()),
        selection=dict(rule="over the pooled corpora and the stored cotangents, among the group's sampled (state, "
                            "cotangent) pairs with every source finite and valid truth columns in the block: argmin "
                            "of max over sources of |log10(e) - log10(pooled median e)|, e the floored relative error "
                            "of the block (accuracy/gradient.py)",
                       corpora=a.corpora, candidates=int(sum(int(v[5].sum()) for v in per_corpus.values())),
                       max_log10_deviation=dmin),
        valid=((GT["bad"] == 0)[w, sl]).tolist(),
        truth=dict(label="MuJoCo C 3.17.0 float64 central finite differences (truth)", path=GT["path"],
                   vjp=GT["fd"][w, c, sl].tolist(), floor=float(flo)),
        sources={eng: dict(info[eng], vjp=per_src[eng][1][w, c].tolist(), error=float(per_src[eng][0][w, c]),
                           pooled_median=med[eng], share_of_pairs_with_smaller_error=float(
                               np.mean(pooled[eng] < per_src[eng][0][w, c])))
                 for eng in a.sources},
        provenance=dict(harness=prov.harness_commit(), script=prov.rel(Path(__file__)),
                        script_sha256=prov.sha256(Path(__file__)), truth_sha256=prov.sha256(tr.path),
                        argv=sys.argv[1:] if argv is None else list(argv)))
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{corpus} state {out['corpus_state']} ({a.group}, {a.block}, cotangent {c}; "
          f"{out['selection']['candidates']} candidates, max |log10 deviation| {dmin:.3f})")
    for eng in a.sources:
        s = out["sources"][eng]
        print(f"  {eng:10s} error {s['error']:.3e}  pooled median {s['pooled_median']:.3e}  "
              f"smaller on {100 * s['share_of_pairs_with_smaller_error']:.0f}%")
    print(f"  truth floor {flo:.3e} -> {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
