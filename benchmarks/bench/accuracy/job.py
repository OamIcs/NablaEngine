"""Score the raw dumps of a harness job document (dk-bench/1, schema.py) against the truth files and fill its
`accuracy` (forward) and `gradient` lists. Runs in the dk environment after the job finished (the driver rewrites the
document while a job runs).

For every dump cell with status ok (kind "dump": its `raw` npz holds index, out_*, grad<c>_*, jacobian, its `contents`
say which): forward entries (forward.score against the forward truth) when it has outputs, gradient entries
(gradient.score against the gradient truth; VJPs from the gradients when the dump has them, the Jacobian's own
entries too) when it has gradients or a Jacobian and the corpus has a gradient truth. Each entry carries the cell id
(`cell`), engine, variant, label, iterations, nworld and process. Once per corpus: the MuJoCo C float64 floor rows
(it10, it50). Run-to-run rows: the forward of two processes of the same (engine, variant, corpus, nworld, iterations),
the second against the first. Paired gradient rows: each engine's VJP errors against dk's on the same corpus,
nworld and iterations (gradient.paired).

Gradient semantics of a source (gradient.plan): the engine's recorded options (engines.<id>.provenance.options
clamp_kv / normal, as the dk adapters record them), else SEMANTICS below, else the exact mode. MuJoCo C FD rows: the
construction and step from the cell's opts, else the mjc adapter's variant table (MJC_VARIANTS), else its recorded
provenance (fd, eps); options "it<iterations>" (the cell's solver options, never the truth's tight ones).

  python -m benchmarks.bench.accuracy.job runs/bench/<machine>/<job>.json            # print a summary
  python -m benchmarks.bench.accuracy.job runs/bench/<machine>/<job>.json --write    # fill the document in place
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov, schema
from benchmarks.bench.accuracy import forward as AF
from benchmarks.bench.accuracy import gradient as AG

# engines whose provenance does not record clamp_kv / normal; mjw_pr1535_fix keeps the actuator damping -kv on the
# qvel path of clamped dofs (adapters/mjw_pr.py semantics), which is clamp_kv 0
SEMANTICS = {"dk_iface": dict(clamp_kv=0, normal=3), "mjw_pr1535_fix": dict(clamp_kv=0, normal=0)}
MJC_VARIANTS = {"default": dict(fd="chain_forward"), "centered": dict(fd="chain_centered")}  # adapters/mjc.py
CELL_KEYS = ("engine", "variant", "label", "native", "iterations", "nworld", "process", "raw")
PAIRED_REFERENCE = "dk"


def _semantics(doc, engine):
    opts = (((doc.get("engines") or {}).get(engine) or {}).get("provenance") or {}).get("options") or {}
    if "clamp_kv" in opts and "normal" in opts:
        return dict(clamp_kv=int(opts["clamp_kv"]), normal=int(opts["normal"])), "recorded"
    if engine in SEMANTICS:
        return dict(SEMANTICS[engine]), "table"
    return dict(AG.EXACT), "assumed exact"


def _fd_meta(doc, cell):
    if cell["engine"] != "mjc":
        return None
    opts = cell.get("opts") or {}
    var = MJC_VARIANTS.get(cell["variant"], {})
    pv = ((doc.get("engines") or {}).get("mjc") or {}).get("provenance") or {}
    fd = opts.get("fd") or var.get("fd") or pv.get("fd") or "chain_forward"
    eps = float(opts.get("eps") or var.get("eps") or pv.get("eps") or 1e-6)
    construction, kind = fd.split("_", 1)
    return dict(construction=construction, centered=kind == "centered", eps=eps, options=f"it{cell['iterations']}")


def score_job(doc, cache=None, jobs=None, floors=True, run_to_run=True, paired=True, log=print):
    """(accuracy entries, gradient entries) of a job document (module docstring)."""
    from benchmarks.bench.truth import store
    truths, gts, acc, grad = {}, {}, [], []
    fwd_runs, grad_states = {}, {}
    for cell in doc.get("cells", []):
        if cell.get("kind") != "dump" or schema.parse_status(cell.get("status", "error"))[0] != "ok":
            continue
        corpus = cell["corpus"]
        if corpus not in truths:
            try:
                truths[corpus] = store.load(corpus)
            except FileNotFoundError as e:
                truths[corpus] = None
                log(f"[accuracy] {corpus}: {e}")
        tr = truths[corpus]
        if tr is None or not cell.get("raw"):
            continue
        p = P.REPO / cell["raw"]
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = dict({k: cell.get(k) for k in CELL_KEYS}, cell=cell["id"])
        tag = dict(cell=cell["id"], nworld=cell["nworld"], process=cell["process"])  # carried by every entry
        contents = cell.get("contents") or []
        if "outputs" in contents:
            src = AF.source_from_dump(d, **meta)
            ent, _ = AF.score(src, tr, cache=cache, jobs=jobs)
            acc += [dict(e, **tag) for e in ent]
            fwd_runs.setdefault((cell["engine"], cell["variant"], corpus, cell["nworld"], cell["iterations"]),
                                []).append((cell["process"], src))
        if ("gradient" in contents or "jacobian" in contents) and tr.has_gradient:
            if corpus not in gts:
                gts[corpus] = AG.truth_from_store(tr)
            sem, how = _semantics(doc, cell["engine"])
            src = AG.source_from_dump(d, **meta, semantics=sem, fd=_fd_meta(doc, cell),
                                      not_supported=cell.get("gradient_not_supported"))
            ent, st = AG.score(src, gts[corpus])
            for e in ent:
                e["semantics_source"] = how
                e.update(tag)
            grad += ent
            grad_states[(cell["engine"], cell["variant"], corpus, cell["nworld"], cell["iterations"],
                         cell["process"])] = st
    if floors:
        for corpus, tr in truths.items():
            if tr is None:
                continue
            for s in AF.floor_sources(tr, cache=cache, jobs=jobs).values():
                acc += AF.score(s, tr, cache=cache, jobs=jobs)[0]
    if run_to_run:
        for (eng, var, corpus, nw, it), runs in fwd_runs.items():
            runs = sorted(runs, key=lambda x: x[0])
            for proc, src in runs[1:]:
                ref = dict(runs[0][1], name=f"run-to-run (process {runs[0][0]})")
                ent, _ = AF.score(src, truths[corpus], reference=ref, subsets=("all", "model_ok"), cache=cache,
                                  jobs=jobs)
                acc += [dict(e, run_to_run=True) for e in ent]
    if paired:
        for (eng, var, corpus, nw, it, proc), st in grad_states.items():
            ref = grad_states.get((PAIRED_REFERENCE, "default", corpus, nw, it, 0))
            if ref is None or (eng, var) == (PAIRED_REFERENCE, "default"):
                continue
            for e in AG.paired(st, ref, label_a=f"{eng}~{var}", label_b=f"{PAIRED_REFERENCE}~default"):
                grad.append(dict(e, engine=eng, variant=var, corpus=corpus, nworld=nw, iterations=it,
                                 process=proc))
    return acc, grad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("doc", help="job document (dk-bench/1)")
    ap.add_argument("--write", action="store_true", help="replace the document's accuracy and gradient lists")
    ap.add_argument("--out", help="write the scored document here instead (implies --write semantics)")
    ap.add_argument("--no-floors", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args(argv)
    doc = schema.read(a.doc)
    acc, grad = score_job(doc, cache=a.cache_dir, jobs=a.jobs, floors=not a.no_floors)
    if a.write or a.out:
        doc["accuracy"], doc["gradient"] = acc, grad
        out = Path(a.out or a.doc)
        if prov.is_committed(out) and not a.out:
            print(f"{prov.rel(out)} is committed: pass --out", file=sys.stderr)
            return 1
        schema.write(out, doc)
    nan = float("nan")
    for e in acc:
        if e.get("subset") in ("all", "model_ok") and "qvel" in e:
            q, r = e["qvel"], e["qpos_rad"]
            print(f"forward {e.get('engine')}~{e.get('variant')} {e['corpus']} n{e.get('nworld', '-')} "
                  f"it{e.get('iterations')} "
                  f"{e['subset']:9s} vs {e['reference']}: qvel median {q.get('median', nan):.2e} p99 "
                  f"{q.get('p99', nan):.2e}, qpos_rad median {r.get('median', nan):.2e}")
    for e in grad:
        if e.get("kind") == "gradient" and e.get("group") == "G0" and e.get("subset") == "all" and "median" in e:
            print(f"gradient {e.get('engine')}~{e.get('variant')} {e['corpus']} n{e.get('nworld', '-')} "
                  f"it{e.get('iterations')} G0 "
                  f"{e['block']:4s} {e['object']:8s}: median {e['median']:.2e} p99 {e['p99']:.2e}")
    print(json.dumps(dict(accuracy=len(acc), gradient=len(grad))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
