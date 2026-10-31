"""The paper's accuracy tables on the held-out corpora: one JSON with every number and its provenance, and a compact
markdown rendering of it. Reads finished harness jobs (dk-bench/1 documents of `python -m benchmarks.bench run --part
accuracy`, raw dumps of outputs and gradients at 4096 worlds, i.e. every corpus state) and one MuJoCo C finite-
difference sweep (benchmarks.bench.accuracy.fd_sweep), and scores them with the harness's own scorers
(accuracy/forward.py, accuracy/gradient.py) against the MuJoCo C float64 truth files. numpy and mujoco; no GPU.

Sources (engine, variant and FD step of a dump cell):
  dk                ours, exact gradient mode (clamp_kv 1, normal 0); its forward is the one of both modes
  dk_iface          ours, the interface defaults (clamp_kv 0, normal 3): gradients only (its forward is checked bitwise
                    against dk's)
  mjw               MuJoCo Warp 3.17.0, native: forward only (no backward)
  mjw_pr1535        MuJoCo Warp + PR #1535 @357a75d, as published
  mjw_pr1535_fix    MuJoCo Warp + PR #1535 @4af093f (clamp_kv 0 semantics: accuracy/job.py SEMANTICS)
  mjx               MJX 3.17.0, float32, native; gradient cot^T J of its forward-mode Jacobian
  mjc               MuJoCo C mjd_transitionFD chained over the substeps, forward / centered, at eps 1e-6 and at the
                    eps chosen with the truth (the sweep's best on --choose-on; the jobs that ran it must match)
  genesis           Genesis 1.4.5, float32, native (adapters/genesis.py: the model's contacts as contype / conaffinity,
                    its recorded deviations); gradient = its native reverse mode, initial qvel only (the qpos-tangent
                    and ctrl blocks are not supported)
Forward rows: every source above with outputs, MuJoCo C float64 at 10 and 50 iterations (the truth file's floor
runs: solver truncation alone) and run-to-run rows (later processes of the same unit against the first).

Forward metrics (accuracy/forward.py): qpos_rad (tangent-space error without the base translation, rad, absolute),
qpos_m (base translation, m), qvel (relative); median, p90, p99, max, share above 1e-3 over the finite worlds,
non-finite worlds counted. Subsets: all, model_ok, each group over all its states and over its model-ok states.
Gradient metrics (accuracy/gradient.py): VJPs of the corpus's 3 stored cotangents on the truth's sampled states,
relative error per block (qpos tangent, qvel, ctrl) floored at the truth's own floor, cosine; per group and subset
(all sampled states, model-ok ones). A source whose semantics has no matching truth on a group is recorded as not
comparable (the interface defaults and the 4af093f tree on G4: the truth has no clamp_kv 0 Jacobian); for information its
G4 errors against the exact truth are kept under gradient_vs_exact.
Pooled rows concatenate the corpora (forward: every state of each; gradient: every sampled state of each).
Non-finite counts: every gradient dump on every corpus state with the stored cotangents (a partial gradient on the
blocks it supports; Genesis's aborted backward passes, retried and bisected by its adapter, are NaN there).
Sources far from the truth by construction (FAR_SOURCES: a different contact model) are left out of the per-state
list of model-ok states above 1e-3 (their count is kept) and of the other sources' annotations.

  python -m benchmarks.bench.report.paper_accuracy --jobs runs/bench/<dir>/ --fd-sweep runs/bench/<dir>/fd.json \
      --out-json docs/results/<name>.json --out-md docs/results/<name>.md
"""

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov, schema
from benchmarks.bench.accuracy import forward as AF
from benchmarks.bench.accuracy import gradient as AG
from benchmarks.bench.accuracy import job as AJ
from benchmarks.bench.accuracy import stats

CORPORA = ("mf_s0", "mf_s1", "mf_s2", "mf_s3")
GROUPS = ("G0", "G4", "G5", "G6", "G1")
TABLE_GROUPS = ("G0", "G4", "G5", "G6")
FWD_SUBSETS = ("all", "model_ok") + GROUPS + tuple(f"{g}&model_ok" for g in GROUPS)
FWD_METRICS = ("qpos_rad", "qpos_m", "qvel")
EPS_DEFAULT = 1e-6
FAR_SOURCES = ("genesis",)
SCRIPT_FILES = ("benchmarks/bench/report/paper_accuracy.py", "benchmarks/bench/accuracy/forward.py",
                "benchmarks/bench/accuracy/gradient.py", "benchmarks/bench/accuracy/stats.py",
                "benchmarks/bench/accuracy/fd_sweep.py", "benchmarks/bench/accuracy/job.py",
                "benchmarks/bench/adapters/mjc_fd.py", "benchmarks/bench/truth/store.py")
LABELS = {
    "ours": "ours (forward of both gradient modes)",
    "ours_exact": "ours, exact mode (clamp_kv=1, normal=0)",
    "ours_interface": "ours, interface defaults (clamp_kv=0, normal=3)",
    "mjw": "MuJoCo Warp 3.17.0 (native)",
    "pr_357a75d": "MuJoCo Warp + PR #1535 @357a75d (as published)",
    "pr_4af093f": "MuJoCo Warp + PR #1535 @4af093f",
    "mjx": "MJX 3.17.0 (float32, native)",
    "mjx_grad": "MJX 3.17.0, forward-mode Jacobian (cot^T J)",
    "mjc": "MuJoCo C 3.17.0 float64 (the cell's iterations)",
    "mjc_floor_it10": "MuJoCo C float64, 10 it (solver-truncation floor)",
    "mjc_floor_it50": "MuJoCo C float64, 50 it (solver-truncation floor)",
    "genesis": "Genesis 1.4.5 (float32, native)",
    "genesis_grad": "Genesis 1.4.5, native reverse mode (initial qvel only)",
}


# ------------------------------------------------------------------------------------------------- inputs

def load_docs(paths):
    files = []
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else P.REPO / p
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    docs = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        if isinstance(d, dict) and str(d.get("schema", "")).startswith("dk-bench/") and d.get("cells"):
            docs.append((f, d))
    return docs


def fd_eps(doc, cell):
    m = AJ._fd_meta(doc, cell)
    return None if m is None else m["eps"]


def source_key(doc, cell, best):
    """The source id of a dump cell (module docstring), or None for cells outside the tables."""
    e, v = cell["engine"], cell["variant"]
    if e == "dk" and v == "default":
        return "ours_exact"
    if e == "dk_iface" and v == "default":
        return "ours_interface"
    if e == "mjw" and v == "default":
        return "mjw"
    if e == "mjw_pr1535" and v == "default":
        return "pr_357a75d"
    if e == "mjw_pr1535_fix" and v == "default":
        return "pr_4af093f"
    if e == "mjx" and v == "default":
        return "mjx"
    if e == "genesis" and v == "default":
        return "genesis"
    if e == "mjc" and v in ("default", "centered"):
        kind = "forward" if v == "default" else "centered"
        eps = fd_eps(doc, cell)
        if eps == EPS_DEFAULT:
            return f"mjc_fd_{kind}_1e-6"
        if best and eps == best.get(kind):
            return f"mjc_fd_{kind}_best"
        return f"mjc_fd_{kind}_eps{eps:g}"
    return None


def source_label(key, best=None):
    if key in LABELS:
        return LABELS[key]
    if key.startswith("mjc_fd_"):
        _, _, kind, eps = key.split("_", 3)
        if eps == "best":
            return (f"MuJoCo C mjd_transitionFD chained, {kind}, eps {best[kind]:g} (chosen with truth)")
        return f"MuJoCo C mjd_transitionFD chained, {kind}, eps {eps}"
    return key


def choose_eps(sweep, corpus, group="G0", subset="all", option="headline"):
    """{forward: eps, centered: eps, criterion, scores}: per chained FD kind, the sweep's eps with the smallest mean
    over the three blocks of log10(median floored relative error) on one corpus's group (chosen with the truth)."""
    res = sweep["corpora"][corpus]["results"]
    out, scores = {}, {}
    for kind in ("forward", "centered"):
        best = None
        for eps in sweep["eps"]:
            r = res.get(f"{option}/chain_{kind}/{eps:g}")
            if r is None:
                continue
            g = r["groups"][group][subset]
            s = sum(math.log10(max(g[b]["median"], 1e-300)) for b in AG.BLOCKS) / len(AG.BLOCKS)
            scores[f"chain_{kind}/{eps:g}"] = round(s, 4)
            if best is None or s < best[1]:
                best = (eps, s)
        out[kind] = best[0]
    out["criterion"] = (f"the eps of {', '.join(f'{e:g}' for e in sweep['eps'])} with the smallest mean over the "
                        f"qpos / qvel / ctrl blocks of log10(median floored relative error), MuJoCo C chained FD at "
                        f"the {option} options, {corpus} {group} sampled states ({subset}): chosen with the truth")
    out["scores"] = scores
    return out


def read_dump(cell):
    with np.load(P.REPO / cell["raw"], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------- forward

def _round(x, n=4):
    if isinstance(x, float):
        return float(f"{x:.{n}g}") if math.isfinite(x) else x
    if isinstance(x, dict):
        return {k: _round(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, n) for v in x]
    return x


def fwd_entry(err, bad, sel):
    e = dict(states=int(sel.sum()), nonfinite=int((bad & sel).sum()))
    for k in FWD_METRICS:
        s = stats.summary(err[k][sel])
        s.pop("nonfinite", None)
        e[k] = s
    return e


def score_forward(src, tr, model, masks, reference=None):
    """(per-corpus {subset: entry}, per-world pieces for pooling) of one forward source."""
    _, pw = AF.score(src, tr, reference=reference, subsets=("all",), model=model)
    common = pw["index"]
    if len(common) != tr.n:
        raise ValueError(f"{tr.name}: the source covers {len(common)} of {tr.n} states")
    err = {k: pw[k] for k in FWD_METRICS}
    bad = pw["nonfinite"]
    out = {name: fwd_entry(err, bad, m[common]) for name, (m, _) in masks.items() if m is not None}
    return out, dict(err=err, bad=bad, masks={name: m[common] for name, (m, _) in masks.items() if m is not None})


def pool_forward(pieces):
    out = {}
    names = pieces[0]["masks"].keys()
    for name in names:
        err = {k: np.concatenate([p["err"][k] for p in pieces]) for k in FWD_METRICS}
        bad = np.concatenate([p["bad"] for p in pieces])
        sel = np.concatenate([p["masks"][name] for p in pieces])
        out[name] = fwd_entry(err, bad, sel)
    return out


# ------------------------------------------------------------------------------------------------- gradient

def merge_truths(gts, n_states):
    """One gradient truth of several corpora (pooled rows): indices offset by k * n_states, no Jacobian."""
    m = dict(name="pooled", path=[g["path"] for g in gts], format=gts[0]["format"], blocks=gts[0]["blocks"],
             jac_index=np.zeros(0, np.int64), jacobian=None, jacobian_dcol=None)
    m["index"] = np.concatenate([g["index"].astype(np.int64) + k * n_states for k, g in enumerate(gts)])
    for key in ("fd", "floor", "bad", "model_ok", "clamped", "axis_lt_1mm", "cot_qpos", "cot_qvel"):
        m[key] = np.concatenate([g[key] for g in gts])
    names = sorted(set().union(*[g["groups"] for g in gts]))
    m["groups"] = {n: np.concatenate([g["groups"].get(n, np.zeros(len(g["index"]), bool)) for g in gts])
                   for n in names}
    return m


def compact_grad(entries, brief=False):
    """{group: {subset: {block | status}}} of gradient.score entries (object vjp); brief: the floored median / p99 /
    max, cos_min and the raw median per block (per-corpus rows)."""
    if brief:
        out = compact_grad(entries)
        for slot in out.values():
            for sub in slot.values():
                for b in AG.BLOCKS:
                    if b in sub and "status" not in sub[b]:
                        x = sub[b]
                        sub[b] = dict({k: x.get(k) for k in ("n", "median", "p99", "max", "cos_min")},
                                      raw_median=x["raw"].get("median"))
        return out
    out = {}
    for e in entries:
        if e.get("object") != "vjp":
            continue
        slot = out.setdefault(e["group"], {})
        if e.get("status") and e.get("block") is None:
            slot[e["subset"]] = dict(status=e["status"], note=e.get("note"), states=e.get("states"))
            continue
        s = slot.setdefault(e["subset"], dict(group_states=e["group_states"], states=e["states"],
                                              usable_states=e["usable_states"],
                                              nonfinite_states=e["nonfinite_states"],
                                              missing_states=e["missing_states"],
                                              excluded_model_mismatch=e.get("excluded_model_mismatch"),
                                              **{k: e[k] for k in e if k.startswith("excluded_")
                                                 and k != "excluded_model_mismatch"}))
        if e.get("status"):  # a block the source does not support
            s[e["block"]] = dict(status=e["status"], note=e.get("note"))
            continue
        s[e["block"]] = dict({k: e.get(k) for k in ("n", "median", "p90", "p99", "max", stats.share_key(),
                                                    "cos_min", "cos_p1")},
                             raw={k: e["raw"].get(k) for k in ("median", "p90", "p99", "max")},
                             truth_floor_median=(e.get("truth_floor") or {}).get("median"),
                             share_at_floor=e.get("share_at_floor"), valid_col_share=e.get("valid_col_share"),
                             states_without_valid_column=e.get("states_without_valid_column"))
    return _round(out)


# ------------------------------------------------------------------------------------------------- main build

def build(docs, sweep, sweep_path, choose_on, log=print):
    from benchmarks.bench.truth import store
    best = choose_eps(sweep, choose_on)
    truths = {c: store.load(c) for c in CORPORA}
    model = AF._model(truths[CORPORA[0]])
    masks = {c: AF.subset_masks(truths[c], FWD_SUBSETS) for c in CORPORA}
    gts = {c: AG.truth_from_store(truths[c]) for c in CORPORA}
    S = truths[CORPORA[0]].n
    pooled_gt = merge_truths([gts[c] for c in CORPORA], S)

    cells = {}  # (key, corpus, iterations, process) -> (doc path, doc, cell)
    for path, doc in docs:
        for c in doc["cells"]:
            if c.get("kind") != "dump" or schema.parse_status(c.get("status", "error"))[0] != "ok":
                continue
            if c.get("model") != "mjlab" or c["corpus"] not in CORPORA or c["nworld"] != S:
                continue
            key = source_key(doc, c, best)
            if key is None:
                continue
            slot = (key, c["corpus"], c["iterations"], c["process"])
            if slot in cells:
                raise ValueError(f"two dumps for {slot}: {prov.rel(cells[slot][0])} and {prov.rel(path)}")
            cells[slot] = (path, doc, c)
    keys = sorted({k for k, *_ in cells})
    its = sorted({it for _, _, it, _ in cells})
    log(f"[paper_accuracy] sources {keys}, iterations {its}, {len(cells)} dumps")

    out = dict(forward=dict(per_corpus={}, pooled={}, model_ok_above_1e_3={}),
               gradient=dict(per_corpus={}, pooled={}),
               gradient_vs_exact=dict(per_corpus={}, pooled={}), nonfinite={}, checks={}, missing=[])
    dumps, qv = {}, {}  # qv: per-state qvel errors (source, corpus, iterations) for the outlier annotation

    def dump(slot):
        if slot not in dumps:
            dumps[slot] = read_dump(cells[slot][2])
        return dumps[slot]

    # forward: engines (process 0), floors, run-to-run
    fwd_sources = [k for k in keys if (k != "ours_interface" and not k.startswith("mjc_fd_"))
                   or k == "mjc_fd_forward_1e-6"]  # one forward per engine (the other mjc rows share it)
    for it in its:
        for key in fwd_sources:
            name = "ours" if key == "ours_exact" else ("mjc" if key == "mjc_fd_forward_1e-6" else key)
            pieces, per = [], {}
            for c in CORPORA:
                slot = (key, c, it, 0)
                if slot not in cells:
                    out["missing"].append(dict(kind="forward", source=name, corpus=c, iterations=it))
                    continue
                d = dump(slot)
                src = AF.source_from_dump(d, engine=cells[slot][2]["engine"])
                per[c], pc = score_forward(src, truths[c], model, masks[c])
                pieces.append(pc)
                big = np.flatnonzero(pc["masks"]["model_ok"] & (np.nan_to_num(pc["err"]["qvel"], nan=np.inf)
                                                                 > stats.THRESHOLD))
                if name in FAR_SOURCES:  # a different model: the count only
                    out["forward"].setdefault("model_ok_above_1e_3_count", {}).setdefault(f"it{it}", {}).setdefault(
                        name, {})[c] = int(len(big))
                    continue
                qv[(name, c, it)] = pc["err"]["qvel"]
                if len(big):  # model-ok states above the threshold: which ones, and how far
                    out["forward"]["model_ok_above_1e_3"].setdefault(f"it{it}", {}).setdefault(name, []).extend(
                        dict(corpus=c, state=int(i), qvel=_round(float(pc["err"]["qvel"][i])),
                             qpos_rad=_round(float(pc["err"]["qpos_rad"][i]))) for i in big)
            if pieces:
                out["forward"]["per_corpus"].setdefault(f"it{it}", {})[name] = _round(per)
                if len(pieces) == len(CORPORA):
                    out["forward"]["pooled"].setdefault(f"it{it}", {})[name] = _round(pool_forward(pieces))
                log(f"[paper_accuracy] forward {name} it{it}: {len(pieces)} corpora")
        # floors: MuJoCo C float64 at the floor options, against the truth
        flo = f"it{it}"
        pieces, per = [], {}
        for c in CORPORA:
            fs = AF.floor_sources(truths[c])
            if flo not in fs:
                continue
            per[c], pc = score_forward(fs[flo], truths[c], model, masks[c])
            qv[(f"mjc_floor_{flo}", c, it)] = pc["err"]["qvel"]
            pieces.append(pc)
        if pieces:
            out["forward"]["per_corpus"].setdefault(f"it{it}", {})[f"mjc_floor_{flo}"] = _round(per)
            out["forward"]["pooled"].setdefault(f"it{it}", {})[f"mjc_floor_{flo}"] = _round(pool_forward(pieces))
        # run-to-run: later processes against process 0 of the same unit
        for key in fwd_sources:
            pieces, per = [], {}
            for c in CORPORA:
                procs = sorted(p for k, cc, i, p in cells if (k, cc, i) == (key, c, it))
                if len(procs) < 2:
                    continue
                ref = AF.source_from_dump(dump((key, c, it, procs[0])))
                ref["name"] = f"process {procs[0]}"
                for p in procs[1:]:
                    src = AF.source_from_dump(dump((key, c, it, p)))
                    a, b = dump((key, c, it, p)), dump((key, c, it, procs[0]))
                    e, pc = score_forward(src, truths[c], model, masks[c], reference=ref)
                    e["bitwise_equal"] = bool(np.array_equal(a["out_qpos"], b["out_qpos"])
                                              and np.array_equal(a["out_qvel"], b["out_qvel"]))
                    per[f"{c}/p{p}_vs_p{procs[0]}"] = e
                    pieces.append(pc)
            if pieces:
                name = "ours" if key == "ours_exact" else ("mjc" if key == "mjc_fd_forward_1e-6" else key)
                out["forward"]["per_corpus"].setdefault(f"it{it}", {})[f"run_to_run_{name}"] = _round(per)
                out["forward"]["pooled"].setdefault(f"it{it}", {})[f"run_to_run_{name}"] = _round(
                    pool_forward(pieces))

    # the other sources' errors at each model-ok state where a source is above the threshold
    for itk, rows in out["forward"]["model_ok_above_1e_3"].items():
        it = int(itk[2:])
        for name, lst in rows.items():
            for e in lst:
                e["others"] = {o: _round(float(qv[(o, e["corpus"], it)][e["state"]]))
                               for (o, c, i) in qv if c == e["corpus"] and i == it and o != name}

    # gradient: sampled truth states, per corpus and pooled; semantics as the harness's job scorer assigns them
    grad_sources = [k for k in keys if k != "mjw"]
    for it in its:
        for key in grad_sources:
            per, vjps, per_exact = {}, [], {}
            sem = None
            for c in CORPORA:
                slot = (key, c, it, 0)
                if slot not in cells:
                    out["missing"].append(dict(kind="gradient", source=key, corpus=c, iterations=it))
                    continue
                path, doc, cell = cells[slot]
                d = dump(slot)
                sem, how = AJ._semantics(doc, cell["engine"])
                src = AG.source_from_dump(d, engine=cell["engine"], variant=cell["variant"],
                                          fd=AJ._fd_meta(doc, cell),
                                          not_supported=cell.get("gradient_not_supported"))
                ent, _ = AG.score(src, gts[c], semantics=sem)
                per[c] = compact_grad(ent, brief=True)
                if sem != AG.EXACT:
                    ent_x, _ = AG.score(src, gts[c], semantics=AG.EXACT)
                    per_exact[c] = compact_grad([e for e in ent_x if e.get("group") == "G4"], brief=True)
                vjps.append((c, src))
            if not per:
                continue
            g = out["gradient"]["per_corpus"].setdefault(f"it{it}", {})
            g[key] = per
            meta = dict(semantics=sem, semantics_source=how, fd=AJ._fd_meta(*cells[(key, CORPORA[0], it, 0)][1:])
                        if (key, CORPORA[0], it, 0) in cells else None)
            if len(vjps) == len(CORPORA):
                src = dict(engine=key, index=np.concatenate([s["index"].astype(np.int64) + k * S
                                                              for k, (_, s) in enumerate(vjps)]),
                           vjp=None, jacobian=None, fd=meta["fd"])
                if all(s.get("vjp") is not None for _, s in vjps):
                    src["vjp"] = np.concatenate([s["vjp"] for _, s in vjps])
                else:  # partial sources: the blocks every corpus's dump has
                    blk = [b for b in AG.BLOCKS if all(b in (s.get("vjp_parts") or {}) for _, s in vjps)]
                    src.update(vjp_parts={b: np.concatenate([s["vjp_parts"][b] for _, s in vjps]) for b in blk},
                               blocks_supported=blk, not_supported=vjps[0][1].get("not_supported"))
                ent, _ = AG.score(src, pooled_gt, semantics=sem)
                out["gradient"]["pooled"].setdefault(f"it{it}", {})[key] = dict(meta, groups=compact_grad(ent))
                if sem != AG.EXACT:
                    ent_x, _ = AG.score(src, pooled_gt, semantics=AG.EXACT)
                    out["gradient_vs_exact"]["pooled"].setdefault(f"it{it}", {})[key] = compact_grad(
                        [e for e in ent_x if e.get("group") == "G4"])
                    out["gradient_vs_exact"]["per_corpus"].setdefault(f"it{it}", {})[key] = per_exact
            log(f"[paper_accuracy] gradient {key} it{it}: {len(vjps)} corpora, semantics {sem} ({how})")

    # non-finite counts of every dump on every corpus state (stored cotangents)
    for (key, c, it, p), (path, doc, cell) in sorted(cells.items(), key=lambda x: str(x[0])):
        d = dump((key, c, it, p))
        e = dict(states=int(len(d["index"])))
        if "out_qpos" in d:
            e["forward_nonfinite_states"] = int((~(np.isfinite(d["out_qpos"]).all(1)
                                                   & np.isfinite(d["out_qvel"]).all(1))).sum())
        have = [b for b in AG.BLOCKS if f"grad0_{b}" in d]
        if have:
            C = 0
            while f"grad{C}_{have[0]}" in d:
                C += 1
            g = np.stack([np.concatenate([d[f"grad{k}_{b}"] for b in have], 1) for k in range(C)], 1)
            e.update(cotangents=C, gradient_nonfinite_states=int((~np.isfinite(g).all((1, 2))).sum()),
                     gradient_nonfinite_values=int((~np.isfinite(g)).sum()), gradient_values=int(g.size))
            if len(have) < len(AG.BLOCKS):
                e["gradient_blocks"] = have
            rep_ = cell.get("gradient_report") or {}
            if "aborted_states" in rep_:
                ab = rep_["aborted_states"]
                e.update(aborted_batches=rep_.get("aborted_batches"), batches=rep_.get("batches"),
                         recovered_on_retry=rep_.get("recovered_on_retry"), bisect_runs=rep_.get("bisect_runs"),
                         aborted_states_per_cotangent={k: len(v) for k, v in ab.items()},
                         aborted_states=sorted({i for v in ab.values() for i in v}))
        else:
            e["gradient"] = "none (the source has no gradient)"
        out["nonfinite"].setdefault(key, {}).setdefault(f"it{it}", {})[f"{c}/p{p}"] = e

    # checks
    ck = out["checks"]
    for it in its:
        for c in CORPORA:
            a, b = (("ours_exact", c, it, 0), ("ours_interface", c, it, 0))
            if a in cells and b in cells:
                ck.setdefault("ours_forward_modes_bitwise", {})[f"{c}/it{it}"] = bool(
                    np.array_equal(dump(a)["out_qpos"], dump(b)["out_qpos"])
                    and np.array_equal(dump(a)["out_qvel"], dump(b)["out_qvel"]))
            m = ("mjc_fd_forward_1e-6", c, it, 0)
            fs = AF.floor_sources(truths[c]).get(f"it{it}")
            if m in cells and fs is not None:
                d = dump(m)
                idx = d["index"].astype(np.int64)
                ck.setdefault("mjc_forward_vs_floor_run", {})[f"{c}/it{it}"] = dict(
                    bitwise=bool(np.array_equal(d["out_qpos"], fs["qpos"][idx])
                                 and np.array_equal(d["out_qvel"], fs["qvel"][idx])),
                    max_abs_qpos=float(np.max(np.abs(d["out_qpos"] - fs["qpos"][idx]))),
                    max_abs_qvel=float(np.max(np.abs(d["out_qvel"] - fs["qvel"][idx]))))
    # gradient run to run: later processes against process 0 of the same unit, per block over every state and cotangent
    for (key, c, it, p), (path, doc, cell) in sorted(cells.items(), key=lambda x: str(x[0])):
        if p == 0 or (key, c, it, 0) not in cells:
            continue
        a, b = dump((key, c, it, p)), dump((key, c, it, 0))
        blk = [x for x in AG.BLOCKS if f"grad0_{x}" in a and f"grad0_{x}" in b]
        if not blk or not np.array_equal(a["index"], b["index"]):
            continue
        C = 0
        while f"grad{C}_{blk[0]}" in a:
            C += 1
        e = {}
        for x in blk:
            g = np.stack([a[f"grad{k}_{x}"] for k in range(C)], 1)
            r = np.stack([b[f"grad{k}_{x}"] for k in range(C)], 1)
            rel = np.linalg.norm(g - r, axis=2) / np.maximum(np.linalg.norm(r, axis=2), 1e-9)
            e[x] = dict(bitwise_equal=bool(np.array_equal(g, r)), states=int(len(rel)),
                        states_differing=int((rel > 0).any(1).sum()),
                        **_round(stats.summary(rel)))
        ck.setdefault("gradient_run_to_run", {}).setdefault(key, {})[f"{c}/it{it}/p{p}_vs_p0"] = e
    # the sweep's chained FD against the harness's mjc dumps (same code, different batch): G0 medians on the sweep's
    # corpus at the headline options (10 iterations)
    sw = sweep["corpora"][choose_on]["results"]
    for key, kind, eps in (("mjc_fd_forward_1e-6", "forward", EPS_DEFAULT),
                           ("mjc_fd_centered_1e-6", "centered", EPS_DEFAULT),
                           ("mjc_fd_forward_best", "forward", best["forward"]),
                           ("mjc_fd_centered_best", "centered", best["centered"])):
        g = out["gradient"]["per_corpus"].get("it10", {}).get(key, {}).get(choose_on, {}).get("G0", {}).get("all")
        r = sw.get(f"headline/chain_{kind}/{eps:g}")
        if g and r:
            ck.setdefault("fd_sweep_vs_harness_mjc", {})[key] = {
                b: dict(sweep=_round(r["groups"]["G0"]["all"][b]["median"]), harness=g[b]["median"])
                for b in AG.BLOCKS}
    out["fd_choice"] = best
    out["fd_sweep"] = sweep_section(sweep, sweep_path, choose_on)
    return out, cells, best


def sweep_section(sweep, path, corpus):
    r = sweep["corpora"][corpus]
    rows = {}
    for k, v in r["results"].items():
        rows[k] = dict(sec=v["sec"], ms_per_world=v["ms_per_world"], threads=v["threads"], states=v["states"],
                       evaluations_per_world=v["evaluations_per_world"], nonfinite=v["nonfinite"],
                       reference_relation=v["reference_relation"], groups=_round(v["groups"]))
    return dict(file=prov.rel(path), sha256=prov.sha256(path), corpus=corpus, states=r["states"],
                group_states=r["group_states"], truth=r["truth"], eps=sweep["eps"], options=sweep["options"],
                modes=sweep["modes"], subset=sweep["subset"], construction=sweep["construction"],
                threads=sweep["threads"], seconds=sweep["seconds"],
                provenance={k: v for k, v in (sweep.get("provenance") or {}).items() if k != "harness"},
                results=rows)


def provenance(docs, cells, sweep_path):
    used = sorted({str(p) for p, _, _ in cells.values()})
    jobs = []
    for path, doc in docs:
        if str(path) not in used:
            continue
        cmds = sorted({c.get("cmd") for c in doc["cells"] if c.get("cmd")})
        jobs.append(dict(file=prov.rel(path), sha256=prov.sha256(path), job=(doc.get("job") or {}).get("name"),
                         created=doc.get("created"), updated=doc.get("updated"), commands=cmds,
                         cells_used=sum(1 for p, _, _ in cells.values() if str(p) == str(path))))
    doc0 = docs[0][1]
    mach = doc0.get("machine") or {}
    envs, engines = {}, {}
    for path, doc in docs:
        if str(path) not in used:
            continue
        for name, env in (doc.get("environments") or {}).items():
            pk = {k: (v or {}).get("version") for k, v in (env.get("packages") or {}).items()}
            if name == "dk":
                pk.pop("mujoco-warp", None)  # installed there but not imported by dk, mjx or mjc
            envs[name] = dict(python=env.get("python"), packages=pk)
        for eid, e in (doc.get("engines") or {}).items():
            pv = e.get("provenance") or {}
            keep = {}
            if eid in ("dk", "dk_iface"):
                b = pv.get("build") or {}
                keep = dict(build_key=b.get("key"), arch=b.get("arch"), version=b.get("version"),
                            options=pv.get("options"), consts_sha256=b.get("consts_sha256"))
            elif eid.startswith("mjw"):
                keep = dict(versions={k: (v.get("version") if isinstance(v, dict) else v)
                                      for k, v in (pv.get("versions") or {}).items()},
                            tree=dict(commit=((pv.get("mujoco_warp") or {}).get("git") or {}).get("commit"),
                                      tree_sha256=(pv.get("mujoco_warp") or {}).get("tree_sha256"),
                                      version=(pv.get("mujoco_warp") or {}).get("version")),
                            device_options=pv.get("device_options"),
                            capacity={k: (pv.get("capacity") or {}).get(k) for k in ("nconmax", "njmax", "naconmax")})
            elif eid == "mjx":
                keep = {k: pv.get(k) for k in ("versions", "precision", "x64", "jac", "xla_flags") if k in pv}
            elif eid == "mjc":
                keep = {k: pv.get(k) for k in ("mujoco", "nthread", "fd", "eps") if k in pv}
            elif eid == "genesis":
                keep = {k: pv.get(k) for k in ("genesis", "quadrants", "torch", "torch_cuda", "mujoco", "precision",
                                               "device", "rigid_options", "rigid_options_grad", "sim_options",
                                               "dispatch_override", "grad_chunk", "grad_retries", "deviations")
                        if k in pv}
                mdl = pv.get("model") or {}
                keep["model"] = dict(edit=mdl.get("edit"), dropped=mdl.get("dropped"), pair_check=mdl.get("pair_check"),
                                     conversion=mdl.get("conversion"))
                keep["dr"] = {k: v for k, v in (pv.get("dr") or {}).items()
                              if k not in ("worlds", "worlds_friction_above_floor", "worlds_friction_below_sole")}
            engines.setdefault(eid, []).append(dict(job=(doc.get("job") or {}).get("name"),
                                                    label=(e.get("adapter") or {}).get("label"),
                                                    semantics=e.get("semantics"), provenance=keep))
    files = {f: prov.sha256(P.REPO / f) for f in SCRIPT_FILES}
    truth = {}
    for c in CORPORA:
        tf, cf = f"benchmarks/data/truth/{c}.npz", f"benchmarks/data/{c}.npz"
        truth[c] = dict(file=tf, sha256=prov.sha256(P.REPO / tf), corpus=cf, corpus_sha256=prov.sha256(P.REPO / cf))
    return dict(date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                platform=dict(role="accuracy platform (no timing in this file)", gpu=mach.get("gpu"),
                              cpu={k: (mach.get("cpu") or {}).get(k) for k in ("model", "sockets", "cores", "threads")},
                              os=mach.get("os")),
                jobs=jobs, environments=envs, engines=engines, truth=truth, scripts_sha256=files,
                fd_sweep=dict(file=prov.rel(sweep_path), sha256=prov.sha256(sweep_path)))


# ------------------------------------------------------------------------------------------------- markdown

def f(x, pct=False):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if pct:
        return "0" if x == 0 else f"{100 * x:.2f}%"
    if x == 0:
        return "0"
    return f"{x:.1e}"


def render(res):
    pv = res["provenance"]
    gpu = (pv["platform"].get("gpu") or {})
    best = res["fd_choice"]
    L = []
    L.append(f"# Accuracy on the held-out corpora (accuracy platform: {gpu.get('name')})")
    L.append("")
    L.append(f"Every number below is in `{res['file']}` (generated {pv['date']} by "
             "`python -m benchmarks.bench.report.paper_accuracy` from the benchmark harness's accuracy jobs). "
             f"GPU {gpu.get('name')} (driver {gpu.get('driver')}), CPU {pv['platform']['cpu'].get('model')}. "
             "The V100 is the accuracy platform only: no timing here.")
    L.append("")
    L.append("Corpora: the four held-out benchmark corpora mf_s0, mf_s1, mf_s2, mf_s3 (4096 states each, "
             "`benchmarks/data/`); one control step = 4 x 5 ms substeps of the mjlab G1 with ctrl held, Newton, "
             "per-world foot friction and torso COM offset. Truth: MuJoCo C 3.17.0 float64, 2000 iterations, "
             "tolerance 1e-13, ls 100 (`benchmarks/data/truth/`). Every engine runs natively (its own package, "
             "unmodified, default options; MuJoCo Warp keeps its internal tolerance floor of 1e-6). "
             + ("Genesis 1.4.5 runs the same model with its 502 contact pairs declared through contype / conaffinity "
                "(Genesis's MJCF importer reads no <pair>); its deviations from the problem are listed in the notes. "
                if "genesis" in res.get("sources", {}) else "")
             + "Pooled rows concatenate the four corpora.")
    L.append("")
    # forward
    L.append("## 1. Forward: one control step against the float64 truth")
    L.append("")
    L.append("qvel: relative error per state; qpos: tangent-space error without the base translation "
             "(mj_differentiatePos, free-joint rotation and hinges, rad, absolute). Statistics over the finite "
             "states, non-finite states counted. Model-ok: MuJoCo C and native MuJoCo Warp 3.17.0 see the same "
             "contacts, zones, limit rows and frames. All 16384 states (4 x 4096) for every engine, MJX included "
             "(no subset was needed).")
    L.append("")
    hdr = ("| source | it | all: n | qvel median | p90 | p99 | max | > 1e-3 | non-finite | qpos (rad) median | p99 "
           "| max | base (m) median | p99 | model-ok: n | qvel median | p90 | p99 | max | > 1e-3 | qpos (rad) p99 |")
    L.append(hdr)
    L.append("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    order = ["ours", "mjw", "pr_357a75d", "pr_4af093f", "mjx", "genesis", "mjc_floor_it10", "mjc_floor_it50",
             "run_to_run_mjw", "run_to_run_ours", "run_to_run_pr_357a75d", "run_to_run_pr_4af093f", "run_to_run_mjx"]
    names = dict(LABELS, run_to_run_mjw="floor: MuJoCo Warp 3.17.0 run to run",
                 run_to_run_ours="ours run to run", run_to_run_pr_357a75d="PR @357a75d run to run",
                 run_to_run_pr_4af093f="PR @4af093f run to run", run_to_run_mjx="MJX run to run")
    sk = stats.share_key()
    for it, rows in res["forward"]["per_corpus"].items():  # what each run-to-run row pools
        for k, per in rows.items():
            if k.startswith("run_to_run_") and per:
                pairs = sorted(per)
                corp = sorted({x.split("/")[0] for x in pairs})
                procs = sorted({x.split("/")[1].split("_vs_")[0][1:] for x in pairs})
                ref = sorted({x.split("_vs_")[1][1:] for x in pairs})
                names[k] = (f"{names.get(k, k)} ({', '.join(corp)}; process {', '.join(procs)} vs "
                            f"{', '.join(ref)})")
    for it, rows in sorted(res["forward"]["pooled"].items(), key=lambda x: int(x[0][2:])):
        for k in order:
            if k not in rows or (k.startswith("mjc_floor_") and k != f"mjc_floor_{it}"):
                continue
            a, m = rows[k]["all"], rows[k]["model_ok"]
            if k.startswith("run_to_run_") and k != "run_to_run_mjw":
                continue
            L.append(f"| {names.get(k, k)} | {it[2:]} | {a['states']} | {f(a['qvel'].get('median'))} | "
                     f"{f(a['qvel'].get('p90'))} | {f(a['qvel'].get('p99'))} | {f(a['qvel'].get('max'))} | "
                     f"{f(a['qvel'].get(sk), True)} | {a['nonfinite']} | {f(a['qpos_rad'].get('median'))} | "
                     f"{f(a['qpos_rad'].get('p99'))} | {f(a['qpos_rad'].get('max'))} | "
                     f"{f(a['qpos_m'].get('median'))} | {f(a['qpos_m'].get('p99'))} | {m['states']} | "
                     f"{f(m['qvel'].get('median'))} | {f(m['qvel'].get('p90'))} | {f(m['qvel'].get('p99'))} | "
                     f"{f(m['qvel'].get('max'))} | {f(m['qvel'].get(sk), True)} | {f(m['qpos_rad'].get('p99'))} |")
    L.append("")
    L.append("Per group (model-ok states of each group; qvel relative error median / p99, qpos rad p99):")
    L.append("")
    gh = ("| source | it | " + " | ".join(f"{g} n | {g} qvel median | {g} p99 | {g} qpos p99" for g in TABLE_GROUPS)
          + " |")
    L.append(gh)
    L.append("|" + "|".join(["---"] * (gh.count("|") - 1)) + "|")
    for it, rows in sorted(res["forward"]["pooled"].items(), key=lambda x: int(x[0][2:])):
        for k in order:
            if k not in rows or (k.startswith("mjc_floor_") and k != f"mjc_floor_{it}"):
                continue
            if k.startswith("run_to_run_") and k != "run_to_run_mjw":
                continue
            cells_ = []
            for g in TABLE_GROUPS:
                e = rows[k][f"{g}&model_ok"]
                cells_.append(f"{e['states']} | {f(e['qvel'].get('median'))} | {f(e['qvel'].get('p99'))} | "
                              f"{f(e['qpos_rad'].get('p99'))}")
            L.append(f"| {names.get(k, k)} | {it[2:]} | " + " | ".join(cells_) + " |")
    L.append("")
    # gradient
    L.append("## 2. Gradients against the float64 finite-difference truth")
    L.append("")
    L.append("VJPs of each corpus's 3 stored output cotangents at the truth's sampled states (up to 64 per group and "
             "corpus, all states of the group, not only model-ok ones), 10 iterations. Error per input block (qpos "
             "tangent / qvel / ctrl) on the valid columns: |g - g_fd| / |g_fd|, floored at the truth's own floor "
             "|fd(1e-7) - fd(1e-6)| / |fd(1e-6)| (no source can look closer than the truth is known); median and p99 "
             "over (state, cotangent) pairs, and the smallest cosine. n = usable states.")
    L.append("")
    gorder = ["ours_exact", "ours_interface", "pr_357a75d", "pr_4af093f", "mjx", "genesis", "mjc_fd_forward_1e-6",
              "mjc_fd_centered_1e-6", "mjc_fd_forward_best", "mjc_fd_centered_best"]
    glabels = dict(LABELS, mjx=LABELS["mjx_grad"], genesis=LABELS["genesis_grad"])
    ns = "not supported"
    gp = res["gradient"]["pooled"].get("it10", {})
    for g in TABLE_GROUPS:
        L.append(f"### {g} (pooled over the four corpora)")
        L.append("")
        h = ("| source | n | qpos median | p99 | min cos | qvel median | p99 | min cos | ctrl median | p99 | "
             "min cos | model-ok n | qpos / qvel / ctrl median (model-ok) |")
        L.append(h)
        L.append("|" + "|".join(["---"] * (h.count("|") - 1)) + "|")
        for k in gorder:
            if k not in gp:
                continue
            lab = source_label(k, best) if k.startswith("mjc_fd_") else glabels.get(k, k)
            e = gp[k]["groups"].get(g, {})
            a, m = e.get("all", {}), e.get("model_ok", {})
            if a.get("status"):
                L.append(f"| {lab} | {a.get('states')} | not comparable: the truth has no clamp_kv=0 "
                         "Jacobian | | | | | | | | | | |")
                continue
            row = [lab, str(a.get("usable_states"))]
            for b in AG.BLOCKS:
                x = a.get(b, {})
                if x.get("status"):
                    row += [ns, "", ""]
                    continue
                row += [f(x.get("median")), f(x.get("p99")), "-" if x.get("cos_min") is None else f"{x['cos_min']:.4f}"]
            row.append(str(m.get("usable_states")))
            row.append(" / ".join(ns if m.get(b, {}).get("status") else f(m.get(b, {}).get("median"))
                                  for b in AG.BLOCKS))
            L.append("| " + " | ".join(row) + " |")
        L.append(f"| {LABELS['mjw']} | - | no backward (MuJoCo Warp has no gradient) | | | | | | | | | | |")
        L.append("")
    # 50 iterations, compact
    gp50 = res["gradient"]["pooled"].get("it50", {})
    if gp50:
        L.append("50 iterations (pooled, all sampled states; qpos / qvel / ctrl median floored error per group):")
        L.append("")
        h = "| source | " + " | ".join(TABLE_GROUPS) + " |"
        L.append(h)
        L.append("|" + "|".join(["---"] * (h.count("|") - 1)) + "|")
        for k in gorder:
            if k not in gp50:
                continue
            lab = source_label(k, best) if k.startswith("mjc_fd_") else glabels.get(k, k)
            cs = []
            for g in TABLE_GROUPS:
                a = gp50[k]["groups"].get(g, {}).get("all", {})
                cs.append("n/c" if a.get("status") else " / ".join(
                    "n/s" if a.get(b, {}).get("status") else f(a.get(b, {}).get("median")) for b in AG.BLOCKS))
            L.append(f"| {lab} | " + " | ".join(cs) + " |")
        L.append("")
    # non-finite
    L.append("## 3. Non-finite outputs and gradients on every corpus state")
    L.append("")
    L.append("All 4 x 4096 states with each corpus's 3 stored cotangents (process 0); a state counts when any of its "
             "outputs, or any entry of its 3 VJPs, is not finite.")
    L.append("")
    h = "| source | it | forward non-finite states | gradient non-finite states | gradient non-finite values |"
    L.append(h)
    L.append("|---|---|---|---|---|")
    for k in ["ours_exact", "ours_interface", "mjw", "pr_357a75d", "pr_4af093f", "mjx", "genesis",
              "mjc_fd_forward_1e-6", "mjc_fd_centered_1e-6", "mjc_fd_forward_best", "mjc_fd_centered_best"]:
        if k not in res["nonfinite"]:
            continue
        lab = source_label(k, best) if k.startswith("mjc_fd_") else glabels.get(k, k)
        for it, per in sorted(res["nonfinite"][k].items(), key=lambda x: int(x[0][2:])):
            p0 = {c: v for c, v in per.items() if c.endswith("/p0")}
            fw = sum(v.get("forward_nonfinite_states", 0) for v in p0.values())
            if any("gradient_nonfinite_states" in v for v in p0.values()):
                gs = sum(v.get("gradient_nonfinite_states", 0) for v in p0.values())
                gv = sum(v.get("gradient_nonfinite_values", 0) for v in p0.values())
                n = sum(v["states"] for v in p0.values())
                blk = sorted({b for v in p0.values() for b in v.get("gradient_blocks", [])}, key=AG.BLOCKS.index)
                tag = f" ({', '.join(blk)} block only)" if blk else ""
                L.append(f"| {lab}{tag} | {it[2:]} | {fw} of {n} | {gs} of {n} | {gv} |")
            else:
                n = sum(v["states"] for v in p0.values())
                L.append(f"| {lab} | {it[2:]} | {fw} of {n} | no gradient | - |")
    L.append("")
    # FD sweep
    sw = res["fd_sweep"]
    L.append(f"## 4. MuJoCo C finite differences across step sizes ({sw['corpus']}, G0, {sw['states']} states)")
    L.append("")
    L.append("Raw (unfloored) relative VJP error per block, median / p90, against the truth; floored medians are in "
             "the JSON. chain = mjd_transitionFD per substep chained over the control step (MuJoCo C's native "
             "derivative, the benchmark's MuJoCo C row); direct = finite differences of the whole control step (the "
             "truth's construction). Headline options: 10 iterations, tolerance 1e-8, ls 8. ms per state: wall time "
             f"of the batch on {sw['threads']} CPU threads.")
    L.append("")
    h = "| construction | eps | qpos median | p90 | qvel median | p90 | ctrl median | p90 | ms per state |"
    L.append(h)
    L.append("|---|---|---|---|---|---|---|---|---|")
    for mode in sw["modes"]:
        for eps in sw["eps"]:
            r = sw["results"].get(f"headline/{mode}/{eps:g}")
            if not r:
                continue
            g = r["groups"]["G0"][sw["subset"]]
            rel = r.get("reference_relation")
            tag = f" ({rel})" if rel else ""
            mark = ""
            if (mode, eps) in (("chain_forward", best["forward"]), ("chain_centered", best["centered"])):
                mark = " (chosen)"
            L.append(f"| {mode.replace('_', ' ')}{tag} | {eps:g}{mark} | " + " | ".join(
                f"{f(g[b]['raw']['median'])} | {f(g[b]['raw']['p90'])}" for b in AG.BLOCKS) +
                f" | {r['ms_per_world']:.2f} |")
    L.append("")
    tight = sw["results"].get("tight/direct_centered/1e-06")
    if tight:
        g = tight["groups"]["G0"][sw["subset"]]
        L.append(f"Tight options (2000 iterations, tolerance 1e-13, ls 100), direct centered at 1e-6 "
                 f"({tight.get('reference_relation')}): qpos / qvel / ctrl raw median "
                 + " / ".join(f(g[b]["raw"]["median"]) for b in AG.BLOCKS)
                 + "; every tight-option row is in the JSON (`fd_sweep.results`, keys `tight/...`).")
        L.append("")
    # reproduce
    L.append("## Reproduce")
    L.append("")
    L.append("The harness jobs (one engine group per GPU; MuJoCo C on the CPU), the sweep and this rendering:")
    L.append("")
    L.append("```")
    for j in pv["jobs"]:
        for c in j["commands"]:
            L.append(c)
    fs = res["fd_sweep"]
    L.append(f"python -m benchmarks.bench.accuracy.fd_sweep --corpora {fs['corpus']} --groups "
             f"{','.join(fs['group_states'])} --threads {fs['threads']} --out {fs['file']}")
    L.append(f"python -m benchmarks.bench.report.paper_accuracy --jobs {' '.join(j['file'] for j in pv['jobs'])} "
             f"--fd-sweep {fs['file']} --out-json {res['file']} --out-md {res['file'][:-5]}.md")
    L.append("```")
    L.append("")
    # notes
    L.append("## Notes")
    L.append("")
    for n in res["notes"]:
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


def notes(res):
    best = res["fd_choice"]
    ck = res["checks"]
    nb = []
    nb.append("Platform: every number was produced on the V100 accuracy platform (GPU engines) and its host CPU "
              "(MuJoCo C); this file has no timing except the FD sweep's cost column.")
    nb.append("Subsets: every forward row covers all 4096 states of each corpus, MJX included; every gradient row "
              "covers all sampled truth states of each group (G0 / G4 / G5 / G6: 64 per corpus, 256 pooled). "
              "Model-ok columns restrict to states where MuJoCo C and native MuJoCo Warp 3.17.0 agree on the "
              "discrete structure.")
    nb.append("Floors: the forward solver-truncation floor is MuJoCo C float64 itself at 10 / 50 iterations "
              "(tolerance 1e-8, ls 8) against its converged solution; the MuJoCo Warp run-to-run floor compares "
              "processes 1 and 2 with process 0 of the same unit (mf_s0 and mf_s1, 10 iterations: the units run in "
              "three processes). The gradient floor is the truth's own step-size floor, applied to every source.")
    nb.append("The interface defaults (clamp_kv=0, normal=3) and the @4af093f tree (which keeps the actuator damping -kv on "
              "the qvel path of clamped dofs, i.e. clamp_kv=0) are not comparable on G4: the truth is "
              "the exact derivative and has no clamp_kv=0 Jacobian. Their G4 errors against the exact truth are "
              "kept, for information only, under `gradient_vs_exact`.")
    nb.append(f"MuJoCo C FD rows: chained mjd_transitionFD at eps 1e-6 (the default) and at the eps chosen with the "
              f"truth, forward {best['forward']:g} and centered {best['centered']:g} ({best['criterion']}). The "
              "centered rows at 1e-6 and 1e-7 are semi-self-referential (centered differences of MuJoCo C's own "
              "float64 map at the truth's steps).")
    nb.append("MJX's gradient is its native forward-mode Jacobian (jax.jvp through every solver iteration; MJX has "
              "no reverse mode), contracted with the same cotangents (cot^T J). MuJoCo Warp 3.17.0 has no backward.")
    nb.append("PR #1535 @357a75d is the published head of the unmerged upstream pull request; @4af093f is that head "
              "plus two follow-up fixes on the pull request's branch (pyramidal edge rule, clamp multiplier). Both "
              "trees run unmodified.")
    rr = {}
    for it, rows in res["forward"]["per_corpus"].items():
        for k, per in rows.items():
            if k.startswith("run_to_run_"):
                v = rr.setdefault(k[len("run_to_run_"):], [0, 0])
                v[0] += sum(bool(x.get("bitwise_equal")) for x in per.values())
                v[1] += len(per)
    if rr:
        nb.append("Run to run (later processes against process 0, bitwise identical / compared): " + ", ".join(
            f"{LABELS.get(k, k)} {a} / {n}" for k, (a, n) in sorted(rr.items())) + ".")
    if ck.get("ours_forward_modes_bitwise"):
        v = ck["ours_forward_modes_bitwise"]
        nb.append(f"Our forward is bitwise identical in both gradient modes on {sum(v.values())} of {len(v)} "
                  "(corpus, iterations) dumps.")
    if ck.get("mjc_forward_vs_floor_run"):
        v = ck["mjc_forward_vs_floor_run"]
        nb.append(f"The harness's MuJoCo C forward equals the truth file's floor run bitwise on "
                  f"{sum(x['bitwise'] for x in v.values())} of {len(v)} (corpus, iterations) dumps.")
    if ck.get("fd_sweep_vs_harness_mjc"):
        v = ck["fd_sweep_vs_harness_mjc"]
        same = all(x["sweep"] == x["harness"] for r in v.values() for x in r.values())
        nb.append("The FD sweep's chained rows and the harness's MuJoCo C dumps agree on mf_s0 G0 "
                  + ("(identical medians)." if same else "(medians differ: see checks.fd_sweep_vs_harness_mjc)."))
    mo = res["forward"].get("model_ok_above_1e_3") or {}
    if mo:
        counts = {}
        for itk in sorted(mo, key=lambda x: int(x[2:])):
            for name, lst in mo[itk].items():
                counts.setdefault(name, {})[itk] = len(lst)
        parts = [f"{LABELS.get(n, n)} " + " / ".join(str(v.get(k, 0)) for k in sorted(mo, key=lambda x: int(x[2:])))
                 for n, v in sorted(counts.items())]
        nb.append("Model-ok states with a qvel error above 1e-3 at " + " / ".join(k[2:] for k in sorted(
            mo, key=lambda x: int(x[2:]))) + " iterations (each listed under `forward.model_ok_above_1e_3` with the "
                  "other sources' errors at the same state): " + "; ".join(parts) + ".")
        ours50 = (mo.get("it50") or {}).get("ours") or []
        if ours50:
            mx = max(e["qvel"] for e in ours50)
            mjw = max(e["others"].get("mjw", 0) or 0 for e in ours50)
            flo = max(e["others"].get("mjc_floor_it50", 0) or 0 for e in ours50)
            it10 = {(e["corpus"], e["state"]): e["qvel"] for e in (mo.get("it10") or {}).get("ours") or []}
            same10 = [e for e in ours50 if (e["corpus"], e["state"]) in it10]
            chg = max((abs(it10[(e["corpus"], e["state"])] - e["qvel"]) / e["qvel"] for e in same10), default=None)
            nb.append(f"Our {len(ours50)} model-ok states above 1e-3 at 50 iterations (largest {mx:.1e}) are above it "
                      f"at 10 iterations too ({len(same10)} of {len(ours50)}, largest relative change of the error "
                      f"between 10 and 50 iterations {chg:.1e}); at those states MuJoCo Warp 3.17.0 is within "
                      f"{mjw:.1e} and MuJoCo C float64 at 50 iterations within {flo:.1e} of the truth. So this is not "
                      "solver truncation. They are contact-activation flips at the distance threshold: on the one "
                      "that differs already in the first substep (mf_s2 state 0), MuJoCo C (float64) activates a "
                      "second plane-capsule contact of a foot at a distance of -3.2e-8 m, which our float32 step "
                      "evaluates on the other side of zero; the foot moves into the floor, so that contact's damping "
                      "changes the step. Any float32 engine has such flips. 'Model-ok' is defined by MuJoCo C and "
                      "MuJoCo Warp agreeing, so it removes MuJoCo Warp's own flips but not ours; on all states the "
                      "share above 1e-3 is about the same for both (see the 'all' columns), and that is the "
                      "comparison to quote.")
    nb += genesis_notes(res)
    for n in res.get("extra_notes") or []:
        nb.append(n)
    if res["missing"]:
        nb.append(f"Missing dumps: {len(res['missing'])} (listed under `missing`).")
    return nb


def genesis_notes(res):
    """The notes of the Genesis rows (none without them)."""
    eng = (res["provenance"]["engines"].get("genesis") or [None])[0]
    if eng is None:
        return []
    pv = eng["provenance"]
    nb = []
    mdl = pv.get("model") or {}
    pc = mdl.get("pair_check") or {}
    nb.append("Genesis 1.4.5 (genesis-world from PyPI, quadrants 1.3.3, unmodified): its MJCF importer reads no <pair> "
              "and imports contype = conaffinity = 0 geoms as visual only, so the adapter writes the problem's model "
              "as an MJCF with contype = conaffinity = 1 on the robot geoms and no <pair> block; Genesis's default "
              "collision filter (self collision on, parent-child and qpos0-touching pairs off) then gives "
              f"{pc.get('genesis_pairs')} pairs, {'exactly' if pc.get('pair_set_equal') else 'not'} the model's "
              f"{pc.get('model_pairs')} explicit pairs. Options: Newton, the cell's iterations, ls 8, tolerance 1e-8, "
              "pyramidal cone, integrator implicitfast, 4 x 5 ms (SimOptions(dt=0.02, substeps=4)), PD targets held "
              "(control_dofs_position with the model's kp / kv and force ranges), constraint_timeconst = the model's "
              "solref time constant 0.02.")
    devs = pv.get("deviations") or []
    if devs:
        nb.append("Genesis's deviations from the problem (recorded by its adapter): " + "; ".join(devs) + ".")
    fric = []
    for c in CORPORA:
        try:
            fric.append(P.load_corpus(c).arrays["friction"].astype(np.float64))
        except Exception:  # noqa: BLE001
            pass
    dr = pv.get("dr") or {}
    if fric and dr:
        f_ = np.concatenate(fric)
        nb.append(f"Genesis's DR: the floor-sole friction of every world is exact (per-world friction ratios on the "
                  f"floor and sole links; Genesis takes the larger of two geoms' friction x ratio); in "
                  f"{int(np.sum(f_ > dr['floor_friction']))} of {len(f_)} states (friction above the floor's "
                  f"{dr['floor_friction']:g}) and {int(np.sum(f_ < dr['sole_friction']))} (below the soles' "
                  f"{dr['sole_friction']:g}) the soles' contacts with other robot geoms then use the scaled sole "
                  "friction too. Torso COM through set_links_COM (Genesis recomputes its inverse weights).")
    nb.append("Genesis's gradient is its native reverse mode (scene.backward) with respect to the initial qvel: the "
              "only integrator Genesis differentiates is approximate_implicitfast (its forward rows use "
              "implicitfast), and it has no gradient path for the joint positions or the PD targets, so the qpos and "
              "ctrl blocks are not supported (Genesis's own messages in the dump cells' gradient_not_supported).")
    ab = [v for per in (res["nonfinite"].get("genesis") or {}).values() for k, v in per.items()]
    if ab and any("aborted_batches" in v for v in ab):
        nab = sum(v.get("aborted_batches") or 0 for v in ab)
        nbt = sum(v.get("batches") or 0 for v in ab)
        rec = sum(v.get("recovered_on_retry") or 0 for v in ab)
        bis = sum(1 for v in ab if v.get("bisect_runs"))
        nst = sum(len(v.get("aborted_states") or []) for v in ab)
        nb.append(f"Genesis's NaN-gradient guard (GenesisException 'Nan grad in qpos or dofs_vel found', which aborts "
                  f"the whole batch) stopped {nab} of {nbt} backward batches of {pv.get('grad_chunk')} worlds on the "
                  f"first attempt (every Genesis dump: {len(ab)} units, 3 cotangents each); the adapter retries a "
                  f"batch once, then bisects it to single states: {rec} retries succeeded, {bis} unit(s) needed "
                  "bisection, "
                  + (f"and {nst} states stayed aborted (a NaN gradient, counted in section 3)." if nst else
                     "and no state stayed aborted.") + " The guard does not fire on the same batches in every "
                  "process.")
    rr = (res["checks"].get("gradient_run_to_run") or {}).get("genesis") or {}
    if rr:
        q = [v["qvel"] for v in rr.values() if "qvel" in v]
        nb.append(f"Genesis's forward is bitwise identical from process to process, its gradient is not: over "
                  f"{len(q)} later processes against process 0 (mf_s0, mf_s1, 10 iterations) the qvel-block VJPs "
                  f"differ on {min(v['states_differing'] for v in q)}-{max(v['states_differing'] for v in q)} of "
                  f"{q[0]['states']} states, relative difference median {max(v['median'] for v in q):.1e}, p99 up to "
                  f"{max(v['p99'] for v in q):.1e}, max {max(v['max'] for v in q):.1e} "
                  "(`checks.gradient_run_to_run`).")
    if res.get("forward", {}).get("model_ok_above_1e_3_count"):
        nb.append("Genesis's forward differs from MuJoCo's on most states (another contact model, frictional "
                  "condim-1 pairs, its own solver): its model-ok states above 1e-3 are counted under "
                  "`forward.model_ok_above_1e_3_count`, not listed.")
    return nb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", nargs="+", required=True, help="job documents or directories of them")
    ap.add_argument("--fd-sweep", required=True, help="benchmarks.bench.accuracy.fd_sweep JSON")
    ap.add_argument("--choose-on", default="mf_s0", help="the sweep corpus whose G0 states choose the best eps")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--note", action="append", default=[], help="an extra note for the JSON and the markdown (repeat)")
    a = ap.parse_args(argv)
    for o in (a.out_json, a.out_md):
        if prov.is_committed(Path(o)):
            print(f"{o} is committed: choose another path", file=sys.stderr)
            return 1
    docs = load_docs(a.jobs)
    sp = Path(a.fd_sweep)
    sp = sp if sp.is_absolute() else P.REPO / sp
    sweep = json.loads(sp.read_text())
    res, cells, best = build(docs, sweep, sp, a.choose_on)
    for kind in ("forward", "centered"):
        if not any(k == f"mjc_fd_{kind}_best" for k, *_ in cells):
            res["missing"].append(dict(kind="gradient", source=f"mjc_fd_{kind}_best",
                                       note=f"no MuJoCo C dump at the chosen eps {best[kind]:g}"))
    res = dict(schema="dk-paper-accuracy/1", title="Accuracy on the held-out corpora", file=prov.rel(a.out_json),
               provenance=provenance(docs, cells, sp), corpora=list(CORPORA),
               sources={k: source_label(k, best) for k in sorted({k for k, *_ in cells})}, **res)
    res["extra_notes"] = list(a.note)
    res["notes"] = notes(res)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(schema.clean(res), indent=1) + "\n")
    Path(a.out_md).write_text(render(res) + "\n")
    print(json.dumps(dict(json=prov.rel(a.out_json), md=prov.rel(a.out_md), sources=len(res["sources"]),
                          missing=len(res["missing"]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
