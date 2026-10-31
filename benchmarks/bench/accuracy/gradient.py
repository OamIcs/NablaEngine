"""Gradient accuracy scorer: a source's gradients of one control step on the truth's sampled states against the MuJoCo
C float64 finite-difference truth (truth/store.py), per group, state subset and input block, with the truth's own
floor. numpy only (the truth's corpus is read through benchmarks.bench.problem).

Objects compared:
  vjp       the VJPs of the corpus's C = 3 stored output cotangents (cot_qpos (qpos form), cot_qvel), (N, C, ncol),
            ncol = 2 nv + nu in the columns qpos tangent | qvel | ctrl. A VJP source gives them directly (the harness's
            gradient dump grad<c>_<qpos|qvel|ctrl>, qpos in the tangent form); a Jacobian source (B, nq + nv, ncol)
            (op jac: rows qpos' in qpos form then qvel') is contracted with the same cotangents (the truth's own
            arithmetic), so every source is scored on the same object
  jacobian  a Jacobian source against the truth's full Jacobian on the states that keep it (the truth file's
            jac_index; default the G0 sample), per input block over all 71 output rows
Per state, cotangent and block (qpos 35 / qvel 35 / ctrl 29), on the block's valid columns (truth bits 0):
  e = |g - fd| / max(|fd|, 1e-9), cos = g . fd / (|g| |fd|)
  floor f = |fd(h = 1e-7) - fd(h = 1e-6)| / max(|fd|, 1e-9)       the truth's own uncertainty on those columns
  floored e_f = max(e, f): no source can be shown closer to the truth than the truth's floor
  (jacobian: Frobenius norms over the block's valid columns; f from the stored column norms of J(1e-6) - J(1e-7),
  at least 2^-24, the float32 storage of the truth Jacobian)
Statistics over the usable (state, cotangent) pairs of a group and subset (stats.summary): median, p90, p99, max,
share above 1e-3, cos_min, cos_p1; the top-level numbers of an entry are the floored ones, `raw` the unfloored ones,
`truth_floor` the floor's own; plus valid_col_share, states without a valid column, non-finite states (a source state
with a non-finite gradient is counted and left out), missing states (not in the source).

Groups: the truth's sampled G0, G4, G5, G6 and G1 (report only) states (truth/mjc.py GROUP_DEFS, MuJoCo C only).
Subsets: all sampled states of the group, and model_ok (MuJoCo C and native MuJoCo Warp 3.17.0 agree on contacts,
zones and frames). Semantics of the source (clamp_kv, normal; the truth is the exact derivative, clamp_kv 1 / normal
0): clamp_kv 0 (the -kv term kept on clamped dofs) has no matching truth on G4 (not evaluated) and leaves out the
clamped states of G1; normal != 0 (the self-contact normal frozen below a distance) has no matching truth on G1 (not
evaluated). MuJoCo C's own FD rows record reference_relation(): "self-reference" for the truth's own construction
(direct, centered, tight options, h = 1e-6), "semi-self-referential" for centered differences of MuJoCo C's float64
map at the truth's steps (chain centered 1e-6 among them).

Partial sources: an engine whose native gradient covers only some input blocks dumps those blocks only (grad_blocks
lists them); its other blocks are scored as not supported (entries with status not_supported and the engine's reason,
the dump cell's gradient_not_supported), and a state counts as non-finite on its supported blocks only.

Paired view (paired): two sources on the same truth states and cotangents, per group, subset and block: the ratio of
their median floored errors, the median per-pair ratio and the share of pairs where the first is at least as close.

  from benchmarks.bench.accuracy import gradient as AG
  GT = AG.truth_from_store(store.load("mf_s0"))
  src = AG.source_from_dump("runs/bench/x/raw/<cell>.npz", engine="dk")
  entries, per_state = AG.score(src, GT)

  python -m benchmarks.bench.accuracy.gradient --truth mf_s0 --raw DUMP --engine dk --out X.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov
from benchmarks.bench.accuracy import stats

BLOCKS = ("qpos", "qvel", "ctrl")
DEFAULT_SUBSETS = ("all", "model_ok")
EXACT = dict(clamp_kv=1, normal=0)
F32_FLOOR = 2.0 ** -24  # the float32 storage of the truth Jacobian
LABEL_G4, LABEL_G1_1MM = 16, 2  # truth/mjc.py LABEL_BITS: G4_clamped, G1_axis_lt_1mm


# ------------------------------------------------------------------------------------------------- truth

def truth_from_store(tr):
    """The gradient truth of a store.Truth with its corpus's stored cotangents at the sampled states."""
    g = tr.gradient()
    c = tr.corpus()
    idx = g["index"]
    return dict(name=tr.name, path=prov.rel(tr.path), format=tr.meta["format"], index=idx, fd=g["fd_vjp"],
                floor=g["fd_vjp_floor"], bad=g["bad"], groups=g["groups"], model_ok=g["model_ok"],
                clamped=(g["labels"] & LABEL_G4) != 0, axis_lt_1mm=(g["labels"] & LABEL_G1_1MM) != 0,
                jac_index=g["jac_index"], jacobian=g["jacobian"], jacobian_dcol=g["jacobian_dcol"], blocks=g["blocks"],
                cot_qpos=c.cot_qpos[idx], cot_qvel=c.cot_qvel[idx])



# ------------------------------------------------------------------------------------------------- sources

def source_from_dump(path_or_dict, **meta):
    """A source from a raw dump (npz path or dict): index (B,), vjp (B, C, ncol) from grad<c>_<qpos|qvel|ctrl> and / or
    jacobian (B, nq + nv, ncol); meta: engine, variant, label, iterations, semantics, fd, ..."""
    d = path_or_dict
    if not isinstance(d, dict):
        p = Path(d)
        p = p if p.is_absolute() else P.REPO / p
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta.setdefault("raw", prov.rel(p))
    out = dict(meta, index=np.asarray(d["index"], np.int64), vjp=None, jacobian=None)
    present = [b for b in BLOCKS if f"grad0_{b}" in d]
    C = 0
    while present and f"grad{C}_{present[0]}" in d:
        C += 1
    if C and len(present) == len(BLOCKS):
        out["vjp"] = np.stack([np.concatenate([np.asarray(d[f"grad{c}_{b}"], np.float64) for b in BLOCKS], 1)
                               for c in range(C)], 1)
    elif C:  # a partial gradient: the blocks the engine supports, laid out against the truth's columns later
        if "grad_blocks" in d and sorted(map(str, d["grad_blocks"])) != sorted(present):
            raise ValueError(f"the dump's grad_blocks {list(d['grad_blocks'])} differ from its arrays {present}")
        out["vjp_parts"] = {b: np.stack([np.asarray(d[f"grad{c}_{b}"], np.float64) for c in range(C)], 1)
                            for b in present}
        out["blocks_supported"] = present
    if "jacobian" in d:
        out["jacobian"] = np.asarray(d["jacobian"], np.float64)
    if out["vjp"] is None and out["jacobian"] is None and not out.get("vjp_parts"):
        raise KeyError("the dump has neither gradients (grad<c>_*) nor a jacobian")
    return out


def _full_vjp(src, GT):
    """A partial source's VJPs (B, C, ncol) in the truth's column layout, nan on the blocks it does not support."""
    parts = src["vjp_parts"]
    B, C = next(iter(parts.values())).shape[:2]
    ncol = GT["fd"].shape[2]
    v = np.full((B, C, ncol), np.nan)
    for b, sl in GT["blocks"].items():
        if b in parts:
            if parts[b].shape[2] != sl.stop - sl.start:
                raise ValueError(f"source block {b} has {parts[b].shape[2]} columns, the truth {sl.stop - sl.start}")
            v[:, :, sl] = parts[b]
    return v


def _rows(src_index, truth_index):
    """Rows of the source for each truth state (-1: missing); the first occurrence of a repeated state."""
    order = np.argsort(src_index, kind="stable")
    s = src_index[order]
    pos = np.searchsorted(s, truth_index)
    pos = np.minimum(pos, len(s) - 1)
    hit = (len(s) > 0) & (s[pos] == truth_index)
    return np.where(hit, order[pos], -1)


def contract(J, cot_qpos, cot_qvel):
    """VJPs (N, C, ncol) of Jacobians J (N, nq + nv, ncol) for cotangents (N, C, nq), (N, C, nv) (truth/mjc.vjp's
    arithmetic)."""
    nq = cot_qpos.shape[2]
    cq = lambda c: np.ascontiguousarray(cot_qpos[:, c]).astype(np.float64)  # noqa: E731
    cv = lambda c: np.ascontiguousarray(cot_qvel[:, c]).astype(np.float64)  # noqa: E731
    return np.stack([np.einsum("wr,wrc->wc", cq(c), J[:, :nq]) + np.einsum("wr,wrc->wc", cv(c), J[:, nq:])
                     for c in range(cot_qpos.shape[1])], 1)


def vjp_at_truth(src, GT, prefer="vjp"):
    """(g (N, C, ncol) with nan rows for missing states, missing (N,) bool, how: 'vjp' | 'jacobian')."""
    rows = _rows(np.asarray(src["index"], np.int64), GT["index"])
    missing = rows < 0
    N, C, ncol = GT["fd"].shape
    g = np.full((N, C, ncol), np.nan)
    have = ~missing
    if src.get("vjp") is None and src.get("vjp_parts"):
        src = dict(src, vjp=_full_vjp(src, GT))
    use_vjp = src.get("vjp") is not None and (prefer == "vjp" or src.get("jacobian") is None)
    if use_vjp:
        v = src["vjp"]
        if v.shape[1] < C or v.shape[2] != ncol:
            raise ValueError(f"source gradients {v.shape[1:]} do not fit the truth's ({C}, {ncol})")
        g[have] = v[rows[have], :C]
        how = "vjp"
    else:
        J = src["jacobian"]
        if J.shape[2] != ncol:
            raise ValueError(f"source Jacobian {J.shape[1:]} does not fit the truth's {ncol} columns")
        g[have] = contract(J[rows[have]], GT["cot_qpos"][have], GT["cot_qvel"][have])
        how = "jacobian"
    return g, missing, how


# ------------------------------------------------------------------------------------------------- errors

def block_errors(g, fd, valid, blocks):
    """{block: (e, cos) (N, C)} on the valid columns of each block, nan where a state has none."""
    out = {}
    for name, sl in blocks.items():
        e = np.full(g.shape[:2], np.nan)
        cos = np.full(g.shape[:2], np.nan)
        for w in range(len(g)):
            v = valid[w, sl]
            if not v.any():
                continue
            a, b = g[w, :, sl][:, v], fd[w, :, sl][:, v]
            nb = np.linalg.norm(b, axis=1)
            e[w] = np.linalg.norm(a - b, axis=1) / np.maximum(nb, 1e-9)
            cos[w] = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * nb, 1e-30)
        out[name] = (e, cos)
    return out


def jacobian_errors(Js, Jt, dcol, valid, blocks):
    """{block: (e, floor) (Nj,)}: Frobenius relative error over the block's valid columns and the truth's floor."""
    out = {}
    for name, sl in blocks.items():
        e = np.full(len(Js), np.nan)
        f = np.full(len(Js), np.nan)
        for w in range(len(Js)):
            v = valid[w, sl]
            if not v.any():
                continue
            a, b = Js[w][:, sl][:, v], Jt[w][:, sl][:, v]
            nb = max(np.linalg.norm(b), 1e-9)
            e[w] = np.linalg.norm(a - b) / nb
            f[w] = max(np.linalg.norm(dcol[w, sl][v]) / nb, F32_FLOOR)
        out[name] = (e, f)
    return out


def reference_relation(fd):
    """'self-reference' | 'semi-self-referential' | None for a MuJoCo C FD source described by fd = dict(construction
    'chain' | 'direct', centered, eps, options 'train' | 'tight') (module docstring)."""
    if not fd:
        return None
    eps, centered = float(fd.get("eps", 0)), bool(fd.get("centered"))
    if fd.get("construction") == "direct" and centered and eps == 1e-6 and fd.get("options") == "tight":
        return "self-reference"
    if centered and eps in (1e-6, 1e-7):
        return "semi-self-referential"
    return None


def plan(group, semantics):
    """(not_evaluated reason or None, {exclusion: mask key}) of a group for a source's semantics."""
    kv, nrm = int(semantics.get("clamp_kv", 1)), int(semantics.get("normal", 0))
    excl = {}
    if nrm != 0 and group == "G1":
        return (f"normal={nrm}: the source freezes the self-contact normal below its threshold, where G1's states are "
                "(axis distance < 5 mm); the truth has the exact normal derivative"), excl
    if kv == 0 and group == "G4":
        return ("clamp_kv=0: the source keeps the -kv term on clamped dofs; the truth is the exact derivative "
                "(no kv0 truth in this file)"), excl
    if kv == 0:
        excl["clamped"] = "clamped"
    if nrm != 0:
        excl["axis_lt_1mm"] = "axis_lt_1mm"
    return None, excl


def _summ(e, cos=None):
    s = stats.summary(e, cos)
    s.pop("nonfinite", None)
    return s


def score(src, GT, semantics=None, subsets=DEFAULT_SUBSETS, prefer="vjp"):
    """(entries, per_state) of one source (module docstring). semantics: dict(clamp_kv, normal), default the source's
    src["semantics"] or the exact mode."""
    sem = dict(EXACT, **(semantics or src.get("semantics") or {}))
    g, missing, how = vjp_at_truth(src, GT, prefer)
    valid = GT["bad"] == 0
    blocks = GT["blocks"]
    err = block_errors(g, GT["fd"], valid, blocks)
    flo = block_errors(GT["floor"], GT["fd"], valid, blocks) if GT.get("floor") is not None else None
    supported = src.get("blocks_supported")
    if supported is None:
        nonfin = ~missing & ~np.isfinite(g).all((1, 2))
    else:  # a partial source: finite on the blocks it supports
        cols = np.zeros(g.shape[2], bool)
        for b, sl in blocks.items():
            cols[sl] = b in supported
        nonfin = ~missing & ~np.isfinite(g[:, :, cols]).all((1, 2))
    rel = reference_relation(src.get("fd"))
    base = {k: src.get(k) for k in ("engine", "variant", "label", "iterations", "raw") if src.get(k) is not None}
    if src.get("fd"):
        base["fd"] = src["fd"]
    if rel:
        base["reference_relation"] = rel
    N, C, _ = g.shape
    entries, per_state = [], dict(index=GT["index"], missing=missing, nonfinite=nonfin, semantics=sem, how=how)
    efl = {}
    for b in blocks:
        e, cos = err[b]
        f = flo[b][0] if flo is not None else np.full_like(e, np.nan)
        efl[b] = np.where(np.isfinite(f), np.fmax(e, f), e)
        per_state.update({f"e_{b}": e, f"cos_{b}": cos, f"ef_{b}": efl[b], f"floor_{b}": f})
    mo = GT.get("model_ok")
    excl_masks = dict(clamped=GT.get("clamped"), axis_lt_1mm=GT.get("axis_lt_1mm"))
    for gname, gm in GT["groups"].items():
        reason, excl = plan(gname, sem)
        for sub in subsets:
            ent = dict(kind="gradient", object="vjp", how=how, **base, corpus=GT["name"], truth=GT["path"],
                       group=gname, subset=sub, semantics=sem)
            if sub == "model_ok":
                if mo is None:
                    entries.append(dict(ent, status="not_available", note="the truth has no model check"))
                    continue
                use = gm & mo
            elif sub == "all":
                use = gm.copy()
            else:
                raise KeyError(f"unknown subset {sub!r} (all, model_ok)")
            ent.update(group_states=int(gm.sum()), states=int(use.sum()), excluded_model_mismatch=int(
                (gm & ~mo).sum()) if mo is not None else None)
            if reason:
                entries.append(dict(ent, status="not_evaluated", note=reason))
                continue
            for k in excl:
                m = excl_masks.get(k)
                if m is not None:
                    ent[f"excluded_{k}"] = int((use & m).sum())
                    use &= ~m
            usable = use & ~missing & ~nonfin
            ent.update(missing_states=int((use & missing).sum()), nonfinite_states=int((use & nonfin).sum()),
                       usable_states=int(usable.sum()), n_cot=C)
            for b, sl in blocks.items():
                if supported is not None and b not in supported:
                    entries.append(dict(ent, block=b, status="not_supported",
                                        note=(src.get("not_supported") or {}).get(b, "not supported by the source")))
                    continue
                e, cos = err[b]
                vb = valid[use][:, sl]
                eb = dict(ent, block=b, valid_col_share=float(vb.mean()) if vb.size else None,
                          states_without_valid_column=int((~vb.any(1)).sum()) if vb.size else 0)
                fl = _summ(efl[b][usable], cos[usable])
                eb.update(fl)
                eb["raw"] = _summ(e[usable], cos[usable])
                eb["truth_floor"] = _summ(per_state[f"floor_{b}"][usable]) if flo is not None else None
                eb["share_at_floor"] = (float(np.mean((e[usable] <= per_state[f"floor_{b}"][usable])[
                    np.isfinite(e[usable])])) if flo is not None and np.isfinite(e[usable]).any() else None)
                entries.append(eb)
    # the full Jacobian on the states that keep it
    if src.get("jacobian") is not None and GT.get("jacobian") is not None and len(GT["jac_index"]):
        ji = GT["jac_index"]
        rows = _rows(np.asarray(src["index"], np.int64), GT["index"][ji])
        have = rows >= 0
        Js = np.full((len(ji),) + GT["jacobian"].shape[1:], np.nan)
        Js[have] = src["jacobian"][rows[have]]
        je = jacobian_errors(Js, GT["jacobian"].astype(np.float64), GT["jacobian_dcol"].astype(np.float64),
                             valid[ji], blocks)
        jnf = have & ~np.isfinite(Js).all((1, 2))
        for gname, gm in GT["groups"].items():
            reason, excl = plan(gname, sem)
            gj = gm[ji]
            if not gj.any() or reason:
                continue
            for sub in subsets:
                if sub == "model_ok" and mo is None:
                    continue
                use = gj & (mo[ji] if sub == "model_ok" else True)
                usable = use & have & ~jnf
                for b in blocks:
                    e, f = je[b]
                    ent = dict(kind="gradient", object="jacobian", how="jacobian", **base, corpus=GT["name"],
                               truth=GT["path"], group=gname, subset=sub, semantics=sem, block=b,
                               states=int(use.sum()), usable_states=int(usable.sum()),
                               nonfinite_states=int((use & jnf).sum()), missing_states=int((use & ~have).sum()))
                    ent.update(_summ(np.where(np.isfinite(f), np.fmax(e, f), e)[usable]))
                    ent["raw"] = _summ(e[usable])
                    ent["truth_floor"] = _summ(f[usable])
                    entries.append(ent)
    per_state["groups"] = GT["groups"]
    per_state["model_ok"] = mo
    return entries, per_state


def paired(a, b, subsets=DEFAULT_SUBSETS, blocks=BLOCKS, label_a=None, label_b=None):
    """Entries of the paired view of two score() per_state results on the same truth (module docstring)."""
    if not np.array_equal(a["index"], b["index"]):
        raise ValueError("paired: the two scores are on different truth states")
    out = []
    for gname, gm in a["groups"].items():
        for sub in subsets:
            if sub == "model_ok" and a.get("model_ok") is None:
                continue
            use = gm & (a["model_ok"] if sub == "model_ok" else True)
            for blk in blocks:
                x, y = a[f"ef_{blk}"][use], b[f"ef_{blk}"][use]
                ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                ent = dict(kind="gradient_paired", a=label_a, b=label_b, group=gname, subset=sub, block=blk,
                           pairs=int(ok.sum()))
                if ok.any():
                    ent.update(median_ratio=stats.median_ratio(x[ok], y[ok]),
                               pair_ratio_median=float(np.exp(np.median(np.log(x[ok] / y[ok])))),
                               share_a_le_b=float(np.mean(x[ok] <= y[ok])))
                out.append(ent)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", required=True, help="corpus name or truth file")
    ap.add_argument("--raw", action="append", default=[], help="raw dump npz (repeat)")
    ap.add_argument("--engine", action="append", default=[])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--clamp-kv", action="append", type=int, default=[], help="source semantics (default 1)")
    ap.add_argument("--normal", action="append", type=int, default=[], help="source semantics (default 0)")
    ap.add_argument("--prefer", default="vjp", choices=("vjp", "jacobian"),
                    help="which object of a dump that has both gives the VJPs")
    ap.add_argument("--paired", action="store_true", help="also the paired view of every pair of sources")
    ap.add_argument("--out", help="JSON of the entries (default: stdout)")
    a = ap.parse_args(argv)
    from benchmarks.bench.truth import store
    GT = truth_from_store(store.load(a.truth))
    out, ps = [], []
    for i, r in enumerate(a.raw):
        pick = lambda lst, d=None: lst[i] if i < len(lst) else d  # noqa: E731
        src = source_from_dump(r, engine=pick(a.engine, f"source{i}"), variant=pick(a.variant, "default"),
                               semantics=dict(clamp_kv=pick(a.clamp_kv, 1), normal=pick(a.normal, 0)))
        ent, st = score(src, GT, prefer=a.prefer)
        out += ent
        ps.append((f"{src['engine']}~{src['variant']}", st))
    if a.paired:
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                out += paired(ps[i][1], ps[j][1], label_a=ps[i][0], label_b=ps[j][0])
    text = json.dumps(dict(truth=GT["path"], entries=out), indent=1)
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
