"""Gradient check of the JAX interface: the VJPs of one control step through nablaengine.jax_ffi against the MuJoCo C
float64 finite-difference truth of the benchmark corpora, per state group, with pass thresholds.

Per corpus (default mf_s0, mf_s1): the truth file benchmarks/data/truth/<corpus>.npz (benchmarks/bench/truth: the
sampled states of the groups G0 (normal), G4 (a PD force at its limit), G5 (a joint limit), G6 (a self contact) and
G1 (self-contact axes within 5 mm, reported only), and per state the VJPs of the corpus's three stored output
cotangents through the float64 MuJoCo C step, centered differences at h = 1e-6, with their floor
|fd(1e-7) - fd(1e-6)|). The engine: jax_ffi.forward(tape=True) on those states (4 substeps, --iterations, default 50,
the truth's reference count) and jax_ffi.backward of the same three cotangents in the exact gradient mode
(clamp_kv=1: no -kv term on the dofs whose PD force is clamped; normal=0: the exact self-contact normal; the truth's
semantics), the qpos gradient mapped to the tangent form at the input state. Scoring: the benchmark's gradient scorer
(benchmarks/bench/accuracy/gradient.py: per group, subset and input block (qpos tangent 35, qvel 35, ctrl 29) on the
block's valid columns, e = |g - fd| / max(|fd|, 1e-9) floored at the truth's own floor, and the cosine).

Gates (THRESHOLDS, from the levels measured on mf_s0 and mf_s1 with a margin of about 3): per gated group (--groups,
default G0, G4, G5, G6), on the subset model_ok (MuJoCo C and native MuJoCo Warp agree on the state's contacts, zones
and frames; the subset "all", whose other states differ in the contact model itself, is reported) and per block: the
floored p99 <= THRESHOLDS[block] and the 1st percentile of the cosine >= COS_P1; usable states in every gated group;
every gradient finite. Self-test (perturbed_must_fail): the same gradients moved by --perturb (default 3e-3) relative
per state, cotangent and block, in a random direction, must fail the G0 gate. Exit status 0 when every gate passes and
the self-test fails, 1 otherwise. Other gradient modes (--clamp-kv 0 --normal 3, the interface default) are scored where the truth applies
(benchmarks/bench/accuracy/gradient.py plan: G4 not evaluated with clamp_kv 0, G1 not with normal != 0) against the
same thresholds.

  PYTHONPATH=src python scripts/gradient_check.py --out runs/gradient_check.json
  PYTHONPATH=src python scripts/gradient_check.py --corpora mf_s0 --groups G0 --out runs/gradient_quick.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the benchmark package (benchmarks/)

from benchmarks.bench.accuracy import gradient as BG  # noqa: E402
from benchmarks.bench.truth import store  # noqa: E402

from nablaengine import fixtures, provenance  # noqa: E402
from nablaengine.reference.adjoint_groups import qpos_tangent  # noqa: E402

INPUTS = ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")
GROUPS = ("G0", "G4", "G5", "G6")
BLOCKS = BG.BLOCKS
SUBSET = "model_ok"
# per block: the bound of the floored relative error's p99 in every gated group (model_ok), and of the cosine's 1st
# percentile. Measured (V100, exact mode, 50 iterations; mf_s0 / mf_s1, the largest over G0, G4, G5, G6): p99 qpos
# 5.2e-4 / 6.4e-4, qvel 1.1e-4 / 1.5e-4, ctrl 1.5e-5 / 2.4e-5; max qpos 1.0e-3 / 6.8e-4, qvel 1.3e-4 / 2.1e-4, ctrl
# 2.3e-5 / 3.2e-5; the smallest cosine 0.9999995 / 0.9999998. The self-test's 3e-3 perturbation exceeds each bound.
THRESHOLDS = dict(qpos=2e-3, qvel=5e-4, ctrl=1e-4)
COS_P1 = 0.99999


def engine_vjps(corpus, index, iterations, clamp_kv, normal):
    """{index, grad{c}_qpos (tangent form) / qvel / ctrl (N, .) float64, nonfinite_rows}: jax_ffi's VJPs of the
    corpus's three stored cotangents at its states `index`."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    F.init()
    z = fixtures.load_corpus(corpus, adjoint=True)
    a = [jnp.asarray(np.asarray(z[k])[index], jnp.float32) for k in INPUTS]
    fwd = jax.jit(lambda *x: F.forward(*x, iterations=iterations, tape=True))
    bwd = jax.jit(lambda tp, fr, com, cq, cv: F.backward(tp, fr, com, cq, cv, clamp_kv=clamp_kv, normal=normal))
    tp = fwd(*a)[4]
    q64 = np.asarray(z["qpos"], np.float64)[index]
    out = dict(index=np.asarray(index, np.int64))
    F.read_diagnostics(reset=True)
    for c in range(3):
        gq, gv, gu, _ = bwd(tp, a[4], a[5], jnp.asarray(np.asarray(z[f"cot{c}_qpos"])[index]),
                            jnp.asarray(np.asarray(z[f"cot{c}_qvel"])[index]))
        out[f"grad{c}_qpos"] = qpos_tangent(np.asarray(gq, np.float64), q64)
        out[f"grad{c}_qvel"] = np.asarray(gv, np.float64)
        out[f"grad{c}_ctrl"] = np.asarray(gu, np.float64)
    out["nonfinite_rows"] = int(F.read_diagnostics(reset=True)["nonfinite_rows"])
    return out


def perturb(src, rel, seed=0):
    """src's VJPs moved by rel relative per state, cotangent and block, in a random direction."""
    rng = np.random.default_rng(seed)
    out = dict(src, vjp=src["vjp"].copy())
    v = out["vjp"]
    nv, nu = (v.shape[2] - 29) // 2, 29
    for sl in (slice(0, nv), slice(nv, 2 * nv), slice(2 * nv, 2 * nv + nu)):
        b = v[:, :, sl]
        u = rng.standard_normal(b.shape)
        u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-30)
        v[:, :, sl] = b + rel * np.linalg.norm(b, axis=-1, keepdims=True) * u
    return out


def summarize(entries):
    """{group: {subset: {block: numbers}}} of the scorer's VJP entries."""
    keep = ("status", "note", "usable_states", "n", "median", "p99", "max", "cos_min", "cos_p1", "valid_col_share")
    out = {}
    for e in entries:
        if e.get("object") != "vjp":
            continue
        d = out.setdefault(e["group"], {}).setdefault(e["subset"], {})
        if e.get("status") in ("not_evaluated", "not_available"):
            d["status"] = e["status"]
            d["note"] = e.get("note")
            continue
        d[e["block"]] = {k: e[k] for k in keep if k in e}
        d[e["block"]]["truth_floor_p99"] = (e.get("truth_floor") or {}).get("p99")
    return out


def gate(summary, groups, thresholds=THRESHOLDS, subset=SUBSET):
    """(pass, {group: verdict}) of the gated groups (module docstring)."""
    verdicts, ok = {}, True
    for g in groups:
        s = summary.get(g, {}).get(subset)
        if s is None:
            verdicts[g] = dict(pass_=False, reason="not in the truth")
        elif s.get("status") == "not_evaluated":
            verdicts[g] = dict(pass_=None, reason=s.get("note"))
            continue
        else:
            bad = []
            for b in BLOCKS:
                e = s.get(b, {})
                if not e.get("usable_states"):
                    bad.append(f"{b}: no usable states")
                    continue
                if not e["p99"] <= thresholds[b]:
                    bad.append(f"{b} p99 {e['p99']:.3g} > {thresholds[b]:g}")
                if not e.get("cos_p1", 1.0) >= COS_P1:
                    bad.append(f"{b} cos_p1 {e['cos_p1']:.9f} < {COS_P1}")
            verdicts[g] = dict(pass_=not bad, failed=bad)
        ok &= bool(verdicts[g]["pass_"])
    return ok, verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpora", default="mf_s0,mf_s1")
    ap.add_argument("--groups", default=",".join(GROUPS), help="the gated groups (the others are reported)")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--clamp-kv", type=int, default=1)
    ap.add_argument("--normal", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=3e-3, help="the self-test's relative perturbation (0: none)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    groups = [g for g in a.groups.split(",") if g]
    sem = dict(clamp_kv=a.clamp_kv, normal=a.normal)
    res = dict(iterations=a.iterations, semantics=sem, gated_groups=groups, subset=SUBSET, thresholds=THRESHOLDS, cos_p1=COS_P1,
               corpora={})
    ok = True
    for corpus in a.corpora.split(","):
        GT = BG.truth_from_store(store.load(corpus))
        d = engine_vjps(corpus, GT["index"], a.iterations, a.clamp_kv, a.normal)
        src = BG.source_from_dump(d, engine="dk", variant=f"kv{a.clamp_kv}_n{a.normal}", iterations=a.iterations,
                                  semantics=sem)
        entries, ps = BG.score(src, GT, sem)
        summ = summarize(entries)
        finite = bool(np.isfinite(src["vjp"]).all()) and d["nonfinite_rows"] == 0
        g_ok, verdicts = gate(summ, groups)
        r = dict(truth=GT["path"], states=int(len(GT["index"])), finite=finite, nonfinite_rows=d["nonfinite_rows"],
                 groups=summ, gates=verdicts)
        if a.perturb > 0 and "G0" in groups:
            pe, _ = BG.score(perturb(src, a.perturb), GT, sem)
            p_ok, p_verdict = gate(summarize(pe), ["G0"])
            r["perturbed_must_fail"] = dict(rel=a.perturb, G0=p_verdict["G0"], failed=not p_ok)
            g_ok &= not p_ok
        r["pass"] = bool(g_ok and finite)
        ok &= r["pass"]
        res["corpora"][corpus] = r
        for g, v in summ.items():
            s = v.get(SUBSET, {})
            line = " ".join(f"{b} p99 {s[b]['p99']:.2e} cos_p1 {s[b]['cos_p1']:.8f}" if s[b].get("n") else
                            f"{b} no usable states" for b in BLOCKS if b in s)
            print(f"{corpus} {g} {SUBSET}: {line or s.get('status')} {verdicts.get(g, {}).get('pass_', 'reported')}",
                  flush=True)
    res["pass"] = bool(ok)
    res["provenance"] = provenance.provenance()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"gradient check ({a.corpora}; groups {','.join(groups)}): {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
