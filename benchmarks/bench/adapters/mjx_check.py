"""Self-check of the MJX adapter (benchmarks/bench/adapters/mjx.py) against MuJoCo C float64 and against JAX itself:

  1. tangent map: mjx.tangent_map (the qpos-tangent input directions of op jac) equals the centered difference of
     mj_integratePos(qpos, +-h e_j) (float64, h = 1e-7) on the selected states;
  2. forward: the adapter's outputs() (MJX float32) against MuJoCo C float64 at the same options (problem.load_model,
     problem.DR per world, `substeps` mj_step with ctrl held), per world ||a - b|| / max(||b||, 1e-9) over qpos (nq)
     and qvel (nv), plus the qpos error in the tangent space (mj_differentiatePos)
     without the base translation; median / p99 / max over the worlds;
  3. Jacobian: on --jac-states states of group G0 (MuJoCo C labels of the corpus: contact, no self-contact, no clamped
     actuator, no limit row, solver converged; any states for a corpus without labels), the adapter's jacobian()
     (chunked lax.map of jax.jvp) against (a) a direct, unchunked jax.vmap(jax.jvp) of the same control step along a
     every column (float32 rounding: the two executables fuse and round differently, which the solver iterations
     amplify in ill-conditioned directions) and (b) MuJoCo C float64 centered differences of the whole control
     step with step --fd-eps (qpos perturbed with mj_integratePos), per block (qpos tangent / qvel / ctrl): relative
     Frobenius error and cosine per state;
  4. reverse mode: the message the adapter recorded for bwd / vjp (the exception JAX raised for jax.vjp).

  python -m benchmarks.bench.adapters.mjx_check --out runs/bench/test_mjx/check_mjlab.json
  python -m benchmarks.bench.adapters.mjx_check --problem menagerie_g1 --corpus menagerie_s0 --iterations native \\
      --out runs/bench/test_mjx/check_menagerie.json
  python -m benchmarks.bench.adapters.mjx_check --corpus mf_s1 --index 11,21,33 ...   # explicit corpus rows

Gates (exit status 1 when one fails): tangent map within 1e-7 of the difference quotient (relative to the column's
norm); finite MJX outputs; median forward qvel error below --fwd-gate (default 1e-3: catches a wrong model, DR or
state layout, not MJX's float32 and solver differences); chunked Jacobian columns equal to the direct jax.jvp: per
column relative difference median below --jvp-gate-median (1e-5) and max below --jvp-gate-max (1e-3) (a wrong
column order, tangent or padding gives O(1); float32 rounding gives a median near 1e-7 and isolated 1e-5..1e-4). The
MuJoCo C FD comparison is reported, not gated (MJX runs in float32 and its solver is not MuJoCo C's). Run it on one
GPU (CUDA_VISIBLE_DEVICES); the result JSON records repository-relative paths and the provenance.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.bench import prov, schema
from benchmarks.bench import problem as P

G0_FALSE = ("label_self_contact", "label_clamped", "label_limit", "label_nonconv")


def _stats(e):
    e = np.asarray(e, np.float64)
    return dict(median=float(np.median(e)), p99=float(np.quantile(e, 0.99)), max=float(e.max()), n=int(e.size))


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-9)


def states_at(corpus, idx):
    """problem.States of the corpus rows idx (in that order)."""
    idx = np.asarray(idx, np.int64)
    a = corpus.arrays
    return P.States(corpus=corpus.name, subset=0, nworld=len(idx), seed=-1, index=idx, tiled=1.0, qpos=a["qpos"][idx],
                    qvel=a["qvel"][idx], qacc_warmstart=a["qacc_warmstart"][idx], ctrl=a["ctrl"][idx],
                    friction=a["friction"][idx] if "friction" in a else None,
                    com=a["com"][idx] if "com" in a else None,
                    cot_qpos=corpus.cot_qpos[idx] if corpus.cot_qpos is not None else None,
                    cot_qvel=corpus.cot_qvel[idx] if corpus.cot_qvel is not None else None)


class MjcStep:
    """MuJoCo C float64 control step of a problem from float32 corpus inputs (exactly converted)."""

    def __init__(self, prob, iterations):
        import mujoco
        self.mujoco = mujoco
        self.m = P.load_model(prob, iterations)
        self.d = mujoco.MjData(self.m)
        self.dr = P.DR(self.m, prob)
        self.substeps = prob.substeps

    def __call__(self, qpos, qvel, warm, ctrl, friction=None, com=None):
        mj, m, d = self.mujoco, self.m, self.d
        if self.dr.enabled:
            self.dr.apply(m, friction, com)
        mj.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.ctrl[:] = qpos, qvel, warm, ctrl
        for _ in range(self.substeps):
            mj.mj_step(m, d)
        return d.qpos.copy(), d.qvel.copy()

    def world(self, s, w):
        x = s.inputs(np.float64)
        dr = (x["friction"][w], x["com"][w]) if "friction" in x else (None, None)
        return x["qpos"][w], x["qvel"][w], x["qacc_warmstart"][w], x["ctrl"][w], dr

    def fd_jacobian(self, s, w, eps):
        """(nq + nv, 2 nv + nu) centered differences of the control step of world w (qpos via mj_integratePos)."""
        q, v, a, u, (f, c) = self.world(s, w)
        nq, nv, nu = self.m.nq, self.m.nv, self.m.nu
        J = np.zeros((nq + nv, 2 * nv + nu))
        for j in range(2 * nv + nu):
            outs = []
            for sgn in (1.0, -1.0):
                qq, vv, uu = q.copy(), v.copy(), u.copy()
                if j < nv:
                    dq = np.zeros(nv)
                    dq[j] = 1.0
                    self.mujoco.mj_integratePos(self.m, qq, dq, sgn * eps)
                elif j < 2 * nv:
                    vv[j - nv] += sgn * eps
                else:
                    uu[j - 2 * nv] += sgn * eps
                oq, ov = self(qq, vv, a, uu, f, c)
                outs.append(np.concatenate([oq, ov]))
            J[:, j] = (outs[0] - outs[1]) / (2 * eps)
        return J


def check_tangent_map(m, qpos, h=1e-7):
    import mujoco
    from benchmarks.bench.adapters import mjx as A
    M = A.tangent_map(m, qpos)
    worst = 0.0
    for b in range(len(qpos)):
        for j in range(m.nv):
            e = np.zeros(m.nv)
            e[j] = 1.0
            qp, qm = qpos[b].astype(np.float64), qpos[b].astype(np.float64)
            mujoco.mj_integratePos(m, qp, e, h)
            mujoco.mj_integratePos(m, qm, e, -h)
            fd = (qp - qm) / (2 * h)
            worst = max(worst, float(np.linalg.norm(M[b, :, j] - fd) / max(np.linalg.norm(fd), 1e-12)))
    return worst


def block_errors(J, R, nv):
    """Per block of columns (qpos tangent / qvel / ctrl): relative Frobenius error and cosine of J against R."""
    out = {}
    for name, sl in (("qpos", slice(0, nv)), ("qvel", slice(nv, 2 * nv)), ("ctrl", slice(2 * nv, None))):
        a, b = J[:, sl].ravel(), R[:, sl].ravel()
        out[name] = dict(rel=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
                         cos=float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)))
    return out


def direct_jvp_columns(adapter, cols):
    """Columns cols of the Jacobian by a direct, unchunked jax.vmap over worlds of jax.jvp (one direction each)."""
    jax, jnp = adapter.jax, adapter.jnp
    from benchmarks.bench.adapters import mjx as A
    T = A.tangent_map(adapter.m, adapter.qpos_host)  # (B, nq, nv)
    nq, nv, nu = adapter.nq, adapter.nv, adapter.nu
    dt = adapter.dtype
    step = adapter._step_world

    def world(m, q, v, w, u, tq, tv, tu):
        def f(q, v, u):
            o = step(m, q, v, w, u)
            return jnp.concatenate([o[0], o[1]])
        return jax.jvp(f, (q, v, u), (tq, tv, tu))[1]

    fn = jax.jit(adapter._batched(world, extra_axes=(0, 0, 0)))
    out = {}
    B = adapter.nworld
    for j in cols:
        tq, tv, tu = np.zeros((B, nq), dt), np.zeros((B, nv), dt), np.zeros((B, nu), dt)
        if j < nv:
            tq[:] = T[:, :, j].astype(dt)
        elif j < 2 * nv:
            tv[:, j - nv] = 1
        else:
            tu[:, j - 2 * nv] = 1
        out[j] = np.asarray(fn(*adapter.args, tq, tv, tu), np.float64)  # (B, nq + nv)
    return out


def run(a):
    from benchmarks.bench.adapters import mjx as A
    prob = P.get(a.problem, a.model_path)
    iterations = [None if it == "native" else int(it) for it in a.iterations.split(",")]
    corpus = P.load_corpus(a.corpus)
    if a.index:
        idx = np.array([int(x) for x in a.index.split(",")], np.int64)
    else:
        idx = P.selection(corpus.n, a.n, 0)
    res = dict(schema="dk-bench-mjx-check/1", created=schema.now(), cmd=prov.scrub(" ".join(sys.argv)),
               problem=prob.describe(), corpus=corpus.describe(), index=idx.tolist(), gates={}, runs={})
    ok = True
    s = states_at(corpus, idx)

    # 1. tangent map
    m0 = P.load_model(prob, iterations[0])
    worst = check_tangent_map(m0, s.qpos.astype(np.float64))
    res["tangent_map_max_rel"] = worst
    res["gates"]["tangent_map"] = worst < 1e-7
    print(f"tangent map vs mj_integratePos difference quotient: max rel {worst:.2e}", flush=True)

    # G0 states for the Jacobian
    ex = corpus.extra
    if all(k in ex for k in G0_FALSE) and "label_ncon_max" in ex:
        g0 = np.ones(corpus.n, bool)
        for k in G0_FALSE:
            g0 &= ~ex[k].astype(bool)
        g0 &= ex["label_ncon_max"] > 0
        cand = np.flatnonzero(g0)
        group = "G0 (corpus labels: contact, no self-contact, clamp, limit or non-convergence)"
    else:
        cand = np.arange(corpus.n)
        group = "any (the corpus has no labels)"
    jidx = cand[P.selection(len(cand), a.jac_states, 0)] if a.jac_states else np.array([], np.int64)

    for it in iterations:
        key = f"it{'native' if it is None else it}"
        R = dict(iterations=it)
        mjc = MjcStep(prob, it)
        # 2. forward
        ad = A.MjxAdapter()
        t0 = time.perf_counter()
        R["setup"] = ad.setup(prob, s, len(idx), it, **a.opts)
        R["setup_wall_s"] = time.perf_counter() - t0
        o = ad.outputs()
        ref = []
        for w in range(len(idx)):
            q, v, acc, u, (f, c) = mjc.world(s, w)
            ref.append(mjc(q, v, acc, u, f, c))
        rq, rv = np.array([r[0] for r in ref]), np.array([r[1] for r in ref])
        fin = bool(np.isfinite(o["qpos"]).all() and np.isfinite(o["qvel"]).all())
        tan = []
        for w in range(len(idx)):
            dq = np.zeros(mjc.m.nv)
            mjc.mujoco.mj_differentiatePos(mjc.m, dq, 1.0, rq[w], o["qpos"][w])
            tan.append(np.linalg.norm(dq[3:]))  # both tracks: the free joint first
        R["forward"] = dict(finite=fin, vs_mjc_same_options_qpos=_stats(rel_err(o["qpos"], rq)),
                            vs_mjc_same_options_qvel=_stats(rel_err(o["qvel"], rv)),
                            qpos_tangent_without_base_translation=_stats(tan),
                            metric="per world ||mjx - mjc|| / max(||mjc||, 1e-9); tangent: |differentiatePos|[3:]")
        R["diagnostics"] = ad.diagnostics()
        R["reverse_mode"] = ad.not_supported()["vjp"]
        R["provenance"] = ad.provenance()
        gate_f = fin and R["forward"]["vs_mjc_same_options_qvel"]["median"] < a.fwd_gate
        res["gates"][f"{key}_forward"] = gate_f
        print(f"{key} forward vs MuJoCo C: qpos {R['forward']['vs_mjc_same_options_qpos']}, qvel "
              f"{R['forward']['vs_mjc_same_options_qvel']}", flush=True)
        ad.close()
        del ad

        # 3. Jacobian on G0 states
        if len(jidx):
            sj = states_at(corpus, jidx)
            ad = A.MjxAdapter()
            ad.setup(prob, sj, len(jidx), it, **a.opts)
            t0 = time.perf_counter()
            J = ad.jacobian()
            jac_s = time.perf_counter() - t0
            cols = list(range(ad.ndir))
            direct = direct_jvp_columns(ad, cols)
            D = np.stack([direct[j] for j in cols], 2)  # (B, nq + nv, ndir)
            cr = np.linalg.norm(J - D, axis=1) / np.maximum(np.linalg.norm(D, axis=1), 1e-30)  # (B, ndir)
            jvp_stats = dict(_stats(cr.ravel()), worst=[int(x) for x in np.unravel_index(cr.argmax(), cr.shape)])
            fd = []
            for w in range(len(jidx)):
                Rf = mjc.fd_jacobian(sj, w, a.fd_eps)
                fd.append(dict(index=int(jidx[w]), **block_errors(J[w], Rf, ad.nv)))
            R["jacobian"] = dict(
                group=group, index=jidx.tolist(), seconds=jac_s, plan=ad.provenance().get("jac_dump"),
                direct_jvp_rel_per_column=jvp_stats, fd_eps=a.fd_eps, vs_mjc_fd=fd,
                vs_mjc_fd_median={blk: dict(rel=float(np.median([x[blk]["rel"] for x in fd])),
                                            cos=float(np.median([x[blk]["cos"] for x in fd])))
                                  for blk in ("qpos", "qvel", "ctrl")},
                finite=bool(np.isfinite(J).all()))
            res["gates"][f"{key}_jac_vs_direct_jvp"] = bool(jvp_stats["median"] < a.jvp_gate_median and
                                                            jvp_stats["max"] < a.jvp_gate_max)
            print(f"{key} jac ({len(jidx)} states, {jac_s:.1f} s): chunked vs direct jvp per column rel median "
                  f"{jvp_stats['median']:.2e} max {jvp_stats['max']:.2e}; vs "
                  f"MuJoCo C FD (median per block) {R['jacobian']['vs_mjc_fd_median']}", flush=True)
            ad.close()
            del ad
        res["runs"][key] = R
    ok = all(res["gates"].values())
    res["ok"] = ok
    res["provenance"] = prov.provenance()
    res["gpu"] = prov.gpu()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema.clean(res), indent=1, allow_nan=False))
    print(f"gates: {res['gates']} -> {'ok' if ok else 'FAIL'}; wrote {prov.rel(out)}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problem", default="mjlab", choices=sorted(P.PROBLEMS))
    ap.add_argument("--model-path", default=None, help="the problem's model file (default: the problem's)")
    ap.add_argument("--corpus", default="mf_s0")
    ap.add_argument("--n", type=int, default=64, help="states for the forward check (problem.selection subset 0)")
    ap.add_argument("--index", default=None, help="explicit corpus rows (comma-separated) instead of --n")
    ap.add_argument("--iterations", default="10,50", help="comma-separated; 'native' = the model's own (Track B)")
    ap.add_argument("--jac-states", type=int, default=4, help="G0 states for the Jacobian check (0: none)")
    ap.add_argument("--fd-eps", type=float, default=1e-6)
    ap.add_argument("--fwd-gate", type=float, default=1e-3)
    ap.add_argument("--jvp-gate-median", type=float, default=1e-5)
    ap.add_argument("--jvp-gate-max", type=float, default=1e-3)
    ap.add_argument("--opt", action="append", default=[], help="KEY=VALUE adapter option (JSON value)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.opts = {}
    for kv in a.opt:
        k, _, v = kv.partition("=")
        try:
            a.opts[k] = json.loads(v)
        except ValueError:
            a.opts[k] = v
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
