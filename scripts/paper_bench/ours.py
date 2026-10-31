"""Part `ours`: our engine through nablaengine.jax_ffi under jax.jit (the main environment).

Per corpus at --nworld (default 4096), 10 iterations, 4 substeps, AOT-compiled (compile time recorded per op):
  fwd        jax_ffi.step without differentiation (dk_step: the forward, no tape)
  fwd_tape   jax_ffi.forward(tape=True) (dk_step_fwd: the forward that writes the tape, the forward of a gradient)
  per gradient mode (exact: clamp_kv=1, normal=0, the headline; interface: clamp_kv=0, normal=3, the interface
  defaults):
    bwd      jax_ffi.backward on the tape (the adjoint alone), the corpus's stored cotangent 0
    vjp      jax.vjp of jax_ffi.step + the pullback, one jit (forward with tape + adjoint)
    jac_bwd  the Jacobian-equivalent: 71 backward calls on one tape (one-hot cotangents, one per output row of the
             71 x 99 Jacobian d(qpos' (36), qvel' (35)) / d(qpos tangent, qvel, ctrl)), lax.scan in one jit
    jac      forward with tape + jac_bwd in one jit (the whole Jacobian from the state); also per world
--sweep: fwd and vjp (exact mode) at each nworld on the first corpus (states tiled past the corpus size).
Records jax_ffi.init()'s build info (built now or loaded from the cache, its time) and launch_info().

  PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/paper_bench/ours.py --out <dir>/ours.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODES = {"exact": (1, 0), "interface": (0, 3)}
NOUT = 71  # qpos' (36) + qvel' (35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default=None, help="comma-separated (default: all of common.corpora())")
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--sweep", default="512,1024,4096,16384")
    ap.add_argument("--no-jac", action="store_true")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from nablaengine import jax_ffi as F
    from nablaengine import provenance

    res = C.Result(a.out, part="ours", what="nablaengine.jax_ffi under jax.jit", nworld=a.nworld,
                   iterations=a.iterations, substeps=a.substeps, modes={k: dict(clamp_kv=v[0], normal=v[1])
                                                                         for k, v in MODES.items()},
                   env=C.basic_env(("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda13-plugin", "mujoco", "numpy")),
                   provenance=provenance.provenance())
    t0 = time.perf_counter()
    info = F.init()
    res.data["init"] = dict(seconds=round(time.perf_counter() - t0, 3), built=info.get("built"),
                            key=info.get("key"), arch=info.get("arch"), arch_flags=info.get("arch_flags"),
                            forward_defines=info.get("forward_defines"), front_end=info.get("front_end"),
                            forward_unit=info.get("forward_unit"), library=provenance.rel(info.get("path")),
                            cache_dir=provenance.rel(F.cache_dir()), commit=info.get("commit"),
                            dirty=info.get("dirty"), sizeof_adj=info.get("sizeof_adj"))
    res.save()
    IT, SUB = a.iterations, a.substeps
    dev = jax.devices()[0]

    def put(st):
        arrs = tuple(jax.device_put(jnp.asarray(st[k]), dev) for k in C.STATE_KEYS)
        cq = jax.device_put(jnp.asarray(st["cot_qpos"]), dev)
        cv = jax.device_put(jnp.asarray(st["cot_qvel"]), dev)
        return arrs, cq, cv

    def aot(fn, *args):
        t = time.perf_counter()
        c = jax.jit(fn).lower(*args).compile()
        return c, round(time.perf_counter() - t, 3)

    def bench(c, args, **kw):
        box = [None]

        def call():
            box[0] = c(*args)
        r = C.timeit(call, lambda: jax.block_until_ready(box[0]), latency=0 if C.QUICK else 10, **kw)
        return r, box[0]

    def step_fn(q, v, w, u, f, c):
        return F.step(q, v, w, u, f, c, iterations=IT, substeps=SUB)

    def fwd_tape(q, v, w, u, f, c):
        return F.forward(q, v, w, u, f, c, iterations=IT, substeps=SUB, tape=True)

    def make_mode(kv, n):
        def bwd(tp, f, c, cq, cv):
            return F.backward(tp, f, c, cq, cv, substeps=SUB, clamp_kv=kv, normal=n)

        def vjp(q, v, w, u, f, c, cq, cv):
            def g(q, v, w, u, f, c):
                return F.step(q, v, w, u, f, c, iterations=IT, substeps=SUB, clamp_kv=kv, normal=n)
            outs, pull = jax.vjp(g, q, v, w, u, f, c)
            return pull((cq, cv, jnp.zeros_like(outs[2])))

        def jac_bwd(tp, f, c):
            B = tp.shape[0]
            eye = jnp.eye(NOUT, dtype=jnp.float32)

            def body(_, e):
                gq, gv, gu, _i = bwd(tp, f, c, jnp.broadcast_to(e[:36], (B, 36)), jnp.broadcast_to(e[36:], (B, 35)))
                return None, (gq, gv, gu)
            return jax.lax.scan(body, None, eye)[1]

        def jac(q, v, w, u, f, c):
            o = fwd_tape(q, v, w, u, f, c)
            return o[:3], jac_bwd(o[4], f, c)
        return bwd, vjp, jac_bwd, jac

    for name in C.corpora(a.corpora):
        try:
            st = C.load_states(name, nworld=a.nworld)
            args, cq, cv = put(st)
            cell = dict(corpus=st["_source"], compile_s={})
            c_step, cell["compile_s"]["fwd"] = aot(step_fn, *args)
            r, out = bench(c_step, args)
            cell["fwd"] = r
            cell["finite_fwd"] = bool(np.isfinite(np.asarray(out[1])).all())
            c_ft, cell["compile_s"]["fwd_tape"] = aot(fwd_tape, *args)
            r, out = bench(c_ft, args)
            cell["fwd_tape"] = r
            o = c_ft(*args)
            tp = o[4]
            inf = np.asarray(o[3])
            cell["forward_info"] = dict(newton_iterations_mean=float(inf[:, 0].mean()),
                                        contacts_mean=float(inf[:, 1].mean()), rows_max=int(inf[:, 2].max()),
                                        unsupported_worlds=int((inf[:, 3] != 0).sum()))
            for mode, (kv, n) in MODES.items():
                bwd, vjp, jac_bwd, jac = make_mode(kv, n)
                m = cell.setdefault(mode, {})
                c_b, cell["compile_s"][f"{mode}.bwd"] = aot(bwd, tp, args[4], args[5], cq, cv)
                m["bwd"], g = bench(c_b, (tp, args[4], args[5], cq, cv))
                m["finite_bwd"] = bool(all(np.isfinite(np.asarray(x)).all() for x in g[:3]))
                c_v, cell["compile_s"][f"{mode}.vjp"] = aot(vjp, *args, cq, cv)
                m["vjp"], _ = bench(c_v, args + (cq, cv))
                if not a.no_jac:
                    c_jb, cell["compile_s"][f"{mode}.jac_bwd"] = aot(jac_bwd, tp, args[4], args[5])
                    m["jac_bwd"], _ = bench(c_jb, (tp, args[4], args[5]))
                    c_j, cell["compile_s"][f"{mode}.jac"] = aot(jac, *args)
                    m["jac"], _ = bench(c_j, args)
                    for k in ("jac_bwd", "jac"):
                        m[k]["us_per_world"] = round(1e3 * m[k]["ms"]["median"] / a.nworld, 5)
                        m[k]["backward_calls"] = NOUT
                res.cell(f"{name}/nw{a.nworld}", cell)
                del c_b, c_v
            del tp, o
        except Exception as e:  # noqa: BLE001  (recorded; the next corpus goes on)
            res.error(f"{name}/nw{a.nworld}", e)
    # nworld sweep: fwd and vjp, exact mode, first corpus
    first = next(iter(C.corpora(a.corpora)))
    bwd, vjp, _, _ = make_mode(*MODES["exact"])
    for nw in [int(x) for x in a.sweep.split(",") if x.strip()]:
        try:
            st = C.load_states(first, nworld=nw)
            args, cq, cv = put(st)
            cell = dict(corpus=st["_source"], mode="exact", compile_s={})
            c_step, cell["compile_s"]["fwd"] = aot(step_fn, *args)
            cell["fwd"], _ = bench(c_step, args)
            c_v, cell["compile_s"]["vjp"] = aot(vjp, *args, cq, cv)
            cell["vjp"], _ = bench(c_v, args + (cq, cv))
            res.cell(f"sweep/{first}/nw{nw}", cell)
            del c_step, c_v, args
        except Exception as e:  # noqa: BLE001
            if C.is_oom(e):
                C.record_oom(res, f"sweep/{first}/nw{nw}", e, nw)
            else:
                res.error(f"sweep/{first}/nw{nw}", e)
    try:
        res.data["launch_info"] = F.launch_info()
    except Exception as e:  # noqa: BLE001
        res.error("launch_info", e)
    res.save()
    return 1 if res.data["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
