"""Parts `mjx`, `mjx_triton`: native MJX 3.17 (mujoco-mjx, impl='jax') per control step, its forward-mode
Jacobian, and the reverse-mode error text.

Everything is MJX's public API as its users run it: mujoco.MjModel -> mjx.put_model(m, impl='jax') -> mjx.make_data ->
jax.jit(jax.vmap(control_step)), control_step = jax.lax.scan of `substeps` x mjx.step (ctrl held, MuJoCo Playground's
pattern), all state outputs (qpos', qvel', qacc_warmstart') returned. No monkeypatch, no contact cap the model does not
declare. Per-world DR on mjlab (pair_friction[foot_pairs, 0:2] = friction, body_ipos[torso] += com) is the standard MJX
pattern: the randomized Model fields carry a leading world axis (vmap in_axes 0 on them).
  mjlab      fixtures/mjlab/g1_train.mjb, iterations 10, sensors disabled, 4 x 5 ms substeps, corpus states
  menagerie  MuJoCo Menagerie unitree_g1/scene_mjx.xml (MJX's own G1, $PAPER_BENCH_PREFIX/menagerie) at its native
             options (4 ms, Newton 5 / ls 8, sensors as declared), 5 substeps = 20 ms; states from MuJoCo C float64
             rollouts from the "home" keyframe with noise (native_states, seed 0)
Forward: AOT compile (lower + compile timed), then common.timeit rounds of N calls + block_until_ready.
Jacobian (mjlab, the first corpus): MJX's native forward mode, jax.jvp of the control step vmapped over input
directions (= jax.jacfwd), the 99 directions of x = (qpos tangent 35, qvel 35, ctrl 29) mapped to qpos by
dqpos/dtangent at the world's state (free joint: 1/2 q (x) (0, e_i)), outputs (qpos' 36, qvel' 35) = the 71 x 99
Jacobian; chunked over directions (--jac-chunk), worlds vmapped; per-world time = time of all chunks / B, at each B of
--jac-batches that fits the time budget (--jac-budget-s, estimated from the previous B). Reverse mode: jax.vjp + the
pullback at B = 2; its exception text is recorded (MJX's solver uses lax.while_loop).

  PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/paper_bench/mjx_native.py --out <dir>/mjx.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

STATE = ("qpos", "qvel", "qacc_warmstart", "ctrl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default=None)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--menagerie", default=None,
                    help="Menagerie checkout (default $PAPER_BENCH_PREFIX/menagerie); missing: Track B skipped")
    ap.add_argument("--menagerie-states", type=int, default=1024)
    ap.add_argument("--jac-batches", default="64,256,1024")
    ap.add_argument("--jac-chunk", type=int, default=33)
    ap.add_argument("--jac-budget-s", type=float, default=240.0)
    ap.add_argument("--parts", default="mjlab,menagerie,jac,reverse")
    a = ap.parse_args()
    parts = a.parts.split(",")

    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx

    res = C.Result(a.out, part="mjx", impl="jax", nworld=a.nworld,
                   env=C.basic_env(("mujoco", "mujoco-mjx", "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda13-plugin",
                                    "numpy")),
                   device=jax.devices()[0].device_kind, mujoco_version=mujoco.__version__)
    import importlib.metadata as md
    try:
        if md.version("mujoco-mjx") != "3.17.0" or mujoco.__version__ != "3.17.0":
            res.data["version_warning"] = (f"mujoco {mujoco.__version__} / mujoco-mjx {md.version('mujoco-mjx')}, not "
                                           "3.17.0: the row label does not apply")
    except md.PackageNotFoundError:
        pass
    prefix = os.environ.get("PAPER_BENCH_PREFIX") or str(C.REPO / "runs" / "paper_bench_env")
    men = Path(a.menagerie or Path(prefix) / "menagerie")
    meta = C.meta()

    class ControlStep:
        """One control step (substeps x mjx.step, ctrl held) of one world; batched with jax.vmap."""

        def __init__(self, mjm, substeps, dr):
            self.mjm, self.substeps, self.dr = mjm, substeps, dr
            self.mx = mjx.put_model(mjm, impl="jax")
            self.d0 = mjx.make_data(self.mx)
            if dr:
                self.foot = np.asarray(meta["foot_pairs"], np.int32)
                self.torso = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, meta["torso_body"])
                ax = jax.tree_util.tree_map(lambda _: None, self.mx)
                self.m_axes = ax.tree_replace({"pair_friction": 0, "body_ipos": 0})
            else:
                self.m_axes = None

        def static_sizes(self):
            return dict(ncon=int(self.d0._impl.contact.dist.shape[0]), nefc=int(self.d0._impl.efc_J.shape[0]))

        def model(self, friction=None, com=None):
            if not self.dr:
                return self.mx
            dt = self.mx.pair_friction.dtype
            friction, com = jnp.asarray(friction, dt), jnp.asarray(com, dt)
            B = friction.shape[0]
            pf = jnp.broadcast_to(self.mx.pair_friction, (B,) + self.mx.pair_friction.shape)
            pf = pf.at[:, self.foot, 0].set(friction[:, None]).at[:, self.foot, 1].set(friction[:, None])
            ip = jnp.broadcast_to(self.mx.body_ipos, (B,) + self.mx.body_ipos.shape)
            ip = ip.at[:, self.torso].add(com)
            return self.mx.tree_replace({"pair_friction": pf, "body_ipos": ip})

        def f1(self, m, qpos, qvel, warm, ctrl):
            d = self.d0.replace(qpos=qpos, qvel=qvel, qacc_warmstart=warm, ctrl=ctrl)
            d = jax.lax.scan(lambda d, _: (mjx.step(m, d), None), d, (), self.substeps)[0]
            return d.qpos, d.qvel, d.qacc_warmstart

        def vmap(self, fn, nextra=0):
            mx_ax = self.m_axes if self.dr else None
            return jax.vmap(fn, in_axes=(mx_ax,) + (0,) * (4 + nextra))

        def args(self, st, n):
            dt = self.mx.body_ipos.dtype
            idx = np.arange(n) % len(st["qpos"])
            arr = tuple(jnp.asarray(np.asarray(st[k])[idx], dt) for k in STATE)
            m = self.model(np.asarray(st["friction"])[idx], np.asarray(st["com"])[idx]) if self.dr else self.mx
            return (m,) + arr

    def aot(fn, args):
        t = time.perf_counter()
        lo = jax.jit(fn).lower(*args)
        t1 = time.perf_counter()
        c = lo.compile()
        t2 = time.perf_counter()
        return c, dict(trace_lower=round(t1 - t, 3), compile=round(t2 - t1, 3))

    def bench(c, args, **kw):
        box = [None]

        def call():
            box[0] = c(*args)
        return C.timeit(call, lambda: jax.block_until_ready(box[0]), **kw), box[0]

    def memory(c):
        try:
            ma = c.memory_analysis()
            return dict(temp=ma.temp_size_in_bytes, argument=ma.argument_size_in_bytes,
                        output=ma.output_size_in_bytes)
        except Exception:  # noqa: BLE001
            return None

    def peak():
        s = jax.devices()[0].memory_stats() or {}
        return s.get("peak_bytes_in_use")

    mj = None
    if "mjlab" in parts or "jac" in parts or "reverse" in parts:
        mjm = mujoco.MjModel.from_binary_path(str(C.FIXTURES / "g1_train.mjb"))
        mjm.opt.iterations = 10
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        mj = ControlStep(mjm, 4, True)
        o = mjm.opt
        res.data["mjlab_model"] = dict(path="fixtures/mjlab/g1_train.mjb", substeps=4, timestep=o.timestep,
                                       iterations=o.iterations, ls_iterations=o.ls_iterations,
                                       tolerance=o.tolerance, sensors_disabled=True, static=mj.static_sizes(),
                                       dtype=str(mj.mx.body_ipos.dtype))
        res.save()
    if "mjlab" in parts:
        fwd = mj.vmap(mj.f1)
        compiled = None
        for name in C.corpora(a.corpora):
            key = f"mjlab/{name}/nw{a.nworld}"
            try:
                st = C.load_states(name)
                args = mj.args(st, a.nworld)
                cell = dict(corpus=st["_source"])
                if compiled is None:
                    compiled, cs = aot(fwd, args)
                    res.data["mjlab_compile_s"] = cs
                    res.data["mjlab_compiled_memory"] = memory(compiled)
                r, out = bench(compiled, args)
                cell["forward"] = r
                cell["finite"] = bool(np.isfinite(np.asarray(out[1])).all())
                cell["peak_bytes"] = peak()
                res.cell(key, cell)
            except Exception as e:  # noqa: BLE001
                res.error(key, e)
        del compiled
    if "menagerie" in parts:
        key = f"menagerie/nw{a.nworld}"
        xml = men / "unitree_g1" / "scene_mjx.xml"
        try:
            if not xml.exists():
                raise FileNotFoundError(f"{C.rel(xml)} missing (run scripts/paper_bench/setup.sh)")
            mm = mujoco.MjModel.from_xml_path(str(xml))
            ms = ControlStep(mm, max(1, int(round(0.02 / mm.opt.timestep))), False)
            t = time.perf_counter()
            nst = 128 if C.QUICK else a.menagerie_states
            st = native_states(mm, nst, seed=0)
            o = mm.opt
            cell = dict(model=dict(file="unitree_g1/scene_mjx.xml", commit=_read(men / ".paper_bench_ok"),
                                   substeps=ms.substeps, timestep=o.timestep, iterations=o.iterations,
                                   ls_iterations=o.ls_iterations, tolerance=o.tolerance, solver=int(o.solver),
                                   integrator=int(o.integrator), cone=int(o.cone), nsensor=int(mm.nsensor),
                                   static=ms.static_sizes()),
                        states=dict(source="MuJoCo C float64 rollouts from keyframe 'home' + noise, seed 0",
                                    count=nst, seconds=round(time.perf_counter() - t, 2)))
            args = ms.args(st, a.nworld)
            c, cell["compile_s"] = aot(ms.vmap(ms.f1), args)
            cell["compiled_memory"] = memory(c)
            cell["forward"], out = bench(c, args)
            cell["finite"] = bool(np.isfinite(np.asarray(out[1])).all())
            res.cell(key, cell)
            del c
        except Exception as e:  # noqa: BLE001
            res.error(key, e)
    if "jac" in parts:
        first = next(iter(C.corpora(a.corpora)))
        st = C.load_states(first)
        K = a.jac_chunk
        nchunk = -(-99 // K)

        def jac_chunk(m, q, v, w, u, tq, tv, tu):  # one world, K directions
            def one(dq, dv, du):
                return jax.jvp(lambda q, v, u: mj.f1(m, q, v, w, u)[:2], (q, v, u), (dq, dv, du))[1]
            return jax.vmap(one)(tq, tv, tu)

        fj = mj.vmap(jac_chunk, nextra=3)
        est = None
        for B in [int(x) for x in a.jac_batches.split(",") if x.strip()]:
            key = f"jacfwd/{first}/B{B}"
            try:
                if est is not None and est * B * (2 + 3 + 1) > a.jac_budget_s:
                    res.cell(key, dict(skipped="time", estimate_s=round(est * B * 6, 1)))
                    continue
                args = mj.args(st, B)
                T = directions(np.asarray(args[1]), np.float32)  # (B, 99, 36 + 35 + 29)
                T = np.concatenate([T, np.zeros((B, nchunk * K - 99, T.shape[2]), T.dtype)], 1)
                chunks = []
                for i in range(nchunk):
                    s = T[:, i * K:(i + 1) * K]
                    chunks.append((jnp.asarray(s[..., :36]), jnp.asarray(s[..., 36:71]), jnp.asarray(s[..., 71:])))
                c, cs = aot(fj, args + chunks[0])
                box = [None]

                def call():
                    box[0] = [c(*args, *ch) for ch in chunks]
                r = C.timeit(call, lambda: jax.block_until_ready(box[0]), n=1, rounds=2 if C.QUICK else 3,
                             warm_calls=1, warm_s=0.0)
                tq = np.concatenate([np.asarray(x[0]) for x in box[0]], 1)[:, :99]
                tv = np.concatenate([np.asarray(x[1]) for x in box[0]], 1)[:, :99]
                cell = dict(corpus=st["_source"], B=B, chunk=K, chunks=nchunk, directions=99, outputs=71,
                            compile_s=cs, compiled_memory=memory(c), jacobian=r,
                            ms_per_world=round(r["ms"]["median"] / B, 4), finite=bool(np.isfinite(tq).all()
                                                                                       and np.isfinite(tv).all()),
                            peak_bytes=peak())
                est = r["ms"]["median"] / 1e3 / B
                res.cell(key, cell)
                del c, box
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
                    res.cell(key, dict(skipped="oom", message=msg[:300]))
                    break
                res.error(key, e)
    if "reverse" in parts:
        key = "reverse_mode"
        try:
            st = C.load_states(next(iter(C.corpora(a.corpora))))
            m, q, v, w, u = mj.args(st, 2)
            f = mj.vmap(mj.f1)
            out = None
            try:
                prim, pull = jax.vjp(lambda q, v, u: f(m, q, v, w, u)[:2], q, v, u)
                g = jax.jit(pull)((jnp.ones_like(prim[0]), jnp.ones_like(prim[1])))
                out = dict(supported=True, finite=bool(all(np.isfinite(np.asarray(x)).all() for x in g)))
            except Exception as e:  # noqa: BLE001
                out = dict(supported=False, type=type(e).__name__, message=str(e).split("\n")[0][:600], B=2,
                           call="jax.vjp(control step) then its pullback under jax.jit")
            res.cell(key, out)
        except Exception as e:  # noqa: BLE001
            res.error(key, e)
    return 1 if res.data["errors"] else 0


def _read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def quat_mul(a, b):
    import numpy as np
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def directions(qpos, dtype):
    """(B, nq) -> (B, 99, nq + nv + nu) input tangents: qpos tangent j (free joint + hinges: dqpos/dtangent, as
    mjc.py's tangent_to_qpos), then the unit vectors of qvel and ctrl (G1: nq 36, nv 35, nu 29)."""
    import numpy as np
    B, nq = qpos.shape
    nv, nu = nq - 1, 29
    M = np.zeros((B, nq, nv))
    M[:, :3, :3] = np.eye(3)
    for i in range(3):
        e = np.zeros(4)
        e[1 + i] = 0.5
        M[:, 3:7, 3 + i] = quat_mul(qpos[:, 3:7].astype(np.float64), np.broadcast_to(e, (B, 4)))
    M[:, 7:, 6:] = np.eye(nq - 7)
    T = np.zeros((B, 2 * nv + nu, nq + nv + nu))
    T[:, :nv, :nq] = np.swapaxes(M, 1, 2)
    T[:, nv:2 * nv, nq:nq + nv] = np.eye(nv)
    T[:, 2 * nv:, nq + nv:] = np.eye(nu)
    return T.astype(dtype)


def native_states(m, n, seed=0, t_range=(0.2, 1.0), noise=(0.05, 0.1, 0.1)):
    """n contact states of a native model: keyframe 'home' +
    N(0, noise[0]^2) joint offsets and N(0, noise[1]^2) joint velocities, ctrl = keyframe ctrl + N(0, noise[2]^2) x ctrl
    half-range resampled every 20 ms, for a uniform random time in t_range, MuJoCo C float64."""
    import mujoco
    import numpy as np
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    half = 0.5 * (m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0])
    out = {f: [] for f in STATE}
    nsub = max(1, int(round(0.02 / m.opt.timestep)))
    for _ in range(n):
        mujoco.mj_resetDataKeyframe(m, d, k)
        d.qpos[7:] += rng.normal(0, noise[0], m.nq - 7)
        d.qvel[6:] += rng.normal(0, noise[1], m.nv - 6)
        T = rng.uniform(*t_range)
        for _c in range(int(T / 0.02)):
            d.ctrl[:] = np.clip(m.key_ctrl[k] + rng.normal(0, noise[2], m.nu) * half, *m.actuator_ctrlrange.T)
            for _s in range(nsub):
                mujoco.mj_step(m, d)
        for f in STATE:
            out[f].append(getattr(d, f).copy())
    return {f: np.asarray(v) for f, v in out.items()}


if __name__ == "__main__":
    sys.exit(main())
