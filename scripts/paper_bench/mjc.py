"""Part `mjc`: MuJoCo C 3.17 (CPU, float64) per control step on the mjlab G1: forward, and its native derivative tooling
(finite differences, mjd_transitionFD).

Model: fixtures/mjlab/g1_train.mjb, iterations 10, tolerance 1e-8, ls_iterations 8, sensors disabled; per-world DR
(foot-floor pair friction [:2], torso body_ipos += com; no mj_setConst, as MuJoCo Warp). Inputs: the corpus states
(float32 rounded, as every engine gets them), qacc_warmstart included.
  forward       mujoco.batch.Batch (C thread pool), nthread = the allocated cores (common.cpu_threads():
                len(os.sched_getaffinity(0)), capped by the scheduler's CPU count when one is set;
                PAPER_BENCH_MJC_THREADS overrides; never os.cpu_count()), --nworld worlds, Batch.step(nstep=4); the
                state load is outside the timed region
  forward_1t    the same with nthread = 1 at --single-B worlds (64)
  fd_forward / fd_centered  the Jacobian d(qpos' (qpos form), qvel') / d(qpos tangent, qvel, ctrl) (71 x 99) by
                mjd_transitionFD per substep (eps --eps, forward / centered differences, stage skipping), chained
                A4 A3 A2 A1 and sum_k A4..A_{k+1} B_k, the output tangent mapped to qpos form; worlds split over
                Python threads (the binding releases the GIL inside the C calls), --fd-B worlds; per-world time and the
                linear extrapolation to --nworld worlds (labelled as such); on the first corpus also measured at
                --nworld worlds (--fd-full-B, N = 1 call per round, --fd-rounds kept rounds; fd_*_full) and with one
                thread at --single-B worlds (fd_*_1t)
Timing: warm-up (>= 5 calls and >= 1 s; FD: one call), then rounds of one call each (round 0 dropped); a CPU engine
needs no synchronization.

  PYTHONPATH= python scripts/paper_bench/mjc.py --out <dir>/mjc.json
"""
import argparse
import copy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NSUB = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default=None)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--fd-B", type=int, default=512)
    ap.add_argument("--single-B", type=int, default=64)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--fd-rounds", type=int, default=3)
    ap.add_argument("--no-fd", action="store_true", help="forward only (the repeat processes of the headline cells)")
    ap.add_argument("--fd-full-B", type=int, default=None, help="first corpus: FD measured at this B (default --nworld"
                    "; QUICK 1024; 0: off)")
    a = ap.parse_args()
    if a.fd_full_B is None:
        a.fd_full_B = 1024 if C.QUICK else a.nworld

    import mujoco
    import numpy as np
    from mujoco import batch as mjbatch

    T, rule = C.cpu_threads()
    res = C.Result(a.out, part="mjc", nthread=T, nthread_rule=rule, nworld=a.nworld, fd_B=a.fd_B, eps=a.eps,
                   fd_full_B=a.fd_full_B, env=C.basic_env(("mujoco", "numpy")),
                   mujoco_version=mujoco.__version__)
    if mujoco.__version__ != "3.17.0":
        res.data["version_warning"] = f"mujoco {mujoco.__version__}, not 3.17.0: the row label does not apply"
    meta = C.meta()
    m = mujoco.MjModel.from_binary_path(str(C.FIXTURES / "g1_train.mjb"))
    m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations = 10, 1e-8, 8
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, meta["torso_body"])
    foot = np.asarray(meta["foot_pairs"], int)
    res.data["opt"] = dict(iterations=int(m.opt.iterations), tolerance=float(m.opt.tolerance),
                           ls_iterations=int(m.opt.ls_iterations), solver=int(m.opt.solver),
                           integrator=int(m.opt.integrator), cone=int(m.opt.cone), timestep=float(m.opt.timestep))
    res.save()

    def batch(s, n, nthread):
        b = mjbatch.Batch(m, nsim=n, nthread=nthread)
        pf, bi = b.expand("pair_friction"), b.expand("body_ipos")
        pf[:] = m.pair_friction
        pf[:, foot, 0] = s["friction"][:n, None]
        pf[:, foot, 1] = s["friction"][:n, None]
        bi[:] = m.body_ipos
        bi[:, torso] = m.body_ipos[torso] + s["com"][:n]

        def load():
            b.time[:] = 0.0
            b.qpos[:], b.qvel[:] = s["qpos"][:n], s["qvel"][:n]
            b.ctrl[:], b.qacc_warmstart[:] = s["ctrl"][:n], s["qacc_warmstart"][:n]
            b.qfrc_applied[:] = 0.0
        return b, load

    def time_cpu(fn, pre=None, rounds=None, warm_calls=None, warm_s=None):
        rounds = C.ROUNDS if rounds is None else rounds
        warm_calls = C.WARM_CALLS if warm_calls is None else warm_calls
        warm_s = C.WARM_S if warm_s is None else warm_s
        t0, k = time.perf_counter(), 0
        while k < warm_calls or time.perf_counter() - t0 < warm_s:
            if pre:
                pre()
            fn()
            k += 1
        per = []
        for _ in range(rounds + 1):
            if pre:
                pre()
            t = time.perf_counter()
            fn()
            per.append(time.perf_counter() - t)
        return dict(ms=C.stats_ms(per[1:]), n=1, kept_rounds=rounds, warmup_calls=k,
                    warmup_s=round(time.perf_counter() - t0 - sum(per), 3))

    def dr_models(s, n):
        out = []
        for f, c in zip(s["friction"][:n], s["com"][:n]):
            mm = copy.copy(m)
            mm.body_ipos[torso] = m.body_ipos[torso] + c
            mm.pair_friction[foot, :2] = f
            out.append(mm)
        return out

    pool = ThreadPoolExecutor(T)

    def fd_cell(s, B, centered, nthread, rounds):
        """The chained-FD Jacobian of the first B states, worlds split over nthread Python threads (the binding
        releases the GIL in the C calls): timing dict with per-world time."""
        models = dr_models(s, B)
        J = np.zeros((B, m.nq + m.nv, 2 * m.nv + m.nu))

        def work(ix):
            d = mujoco.MjData(m)
            for i in ix:
                J[i] = chain_one(models[i], d, s["qpos"][i], s["qvel"][i], s["ctrl"][i], s["qacc_warmstart"][i],
                                 a.eps, centered)

        def fd():
            if nthread == 1:
                work(np.arange(B))
            else:
                list(pool.map(work, np.array_split(np.arange(B), nthread)))
        r = time_cpu(fd, rounds=rounds, warm_calls=1, warm_s=0.0)
        nev = 2 * (2 * m.nv + m.nu) + 1 if centered else (2 * m.nv + m.nu) + 1
        r.update(B=B, nthread=nthread, eps=a.eps, ms_per_world=round(r["ms"]["median"] / B, 5),
                 evaluations=f"{NSUB} x mjd_transitionFD ({nev} evaluations each, stage skipping)",
                 finite=bool(np.isfinite(J).all()))
        return r

    fd_rounds = 2 if C.QUICK else a.fd_rounds
    for ci, name in enumerate(C.corpora(a.corpora)):
        s = {k: (None if v is None else np.asarray(v, np.float64)) if not k.startswith("_") else v
             for k, v in C.load_states(name, nworld=a.nworld).items()}
        cell = dict(corpus=s["_source"], nthread=T)
        key = f"{name}/nw{a.nworld}"
        try:
            b, load = batch(s, a.nworld, T)
            cell["forward"] = time_cpu(lambda: b.step(nstep=NSUB), pre=load)
            cell["forward"]["us_per_world_substep"] = round(1e3 * cell["forward"]["ms"]["median"] / (a.nworld * NSUB),
                                                            4)
            cell["finite"] = bool(np.isfinite(b.qvel).all())
            del b
            n1 = a.single_B
            b1, load1 = batch(s, n1, 1)
            cell["forward_1t"] = time_cpu(lambda: b1.step(nstep=NSUB), pre=load1)
            cell["forward_1t"]["B"] = n1
            cell["forward_1t"]["us_per_world_substep"] = round(
                1e3 * cell["forward_1t"]["ms"]["median"] / (n1 * NSUB), 4)
            del b1
            res.cell(key, cell)
            if a.no_fd:
                continue
            for mode, centered in (("fd_forward", False), ("fd_centered", True)):
                r = fd_cell(s, a.fd_B, centered, T, fd_rounds)
                r["extrapolated_ms_at_nworld"] = dict(nworld=a.nworld, ms=round(r["ms_per_world"] * a.nworld, 3),
                                                      note=f"linear extrapolation from B = {a.fd_B}, not measured")
                cell[mode] = r
                res.cell(key, cell)
            if ci == 0:  # measured at the full size (N = 1, R = fd_rounds) and single-threaded at small B
                for mode, centered in (("fd_forward", False), ("fd_centered", True)):
                    if a.fd_full_B:
                        cell[f"{mode}_full"] = fd_cell(s, min(a.fd_full_B, a.nworld), centered, T, fd_rounds)
                        res.cell(key, cell)
                    cell[f"{mode}_1t"] = fd_cell(s, a.single_B, centered, 1, fd_rounds)
                    res.cell(key, cell)
        except Exception as e:  # noqa: BLE001
            res.error(key, e)
    pool.shutdown()
    return 1 if res.data["errors"] else 0


def quat_mul(a, b):
    import numpy as np
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def tangent_to_qpos(q):
    """(nq,) -> (nq, nv): dqpos/d(tangent) at q for the G1 (free joint first, then hinges)."""
    import numpy as np
    nq, nv = len(q), len(q) - 1
    M = np.zeros((nq, nv))
    M[:3, :3] = np.eye(3)
    for i in range(3):
        e = np.zeros(4)
        e[1 + i] = 0.5
        M[3:7, 3 + i] = quat_mul(q[3:7], e)
    M[7:, 6:] = np.eye(nq - 7)
    return M


def chain_one(m, d, q, v, u, w, eps, centered):
    """One world's control-step Jacobian (nq + nv, 2 nv + nu) from per-substep mjd_transitionFD (fdlib.chain_one)."""
    import mujoco
    import numpy as np
    nv, nu = m.nv, m.nu
    d.time = 0.0
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, w
    d.qfrc_applied[:] = 0.0
    A, Bm = np.zeros((2 * nv, 2 * nv)), np.zeros((2 * nv, nu))
    Jx, Ju = np.eye(2 * nv), np.zeros((2 * nv, nu))
    for _ in range(NSUB):
        mujoco.mjd_transitionFD(m, d, eps, centered, A, Bm, None, None)  # restores d's state
        Jx = A @ Jx
        Ju = A @ Ju + Bm
        mujoco.mj_step(m, d)
    Jt = np.concatenate([Jx, Ju], 1)
    return np.concatenate([tangent_to_qpos(d.qpos.copy()) @ Jt[:nv], Jt[nv:]], 0)


if __name__ == "__main__":
    sys.exit(main())
