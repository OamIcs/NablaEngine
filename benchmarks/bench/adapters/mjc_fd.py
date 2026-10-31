"""MuJoCo C finite-difference Jacobians of one control step (substeps x mj_step, ctrl held, qfrc_applied 0), the
helper of the mjc adapter (adapters/mjc.py). numpy and mujoco only.

Inputs x = (qpos tangent (nv), qvel (nv), ctrl (nu)); outputs y = (qpos' in qpos form (nq), qvel' (nv)); J is
(B, nq + nv, 2 nv + nu), the layout of the harness's op jac (adapters/base.py) and of truth/fdlib.fd_job's J.

  chain   MuJoCo C's native derivative: per substep mjd_transitionFD(m, d, eps, centered, A, B) (A = d(dq', v') /
          d(dq, v), B = d(dq', v')/du, with its stage skipping), chained J_x = A_S ... A_1 and
          J_u = sum_k A_S ... A_(k+1) B_k along the nominal trajectory (mj_step between the substeps); the output
          tangent mapped to qpos form by
          d qpos' / d tangent at qpos' (base.tangent_to_qpos). Worlds run on a Python thread pool (the binding
          releases the GIL inside the C calls; MuJoCo C has no batched derivative API), in dynamic chunks; each pool
          thread keeps its own MjModel copy, on which every world's DR is set before its derivatives, and its own
          MjData.
  direct  for truth cross-checks only: every perturbed input (+-eps along each column; qpos along the tangent as
          mj_integratePos, base.perturbed_qpos) is its own simulation of one mujoco.batch.Batch (set_models: each
          world's DR model), stepped the whole control step, so the warm starts of later substeps follow the
          perturbation (the construction of the gradient truth); forward or centered differences of qpos' in qpos form.

"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from benchmarks.bench.adapters import base

CHUNKS_PER_THREAD = 4  # dynamic load balancing of the chain: about this many chunks of worlds per thread


def check_model(m):
    """NotImplementedError unless the model is a free joint followed by hinge / slide joints without actuator
    activations (the tangent maps of base.py)."""
    jt = np.asarray(m.jnt_type)
    ok = (m.njnt >= 1 and jt[0] == mujoco.mjtJoint.mjJNT_FREE
          and np.all((jt[1:] == mujoco.mjtJoint.mjJNT_HINGE) | (jt[1:] == mujoco.mjtJoint.mjJNT_SLIDE))
          and m.nq == m.nv + 1 and m.na == 0)
    if not ok:
        raise NotImplementedError("mjc: the FD Jacobian's tangent maps expect one free joint followed by hinge / slide "
                                  "joints and no actuator activations (na = 0)")


def chain_one(m, d, q, v, u, w, eps, centered, nsub, A, Bm):
    """One world: (J (nq + nv, 2 nv + nu), its nominal outputs qpos', qvel'). d: the thread's MjData; m: the world's
    model (its DR set); A (2 nv, 2 nv), Bm (2 nv, nu): scratch matrices (overwritten by mjd_transitionFD)."""
    nv = m.nv
    d.time = 0.0
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, w
    d.qfrc_applied[:] = 0.0
    Jx, Ju = np.eye(2 * nv), np.zeros((2 * nv, m.nu))
    for _ in range(nsub):
        mujoco.mjd_transitionFD(m, d, eps, bool(centered), A, Bm, None, None)  # restores d's state
        Jx = A @ Jx
        Ju = A @ Ju + Bm
        mujoco.mj_step(m, d)
    Jt = np.concatenate([Jx, Ju], 1)  # (2 nv, 2 nv + nu): output tangent rows
    J = np.zeros((m.nq + nv, 2 * nv + m.nu))
    J[:m.nq] = base.tangent_to_qpos(d.qpos[None].copy())[0] @ Jt[:nv]
    J[m.nq:] = Jt[nv:]
    return J, np.concatenate([d.qpos, d.qvel])


class ChainFD:
    """The chained mjd_transitionFD Jacobians of B worlds on a thread pool of nthread threads (module docstring)."""

    def __init__(self, m, dr, s, nsub, nthread):
        """m: the problem's MjModel (nominal); dr: problem.DR of it; s: the float64 inputs {qpos, qvel,
        qacc_warmstart, ctrl[, friction, com]} (B rows)."""
        check_model(m)
        self.m, self.dr, self.s, self.nsub = m, dr, s, int(nsub)
        self.B = len(s["qpos"])
        self.T = max(1, int(nthread))
        self.pool = ThreadPoolExecutor(self.T, thread_name_prefix="mjc_fd") if self.T > 1 else None
        self.local = threading.local()
        n = min(self.B, CHUNKS_PER_THREAD * self.T) if self.T > 1 else 1
        self.chunks = np.array_split(np.arange(self.B), n)

    def _thread_state(self):
        st = getattr(self.local, "st", None)
        if st is None:
            m = self.m
            mm = copy.copy(m)
            st = self.local.st = (mm, mujoco.MjData(mm), np.zeros((2 * m.nv, 2 * m.nv)), np.zeros((2 * m.nv, m.nu)))
        return st

    def jacobians(self, eps, centered):
        """(J (B, nq + nv, 2 nv + nu), nominal outputs (B, nq + nv))."""
        m, s = self.m, self.s
        J = np.zeros((self.B, m.nq + m.nv, 2 * m.nv + m.nu))
        out = np.zeros((self.B, m.nq + m.nv))
        dr = self.dr.enabled

        def work(ix):
            mm, d, A, Bm = self._thread_state()
            for i in ix:
                if dr:
                    self.dr.apply(mm, s["friction"][i], s["com"][i])
                J[i], out[i] = chain_one(mm, d, s["qpos"][i], s["qvel"][i], s["ctrl"][i], s["qacc_warmstart"][i],
                                         eps, centered, self.nsub, A, Bm)
        if self.pool is None:
            work(np.arange(self.B))
        else:
            list(self.pool.map(work, self.chunks))
        return J, out

    def describe(self):
        return dict(construction="chain", threads=self.T, chunks=len(self.chunks),
                    scheduling="Python ThreadPoolExecutor, dynamic chunks of worlds; per thread one MjModel copy (each "
                               "world's DR set before its derivatives) and one MjData")

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


class DirectFD:
    """Direct FD of the whole control step through one mujoco.batch.Batch per chunk of worlds (truth cross-checks)."""

    def __init__(self, m, dr, s, nsub, nthread, centered, chunk=256):
        from mujoco import batch as mjbatch
        check_model(m)
        self.m, self.dr, self.s, self.nsub, self.centered = m, dr, s, int(nsub), bool(centered)
        self.B = len(s["qpos"])
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.ncol = 2 * self.nv + self.nu
        self.nper = 1 + self.ncol * (2 if self.centered else 1)
        self.W = max(1, min(int(chunk), self.B))
        self.T = max(1, int(nthread))
        self.b = mjbatch.Batch(m, nsim=self.W * self.nper, nthread=self.T)

    def _models(self, ix):
        if not self.dr.enabled:
            return [self.m] * len(ix)
        return self.dr.models(self.m, self.s["friction"][ix], self.s["com"][ix])

    def jacobians(self, eps):
        s, h, b = self.s, float(eps), self.b
        nq, nv, nu, ncol, nper = self.nq, self.nv, self.nu, self.ncol, self.nper
        J = np.zeros((self.B, nq + nv, ncol))
        out = np.zeros((self.B, nq + nv))
        signs = (1.0, -1.0) if self.centered else (1.0,)
        for c0 in range(0, self.B, self.W):
            ix = np.arange(c0, min(self.B, c0 + self.W))
            n = len(ix) * nper
            mods = [mm for mm in self._models(ix) for _ in range(nper)]
            b.set_models(mods + [mods[-1]] * (b.nsim - n))
            Q = np.repeat(s["qpos"][ix][:, None], nper, 1)
            V = np.repeat(s["qvel"][ix][:, None], nper, 1)
            U = np.repeat(s["ctrl"][ix][:, None], nper, 1)
            for si, sg in enumerate(signs):
                o = 1 + si * ncol
                for wi, i in enumerate(ix):
                    Q[wi, o:o + nv] = base.perturbed_qpos(s["qpos"][i], sg * h)
                V[:, o + nv:o + 2 * nv] += sg * h * np.eye(nv)
                U[:, o + 2 * nv:o + ncol] += sg * h * np.eye(nu)
            b.time[:n] = 0.0
            b.qpos[:n], b.qvel[:n], b.ctrl[:n] = Q.reshape(-1, nq), V.reshape(-1, nv), U.reshape(-1, nu)
            b.qacc_warmstart[:n] = np.repeat(s["qacc_warmstart"][ix], nper, 0)
            b.qfrc_applied[:n] = 0.0
            b.step(ids=np.arange(n), nstep=self.nsub)
            Y = np.concatenate([b.qpos[:n], b.qvel[:n]], 1).reshape(len(ix), nper, nq + nv)
            out[ix] = Y[:, 0]
            if self.centered:
                J[ix] = np.swapaxes((Y[:, 1:1 + ncol] - Y[:, 1 + ncol:]) / (2 * h), 1, 2)
            else:
                J[ix] = np.swapaxes((Y[:, 1:] - Y[:, :1]) / h, 1, 2)
        return J, out

    def describe(self):
        return dict(construction="direct", threads=self.T, chunk_worlds=self.W, simulations_per_world=self.nper,
                    scheduling="mujoco.batch.Batch C thread pool, set_models with each world's DR model")

    def close(self):
        self.b = None
