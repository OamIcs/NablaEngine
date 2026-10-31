"""MuJoCo C adapter (engine id mjc): the official mujoco wheel on the CPU, float64, Tracks A and B.

  python -m benchmarks.bench run --engines mjc --quick --out runs/bench/test_mjc/
  python -m benchmarks.bench run --part cpu --variant mjc=default,centered --out runs/bench/<machine>/

Ops (every call starts from the loaded input states):
  fwd   mujoco.batch.Batch(model, nsim=nworld, nthread).step(nstep=substeps): the state rows (qpos, qvel, ctrl,
        qacc_warmstart, time 0, qfrc_applied 0; per-world DR through Batch.expand("pair_friction") / ("body_ipos"),
        no mj_setConst) are restored from a copy made at setup (one memcpy, timed as setup reset_s) and stepped. Bitwise
        equal to an mj_step loop over one MjData per world (Batch's documented guarantee, checked by the tests).
  jac   MuJoCo C's native derivative, finite differences: mjd_transitionFD chained over the substeps
        (adapters/mjc_fd.py: fd 'chain_forward' (default) or 'chain_centered', step eps), on a Python thread pool of
        nthread threads;
        (nworld, nq + nv, 2 nv + nu) float64, rows qpos' (qpos form) then qvel', columns qpos tangent, qvel, ctrl.
fwd_train, bwd and vjp are not_supported: MuJoCo C has no reverse mode; its VJP is c^T J of the FD Jacobian (op jac),
whose cost does not depend on the cotangent. gradient() (accuracy dumps) is that c^T J.

nthread: len(os.sched_getaffinity(0)) unless given (never Batch's default nthread=0, all CPUs of the machine, which
oversubscribes a node shared under a scheduler); the FD thread pool has the same size.

Options (run --opt mjc.<key>=<value>):
  nthread    threads of the Batch and of the FD pool (default: the process's CPU affinity count)
  fd         'chain_forward' (default) | 'chain_centered' (the jac op and gradient()); 'direct_centered' |
             'direct_forward': FD of the whole control step through Batch rollouts, the gradient truth's construction,
             for truth cross-checks only (never a timing row)
  eps        the FD step (default 1e-6); the "best eps (chosen with truth)" rows pass it explicitly
  fd_chunk   worlds per Batch of the direct modes (default 256)
  tolerance  None (default: the problem's) or the Newton tolerance (variant tol6: 1e-6, MuJoCo Warp's internal value)
Variants: default (chain_forward, eps 1e-6), centered (chain_centered, eps 1e-6), single_thread (nthread 1: the
single-thread rows at small nworld), tol6 (tolerance 1e-6).

Diagnostics (of one control step from the loaded states, stepped substep by substep so every substep's counts are
read; its final state is checked bitwise against fwd's): ncon / nefc per world and substep, Newton iterations per
world and substep (the largest solver_niter over the world's islands), niter_hist (worlds by their largest substep
iteration count), MuJoCo's warnings per world (overflow: worlds with mjWARN_CONTACTFULL or mjWARN_CNSTRFULL).
"""

import os
import time

import mujoco
import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench.adapters import base, mjc_fd

FD_MODES = ("chain_forward", "chain_centered", "direct_centered", "direct_forward")
TIMED_FD = ("chain_forward", "chain_centered")
EPS = 1e-6
BIG_FD_NWORLD = 4096  # MuJoCo C FD at 4096 worlds (and above): one call per round (N = 1), 3 rounds
OVERFLOW_WARNINGS = ("mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL")


def affinity_threads():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


class MjcAdapter(base.Adapter):
    id = "mjc"
    label = f"MuJoCo C {mujoco.__version__} (mujoco.batch, float64, CPU)"
    native = "native"
    device = "cpu"
    tracks = ("A", "B")
    variants = {"default": {}, "centered": {"fd": "chain_centered"}, "single_thread": {"nthread": 1},
                "tol6": {"tolerance": 1e-6}}

    def setup(self, problem, states, nworld, iterations, nthread=None, fd="chain_forward", eps=EPS, fd_chunk=256,
              tolerance=None):
        from mujoco import batch as mjbatch
        if fd not in FD_MODES:
            raise ValueError(f"mjc: fd must be one of {FD_MODES}, got {fd!r}")
        if not float(eps) > 0:
            raise ValueError(f"mjc: eps must be positive, got {eps}")
        t0 = time.perf_counter()
        m = P.load_model(problem, iterations)
        if tolerance is not None:
            m.opt.tolerance = float(tolerance)
        mjc_fd.check_model(m)
        states.check(m.nq, m.nv, m.nu, problem.dr)
        self.m, self.problem, self.nworld = m, problem, int(nworld)
        self.sub, self.iterations = int(problem.substeps), iterations
        self.fd, self.eps, self.fd_chunk, self.tolerance = fd, float(eps), int(fd_chunk), tolerance
        self.nthread_given = nthread
        self.T = int(nthread) if nthread else affinity_threads()
        if self.T < 1:
            raise ValueError(f"mjc: nthread must be >= 1, got {nthread}")
        self.dr = P.DR(m, problem)
        self.s = states.inputs(np.float64)  # float32 corpus values, exactly
        model_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = mjbatch.Batch(m, nsim=self.nworld, nthread=self.T)
        if self.dr.enabled:
            pf, bi = self.dr.batched(self.s["friction"], self.s["com"], np.float64)
            b.expand("pair_friction")[:] = pf
            b.expand("body_ipos")[:] = bi
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        b.time = 0.0
        b.qpos, b.qvel, b.ctrl = self.s["qpos"], self.s["qvel"], self.s["ctrl"]
        b.qacc_warmstart = self.s["qacc_warmstart"]
        b.qfrc_applied = 0.0
        self.b, self.state0 = b, b.state.copy()
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(3):
            b.state[:] = self.state0
        self.reset_s = (time.perf_counter() - t0) / 3
        self.fdj, self.J, self.J_out = None, None, None
        return dict(build_s=build_s, load_s=load_s, model_s=model_s, reset_s=self.reset_s)

    # ------------------------------------------------------------------ ops
    def _fwd(self):
        self.b.state[:] = self.state0
        self.b.step(nstep=self.sub)

    def _fd(self):
        if self.fdj is None:
            if self.fd.startswith("chain"):
                self.fdj = mjc_fd.ChainFD(self.m, self.dr, self.s, self.sub, self.T)
            else:
                self.fdj = mjc_fd.DirectFD(self.m, self.dr, self.s, self.sub, self.T, self.fd == "direct_centered",
                                           self.fd_chunk)
        return self.fdj

    def _jac(self):
        f = self._fd()
        if self.fd.startswith("chain"):
            self.J, self.J_out = f.jacobians(self.eps, self.fd == "chain_centered")
        else:
            self.J, self.J_out = f.jacobians(self.eps)
        return self.J

    def ops(self):
        out = dict(fwd=self._fwd)
        if self.fd in TIMED_FD:
            out["jac"] = self._jac
        return out

    def not_supported(self):
        why = ("MuJoCo C has no reverse mode (no adjoint); its derivative is finite differences: the VJP is c^T J of "
               "the FD Jacobian (op jac), whose cost does not depend on the cotangent")
        out = dict(fwd_train=f"mjc: no forward that keeps residuals for a backward: {why}",
                   bwd=f"mjc: no backward pass: {why}", vjp=f"mjc: no native VJP: {why}")
        if self.fd not in TIMED_FD:
            out["jac"] = f"mjc: fd={self.fd} is for truth cross-checks only, never timed"
        return out

    def prepare(self, op):
        if op == "jac":
            t0 = time.perf_counter()
            self._fd()
            return dict(build_s=time.perf_counter() - t0)
        return {}

    def timing_hints(self, op):
        if op == "jac" and self.nworld >= BIG_FD_NWORLD:
            return dict(reps=1, rounds=3, warmup_calls=1, warmup_s=0.0)
        return {}

    # ------------------------------------------------------------------ outputs, gradients, diagnostics
    def outputs(self):
        self._fwd()
        b = self.b
        return dict(qpos=np.array(b.qpos), qvel=np.array(b.qvel), qacc_warmstart=np.array(b.qacc_warmstart))

    def jacobian(self):
        return self._jac().copy() if self.J is None else self.J.copy()

    def gradient(self, cot_qpos, cot_qvel):
        J = self._jac() if self.J is None else self.J
        return base.vjp_from_jacobian(J, cot_qpos, cot_qvel)

    def diagnostics(self):
        b, B, S = self.b, self.nworld, self.sub
        ncon, nefc, niter = np.zeros((B, S), np.int64), np.zeros((B, S), np.int64), np.zeros((B, S), np.int64)
        names = sorted(w for w in dir(mujoco.mjtWarning) if w.startswith("mjWARN_"))
        warn = {w: np.zeros(B, np.int64) for w in names}
        b.state[:] = self.state0
        sniter, nisland = b.bind("solver_niter"), b.bind("nisland")
        for k in range(S):
            b.step(nstep=1)
            ncon[:, k], nefc[:, k] = b.ncon, b.nefc
            nis = np.maximum(np.asarray(nisland), 1)
            mask = np.arange(sniter.shape[1])[None, :] < nis[:, None]
            niter[:, k] = np.where(mask, sniter, 0).max(1)
            for w in names:
                warn[w] += np.asarray(b.warning[:, int(getattr(mujoco.mjtWarning, w)), 1])
        stepwise = dict(qpos=np.array(b.qpos), qvel=np.array(b.qvel))
        ref = self.outputs()
        same = all(np.array_equal(stepwise[k], ref[k]) for k in stepwise)
        over = np.zeros(B, bool)
        for w in OVERFLOW_WARNINGS:
            if w in warn:
                over |= warn[w] > 0
        nw = niter.max(1)
        return dict(ncon_mean=float(ncon.mean()), ncon_max=int(ncon.max()), nefc_mean=float(nefc.mean()),
                    nefc_max=int(nefc.max()), niter_mean=float(niter.mean()), niter_max=int(niter.max()),
                    niter_hist=np.bincount(nw, minlength=int(nw.max()) + 1).tolist(), overflow=int(over.sum()),
                    unsupported=None,
                    extra=dict(scope="ncon / nefc / niter per world and substep (stepped one substep at a time); "
                                     "niter: the largest solver_niter over the world's islands",
                               iteration_limit_records=int((niter >= self.m.opt.iterations).sum()),
                               warning_worlds={w: int((v > 0).sum()) for w, v in warn.items() if v.any()},
                               stepwise_bitwise_equal_fwd=bool(same)))

    def provenance(self):
        out = dict(engine="mujoco.batch.Batch", mujoco=mujoco.__version__, nthread=self.T,
                   nthread_source="option" if self.nthread_given else "len(os.sched_getaffinity(0))",
                   affinity_cpus=affinity_threads(), batch_nthread=int(self.b.nthread), fd=self.fd, eps=self.eps,
                   fd_impl=self.fdj.describe() if self.fdj is not None else None, reset_s=self.reset_s,
                   options=P.options(self.m), dr=bool(self.dr.enabled), tolerance_override=self.tolerance,
                   numpy=np.__version__)
        if self.fd.startswith("direct"):
            out["fd_chunk"] = self.fd_chunk
        return out

    def semantics(self):
        kind = "centered" if "centered" in self.fd else "forward"
        if self.fd.startswith("chain"):
            grad = (f"finite differences of MuJoCo C's float64 map: mjd_transitionFD per substep ({kind}, eps "
                    f"{self.eps:g}, stage skipping) chained J_x = A_S..A_1, J_u = sum_k A_S..A_(k+1) B_k; the output "
                    "tangent mapped to qpos form at qpos'; VJP = c^T J")
        else:
            grad = (f"direct {kind} finite differences of the whole control step (eps {self.eps:g}) through Batch "
                    "rollouts (truth construction, cross-checks only)")
        return dict(gradient=grad, solver_exit="per world (tolerance or the iteration cap)", precision="float64",
                    capacity="dynamic arena (no fixed contact capacity)")

    def close(self):
        if self.fdj is not None:
            self.fdj.close()
            self.fdj = None

    @classmethod
    def check_env(cls):
        from mujoco import batch as mjbatch  # noqa: F401  (mujoco >= 3.17)
        return dict(mujoco=mujoco.__version__, affinity_threads=affinity_threads(),
                    transitionFD=hasattr(mujoco, "mjd_transitionFD"))


ADAPTERS = {"mjc": MjcAdapter}
