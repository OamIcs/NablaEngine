"""The null adapter: a CUDA-free stand-in engine that exercises the whole driver (worker processes, protocol, schema,
resume, error statuses) without any physics. Not an engine; never in a results table.

Its "control step" is the identity map (qpos' = qpos, qvel' = qvel, warm' = warm) computed with numpy, plus a sleep
of work_ms per call, so its gradient and Jacobian are known exactly and test the tangent helpers of base.py:
gradient = (qpos_tangent(cot_qpos, qpos), cot_qvel, 0), Jacobian = [[tangent_to_qpos(qpos), 0, 0], [0, I, 0]].

  python -m benchmarks.bench run --engines null --quick --out runs/bench/test_null/
  python -m benchmarks.bench run --engines null --opt null.fail=oom ...    # error paths (below)

Options (run --opt null.<key>=<value>):
  work_ms   milliseconds slept per fwd call (default 1.0); vjp sleeps 2 x, jac 4 x
  fail      None (default); "setup": setup raises RuntimeError (status error for every op); "oom": setup raises
            MemoryError (skipped:oom); "op": the vjp op raises (error for vjp only); "hang": setup sleeps for an hour
            (the driver's per-unit timeout: skipped:time)
  no_jac    true: no jac op (not_supported with a message)
"""

import time

import numpy as np

from benchmarks.bench.adapters import base


class NullAdapter(base.Adapter):
    id = "null"
    label = "null adapter (harness self-test, not an engine)"
    native = "labelled"
    device = "cpu"
    tracks = ("A", "B")
    variants = {"default": {}, "slow": {"work_ms": 5.0}}

    def setup(self, problem, states, nworld, iterations, work_ms=1.0, fail=None, no_jac=False):
        if fail not in (None, "setup", "oom", "op", "hang"):
            raise ValueError(f"null: unknown fail mode {fail!r}")
        if fail == "setup":
            raise RuntimeError("null: setup failure requested (--opt null.fail=setup)")
        if fail == "oom":
            raise MemoryError("null: out of memory requested (--opt null.fail=oom)")
        if fail == "hang":
            time.sleep(3600)
        t0 = time.perf_counter()
        self.s = states.inputs(np.float64)
        self.qpos0 = self.s["qpos"]
        base.check_layout(states.nq, states.nv)
        self.nv, self.nu = states.nv, states.nu
        self.cot = states.cotangent(0, np.float64) if states.cot_qpos is not None else None
        self.work = float(work_ms) * 1e-3
        self.fail, self.no_jac = fail, bool(no_jac)
        self.iterations, self.substeps = iterations, problem.substeps
        self.out = None
        return dict(load_s=time.perf_counter() - t0)

    def _fwd(self):
        time.sleep(self.work)
        self.out = dict(qpos=self.s["qpos"] + 0.0, qvel=self.s["qvel"] + 0.0,
                        qacc_warmstart=self.s["qacc_warmstart"] + 0.0)
        return self.out

    def _vjp(self):
        if self.fail == "op":
            raise RuntimeError("null: op failure requested (--opt null.fail=op)")
        self._fwd()
        time.sleep(self.work)
        cq, cv = self.cot
        return self.gradient(cq, cv)

    def _jac(self):
        time.sleep(4 * self.work)
        return self.jacobian()

    def ops(self):
        out = dict(fwd=self._fwd, vjp=self._vjp)
        out["fwd_train"] = out["fwd"]  # the same callable: timed once, recorded as alias_of
        if not self.no_jac:
            out["jac"] = self._jac
        if self.cot is None:
            out.pop("vjp")
        return out

    def not_supported(self):
        out = dict(bwd="null: no separate backward (the identity map has nothing to pull back)")
        if self.no_jac:
            out["jac"] = "null: jac disabled (--opt null.no_jac=true)"
        if self.cot is None:
            out["vjp"] = "null: the corpus has no stored cotangents"
        return out

    def timing_hints(self, op):
        return {"warmup_calls": 3} if op == "jac" else {}

    def outputs(self):
        return {k: v.copy() for k, v in self._fwd().items()}

    def gradient(self, cot_qpos, cot_qvel):
        B = len(cot_qpos)
        return dict(qpos=base.qpos_tangent(cot_qpos, self.qpos0), qvel=np.asarray(cot_qvel, np.float64),
                    ctrl=np.zeros((B, self.nu)))

    def jacobian(self):
        B, nq = self.qpos0.shape
        nv, nu = self.nv, self.nu
        J = np.zeros((B, nq + nv, 2 * nv + nu))
        J[:, :nq, :nv] = base.tangent_to_qpos(self.qpos0)
        J[:, nq:, nv:2 * nv] = np.eye(nv)
        return J

    def diagnostics(self):
        B = len(self.qpos0)
        return dict(ncon_mean=0.0, ncon_max=0, nefc_mean=0.0, nefc_max=0, niter_mean=0.0, niter_max=0,
                    niter_hist=[B], overflow=0, unsupported=0, extra=dict(identity_map=True))

    def provenance(self):
        return dict(version="0", what="identity map + sleep", work_ms=self.work * 1e3, iterations=self.iterations,
                    substeps=self.substeps, numpy=np.__version__)

    def semantics(self):
        return dict(gradient="exact derivative of the identity map", solver_exit="none")


ADAPTERS = {"null": NullAdapter}
