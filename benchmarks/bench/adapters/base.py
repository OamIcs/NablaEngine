"""The interface every engine adapter implements, and the shared tangent-space helpers. Standard library and numpy
only: this module is imported in every engine environment.

Writing an adapter
------------------
Add one file benchmarks/bench/adapters/<name>.py (nothing else to edit) that defines Adapter subclasses and, at top
level, a dict literal with string-literal keys

    ADAPTERS = {"mjw": MjwAdapter, "mjw_tol6": MjwTol6Adapter}

The registry (adapters/__init__.py) reads these keys from the source without importing the module, so the worker
imports only the one module whose engine it runs, inside that engine's interpreter (benchmarks/envs/envs.toml maps the
engine id to the environment). Import heavy dependencies (jax, warp, mujoco_warp, nablaengine) at module top level or
lazily, as you like: the module is only imported where they exist. The class attribute `id` equals its ADAPTERS key.

Class attributes
    id        engine id, also the cell ids' first part (the comparison's engines: dk, dk_jac, mjw, mjw_pr1535,
              mjw_pr1535_fix, mjx, mjc)
    label     the public label, e.g. "MuJoCo Warp 3.17.0 (PyPI)", "MuJoCo Warp + unmerged PR #1535 @357a75d"
    native    'native' (an official release through its public API), 'ours' (NablaEngine), 'unmerged-upstream'
              (published but unmerged upstream code, unmodified), 'labelled' (anything else, never a headline row)
    device    'gpu' | 'cpu' (where the timed work runs; decides the minimum repetitions per round)
    tracks    the problem tracks it supports, ("A",) or ("A", "B")
    variants  {variant name: setup options}: named configurations of one engine that are separate rows (e.g.
              {"default": {}, "triton": {"xla_triton": True}}, {"matched": {"capacity": "matched"}}); "default" must
              exist. Cell ids carry a non-default variant as <engine>~<variant>.

Lifecycle (the worker, one fresh process per unit = engine, variant, problem, corpus, subset, nworld, iterations)
    a = Cls()
    a.setup(problem, states, nworld, iterations, **opts) -> {name: seconds}
        problem  benchmarks.bench.problem.Problem: problem.load_model(problem, iterations) gives the MjModel with
                 the problem's options; problem.DR(m, problem) applies or batches the per-world DR (Track A; a no-op
                 for Track B); problem.substeps substeps per control step, ctrl held, qfrc_applied 0.
        states   problem.States with nworld rows: qpos (nworld, nq), qvel, qacc_warmstart (nworld, nv), ctrl
                 (nworld, nu), friction (nworld,), com (nworld, 3) (None for Track B), cot_qpos (nworld, 3, nq),
                 cot_qvel (nworld, 3, nv) (the stored output cotangents; None if the corpus has none), all float32
                 as stored. states.inputs(np.float64) converts exactly.
        iterations  Newton iterations per substep (int), or None = the model's own (Track B native rows only).
        opts     the variant's options merged with the user's (`run --opt <engine>.<key>=<value>`); raise
                 TypeError / ValueError for an option you do not know.
        Build the model, upload the states to the device and keep them resident there (the timed ops never copy
        from the host), build or load kernels. Return the seconds of the parts you can separate, with these keys
        where they apply: build_s (model / library build), compile_s (JIT or kernel compile), capture_s (CUDA graph
        capture), load_s (state upload). The worker also records the wall time of setup as a whole.
    a.ops() -> {op: callable}
        Enqueue-only callables without arguments for the ops this engine supports natively (schema.OPS):
          fwd        one control step from the loaded states (inference)
          fwd_train  the forward a backward needs (e.g. with the tape / residuals); omit it when it is just fwd
          bwd        the backward alone: pullback of the stored cotangent 0 (states.cotangent(0)) through residuals
                     of a fwd_train made in prepare("bwd")
          vjp        fwd_train + bwd of cotangent 0, every call from the loaded states
          jac        the full Jacobian (nworld, nq + nv, 2 nv + nu) of the control step (rows qpos' in qpos form,
                     then qvel'; columns qpos tangent, qvel, ctrl), however the engine computes it natively
        Every call does the same work: it starts from the loaded input states, never from a previous call's output.
        A call enqueues the work and returns without waiting for it (no host sync, no device-to-host copy); it may
        return its outputs, but keep whatever sync() needs (e.g. the last JAX outputs) on self. A CPU engine whose
        call is synchronous simply runs it. The same callable object for two ops (e.g. ops()["fwd_train"] is
        ops()["fwd"]) is timed once and recorded as alias_of.
    a.not_supported() -> {op: message}
        For each op absent from ops(): why, quoted from the engine where possible (e.g. MJX's reverse-mode
        ValueError, "no backward in any MuJoCo Warp release (docs FAQ, issue #500)"). Ops neither in ops() nor
        here are recorded as not_supported with a generic message.
    a.prepare(op) -> {name: seconds}
        Optional (default: nothing): compile / capture what op needs before it is timed (e.g. a CUDA graph, an AOT
        compile, the residuals for bwd); return {compile_s, capture_s, ...}. The worker then runs one call + sync
        (first_call_s), a warm-up (>= warmup_calls calls and >= warmup_s seconds), and the timed rounds.
    a.sync()
        Block until everything enqueued so far has finished (jax: block_until_ready on the last outputs; warp:
        wp.synchronize_device(); CPU: nothing).
    a.timing_hints(op) -> dict
        Optional protocol overrides for one op, recorded with the cell: reps (calls per round), rounds,
        warmup_calls, warmup_s (e.g. MuJoCo C FD at 4096: {"reps": 1, "rounds": 3, "warmup_calls": 1}).
    a.outputs() -> {"qpos": (nworld, nq), "qvel": (nworld, nv), "qacc_warmstart": (nworld, nv)}
        float64 host arrays of one fwd from the loaded states (run one if needed; not timed). Called after timing.
    a.gradient(cot_qpos, cot_qvel) -> {"qpos": (nworld, nv), "qvel": (nworld, nv), "ctrl": (nworld, nu)}
        float64 host arrays: the native VJP of one control step from the loaded states for output cotangents
        cot_qpos (nworld, nq) (qpos form) and cot_qvel (nworld, nv); the qpos gradient in the 35-dimensional tangent
        form (convert a qpos-form gradient with qpos_tangent below; an FD Jacobian gives it directly). Raise
        NotSupported(message) when the engine has no native gradient. Not timed (accuracy dumps). An engine whose
        native gradient covers only some input blocks returns only those keys (the same ones for every cotangent);
        the dump records which (grad_blocks) and gradient_not_supported()'s reasons for the others, and the scorers
        report those blocks as not supported, never as non-finite.
    a.gradient_not_supported() -> {block: message}
        Optional, with a partial gradient: why each missing block is not supported, from the engine where possible.
        Called after the gradients of every cotangent.
    a.gradient_report() -> dict
        Optional: anything the gradients learned that belongs with the dump (e.g. backward passes the engine aborted
        and how they were retried), recorded in the dump cell.
    a.jacobian() -> (nworld, nq + nv, 2 nv + nu) float64 (Track A: (B, 71, 99)), layout as op "jac"; or raise
        NotSupported(message).
    a.diagnostics() -> dict, of the forward outputs() ran, every key optional (None: not available):
        ncon_mean, ncon_max, nefc_mean, nefc_max (per world, over the substeps the engine reports), niter_mean,
        niter_max, niter_hist (list: count of worlds per iteration count 0..max), overflow (worlds or events that
        exceeded a capacity; any overflow makes the cell an error), unsupported (worlds the engine flags as outside
        its scope), nonfinite_worlds (computed by the worker from outputs() when absent), extra (dict, anything
        engine-specific).
    a.memory() -> dict
        Optional: alloc_peak_bytes (the allocator's own high-water mark), tape_bytes, ... (the worker adds the
        process's NVML MiB and peak RSS).
    a.provenance() -> dict
        Engine-specific: versions, git commit / tree hash of a source tree, wheel tag, effective options (e.g.
        problem.options(m) plus engine options such as nconmax / njmax / graph_conditional / warn_overflow), the
        switch set and environment. Repository-relative paths only (benchmarks.bench.prov.rel).
    a.semantics() -> dict
        Optional: one-line statements recorded with the engine, e.g. {"gradient": "analytic adjoint at the converged
        solution, discrete decisions frozen", "solver_exit": "per batch (graph_conditional)"}.
    a.close()
        Optional: release thread pools, files.
    Cls.check_env() -> dict  (classmethod)
        Optional, for env-check: verify the environment without a model (e.g. the Warp device is CUDA, not a silent
        CPU fallback); return facts to print, raise on a problem.

Errors: raise NotSupported for things the engine cannot do natively (recorded as status not_supported with the
message), MemoryError or the engine's own out-of-memory error for OOM (recognised by is_oom: status skipped:oom), and
anything else for real failures (status error, with the message). Never catch and hide errors to produce a number.

Fairness rules: run every engine other than ours as natively as possible (official releases, public
APIs, documented settings, no monkeypatches, none of our optimisations); the only settings an adapter chooses are the
problem's (model, iterations, substeps, DR) and documented user settings that it records in provenance().
"""

import re

import numpy as np

NATIVE_KINDS = ("native", "ours", "unmerged-upstream", "labelled")
DEVICES = ("gpu", "cpu")
OPS = ("fwd", "fwd_train", "bwd", "vjp", "jac")
GRAD_BLOCKS = ("qpos", "qvel", "ctrl")  # the input blocks of gradient(), in the Jacobian's column order
DIAG_KEYS = ("ncon_mean", "ncon_max", "nefc_mean", "nefc_max", "niter_mean", "niter_max", "niter_hist", "overflow",
             "unsupported", "nonfinite_worlds", "extra")
SETUP_KEYS = ("build_s", "compile_s", "capture_s", "load_s")
HINT_KEYS = ("reps", "rounds", "warmup_calls", "warmup_s")
OOM_RE = re.compile(r"out of memory|out_of_memory|resource_exhausted|resource exhausted|cudaerrormemoryallocation|"
                    r"cuda_error_out_of_memory|failed to allocate|\boom\b")


class NotSupported(Exception):
    """The engine has no native way to do this; str(exc) is the recorded message."""


def is_oom(exc):
    """True for an out-of-memory error (MemoryError, or an engine error whose text says so)."""
    if isinstance(exc, MemoryError):
        return True
    return bool(OOM_RE.search(f"{type(exc).__name__} {exc}".lower()))


class Adapter:
    """Base class: the defaults of the optional hooks (see the module docstring for the contract)."""

    id = ""
    label = ""
    native = "labelled"
    device = "gpu"
    tracks = ("A",)
    variants = {"default": {}}

    def setup(self, problem, states, nworld, iterations, **opts):
        raise NotImplementedError

    def ops(self):
        raise NotImplementedError

    def not_supported(self):
        return {}

    def prepare(self, op):
        return {}

    def sync(self):
        pass

    def timing_hints(self, op):
        return {}

    def outputs(self):
        raise NotImplementedError

    def gradient(self, cot_qpos, cot_qvel):
        raise NotSupported(f"{self.id}: no native gradient")

    def jacobian(self):
        raise NotSupported(f"{self.id}: no native Jacobian")

    def gradient_not_supported(self):
        return {}

    def gradient_report(self):
        return {}

    def diagnostics(self):
        return {}

    def memory(self):
        return {}

    def provenance(self):
        return {}

    def semantics(self):
        return {}

    def close(self):
        pass

    @classmethod
    def check_env(cls):
        return {}

    @classmethod
    def describe(cls):
        """Class attributes as recorded with every cell."""
        return dict(id=cls.id, label=cls.label, native=cls.native, device=cls.device, tracks=list(cls.tracks),
                    variants=sorted(cls.variants))


def validate_class(cls, engine_id=None):
    """Problems of an adapter class's attributes (empty list: fine)."""
    out = []
    if not (isinstance(cls, type) and issubclass(cls, Adapter)):
        return [f"{cls!r} is not an Adapter subclass"]
    if engine_id is not None and cls.id != engine_id:
        out.append(f"{cls.__name__}.id = {cls.id!r}, registered as {engine_id!r}")
    if cls.native not in NATIVE_KINDS:
        out.append(f"{cls.__name__}.native = {cls.native!r} not in {NATIVE_KINDS}")
    if cls.device not in DEVICES:
        out.append(f"{cls.__name__}.device = {cls.device!r} not in {DEVICES}")
    if "default" not in cls.variants:
        out.append(f"{cls.__name__}.variants has no 'default'")
    if not cls.label:
        out.append(f"{cls.__name__}.label is empty")
    return out


# ------------------------------------------------------------------------------------------------- tangent helpers
# The G1 layout: one free joint first (qpos[0:3] position, qpos[3:7] quaternion w, x, y, z; qvel[0:6] linear and
# angular velocity, the angular one in the body frame), then hinges (one qpos and one qvel entry each): nq = nv + 1.

def check_layout(nq, nv):
    if nq != nv + 1:
        raise NotImplementedError(f"the tangent helpers assume a free joint followed by hinges (nq = nv + 1), got "
                                  f"nq {nq}, nv {nv}: use mj_differentiatePos / mj_integratePos for other models")


def quat_mul(a, b):
    """Hamilton product of (..., 4) quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qpos_tangent(g, q):
    """qpos-form gradients g (B, nq) at the states q (B, nq) -> the tangent form (B, nv): the quaternion part as
    1/2 Im(q* (x) g) / |q|^2, the rest unchanged (nablaengine.reference.adjoint_groups.qpos_tangent: the left
    inverse of MuJoCo Warp's dof-to-qpos map; for |q| = 1 exactly g . tangent_to_qpos(q))."""
    g = np.asarray(g, np.float64)
    q = np.asarray(q, np.float64)
    check_layout(g.shape[1], g.shape[1] - 1)
    xi, gq = q[:, 3:7], g[:, 3:7]
    w, x, y, z = xi.T
    a, b, c, d = gq.T
    v = np.stack([w * b - x * a - y * d + z * c, w * c + x * d - y * a - z * b, w * d - x * c + y * b - z * a], 1)
    return np.concatenate([g[:, :3], 0.5 * v / np.sum(xi * xi, 1, keepdims=True), g[:, 7:]], 1)


def tangent_to_qpos(q):
    """(B, nq) -> (B, nq, nv): d qpos / d tangent at q (position: identity; quaternion: 1/2 q (x) (0, e_i); hinges:
    identity), the map mj_integratePos linearises."""
    q = np.asarray(q, np.float64)
    B, nq = q.shape
    nv = nq - 1
    M = np.zeros((B, nq, nv))
    M[:, :3, :3] = np.eye(3)
    for i in range(3):
        e = np.zeros(4)
        e[1 + i] = 0.5
        M[:, 3:7, 3 + i] = quat_mul(q[:, 3:7], e)
    M[:, 7:, 6:] = np.eye(nq - 7)
    return M


def perturbed_qpos(q, h):
    """(nq,) -> (nv, nq): mj_integratePos(q, h e_j, 1) for every tangent direction j (the quaternion renormalised, as
    mj_integratePos does)."""
    q = np.asarray(q, np.float64)
    nq, nv = len(q), len(q) - 1
    out = np.tile(q, (nv, 1))
    out[np.arange(3), np.arange(3)] += h
    for i in range(3):
        r = np.zeros(4)
        r[0], r[1 + i] = np.cos(h / 2), np.sin(h / 2)
        out[3 + i, 3:7] = quat_mul(q[3:7], r)
    out[np.arange(6, nv), np.arange(7, nq)] += h
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def vjp_from_jacobian(J, cot_qpos, cot_qvel):
    """The VJP {qpos (tangent), qvel, ctrl} of Jacobians J (B, nq + nv, 2 nv + nu) for cotangents (B, nq), (B, nv)."""
    J = np.asarray(J, np.float64)
    nq, nv = cot_qpos.shape[1], cot_qvel.shape[1]
    c = np.concatenate([np.asarray(cot_qpos, np.float64), np.asarray(cot_qvel, np.float64)], 1)
    g = np.einsum("bi,bij->bj", c, J)
    if J.shape[1] != nq + nv:
        raise ValueError(f"J has {J.shape[1]} rows, cotangents {nq} + {nv}")
    return dict(qpos=g[:, :nv], qvel=g[:, nv:2 * nv], ctrl=g[:, 2 * nv:])


def jacobian_row_cotangents(nq, nv):
    """The nq + nv unit output cotangents (row r of the Jacobian = the VJP of e_r): (cot_qpos (R, nq), cot_qvel
    (R, nv)), R = nq + nv (71 for the G1: dk_jac's 71 VJPs)."""
    E = np.eye(nq + nv)
    return E[:, :nq], E[:, nq:]


def split_gradient(g, nv, nu):
    """(B, 2 nv + nu) in the Jacobian's column order -> {qpos (tangent), qvel, ctrl}."""
    g = np.asarray(g, np.float64)
    return dict(qpos=g[:, :nv], qvel=g[:, nv:2 * nv], ctrl=g[:, 2 * nv:2 * nv + nu])
