"""The MJX adapter: MuJoCo MJX (the mujoco-mjx package, impl='jax') run the way its users run it, on Track A (the
mjlab G1 with per-world domain randomization) and Track B (Menagerie unitree_g1/scene_mjx.xml, native options).

Everything goes through MJX's public API and plain JAX transforms: no monkeypatch, no replaced solver, no contact cap
that the model does not declare itself (max_contact_points would be a model edit), no MJX-Warp (impl='warp').

  fwd    jax.jit(jax.vmap(control_step)), compiled ahead of time in setup (lower_s + compile_s reported). control_step
         takes (qpos, qvel, qacc_warmstart, ctrl) of one world and runs `substeps` mjx.step calls in a jax.lax.scan with
         ctrl held (MuJoCo Playground's mjx_env.step pattern), returning every state output (qpos', qvel',
         qacc_warmstart'). Per-world DR (Track A) is the standard MJX pattern: a Model whose pair_friction and
         body_ipos carry a leading world axis, vmapped with in_axes 0 on those two fields and None on the others; the
         values are problem.DR.batched(..., float32) (the corpus friction on the foot pairs, float32(ipos0) + com).
  jac    the forward-mode Jacobian (native): jax.vmap over worlds of jax.jvp of control_step along the 2 nv + nu input
         directions (qpos tangent, qvel, ctrl), chunked with jax.lax.map(batch_size=jac_chunk) inside one jitted call
         (jac_chunk = 2 nv + nu is jax.jacfwd). Result (nworld, nq + nv, 2 nv + nu) float32: rows qpos' (qpos form)
         then qvel'. The qpos-tangent directions are d/dh mj_integratePos(qpos, h e_j, 1) at h = 0 (hinge / slide: 1,
         free joint position: identity, free / ball quaternion: 1/2 q (x) (0, e_i) / |q|, since mj_integratePos
         renormalises), computed in float64 from the stored qpos and rounded to the model's dtype.
  bwd, vjp   not supported natively: MJX's Newton / CG solver is a lax.while_loop with a data-dependent exit, which
         JAX cannot transpose. setup attempts jax.vjp of the same control step on two worlds (tracing only:
         jax.eval_shape, nothing runs on the device) and records the exact exception JAX raises as the message.
  fwd_train  not supported (no reverse mode, hence no forward that keeps residuals for a backward).
gradient(cot) = cot^T J of the forward-mode Jacobian (semantics()["gradient"] says so); jacobian() = op jac in float64
(computed over slices of the worlds when the whole batch does not fit; never timed).

Choosing jac_chunk K (directions per lax.map step; each step re-runs the primal once for its K tangents).
  speed   K up to a cap, max(JAC_CHUNK_MIN, ceil(JAC_TARGET_ROWS / nworld)): once nworld x K world-directions fill the
          GPU, a larger K only costs memory (V100, mjlab G1, 64 worlds: K = 1 5.8 s, 11 1.83 s, 33 2.0 s, 99 2.04 s
          per Jacobian)
  memory  the compiled forward's temp bytes per world f give the estimate f x (JAC_MEM_PRIMAL + JAC_MEM_PER_DIRECTION
          x K) per world (mjlab G1, V100: 1.56 MB per world at K = 1, 6.7 at K = 11, 49 at K = 99, f = 0.51 MB),
          within jac_mem_frac of the largest single allocation JAX's allocator can still serve (an executable's temp
          memory is one buffer: the largest free block of the allocator's pool, or a new region within bytes_limit
          and the device's free memory)
The smaller of the two, balanced over the directions (the same number of steps with the smallest K; the last step is
padded with zero directions), is compiled and its memory_analysis (temp + output bytes) checked against the budget
measured again after the compile (XLA's autotuning allocates while compiling and, with XLA_PYTHON_CLIENT_PREALLOCATE=
false, the allocator keeps what it grew). A smaller K and a recompile when it does not fit, up to JAC_PLAN_ATTEMPTS
compiles; MemoryError (skipped:oom) when K = 1 does not fit, without a compile when the K = 1 estimate exceeds
JAC_EARLY_OOM x the budget. The plan is in provenance()["jac"] and memory()["jac"].

Variants and options (run --variant mjx=default,triton; run --opt mjx.<key>=<value>):
  default       XLA's default GPU flags
  triton        xla_triton=True: XLA_FLAGS gets --xla_gpu_triton_gemm_any=true, set in the worker process before
                JAX is imported (this module imports JAX only in setup / check_env); the number of Triton GEMM
                fusions in each compiled executable is recorded (XLA's Triton GEMM emitter needs an Ampere or
                newer GPU: the count is 0 on a V100, sm_70).
  jac_chunk     None (default: the plan above) or an int in 1..2 nv + nu (no memory check)
  jac_mem_frac  0.85 (default): the share of the largest allocation the allocator can serve that a jac executable
                may plan for
  preallocate   None (default: the worker's XLA_PYTHON_CLIENT_PREALLOCATE, which the driver sets to false) or a
                bool: XLA_PYTHON_CLIENT_PREALLOCATE for this worker, set before JAX is imported. JAX's own default is
                true (one region of 75 % of the device memory up front); with false the allocator grows by regions it
                keeps, so a large Jacobian may find less contiguous memory after the forward ran (the plan accounts
                for it; on a 16 GB GPU that limits the chunk at 4096 worlds)
  diagnostics   True (default): after the timed ops, run a second, untimed executable of the same control step that
                also counts the contacts (dist < includemargin, MuJoCo C's criterion) and the constraint rows with a
                non-zero Jacobian row per substep (diagnostics()); False skips it (no ncon / nefc)

  python -m benchmarks.bench run --engines mjx --quick --out runs/bench/test_mjx/
  python -m benchmarks.bench run --engines mjx --variant mjx=default,triton --nworld 4096 --ops fwd --out ...
  python -m benchmarks.bench.adapters.mjx_check --out runs/bench/test_mjx/check.json   # accuracy self-check

Environment (worker): XLA_PYTHON_CLIENT_PREALLOCATE=false (the driver's default) so memory figures are the
allocator's own; XLA_FLAGS is read once, at JAX's backend start-up (the default variant refuses an XLA_FLAGS that
already sets xla_gpu_triton_gemm_any, so the default row is never silently the Triton one). MJX runs in float32 (JAX's
default); JAX_ENABLE_X64=1 would make it float64 and is recorded in provenance().
"""

import importlib.metadata
import math
import os
import sys
import time

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov
from benchmarks.bench.adapters import base

TRITON_FLAG = "--xla_gpu_triton_gemm_any=true"
TRITON_KEY = "--xla_gpu_triton_gemm_any"
JAC_TARGET_ROWS = 1024  # world-directions per lax.map step beyond which a larger chunk only costs memory (V100)
JAC_CHUNK_MIN = 9  # ... but at least this many directions per step, to amortise the primal each step recomputes
JAC_MEM_PRIMAL = 2.1  # per-world jac temp memory, in units of the compiled forward's temp bytes per world: primal
JAC_MEM_PER_DIRECTION = 1.0  # ... and per direction of a chunk
JAC_PLAN_ATTEMPTS = 4
JAC_EARLY_OOM = 1.5  # an estimate above 1.5 x the budget at K = 1 is out of memory without a confirming compile
REVERSE_PROBE_WORLDS = 2
# In-process caches of untimed work shared by the fresh adapters of one worker process (the state subsets of a unit
# run one after another in it): the reverse-mode attempt (a property of the model and options, not of the states) and
# the diagnostics executable (the states are its arguments). The timed executables are never cached.
_REVERSE_CACHE = {}
_DIAG_CACHE = {}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


MJX_VERSION = _version("mujoco-mjx") or "?"


def _xla_flags(triton):
    """Put the Triton GEMM flag into XLA_FLAGS for the triton variant (before JAX is imported); refuse a conflicting
    setting. Returns the effective XLA_FLAGS."""
    flags = os.environ.get("XLA_FLAGS", "")
    words = flags.split()
    has_key = any(w.split("=")[0] == TRITON_KEY for w in words)
    if triton:
        if TRITON_FLAG in words:
            return flags
        if has_key:
            raise ValueError(f"mjx: XLA_FLAGS already sets {TRITON_KEY} to another value ({flags!r})")
        if "jax" in sys.modules:
            raise RuntimeError("mjx: xla_triton needs XLA_FLAGS set before JAX is imported, but JAX was imported "
                               "earlier in this worker process")
        os.environ["XLA_FLAGS"] = (flags + " " + TRITON_FLAG).strip()
    elif has_key:
        raise ValueError(f"mjx: XLA_FLAGS sets {TRITON_KEY} ({flags!r}) in a default-variant run; use the triton "
                         f"variant (run --variant mjx=triton) so the row is labelled")
    return os.environ.get("XLA_FLAGS", "")


def _set_before_jax(name, value):
    """Set an environment variable JAX reads when its backend starts; refuse when JAX is already imported with
    another value."""
    if os.environ.get(name) == value:
        return
    if "jax" in sys.modules:
        raise RuntimeError(f"mjx: {name}={value} must be set before JAX is imported, but JAX was imported earlier in "
                           f"this worker process with {name}={os.environ.get(name)!r}")
    os.environ[name] = value


def _bool(v):
    if isinstance(v, str):
        if v.lower() in ("1", "true", "yes", "on"):
            return True
        if v.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"mjx: not a boolean: {v!r}")
    return bool(v)


def tangent_map(m, qpos):
    """(B, nq) qpos -> (B, nq, nv) float64: d qpos / d tangent of mj_integratePos(qpos, h e_j, 1) at h = 0 for every
    joint type (free, ball: 1/2 q (x) (0, e_i) / |q| on the quaternion; position, hinge, slide: identity)."""
    import mujoco
    q = np.asarray(qpos, np.float64)
    B = len(q)
    M = np.zeros((B, m.nq, m.nv))
    for j in range(m.njnt):
        t, qa, va = int(m.jnt_type[j]), int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        if t == mujoco.mjtJoint.mjJNT_FREE:
            M[:, qa:qa + 3, va:va + 3] = np.eye(3)
            qa, va = qa + 3, va + 3
        if t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            quat = q[:, qa:qa + 4]
            nrm = np.linalg.norm(quat, axis=1)
            for i in range(3):
                e = np.zeros(4)
                e[1 + i] = 0.5
                M[:, qa:qa + 4, va + i] = base.quat_mul(quat, e) / nrm[:, None]
        else:
            M[:, qa, va] = 1.0
    return M


def _count_triton(compiled):
    try:
        return compiled.as_text().count("__triton")
    except Exception:  # noqa: BLE001  (text not available for this executable)
        return None


def _mem_analysis(compiled):
    try:
        ma = compiled.memory_analysis()
    except Exception:  # noqa: BLE001
        return None
    if ma is None:
        return None
    return dict(temp_bytes=int(ma.temp_size_in_bytes), argument_bytes=int(ma.argument_size_in_bytes),
                output_bytes=int(ma.output_size_in_bytes), generated_code_bytes=int(ma.generated_code_size_in_bytes))


def _first_lines(text, n=3):
    """An exception text without JAX's traceback-filtering note."""
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if ln.startswith("----") or "JAX has removed its internal frames" in ln:
            break
        out.append(ln)
    return " ".join(out[:n])


class MjxAdapter(base.Adapter):
    id = "mjx"
    label = f"MJX {MJX_VERSION} (mujoco-mjx, impl='jax')"
    native = "native"
    device = "gpu"
    tracks = ("A", "B")
    variants = {"default": {}, "triton": {"xla_triton": True}}

    # ------------------------------------------------------------------ setup
    def setup(self, problem, states, nworld, iterations, xla_triton=False, jac_chunk=None, jac_mem_frac=0.85,
              diagnostics=True, preallocate=None):
        self.triton = _bool(xla_triton)
        self.xla_flags = _xla_flags(self.triton)
        if preallocate is not None:
            _set_before_jax("XLA_PYTHON_CLIENT_PREALLOCATE", "true" if _bool(preallocate) else "false")
        t0 = time.perf_counter()
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
        self.jax, self.jnp, self.mujoco, self.mjx = jax, jnp, mujoco, mjx
        if jax.default_backend() != "gpu":
            raise RuntimeError(f"mjx: JAX's default backend is {jax.default_backend()!r}, not 'gpu' (JAX_PLATFORMS="
                               f"{os.environ.get('JAX_PLATFORMS')!r}); refusing a silent CPU run")
        self.dev = jax.devices()[0]
        t_import = time.perf_counter() - t0

        self.problem, self.nworld, self.iterations = problem, int(nworld), iterations
        self.substeps = int(problem.substeps)
        self.jac_chunk_opt = None if jac_chunk in (None, "auto", "") else int(jac_chunk)
        self.jac_mem_frac = float(jac_mem_frac)
        if not 0.0 < self.jac_mem_frac <= 1.0:
            raise ValueError(f"mjx: jac_mem_frac must be in (0, 1], got {jac_mem_frac!r}")
        self.want_diag = _bool(diagnostics)

        t0 = time.perf_counter()
        m = P.load_model(problem, iterations)
        self.m = m
        states.check(m.nq, m.nv, m.nu, problem.dr)
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.ndir = 2 * m.nv + m.nu
        if self.jac_chunk_opt is not None and not 1 <= self.jac_chunk_opt <= self.ndir:
            raise ValueError(f"mjx: jac_chunk must be in 1..{self.ndir}, got {jac_chunk!r}")
        self.mx = mjx.put_model(m, impl="jax")
        self.d0 = mjx.make_data(self.mx)
        self.dtype = np.dtype(self.mx.body_ipos.dtype)
        self.dr = problem.dr
        if self.dr:
            axes = jax.tree_util.tree_map(lambda _: None, self.mx)
            self.m_axes = axes.tree_replace({"pair_friction": 0, "body_ipos": 0})
        else:
            self.m_axes = None
        self.static = dict(ncon=int(self.d0._impl.contact.dist.shape[0]), nefc=int(self.d0._impl.efc_J.shape[0]))
        build_s = time.perf_counter() - t0

        # inputs resident on the device (the stored float32 values; the batched model's DR fields)
        t0 = time.perf_counter()
        s = states.inputs()
        self.qpos_host = s["qpos"].astype(np.float64)
        self.state = tuple(jax.device_put(np.asarray(s[k], self.dtype), self.dev)
                           for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"))
        if self.dr:
            pf, bi = P.DR(m, problem).batched(states.friction, states.com, self.dtype)
            self.model = self.mx.tree_replace({"pair_friction": jax.device_put(pf, self.dev),
                                               "body_ipos": jax.device_put(bi, self.dev)})
        else:
            self.model = self.mx
        jax.block_until_ready((self.state, self.model))
        load_s = time.perf_counter() - t0

        # the forward, AOT
        self.args = (self.model,) + self.state
        t0 = time.perf_counter()
        lowered = jax.jit(self._batched(self._step_world)).lower(*self.args)
        lower_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.fwd_exec = lowered.compile()
        compile_s = time.perf_counter() - t0
        self.compile = dict(fwd=dict(lower_s=lower_s, compile_s=compile_s))
        self.mem = dict(fwd=_mem_analysis(self.fwd_exec))
        self.triton_fusions = dict(fwd=_count_triton(self.fwd_exec))

        # reverse mode: attempted once, tracing only
        t0 = time.perf_counter()
        key = self._key()
        if key not in _REVERSE_CACHE:
            _REVERSE_CACHE[key] = self._reverse_probe()
            self.reverse_cached = False
        else:
            self.reverse_cached = True
        self.reverse = _REVERSE_CACHE[key]
        reverse_s = time.perf_counter() - t0

        self.jac_exec, self.jac_plan, self.jac_args, self._J = None, None, None, None
        self._last, self._diag = None, None
        return dict(build_s=build_s, compile_s=lower_s + compile_s, lower_s=lower_s, xla_compile_s=compile_s,
                    load_s=load_s, import_s=t_import, reverse_probe_s=reverse_s)

    def _key(self, *extra):
        p = self.problem
        return (p.id, p.model, self.iterations, self.substeps, self.xla_flags, str(self.dtype)) + extra

    def _step_world(self, m, qpos, qvel, warm, ctrl):
        """One control step of one world: `substeps` mjx.step with ctrl held, every state output."""
        jax, mjx = self.jax, self.mjx
        d = self.d0.replace(qpos=qpos, qvel=qvel, qacc_warmstart=warm, ctrl=ctrl)
        d = jax.lax.scan(lambda d, _: (mjx.step(m, d), None), d, None, length=self.substeps)[0]
        return d.qpos, d.qvel, d.qacc_warmstart

    def _batched(self, f, extra_axes=()):
        """jax.vmap of f(m, qpos, qvel, warm, ctrl, *extra) over worlds, the model batched on its DR fields only."""
        return self.jax.vmap(f, in_axes=(self.m_axes, 0, 0, 0, 0) + tuple(extra_axes))

    def _slice_model(self, a, b):
        if not self.dr:
            return self.model
        return self.model.tree_replace({"pair_friction": self.model.pair_friction[a:b],
                                        "body_ipos": self.model.body_ipos[a:b]})

    def _reverse_probe(self):
        """jax.vjp of the control step (outputs qpos', qvel'; inputs qpos, qvel, ctrl) on the first worlds, traced with
        jax.eval_shape; None when it traces, else {stage, type, message}."""
        jax = self.jax
        k = min(REVERSE_PROBE_WORLDS, self.nworld)
        mk = self._slice_model(0, k)
        q, v, w, u = (x[:k] for x in self.state)
        f = self._batched(self._step_world)

        def out(q, v, u):
            o = f(mk, q, v, w, u)
            return o[0], o[1]

        stage = ["jax.vjp (linearize)"]

        def pulled(q, v, u):
            prim, pull = jax.vjp(out, q, v, u)
            stage[0] = "pullback (transpose)"
            return pull(prim)

        try:
            jax.eval_shape(pulled, q, v, u)
        except Exception as e:  # noqa: BLE001  (the error is the result)
            return dict(stage=stage[0], type=type(e).__name__, message=_first_lines(e))
        return None

    # ------------------------------------------------------------------ ops
    def _fwd(self):
        self._last = self.fwd_exec(*self.args)
        return self._last

    def _jac(self):
        self._last = self.jac_exec(*self.jac_args)
        return self._last

    def ops(self):
        return dict(fwd=self._fwd, jac=self._jac)

    def _reverse_message(self):
        r = self.reverse
        it = "native" if self.iterations is None else self.iterations
        if r is None:
            return (f"mjx: reverse mode is not measured by this adapter: jax.vjp of the control step traced at these "
                    f"options (iterations {it}), which the benchmark's rows do not use; MJX's derivative here is the "
                    f"forward-mode Jacobian (op jac)")
        return (f"MJX {MJX_VERSION} has no native reverse mode: jax.vjp of the control step (iterations {it}) fails "
                f"in the {r['stage']} with {r['type']}: {r['message']}")

    def not_supported(self):
        msg = self._reverse_message()
        return dict(bwd=msg, vjp=msg,
                    fwd_train=f"MJX {MJX_VERSION}: no reverse mode (see bwd / vjp), hence no forward that keeps "
                              f"residuals for a backward; the forward-mode Jacobian (op jac) runs its own forward")

    def prepare(self, op):
        if op != "jac" or self.jac_exec is not None:
            return {}
        t0 = time.perf_counter()
        self.jac_exec, self.jac_plan, self.jac_args, t = self._plan_jac(np.arange(self.nworld))
        out = dict(t)
        out["plan_s"] = time.perf_counter() - t0
        return out

    def timing_hints(self, op):
        if op != "jac":
            return {}
        # one Jacobian is 2 nv + nu tangent passes plus a primal per chunk (seconds per call): one call per round,
        # 3 rounds at large batches, as the MuJoCo C FD cells
        h = dict(reps=1, warmup_calls=1)
        if self.nworld >= 1024:
            h["rounds"] = 3
        return h

    def sync(self):
        if self._last is not None:
            self.jax.block_until_ready(self._last)

    # ------------------------------------------------------------------ the forward-mode Jacobian
    def _jac_fn(self, chunk):
        """f(m, qpos, qvel, warm, ctrl, Tq) -> (B, nq + nv, 2 nv + nu): vmap over worlds of lax.map(batch_size=chunk)
        over the input directions of jax.jvp of the control step. The directions are padded with zero directions to a
        multiple of chunk (fewer than one per lax.map step), so lax.map runs whole chunks only (one loop body)."""
        jax, jnp = self.jax, self.jnp
        nq, nv, nu, ndir = self.nq, self.nv, self.nu, self.ndir
        npad = math.ceil(ndir / chunk) * chunk - ndir
        dt = self.dtype
        step = self._step_world

        def world(m, q, v, w, u, Tq):
            def f(q, v, u):
                o = step(m, q, v, w, u)
                return jnp.concatenate([o[0], o[1]])

            dq = jnp.concatenate([Tq, jnp.zeros((nv + nu + npad, nq), dt)])
            dv = jnp.concatenate([jnp.zeros((nv, nv), dt), jnp.eye(nv, dtype=dt), jnp.zeros((nu + npad, nv), dt)])
            du = jnp.concatenate([jnp.zeros((2 * nv, nu), dt), jnp.eye(nu, dtype=dt), jnp.zeros((npad, nu), dt)])
            cols = jax.lax.map(lambda t: jax.jvp(f, (q, v, u), t)[1], (dq, dv, du), batch_size=chunk)
            return cols[:ndir].T  # (nq + nv, ndir)

        return jax.jit(self._batched(world, extra_axes=(0,)))

    def _jac_args(self, idx):
        """The jac executable's arguments for the worlds idx (an index array; repeats allowed, e.g. padding)."""
        idx = np.asarray(idx)
        T = tangent_map(self.m, self.qpos_host[idx])  # (B, nq, nv)
        Tq = self.jax.device_put(np.ascontiguousarray(np.transpose(T, (0, 2, 1))).astype(self.dtype), self.dev)
        if np.array_equal(idx, np.arange(self.nworld)):
            return self.args + (Tq,)
        j = self.jnp.asarray(idx)
        model = self.model
        if self.dr:
            model = model.tree_replace({"pair_friction": model.pair_friction[j], "body_ipos": model.body_ipos[j]})
        return (model,) + tuple(x[j] for x in self.state) + (Tq,)

    def _plan_jac(self, idx):
        """Compile the jac executable for the worlds idx: (executable, plan, args, {lower_s, compile_s})."""
        B = len(idx)
        args = self._jac_args(idx)
        free, stats = self._free_contiguous()
        budget = None if free is None else self.jac_mem_frac * free
        fwd_mem = self.mem.get("fwd") or {}
        fwd_pw = fwd_mem.get("temp_bytes", 0) / self.nworld if fwd_mem.get("temp_bytes") else None
        plan = dict(directions=self.ndir, worlds=B, budget_bytes=None if budget is None else int(budget),
                    free_contiguous_bytes=free, allocator=stats, fwd_temp_bytes_per_world=fwd_pw, attempts=[])
        cap = self._chunk_cap(B)
        if self.jac_chunk_opt is not None:
            k, plan["rule"] = self.jac_chunk_opt, "jac_chunk option"
        elif budget is None or fwd_pw is None:
            k, plan["rule"] = cap, "speed cap (no memory statistics)"
        else:
            per_world = budget / B / fwd_pw
            k_mem = int(math.floor((per_world - JAC_MEM_PRIMAL) / JAC_MEM_PER_DIRECTION))
            plan.update(rule="min(memory estimate, speed cap)", estimate_chunk=k_mem, speed_cap=cap,
                        estimate_max_worlds_chunk1=int(budget / (fwd_pw * (JAC_MEM_PRIMAL + JAC_MEM_PER_DIRECTION))))
            est1 = B * fwd_pw * (JAC_MEM_PRIMAL + JAC_MEM_PER_DIRECTION)
            if est1 > JAC_EARLY_OOM * budget:  # clearly too large even at K = 1: no compile to confirm it
                raise MemoryError(f"mjx: the forward-mode Jacobian of {B} worlds does not fit: the estimate at chunk 1 "
                                  f"is {est1 / 2**20:.0f} MiB, {budget / 2**20:.0f} MiB available "
                                  f"({self.jac_mem_frac:.2f} x the largest allocation JAX's allocator can serve; not "
                                  f"compiled)")
            k = max(1, min(k_mem, cap))
        times = dict(lower_s=0.0, compile_s=0.0)
        need = None
        for _ in range(JAC_PLAN_ATTEMPTS):
            k = k_tried = self._balanced(k)
            t0 = time.perf_counter()
            lowered = self._jac_fn(k).lower(*args)
            t1 = time.perf_counter()
            ex = lowered.compile()
            t2 = time.perf_counter()
            times["lower_s"] += t1 - t0
            times["compile_s"] += t2 - t1
            ma = _mem_analysis(ex)
            need = None if ma is None else ma["temp_bytes"] + ma["output_bytes"]
            # XLA's autotuning allocates through the same allocator while compiling: with a growing pool
            # (XLA_PYTHON_CLIENT_PREALLOCATE=false) what is left for the temp buffer is known only now
            free_after, st_after = self._free_contiguous()
            if budget is not None and free_after is not None:
                budget = min(budget, self.jac_mem_frac * free_after)
            plan["attempts"].append(dict(chunk=k, need_bytes=need, budget_bytes=None if budget is None else int(budget),
                                         allocator_after_compile=st_after, lower_s=t1 - t0, compile_s=t2 - t1))
            if self.jac_chunk_opt is not None or budget is None or need is None or need <= budget:
                plan["budget_bytes"] = None if budget is None else int(budget)
                plan.update(chunk=k, chunks=math.ceil(self.ndir / k),
                            padded_directions=math.ceil(self.ndir / k) * k - self.ndir, memory=ma, need_bytes=need,
                            triton_fusions=_count_triton(ex))
                return ex, plan, args, times
            if k == 1:
                break
            k = max(1, int(math.floor(k * budget / need * 0.95)))
            del ex
        raise MemoryError(f"mjx: the forward-mode Jacobian of {B} worlds does not fit: chunk {k_tried} needs "
                          f"{need / 2**20:.0f} MiB of temp + output memory, {budget / 2**20:.0f} MiB available "
                          f"({self.jac_mem_frac:.2f} x the largest allocation JAX's allocator can serve)")

    def _free_contiguous(self):
        """(bytes, allocator stats): the largest single allocation JAX's allocator can still serve (an executable's
        temp memory is one buffer): the largest free block of its pool or a new region within bytes_limit and the
        device's free memory (with XLA_PYTHON_CLIENT_PREALLOCATE=false the pool grows by regions that it keeps, so
        after the forward's compile and run it can hold about twice the forward's peak)."""
        stats = self.dev.memory_stats() or {}
        keep = ("bytes_limit", "bytes_in_use", "pool_bytes", "largest_free_block_bytes", "peak_bytes_in_use")
        st = {k: stats.get(k) for k in keep}
        smi = prov._smi(("memory.free",)) or {}
        mib = prov._num(smi.get("memory.free"))
        st["device_free_bytes"] = None if mib is None else int(mib * 2**20)  # all processes (nvidia-smi)
        limit = stats.get("bytes_limit")
        if limit is None:
            return None, st
        pool = stats.get("pool_bytes")
        new_region = limit - (pool if pool is not None else stats.get("bytes_in_use", 0))
        if st["device_free_bytes"] is not None:  # another process on the GPU (not exclusive) takes its share
            new_region = min(new_region, st["device_free_bytes"])
        return max(0, stats.get("largest_free_block_bytes") or 0, new_region), st

    def _chunk_cap(self, B):
        """The largest chunk worth using at B worlds: B x chunk ~ JAC_TARGET_ROWS world-directions per lax.map step
        (larger steps only cost memory), at least JAC_CHUNK_MIN, at most all directions."""
        return max(JAC_CHUNK_MIN, min(self.ndir, math.ceil(JAC_TARGET_ROWS / B)))

    def _balanced(self, k):
        """The smallest chunk with the same number of lax.map steps as k (99 directions: 33 -> 33, 30 -> 25)."""
        n = math.ceil(self.ndir / max(1, k))
        return math.ceil(self.ndir / n)

    def jacobian(self):
        if self._J is not None:
            return self._J
        if self.jac_exec is not None:
            self._J = np.asarray(self.jac_exec(*self.jac_args), np.float64)
            return self._J
        n = 1
        while True:  # untimed: slices of the worlds when the whole batch does not fit
            size = math.ceil(self.nworld / n)
            try:
                ex, plan, args, t = self._plan_jac(np.arange(size))
                break
            except MemoryError:
                if size == 1:
                    raise
                n *= 2
        parts = []
        for a in range(0, self.nworld, size):
            if a:  # the last slice is padded with repeats of the last world
                args = self._jac_args(np.minimum(np.arange(a, a + size), self.nworld - 1))
            parts.append(np.asarray(ex(*args), np.float64)[:min(size, self.nworld - a)])
        self.jac_plan_dump = dict(plan, slices=len(parts), slice_worlds=size, **t)
        self._J = np.concatenate(parts)
        return self._J

    def gradient(self, cot_qpos, cot_qvel):
        return base.vjp_from_jacobian(self.jacobian(), cot_qpos, cot_qvel)

    # ------------------------------------------------------------------ results
    def outputs(self):
        o = self.fwd_exec(*self.args)
        return {k: np.asarray(v, np.float64) for k, v in zip(("qpos", "qvel", "qacc_warmstart"), o)}

    def _diag_fn(self):
        """The control step that also returns, per substep, the contacts within their margin and the constraint rows
        with a non-zero Jacobian row (the rows MuJoCo C would instantiate)."""
        jax, jnp, mjx = self.jax, self.jnp, self.mjx

        def world(m, qpos, qvel, warm, ctrl):
            d = self.d0.replace(qpos=qpos, qvel=qvel, qacc_warmstart=warm, ctrl=ctrl)

            def body(d, _):
                d = mjx.step(m, d)
                c = d._impl.contact
                ncon = jnp.sum(c.dist < c.includemargin) if c.dist.shape[0] else jnp.int32(0)
                nefc = jnp.sum(jnp.any(d._impl.efc_J != 0, axis=1)) if d._impl.efc_J.shape[0] else jnp.int32(0)
                return d, (ncon, nefc)

            d, (ncon, nefc) = jax.lax.scan(body, d, None, length=self.substeps)
            return ncon, nefc

        return jax.jit(self._batched(world))

    def diagnostics(self):
        extra = dict(static_ncon=self.static["ncon"], static_nefc=self.static["nefc"],
                     niter="not exported: MJX's solve() does not write Data.solver_niter",
                     overflow="none possible: MJX allocates every candidate contact of every geom pair statically")
        out = dict(niter_mean=None, niter_max=None, niter_hist=None, overflow=0, unsupported=None, extra=extra)
        if not self.want_diag:
            extra["counts"] = "diagnostics=False"
            return out
        if self._diag is None:
            t0 = time.perf_counter()
            key = self._key(self.nworld)
            if key not in _DIAG_CACHE:
                _DIAG_CACHE[key] = self._diag_fn().lower(*self.args).compile()
            ex = _DIAG_CACHE[key]
            ncon, nefc = (np.asarray(x) for x in ex(*self.args))  # (B, substeps)
            self._diag = dict(ncon=ncon, nefc=nefc, s=time.perf_counter() - t0)
        ncon, nefc = self._diag["ncon"].max(1), self._diag["nefc"].max(1)
        extra.update(counts="per world, the largest count over the substeps", diag_s=round(self._diag["s"], 3))
        out.update(ncon_mean=float(ncon.mean()), ncon_max=int(ncon.max()), nefc_mean=float(nefc.mean()),
                   nefc_max=int(nefc.max()))
        return out

    def memory(self):
        stats = self.dev.memory_stats() or {}
        out = dict(alloc_peak_bytes=stats.get("peak_bytes_in_use"), bytes_limit=stats.get("bytes_limit"),
                   fwd=self.mem.get("fwd"))
        if self.jac_plan is not None:
            out["jac"] = dict(chunk=self.jac_plan.get("chunk"), chunks=self.jac_plan.get("chunks"),
                              memory=self.jac_plan.get("memory"))
        out["note"] = ("alloc_peak_bytes is the JAX allocator's high-water mark of this process (all ops so far); "
                       "fwd / jac: XLA's memory_analysis of each executable")
        return out

    def provenance(self):
        jax = self.jax
        if self.jac_plan is not None:
            self.triton_fusions["jac"] = self.jac_plan.get("triton_fusions")
        out = dict(
            mjx=MJX_VERSION, mujoco=_version("mujoco"), jax=_version("jax"), jaxlib=_version("jaxlib"),
            jax_cuda_plugin=_version("jax-cuda12-plugin") or _version("jax-cuda13-plugin"),
            impl="jax", backend=jax.default_backend(), device_kind=self.dev.device_kind,
            dtype=str(self.dtype), jax_enable_x64=bool(jax.config.jax_enable_x64),
            xla_flags=self.xla_flags or None, xla_triton=self.triton, triton_fusions=dict(self.triton_fusions),
            allocator={k: os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                                                      "XLA_PYTHON_CLIENT_ALLOCATOR")},
            options=P.options(self.m), iterations=self.iterations, substeps=self.substeps,
            control_step="jax.jit(jax.vmap(jax.lax.scan over mjx.step, ctrl held)), compiled ahead of time; outputs "
                         "qpos, qvel, qacc_warmstart",
            dr=("batched mjx.Model: pair_friction and body_ipos with a world axis (vmap in_axes 0), "
                "problem.DR.batched in float32, no mj_setConst") if self.dr else None,
            static_sizes=self.static, compile=self.compile,
            reverse_probe=None if self.reverse is None else dict(self.reverse, cached_in_process=self.reverse_cached),
            jac_rule=dict(target_rows=JAC_TARGET_ROWS, chunk_min=JAC_CHUNK_MIN, mem_primal=JAC_MEM_PRIMAL,
                          mem_per_direction=JAC_MEM_PER_DIRECTION, mem_frac=self.jac_mem_frac,
                          option=self.jac_chunk_opt))
        if self.jac_plan is not None:
            out["jac"] = self.jac_plan
        if getattr(self, "jac_plan_dump", None):
            out["jac_dump"] = self.jac_plan_dump
        return out

    def semantics(self):
        return dict(
            gradient=("forward-mode Jacobian (native): jax.jvp of MJX's control step as executed, i.e. through every "
                      "solver iteration and line search of its lax.while_loop (no implicit differentiation; the "
                      "contact set and other discrete choices of the primal are fixed); gradient(cot) = cot^T J"),
            jacobian=("vmap over worlds of jax.lax.map(batch_size=chunk) over the 2 nv + nu input directions; qpos "
                      "columns along d mj_integratePos(qpos, h e_j) / dh at h = 0"),
            reverse_mode=self._reverse_message(),
            solver_exit=("per world: MJX's lax.while_loop (iteration cap, tolerance on the cost improvement and the "
                         "gradient) under jax.vmap, which runs until every world has exited and keeps each finished "
                         "world's values"),
            precision=f"{self.dtype} (JAX's default)")

    @classmethod
    def check_env(cls):
        import jax
        from mujoco import mjx  # noqa: F401
        backend = jax.default_backend()
        if backend != "gpu":
            raise RuntimeError(f"JAX's default backend is {backend!r}, not 'gpu' (JAX_PLATFORMS="
                               f"{os.environ.get('JAX_PLATFORMS')!r})")
        d = jax.devices()[0]
        return dict(mjx=MJX_VERSION, jax=jax.__version__, backend=backend, device=d.device_kind,
                    devices=len(jax.devices()), x64=bool(jax.config.jax_enable_x64),
                    xla_flags=os.environ.get("XLA_FLAGS") or None)


ADAPTERS = {"mjx": MjxAdapter}
