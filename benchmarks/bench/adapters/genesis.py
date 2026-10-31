"""The Genesis adapter 'genesis': genesis-world 1.4.5 from PyPI (its rigid solver, quadrants 1.3.3), unmodified, through
its public API (gs.Scene, gs.morphs.MJCF, the entity setters and getters, scene.backward). Track A only. Environment
'genesis' (benchmarks/envs/envs.toml, requirements-genesis.txt).

The model. Genesis's MJCF importer compiles the file with MuJoCo and reads the compiled arrays, but it never reads
explicit <pair> contacts, and it imports a geom with contype = conaffinity = 0 as visual only. The mjlab G1 declares all
of its 502 contacts as <pair>s between geoms with contype = conaffinity = 0, so loaded as it is the robot has no
collision geometry at all. The adapter therefore writes an MJCF of the problem's compiled model (MjSpec,
mjcf_from_model) with one edit: contype = conaffinity = 1 on every robot geom, and no <pair> block. Genesis's own
collision filter (its defaults: self collision on, parent-child pairs off, pairs that touch in the neutral
configuration qpos0 off) then gives exactly the 502 explicit pairs: setup checks the pair set against the model's
(pair_set_equal in provenance). What the importer keeps and drops is recorded in provenance()["model"]; the deviations
from the problem in provenance()["deviations"]:
  - condim: Genesis has no per-contact condim; the model's condim-1 (frictionless) pairs become frictional, with the
    friction of the geoms (the larger of the two, as MuJoCo's pair default);
  - solref: the model's (0.02, 1) everywhere; Genesis replaces an MJCF default solref by its own constraint_timeconst,
    which the adapter sets to the model's value (a documented option), so contacts and joint limits use 0.02;
  - no qacc warm start input: every control step starts the constraint solver cold (set_qpos resets it), later
    substeps warm-start from the previous one inside Genesis;
  - Genesis-specific contact generation at its defaults (multi-contact, contact pruning tolerance 0.02, MPR in the
    forward, GJK in the differentiable scene), and Genesis's own inverse weights (recomputed from the model, also
    after the COM shift; MuJoCo keeps invweight0 under the problem's DR);
  - V100 (sm_70) only: quadrants' 'func_solve_decomposed' constraint-solver kernel does not compile for sm_70 (LLVM:
    "Cannot select: intrinsic %llvm.nvvm.activemask"), so the worker sets quadrants' dispatch override
    QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith (Genesis's other solver kernel) before Genesis is
    imported; never on other GPUs.

The problem's options map to RigidOptions: Newton, the cell's iterations, ls_iterations 8, tolerance 1e-8, the
model's cone (pyramidal) and integrator (implicitfast, which Genesis implements "consistent with MuJoCo"), gravity, and
the control step as SimOptions(dt = substeps x timestep, substeps = 4) with the PD targets held: the MJCF's
general / affine actuators become Genesis's position controller (control_dofs_position: force = kp (ctrl - q) - kv
qdot, clamped by the joints' actuatorfrcrange); the corpus ctrl is inside ctrlrange on every state.

Per-world DR through Genesis's batched setters. Torso COM: set_links_COM (body_ipos0[torso] + com per world,
RigidOptions(batch_links_info=True)). Foot friction: Genesis gives a contact the larger of its two geoms' friction x
friction ratio (set_friction_ratio, per world and link), so the floor-sole friction f of every world is set exactly
with ratios on the floor (min(f, floor) / floor) and on the two sole links (min(f, sole) / sole for f <= floor, f /
sole above); the other pairs of the floor keep the model's friction. Unavoidable side effect, recorded per unit
(provenance()["dr"]): the sole capsules' contacts with other robot geoms then use the soles' scaled friction, which
differs from the model's when f > floor friction or f < sole friction.

  fwd    one control step from the loaded states: set_qpos, set_dofs_velocity, control_dofs_position, scene.step()
         (the forward scene: requires_grad=False, the problem's integrator)
  gradient(cot)  Genesis's native backward (SimOptions(requires_grad=True), a second scene of grad_chunk worlds):
         per chunk of states, set the state with qvel as a torch leaf that requires grad, one control step, the
         scalar loss cot_qpos . qpos' + cot_qvel . qvel' from the queried solver state, scene.backward(loss); the
         VJP with respect to the initial qvel is the leaf's .grad. Genesis 1.4.5 differentiates only approximate_
         implicitfast (it raises for any other integrator), so the gradient is that of the approximate integrator's
         step (recorded in semantics()). The qpos-tangent and ctrl blocks are not supported natively
         (gradient_not_supported() records Genesis's own behaviour: set_qpos records no gradient, PD targets raise).
         Genesis aborts a whole backward with GenesisException("Nan grad in qpos or dofs_vel found at step ...") when
         any world's gradient is not finite: such a chunk is retried once, then bisected (each half padded to the
         chunk by repeating its own states) down to single states; a state that still aborts gets a NaN gradient (a
         non-finite state of the scorers). Counts and state indices: gradient_report(). Genesis's forward is
         reproducible bitwise from process to process; its backward is not (the same VJP repeated in one process
         differs on most states, with a heavy tail), and the guard does not fire on the same batches every time.
  bwd, vjp, fwd_train, jac   not supported as the benchmark defines them (a VJP with respect to every input, or the
         full Jacobian); Genesis's partial backward is in the accuracy dumps.

Variants (cell ids genesis~<variant>; options also with run --opt genesis.<key>=<value>):
  default    float32 (gs.init's default precision)
  fp64       precision "64" (labelled: a diagnostic of Genesis's own numerical error, never a headline row)
  approx     integrator approximate_implicitfast in the forward scene too (Genesis's default integrator and the
             gradient's primal; labelled)
Options: precision ("32" | "64"), integrator ("model" | "approximate_implicitfast" | "implicitfast"), grad_chunk
(worlds per differentiable scene, default 512 in float32 and 256 in float64), grad_retries (default 1),
performance_mode (default false: gs.init's default; true: gs.init(performance_mode=True), Genesis's documented
static-array mode, set before the first scene of the process; the same physics).

  python -m benchmarks.bench run --machine <tag> --part accuracy --no-anchor --engines genesis \
      --add-engine genesis=mjx --job acc_genesis --out runs/bench/<dir>
"""

import importlib.metadata
import os
import subprocess
import time

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov
from benchmarks.bench.adapters import base

GENESIS_VERSION = "1.4.5"
QUADRANTS_VERSION = "1.3.3"
SM70_FORCE = "func_solve_body:func_solve_body_monolith"
NAN_GUARD = "Nan grad in qpos or dofs_vel found"
GRAD_CHUNK = {"32": 512, "64": 256}
NOT_SUPPORTED_OP = (f"Genesis {GENESIS_VERSION}'s backward has no gradient path for the joint positions (set_qpos) "
                    "or the PD targets (control_dofs_position), so it gives no VJP or Jacobian with respect to every "
                    "input of the control step; its gradient with respect to the initial qvel is in the accuracy "
                    "dumps (gradient())")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _compute_cap():
    """The compute capability of the first visible GPU as (major, minor), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    idx = 0
    if vis and vis.split(",")[0].strip().isdigit():
        idx = int(vis.split(",")[0])
    if not out:
        return None
    v = out[min(idx, len(out) - 1)]
    try:
        a, b = v.split(".")
        return int(a), int(b)
    except ValueError:
        return None


def _dispatch_override():
    """Set quadrants' dispatch override on sm_70 before Genesis / quadrants are imported (module docstring); returns
    the value in effect."""
    cur = os.environ.get("QD_PERFDISPATCH_FORCE")
    cc = _compute_cap()
    if cc is not None and cc[0] == 7 and cc[1] == 0 and cur is None:
        import sys
        if "quadrants" in sys.modules:
            raise RuntimeError("genesis: QD_PERFDISPATCH_FORCE must be set before quadrants is imported (sm_70)")
        os.environ["QD_PERFDISPATCH_FORCE"] = SM70_FORCE
    return os.environ.get("QD_PERFDISPATCH_FORCE")


# ------------------------------------------------------------------------------------------------- the model

def mjcf_from_model(m, collide_robot=True, pairs=False):
    """An MJCF string of a compiled MjModel (MjSpec; mj_saveLastXML needs an XML-loaded model). Covers what the mjlab
    G1 uses: bodies with explicit inertials, free / hinge joints (armature, damping, frictionloss, limits,
    actuatorfrcrange), primitive geoms, explicit pairs and excludes (pairs=True), joint actuators, keyframes,
    options. collide_robot: contype = conaffinity = 1 on the robot geoms (bodies other than the world) that have
    contype = conaffinity = 0."""
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = "g1_train"
    o = spec.option
    for f in ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance"):
        setattr(o, f, getattr(m.opt, f))
    o.gravity = m.opt.gravity.copy()
    o.wind = m.opt.wind.copy()
    o.magnetic = m.opt.magnetic.copy()
    o.density, o.viscosity = m.opt.density, m.opt.viscosity
    o.integrator, o.cone, o.jacobian = int(m.opt.integrator), int(m.opt.cone), int(m.opt.jacobian)
    o.solver, o.iterations, o.ls_iterations = int(m.opt.solver), int(m.opt.iterations), int(m.opt.ls_iterations)
    o.noslip_iterations, o.ccd_iterations = int(m.opt.noslip_iterations), int(m.opt.ccd_iterations)
    o.disableflags, o.enableflags = int(m.opt.disableflags), int(m.opt.enableflags)
    o.o_margin = m.opt.o_margin
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    spec.compiler.autolimits = True
    spec.compiler.balanceinertia = False
    spec.compiler.degree = False
    T = mujoco.mjtObj
    nm = lambda t, i: mujoco.mj_id2name(m, t, i)  # noqa: E731
    lim = lambda x: mujoco.mjtLimited.mjLIMITED_TRUE if x else mujoco.mjtLimited.mjLIMITED_FALSE  # noqa: E731
    gname = [nm(T.mjOBJ_GEOM, g) or f"geom_{g}" for g in range(m.ngeom)]
    bodies = {0: spec.worldbody}
    for b in range(1, m.nbody):
        bb = bodies[int(m.body_parentid[b])].add_body(name=nm(T.mjOBJ_BODY, b), pos=m.body_pos[b], quat=m.body_quat[b])
        bb.mass, bb.ipos, bb.iquat = float(m.body_mass[b]), m.body_ipos[b], m.body_iquat[b]
        bb.inertia, bb.explicitinertial = m.body_inertia[b], True
        bodies[b] = bb
    for j in range(m.njnt):
        b, d, t = bodies[int(m.jnt_bodyid[j])], int(m.jnt_dofadr[j]), int(m.jnt_type[j])
        if t == mujoco.mjtJoint.mjJNT_FREE:
            b.add_freejoint(name=nm(T.mjOBJ_JOINT, j))
            continue
        jj = b.add_joint(name=nm(T.mjOBJ_JOINT, j), type=t, pos=m.jnt_pos[j], axis=m.jnt_axis[j])
        jj.range, jj.limited = m.jnt_range[j], lim(m.jnt_limited[j])
        jj.armature = float(m.dof_armature[d])
        jj.damping = np.r_[m.dof_damping[d], m.dof_dampingpoly[d]]
        jj.frictionloss = float(m.dof_frictionloss[d])
        jj.stiffness = np.r_[m.jnt_stiffness[j], m.jnt_stiffnesspoly[j]]
        jj.springref = float(m.qpos_spring[m.jnt_qposadr[j]])
        jj.ref = float(m.qpos0[m.jnt_qposadr[j]])
        jj.margin = float(m.jnt_margin[j])
        jj.actfrcrange, jj.actfrclimited = m.jnt_actfrcrange[j], lim(m.jnt_actfrclimited[j])
        jj.solref_limit, jj.solimp_limit = m.jnt_solref[j], m.jnt_solimp[j]
        jj.solref_friction, jj.solimp_friction = m.dof_solref[d], m.dof_solimp[d]
    for g in range(m.ngeom):
        gg = bodies[int(m.geom_bodyid[g])].add_geom(name=gname[g], type=int(m.geom_type[g]), size=m.geom_size[g],
                                                    pos=m.geom_pos[g], quat=m.geom_quat[g])
        ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
        if collide_robot and m.geom_bodyid[g] > 0 and ct == 0 and ca == 0:
            ct, ca = 1, 1
        gg.contype, gg.conaffinity, gg.condim = ct, ca, int(m.geom_condim[g])
        gg.priority, gg.friction, gg.solmix = int(m.geom_priority[g]), m.geom_friction[g], float(m.geom_solmix[g])
        gg.solref, gg.solimp = m.geom_solref[g], m.geom_solimp[g]
        gg.margin, gg.gap = float(m.geom_margin[g]), float(m.geom_gap[g])
        gg.group, gg.rgba = int(m.geom_group[g]), m.geom_rgba[g]
        gg.mass = gg.density = 0.0
    if pairs:
        for p in range(m.npair):
            pp = spec.add_pair(geomname1=gname[int(m.pair_geom1[p])], geomname2=gname[int(m.pair_geom2[p])])
            pp.condim, pp.friction = int(m.pair_dim[p]), m.pair_friction[p]
            pp.solref, pp.solreffriction, pp.solimp = m.pair_solref[p], m.pair_solreffriction[p], m.pair_solimp[p]
            pp.margin, pp.gap = float(m.pair_margin[p]), float(m.pair_gap[p])
        for e in range(m.nexclude):
            s = int(m.exclude_signature[e])
            spec.add_exclude(bodyname1=nm(T.mjOBJ_BODY, (s >> 16) - 1), bodyname2=nm(T.mjOBJ_BODY, (s & 0xFFFF) - 1))
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise NotImplementedError("genesis: only joint actuators are converted")
        aa = spec.add_actuator(name=nm(T.mjOBJ_ACTUATOR, a), target=nm(T.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])))
        aa.trntype = mujoco.mjtTrn.mjTRN_JOINT
        aa.dyntype, aa.gaintype = int(m.actuator_dyntype[a]), int(m.actuator_gaintype[a])
        aa.biastype = int(m.actuator_biastype[a])
        aa.gainprm, aa.biasprm, aa.dynprm = m.actuator_gainprm[a], m.actuator_biasprm[a], m.actuator_dynprm[a]
        aa.gear, aa.ctrlspec = m.actuator_gear[a], int(m.actuator_ctrlspec[a])
        aa.ctrlrange, aa.forcerange = m.actuator_ctrlrange[a], m.actuator_forcerange[a]
        aa.ctrllimited, aa.forcelimited = lim(m.actuator_ctrllimited[a]), lim(m.actuator_forcelimited[a])
    for k in range(m.nkey):
        spec.add_key(name=nm(T.mjOBJ_KEY, k) or f"key{k}", qpos=m.key_qpos[k], ctrl=m.key_ctrl[k])
    spec.compile()
    return spec.to_xml()


MODEL_FIELDS = ("body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
                "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
                "jnt_actfrcrange", "jnt_actfrclimited", "jnt_solref", "jnt_solimp", "dof_armature", "dof_damping",
                "dof_frictionloss", "geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_condim",
                "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "actuator_gainprm",
                "actuator_biasprm", "actuator_ctrlrange", "actuator_trnid", "actuator_gear", "qpos0")


def check_conversion(m, xml):
    """Compile the written MJCF and compare it with the source model: the physics arrays must agree (float32
    printing tolerance); the only differences are the robot geoms' contype / conaffinity and the missing pairs."""
    import mujoco
    m2 = mujoco.MjModel.from_xml_string(xml)
    worst = {}
    for f in MODEL_FIELDS:
        a, b = np.asarray(getattr(m, f), np.float64), np.asarray(getattr(m2, f), np.float64)
        if a.shape != b.shape:
            raise RuntimeError(f"genesis: the written MJCF differs from the model in {f} {a.shape} vs {b.shape}")
        worst[f] = float(np.max(np.abs(a - b))) if a.size else 0.0
    bad = {f: v for f, v in worst.items() if v > 1e-6 * max(1.0, float(np.max(np.abs(getattr(m, f)))))}
    if bad:
        raise RuntimeError(f"genesis: the written MJCF differs from the model: {bad}")
    robot = m.geom_bodyid > 0
    return dict(max_abs_diff=max(worst.values()), fields_compared=len(MODEL_FIELDS),
                robot_geoms_made_colliding=int(np.sum(robot & (m.geom_contype == 0) & (m.geom_conaffinity == 0))),
                pairs_written=int(m2.npair), pairs_in_model=int(m.npair))


def model_record(m):
    """What Genesis's importer keeps and drops for this model (module docstring)."""
    import mujoco
    lost_condim = int(np.sum(m.pair_dim == 1)) if m.npair else 0
    return dict(
        importer="gs.morphs.MJCF(file=<MJCF string>) of the problem's compiled model (mjcf_from_model)",
        edit=("contype = conaffinity = 1 on the robot geoms, no <pair> block: Genesis does not read <pair> and imports "
              "contype = conaffinity = 0 geoms as visual only"),
        kept=("bodies, inertials, free and hinge joints (limits, armature), the general / affine actuators as the PD "
              "controller's kp = gainprm[0], kv = -biasprm[2], force limits from the joints' actuatorfrcrange, geom "
              "friction, solimp"),
        dropped=dict(pairs=int(m.npair), pair_condim1=lost_condim,
                     pair_condim1_note="these pairs are frictionless in MuJoCo and frictional in Genesis",
                     options="the MJCF <option> (Genesis takes the time step and the solver from SimOptions / "
                             "RigidOptions, set from the problem)"),
        cone=mujoco.mjtCone(int(m.opt.cone)).name, integrator=mujoco.mjtIntegrator(int(m.opt.integrator)).name)


# ------------------------------------------------------------------------------------------------- the adapter

class GenesisAdapter(base.Adapter):
    id = "genesis"
    label = f"Genesis {GENESIS_VERSION} (genesis-world, rigid solver)"
    native = "native"
    device = "gpu"
    tracks = ("A",)
    variants = {"default": {}, "fp64": {"precision": "64"}, "approx": {"integrator": "approximate_implicitfast"}}

    def setup(self, problem, states, nworld, iterations, precision="32", integrator="model", grad_chunk=None,
              grad_retries=1, performance_mode=False):
        if problem.id != "mjlab":
            raise base.NotSupported("genesis: Track A (the mjlab G1) only")
        precision = str(precision)
        if precision not in ("32", "64"):
            raise ValueError(f"genesis: precision must be '32' or '64', got {precision!r}")
        if integrator not in ("model", "approximate_implicitfast", "implicitfast"):
            raise ValueError(f"genesis: unknown integrator {integrator!r}")
        self.precision, self.integrator_opt = precision, integrator
        self.grad_chunk = int(grad_chunk) if grad_chunk not in (None, "", "auto") else GRAD_CHUNK[precision]
        self.grad_retries = int(grad_retries)
        self.dispatch_force = _dispatch_override()
        t0 = time.perf_counter()
        import mujoco
        import torch
        import genesis as gs
        import quadrants as qd
        self.mujoco, self.torch, self.gs, self.qd = mujoco, torch, gs, qd
        vers = dict(genesis=_version("genesis-world"), quadrants=_version("quadrants"))
        if vers["genesis"] != GENESIS_VERSION or vers["quadrants"] != QUADRANTS_VERSION:
            raise RuntimeError(f"genesis: needs genesis-world {GENESIS_VERSION} with quadrants {QUADRANTS_VERSION}, "
                               f"found {vers}: use the genesis environment (benchmarks/envs/setup.sh --envs genesis)")
        if getattr(gs, "_initialized", False) and getattr(gs, "tc_float", None) is not None and \
                str(gs.tc_float) != ("torch.float32" if precision == "32" else "torch.float64"):
            gs.destroy()
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu, precision=precision, logging_level="warning",
                    performance_mode=str(performance_mode).lower() in ("1", "true", "yes"))
        if torch.device(gs.device).type != "cuda":
            raise RuntimeError(f"genesis: device {gs.device}, not CUDA; refusing a silent CPU run")
        import_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = P.load_model(problem, iterations)
        states.check(m.nq, m.nv, m.nu, problem.dr)
        self.m, self.problem, self.nworld, self.iterations = m, problem, int(nworld), int(iterations)
        self.substeps = int(problem.substeps)
        self.xml = mjcf_from_model(m)
        self.conversion = check_conversion(m, self.xml)
        self.model_rec = model_record(m)
        solref = np.unique(np.concatenate([m.geom_solref, m.jnt_solref]), axis=0)
        if len(solref) != 1 or solref[0][1] != 1.0:
            raise NotImplementedError(f"genesis: one solref (timeconst, 1) for every geom and joint expected, got "
                                      f"{solref.tolist()}")
        self.timeconst = float(solref[0][0])
        mi = mujoco.mjtIntegrator(int(m.opt.integrator)).name
        self.integrator = {"mjINT_IMPLICITFAST": "implicitfast", "mjINT_EULER": "Euler"}.get(mi)
        if integrator != "model":
            self.integrator = integrator
        if self.integrator is None:
            raise NotImplementedError(f"genesis: no Genesis integrator for the model's {mi}")
        if int(m.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
            raise NotImplementedError("genesis: the adapter maps the pyramidal cone only")
        self.dr = problem.dr
        self.fwd_scene, self.fwd_robot = self._scene(self.nworld, requires_grad=False)
        build_s = time.perf_counter() - t0
        self.pair_check = self._check_pairs(self.fwd_scene)

        t0 = time.perf_counter()
        s = states.inputs()
        dev, f = gs.device, gs.tc_float
        self.qpos = torch.tensor(np.asarray(s["qpos"], np.float64), dtype=f, device=dev)
        self.qvel = torch.tensor(np.asarray(s["qvel"], np.float64), dtype=f, device=dev)
        self.ctrl = torch.tensor(np.asarray(s["ctrl"], np.float64), dtype=f, device=dev)
        self.index = np.asarray(states.index)
        self.friction = None if states.friction is None else np.asarray(states.friction, np.float64)
        self.com = None if states.com is None else np.asarray(states.com, np.float64)
        self._dr_tensors()
        self._apply_com(self.fwd_robot, np.arange(self.nworld), self.nworld)
        self._apply_friction(self.fwd_robot, np.arange(self.nworld), self.nworld)
        self.sync()
        load_s = time.perf_counter() - t0
        self.grad_scene = self.grad_robot = self.grad_n = None
        self.report = dict(batches=0, aborted_batches=0, recovered_on_retry=0, bisect_runs=0, runs=0,
                           aborted_states={}, guard=NAN_GUARD, grad_chunk=self.grad_chunk, retries=self.grad_retries)
        self._outs, self._ncon, self._probes = None, None, None
        self.compile_note = ("Genesis compiles its kernels on first use (scene.build runs one step); its kernel cache "
                             "(the quadrants offline cache) makes later processes faster")
        return dict(build_s=build_s, load_s=load_s, import_s=import_s)

    # ------------------------------------------------------------------ scenes
    def _rigid_options(self, requires_grad):
        gs, prob = self.gs, self.problem
        integ = "approximate_implicitfast" if requires_grad else self.integrator
        return dict(constraint_solver=gs.constraint_solver.Newton, iterations=self.iterations,
                    ls_iterations=int(prob.ls_iterations), tolerance=float(prob.tolerance),
                    integrator=getattr(gs.integrator, integ), friction_cone=gs.friction_cone.pyramidal,
                    constraint_timeconst=self.timeconst, batch_links_info=bool(self.dr),
                    gravity=tuple(float(x) for x in self.m.opt.gravity))

    def _scene(self, n_envs, requires_grad):
        gs = self.gs
        dt = self.substeps * float(self.m.opt.timestep)
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=self.substeps, requires_grad=requires_grad),
                         rigid_options=gs.options.RigidOptions(**self._rigid_options(requires_grad)),
                         show_viewer=False)
        robot = scene.add_entity(gs.morphs.MJCF(file=self.xml))
        scene.build(n_envs=n_envs)
        if robot.n_qs != self.m.nq or robot.n_dofs != self.m.nv:
            raise RuntimeError(f"genesis: the entity has {robot.n_qs} qpos / {robot.n_dofs} dofs, the model "
                               f"{self.m.nq} / {self.m.nv}")
        names = [ln.name for ln in robot.links]
        mj_names = [self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.m.nbody)]
        if names != ["world"] + mj_names[1:]:
            raise RuntimeError("genesis: the entity's links are not the model's bodies in order")
        if [g.link.idx for g in robot.geoms] != [int(b) for b in self.m.geom_bodyid]:
            raise RuntimeError("genesis: the entity's collision geoms are not the model's geoms in order")
        self.act_dofs = list(range(self.m.nv - self.m.nu, self.m.nv))
        return scene, robot

    def _check_pairs(self, scene):
        cl = scene.sim.rigid_solver.collider
        gen = {tuple(sorted(map(int, p))) for p in np.asarray(cl._valid_collision_pairs).tolist()}
        mj = {tuple(sorted((int(a), int(b)))) for a, b in zip(self.m.pair_geom1, self.m.pair_geom2)}
        return dict(genesis_pairs=len(gen), model_pairs=len(mj), pair_set_equal=gen == mj,
                    only_genesis=sorted(gen - mj)[:20], only_model=sorted(mj - gen)[:20])

    # ------------------------------------------------------------------ DR
    def _dr_tensors(self):
        """Per-world friction ratios of (floor link, sole links) and torso COM (module docstring)."""
        m, mujoco = self.m, self.mujoco
        self.dr_rec = None
        if not self.dr:
            return
        dr = P.DR(m, self.problem)
        foot_geoms = np.unique(np.r_[m.pair_geom1[dr.foot], m.pair_geom2[dr.foot]])
        floor = [g for g in foot_geoms if m.geom_bodyid[g] == 0]
        soles = [g for g in foot_geoms if m.geom_bodyid[g] != 0]
        if len(floor) != 1:
            raise NotImplementedError("genesis: the foot pairs must share one world geom (the floor)")
        sole_links = sorted({int(m.geom_bodyid[g]) for g in soles})
        on_links = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sole_links]
        if sorted(on_links) != sorted(soles):
            raise NotImplementedError("genesis: the sole links carry geoms outside the foot pairs")
        F = float(m.geom_friction[floor[0], 0])
        S = np.unique(m.geom_friction[soles, 0])
        if len(S) != 1 or not S[0] <= F:
            raise NotImplementedError("genesis: the soles must share one friction no larger than the floor's")
        S = float(S[0])
        f = self.friction
        r_floor = np.minimum(f, F) / F
        r_sole = np.where(f <= F, np.minimum(f, S), f) / S
        self.dr_links = [0] + sole_links
        self.dr_ratio = np.stack([r_floor] + [r_sole] * len(sole_links), 1)
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.problem.torso_body)
        self.dr_torso = int(torso)
        self.dr_com = m.body_ipos[torso][None].astype(np.float64) + self.com
        others = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom)} - set(self.dr_links))
        self.dr_rec = dict(
            friction=("set_friction_ratio per world on the floor link and the sole links: floor-sole contact friction "
                      "= max(floor x r_floor, sole x r_sole) = the corpus friction exactly"),
            floor_friction=F, sole_friction=S, sole_links=sole_links, other_links=len(others),
            worlds=int(len(f)), worlds_friction_above_floor=int(np.sum(f > F)),
            worlds_friction_below_sole=int(np.sum(f < S)),
            side_effect=("in those worlds the sole capsules' contacts with other robot geoms use the scaled sole "
                         "friction (sole-sole: min(f, sole) or f instead of sole; sole-other: f instead of the "
                         "larger geom friction when f > floor friction)"),
            com=("set_links_COM per world: body_ipos0[torso] + com (RigidOptions(batch_links_info=True)); Genesis "
                 "recomputes its inverse weights from the shifted COM"),
            torso_link=self.problem.torso_body)

    def _apply_com(self, robot, rows, n_envs):
        if not self.dr:
            return
        torch, gs = self.torch, self.gs
        c = torch.tensor(self.dr_com[np.resize(rows, n_envs)], dtype=gs.tc_float, device=gs.device)
        robot.set_links_COM(c[:, None, :], [self.dr_torso])

    def _apply_friction(self, robot, rows, n_envs):
        if not self.dr:
            return
        torch, gs = self.torch, self.gs
        r = torch.tensor(self.dr_ratio[np.resize(rows, n_envs)], dtype=gs.tc_float, device=gs.device)
        robot.set_friction_ratio(r, self.dr_links)

    # ------------------------------------------------------------------ ops
    def _fwd(self):
        r = self.fwd_robot
        r.set_qpos(self.qpos, zero_velocity=False)
        r.set_dofs_velocity(self.qvel)
        r.control_dofs_position(self.ctrl, self.act_dofs)
        self.fwd_scene.step(update_visualizer=False)

    def ops(self):
        return dict(fwd=self._fwd)

    def not_supported(self):
        return dict(fwd_train=NOT_SUPPORTED_OP, bwd=NOT_SUPPORTED_OP, vjp=NOT_SUPPORTED_OP, jac=NOT_SUPPORTED_OP)

    def sync(self):
        self.qd.sync()
        self.torch.cuda.synchronize()

    def outputs(self):
        if self._outs is None:
            self._fwd()
            self.sync()
            r = self.fwd_robot
            self._outs = dict(qpos=r.get_qpos().double().cpu().numpy(),
                              qvel=r.get_dofs_velocity().double().cpu().numpy())
            try:
                c = r.get_contacts()
                vm = c.get("valid_mask")
                self._ncon = None if vm is None else vm.sum(-1).cpu().numpy()
            except Exception as e:  # noqa: BLE001  (a diagnostic only)
                self._ncon = f"{type(e).__name__}: {e}"[:200]
        return dict(self._outs)

    def diagnostics(self):
        out = dict(overflow=0, extra=dict(contacts="Genesis's contacts of the last substep (get_contacts)"))
        if isinstance(self._ncon, np.ndarray):
            out.update(ncon_mean=float(self._ncon.mean()), ncon_max=int(self._ncon.max()))
        elif self._ncon is not None:
            out["extra"]["contacts_error"] = self._ncon
        return out

    # ------------------------------------------------------------------ gradient
    def _ensure_grad_scene(self):
        if self.grad_scene is None:
            t0 = time.perf_counter()
            n = self.grad_n = min(self.grad_chunk, self.nworld)
            self.grad_scene, self.grad_robot = self._scene(n, requires_grad=True)
            self._apply_com(self.grad_robot, np.arange(n), n)  # replaced per chunk
            self.report["grad_scene_build_s"] = time.perf_counter() - t0
            self.report["grad_pair_check"] = self._check_pairs(self.grad_scene)
        return self.grad_scene

    def _run(self, rows, cq, cv):
        """VJP w.r.t. the initial qvel of the worlds rows (len <= chunk; padded with repeats of rows): (len(rows), nv)
        array, or None when Genesis's NaN guard aborted the backward."""
        torch, gs = self.torch, self.gs
        scene, robot = self.grad_scene, self.grad_robot
        n = self.grad_n
        sel = np.resize(rows, n)
        t = torch.as_tensor(sel, device=gs.device)
        self.report["runs"] += 1
        scene.reset()
        self._apply_com(robot, sel, n)
        self._apply_friction(robot, sel, n)
        robot.set_qpos(self.qpos[t], zero_velocity=False)
        v0 = self.qvel[t].clone().requires_grad_(True)
        robot.set_dofs_velocity(v0)
        robot.control_dofs_position(self.ctrl[t], self.act_dofs)
        scene.step(update_visualizer=False)
        st = scene.sim.rigid_solver.get_state()
        loss = (cq[t] * st.qpos).sum() + (cv[t] * st.dofs_vel).sum()
        try:
            scene.backward(loss)
        except gs.GenesisException as e:
            if NAN_GUARD not in str(e):
                raise
            self.report["last_guard_message"] = str(e)[:300]
            return None
        if v0.grad is None:
            raise RuntimeError("genesis: no gradient reached the initial qvel")
        return v0.grad.detach().double().cpu().numpy()[:len(rows)]

    def _solve(self, rows, cq, cv, out, aborted, top=False):
        """Run rows; on Genesis's NaN guard retry, then bisect down to single states (module docstring)."""
        g = self._run(rows, cq, cv)
        if g is None and top:
            self.report["aborted_batches"] += 1
        tries = 0
        while g is None and tries < self.grad_retries:
            tries += 1
            g = self._run(rows, cq, cv)
            if g is not None:
                self.report["recovered_on_retry"] += 1
        if g is not None:
            out[rows] = g
            return
        if len(rows) == 1:
            aborted.append(int(rows[0]))
            return
        h = len(rows) // 2
        for part in (rows[:h], rows[h:]):
            self.report["bisect_runs"] += 1
            self._solve(part, cq, cv, out, aborted)

    def gradient(self, cot_qpos, cot_qvel):
        torch, gs = self.torch, self.gs
        self._ensure_grad_scene()
        n = self.grad_n
        cq = torch.tensor(np.asarray(cot_qpos, np.float64), dtype=gs.tc_float, device=gs.device)
        cv = torch.tensor(np.asarray(cot_qvel, np.float64), dtype=gs.tc_float, device=gs.device)
        out = np.full((self.nworld, self.m.nv), np.nan)
        aborted = []
        c = len(self.report["aborted_states"])
        t0 = time.perf_counter()
        for a in range(0, self.nworld, n):
            self.report["batches"] += 1
            self._solve(np.arange(a, min(a + n, self.nworld)), cq, cv, out, aborted, top=True)
        self.report["aborted_states"][f"cot{c}"] = [int(self.index[i]) for i in aborted]
        self.report.setdefault("seconds", []).append(round(time.perf_counter() - t0, 3))
        return dict(qvel=out)

    def gradient_not_supported(self):
        """{block: message} for the input blocks Genesis does not differentiate, from Genesis itself (run once after
        the gradients: a failed backward can leave entries on Genesis's input tape)."""
        if self._probes is None:
            self._probes = self._probe_inputs()
        return dict(self._probes)

    def _probe_inputs(self):
        torch, gs = self.torch, self.gs
        self._ensure_grad_scene()
        scene, robot = self.grad_scene, self.grad_robot
        n = self.grad_n
        t = torch.as_tensor(np.resize(np.arange(self.nworld), n), device=gs.device)
        out = {}

        def attempt(name, setter):
            scene.reset()
            self._apply_friction(robot, np.resize(np.arange(self.nworld), n), n)
            try:
                leaf = setter()
                scene.step(update_visualizer=False)
                st = scene.sim.rigid_solver.get_state()
                scene.backward((st.dofs_vel ** 2).sum())
                g = leaf.grad
                return ("no gradient path: the leaf's .grad is None after scene.backward" if g is None else
                        "a gradient arrived (not expected)")
            except Exception as e:  # noqa: BLE001  (the engine's own message is the result)
                return f"raises {type(e).__name__}: {prov.scrub(str(e))[:300]}"

        def qpos_leaf():
            q = self.qpos[t].clone().requires_grad_(True)
            robot.set_qpos(q, zero_velocity=False)
            robot.set_dofs_velocity(self.qvel[t])
            robot.control_dofs_position(self.ctrl[t], self.act_dofs)
            return q

        def ctrl_leaf():
            u = self.ctrl[t].clone().requires_grad_(True)
            robot.set_qpos(self.qpos[t], zero_velocity=False)
            robot.set_dofs_velocity(self.qvel[t])
            robot.control_dofs_position(u, self.act_dofs)
            return u

        out["qpos"] = (f"Genesis {GENESIS_VERSION}: set_qpos is not a differentiable input; with a qpos tensor that "
                       f"requires grad: {attempt('qpos', qpos_leaf)}. Its differentiable state setters are "
                       "set_dofs_velocity and the base link's set_pos / set_quat; the base link of this entity is "
                       "the world body (it holds the floor), so no qpos column has a gradient")
        out["ctrl"] = (f"Genesis {GENESIS_VERSION}: control_dofs_position with targets that require grad: "
                       f"{attempt('ctrl', ctrl_leaf)}")
        return out

    def gradient_report(self):
        r = dict(self.report)
        r["aborted_state_count"] = {k: len(v) for k, v in r["aborted_states"].items()}
        return r

    # ------------------------------------------------------------------ records
    def provenance(self):
        gs = self.gs
        return dict(
            genesis=_version("genesis-world"), quadrants=_version("quadrants"), torch=_version("torch"),
            torch_cuda=self.torch.version.cuda, mujoco=_version("mujoco"), precision=self.precision,
            backend=str(getattr(gs.backend, "name", gs.backend)), device=str(gs.device),
            performance_mode=not getattr(gs, "use_ndarray", True),
            dispatch_override=self.dispatch_force,
            options=P.options(self.m), iterations=self.iterations, substeps=self.substeps,
            rigid_options={k: str(getattr(v, "name", v)) for k, v in self._rigid_options(False).items()},
            rigid_options_grad={k: str(getattr(v, "name", v)) for k, v in self._rigid_options(True).items()},
            sim_options=dict(dt=self.substeps * float(self.m.opt.timestep), substeps=self.substeps),
            control_step=("set_qpos, set_dofs_velocity, control_dofs_position(ctrl) on the actuated dofs, one "
                          "scene.step() of `substeps` substeps"),
            model=dict(self.model_rec, conversion=self.conversion, pair_check=self.pair_check),
            dr=self.dr_rec, grad_chunk=self.grad_chunk, grad_retries=self.grad_retries,
            deviations=self.deviations(), compile=self.compile_note)

    def deviations(self):
        d = [
            "model: robot geoms contype = conaffinity = 1 and no <pair> block (Genesis reads no <pair>); Genesis's "
            "default collision filter gives the same 502 pairs (provenance model.pair_check)",
            f"condim: {self.model_rec['dropped']['pair_condim1']} condim-1 (frictionless) pairs are frictional in "
            "Genesis (no per-contact condim)",
            f"solref: Genesis's constraint_timeconst = {self.timeconst:g} (the model's solref) instead of a per-geom "
            "solref (Genesis drops an MJCF default solref in favour of its own time constant)",
            "no qacc warm start input (cold constraint solve at every control step)",
            f"tolerance {self.problem.tolerance:g} passed to Genesis as is (float{self.precision})",
            "contact generation: Genesis's (multi-contact, contact_pruning_tolerance 0.02; MPR in the forward scene, "
            "GJK in the differentiable one), inverse weights recomputed by Genesis",
            "gradient: integrator approximate_implicitfast (the only one Genesis differentiates), w.r.t. the initial "
            "qvel only",
        ]
        if self.integrator != "implicitfast":
            d.append(f"forward integrator {self.integrator} (variant), not the model's implicitfast")
        if self.dr_rec:
            d.append("DR: foot friction through per-world friction ratios on the floor and sole links "
                     f"({self.dr_rec['worlds_friction_above_floor']} worlds above the floor friction, "
                     f"{self.dr_rec['worlds_friction_below_sole']} below the sole friction get scaled sole friction "
                     "on their sole-robot contacts too); COM through set_links_COM")
        if self.dispatch_force:
            d.append(f"QD_PERFDISPATCH_FORCE={self.dispatch_force} (quadrants' dispatch override; set on sm_70, where "
                     "the decomposed constraint-solver kernel does not compile)")
        return d

    def semantics(self):
        return dict(
            gradient=("Genesis's native reverse mode (SimOptions(requires_grad=True), scene.backward) of one control "
                      "step with integrator approximate_implicitfast (Genesis differentiates no other), with respect "
                      "to the initial qvel only; the qpos-tangent and ctrl blocks are not supported"),
            forward=f"integrator {self.integrator}, float{self.precision}",
            solver_exit="Genesis's Newton solver: per world early exit at the tolerance, batch until all exit",
            precision=f"float{self.precision}")

    def memory(self):
        torch = self.torch
        return dict(torch_alloc_peak_bytes=int(torch.cuda.max_memory_allocated()),
                    note="Genesis's quadrants arrays are outside torch's allocator; the worker adds the process's "
                         "NVML memory")

    def close(self):
        for s in (self.grad_scene, self.fwd_scene):
            try:
                if s is not None:
                    s.destroy()
            except Exception:  # noqa: BLE001
                pass

    @classmethod
    def check_env(cls):
        force = _dispatch_override()
        import torch
        import genesis  # noqa: F401
        return dict(genesis=_version("genesis-world"), quadrants=_version("quadrants"), torch=torch.__version__,
                    torch_cuda=torch.version.cuda, cuda=torch.cuda.is_available(),
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    dispatch_override=force)


ADAPTERS = {"genesis": GenesisAdapter}
